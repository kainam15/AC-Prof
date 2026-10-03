"""离线接口验证可指定扩展测试，避免把任务族等同于运行时。"""
import io
import json
import os
import subprocess
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.check_runtime import ONNX_ENVIRONMENT_CHECK, main


class TestRuntimeCheckSelection:
    @pytest.fixture(autouse=True)
    def offline_test_tools(self, monkeypatch):
        monkeypatch.setattr('scripts.check_runtime.prepare_test_wheels', lambda *args: None)

    def test_test_tools_are_installed_offline_outside_the_runtime_image(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as directory, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run',
                 return_value=subprocess.CompletedProcess([], 0)) as run:
            assert main(['--profile', 'nlp-transformers560-cpu', '--output-dir', directory]) == 0
        command = next(call.args[0] for call in run.call_args_list
                       if 'scripts/run_tests.py' in call.args[0])
        assert '/tmp/acprof-tests/bin/python' in command
        assert any('--no-index' in argument and '--require-hashes' in argument for argument in command)
        assert command[command.index('--network') + 1] == 'none'

    @pytest.mark.parametrize('profile_id,task,trust', (('nlp-transformers560-cpu', 'text-generation', False), ('custom-multimodal-cpu', 'audio-text-to-text', True)))
    def test_container_uses_selected_profile_and_adapter_loading_policy(self, profile_id, task, trust):
        from acprof.container.load_policy import registered_policy
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as directory:
            def run(command, **kwargs):
                if 'scripts/run_tests.py' in command:
                    environment = dict(command[i + 1].split('=', 1)
                                       for i, part in enumerate(command) if part == '-e')
                    with patch.dict(os.environ, environment, clear=True):
                        profile, _, allowed = registered_policy(task, 'transformers_pipeline', 'cpu')
                    assert (profile.profile_id) == (profile_id)
                    assert (allowed) == (trust)
                return subprocess.CompletedProcess(command, 0)
            with patch('scripts.check_runtime.prepare_environment_image', return_value=image), patch(
                'scripts.check_runtime.subprocess.run', side_effect=run,
            ):
                assert (main(['--profile', profile_id, '--test-pattern', 'test_fixture.py',
                                       '--output-dir', directory])) == (0)

    @pytest.mark.parametrize('error_case', range(2), ids=["TypeError('invalid runtime result')", 'KeyboardInterrupt()'])
    def test_cleanup_failure_does_not_swallow_unexpected_exception_or_interrupt(self, error_case):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        error = tuple((TypeError('invalid runtime result'), KeyboardInterrupt()))[error_case]
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', side_effect=[
            error, subprocess.CompletedProcess([], 1, stderr='cleanup unavailable'),
        ]):
            with pytest.raises(type(error)):
                main(['--profile', 'onnxruntime-cpu', '--output-dir', temporary])
            result = json.loads((Path(temporary) / 'runtime.json').read_text())
            assert not (result['successful'])
            assert not (result['cleanup']['successful'])

    def test_cleanup_failure_marks_successful_validation_failed(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', side_effect=[
            subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 1, stderr='cleanup unavailable'),
        ]):
            assert (main(['--profile', 'onnxruntime-cpu', '--output-dir', temporary])) == (1)
            result = json.loads((Path(temporary) / 'runtime.json').read_text())
            assert not (result['successful'])
            assert not (result['cleanup']['successful'])

    @pytest.mark.parametrize('forbidden', ('torch', 'transformers'))
    def test_onnx_environment_guard_rejects_installed_torch_or_transformers(self, forbidden):
        with patch('importlib.util.find_spec',
                side_effect=lambda name: object() if name == forbidden else None):
            with pytest.raises(AssertionError, match=forbidden + ' must not be installed'):
                exec(ONNX_ENVIRONMENT_CHECK, {})

    def test_onnx_environment_guard_rejects_missing_dependencies(self):
        with patch('importlib.util.find_spec', return_value=None):
            with pytest.raises(AssertionError, match='required ONNX dependency missing: onnx'):
                exec(ONNX_ENVIRONMENT_CHECK, {})

    def test_timeout_fails_and_removes_named_container(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', side_effect=[
            subprocess.TimeoutExpired(['docker', 'run'], 1), subprocess.CompletedProcess([], 0),
        ]) as run:
            assert (main(['--profile', 'onnxruntime-cpu', '--timeout-seconds', '1',
                                   '--output-dir', temporary])) == (1)
            first = run.call_args_list[0].args[0]
            name = first[first.index('--name') + 1]
            assert (run.call_args_list[-1].args[0]) == (['docker', 'rm', '-f', name])
            result = json.loads((Path(temporary) / 'runtime.json').read_text())
            assert not (result['successful'])
            assert ('timeout') in (result['error'])

    def test_failed_required_tests_block_e2e_and_success(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})

        def execute(command, **kwargs):
            return subprocess.CompletedProcess(command, 1 if 'scripts/run_tests.py' in command else 0)

        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', side_effect=execute), patch(
            'scripts.check_onnx_basic.run_basic_e2e',
        ) as e2e:
            assert (main(['--profile', 'onnxruntime-cpu', '--basic-e2e',
                                   '--output-dir', temporary])) == (1)
            e2e.assert_not_called()
            assert not (json.loads((Path(temporary) / 'runtime.json').read_text())['successful'])

    def test_cleanup_timeout_preserves_failure_report(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', side_effect=[
            subprocess.TimeoutExpired(['docker', 'run'], 1),
            subprocess.TimeoutExpired(['docker', 'rm'], 30),
        ]):
            assert (main(['--profile', 'onnxruntime-cpu', '--output-dir', temporary])) == (1)
            result = json.loads((Path(temporary) / 'runtime.json').read_text())
            assert not (result['successful'])
            assert ('validation timeout') in (result['error'])
            assert not (result['cleanup']['successful'])

    def test_basic_e2e_requires_all_three_task_families_to_succeed(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', return_value=subprocess.CompletedProcess([], 0)), patch(
            'scripts.check_onnx_basic.run_basic_e2e',
            side_effect=lambda *args, scenario='tabular', **kwargs: {'successful': scenario != 'image'},
        ) as e2e:
            assert (main(['--profile', 'onnxruntime-cpu', '--basic-e2e',
                                   '--output-dir', temporary])) == (1)
            assert ([call.kwargs['scenario'] for call in e2e.call_args_list]) == (['tabular', 'image', 'text'])

    @pytest.mark.parametrize('profile', ('onnxruntime-cpu', 'onnxruntime-cv-cpu', 'onnxruntime-nlp-cpu'))
    def test_onnx_profile_defaults_to_required_onnx_tests_and_forbids_torch(self, profile):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as run:
            code = main(['--profile', profile, '--output-dir', temporary])
            assert (code) == (0)
            commands = [call.args[0] for call in run.call_args_list]
            tests = next(command for command in commands if 'scripts/run_tests.py' in command)
            for pattern in ('test_onnx_runtime_optional.py', 'test_onnx_tasks_runtime.py',
                            'test_request_completion.py', 'test_onnx_server_runtime.py'):
                assert (pattern) in (tests)
            assert ('--require-no-skips') in (tests)
            guard = next(command for command in commands if '-c' in command)
            assert ("'torch'") in (guard[-1])
            assert ("'transformers'") in (guard[-1])
            assert ("'onnxruntime'") in (guard[-1])

    @pytest.mark.parametrize('profile', ('onnxruntime-cpu', 'onnxruntime-cv-cpu', 'onnxruntime-nlp-cpu'))
    def test_each_onnx_profile_build_only_records_dependencies_without_claiming_tests(self, profile):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run') as run:
            assert (main(['--profile', profile, '--build-only', '--output-dir', temporary])) == (0)
            run.assert_not_called()
            result = json.loads((Path(temporary) / 'runtime.json').read_text())
            assert (result['successful'])
            assert (result['validation_scope']) == ('dependencies')
            assert (result['test_patterns']) == ([])
            assert (result['device']) is None

    def test_explicit_patterns_override_family_default_and_are_recorded(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as run:
            code = main(['--profile', 'onnxruntime-cpu', '--test-pattern', 'test_custom_runtime.py',
                         '--test-pattern', 'test_custom_validation.py', '--output-dir', temporary])
            assert (code) == (0)
            command = next(call.args[0] for call in run.call_args_list
                           if 'scripts/run_tests.py' in call.args[0])
            assert ('test_custom_runtime.py') in (command)
            assert ('test_custom_validation.py') in (command)
            assert ('test_structured_runtime.py') not in (command)
            result = json.loads((Path(temporary) / 'runtime.json').read_text())
            assert (result['test_patterns']) == (['test_custom_runtime.py', 'test_custom_validation.py'])

    def test_build_only_cannot_claim_test_pattern_validation(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image',
        ) as build, redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as error:
            main(['--profile', 'onnxruntime-cpu', '--build-only', '--test-pattern', 'test_custom.py',
                  '--output-dir', temporary])
        assert (error.value.code) == (2)
        build.assert_not_called()
