import json
import subprocess
import tempfile
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.profiler_support import profiler_container_command
from acprof.host.runtime_images import ImageInfo
from acprof.host.runtime_validation import validate_runtime


class TestRuntimeValidation:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        selection = patch('acprof.host.gpu_device.resolve_gpu_device', return_value={
            'uuid': 'GPU-fixture', 'index': 1, 'name': 'Fixture', 'memory_total_bytes': 8 * 1024 ** 3,
            'pci_bus_id': '00000000:02:00.0',
        })
        selection.start()
        self._request.addfinalizer(partial(selection.stop))
        recovery = patch('acprof.host.runtime_validation.recover_abandoned_containers')
        recovery.start()
        self._request.addfinalizer(partial(recovery.stop))
        self.cid = patch('acprof.host.runtime_validation.owned_container_id', return_value='c' * 64)
        self.cid.start()
        self._request.addfinalizer(partial(self.cid.stop))

    def task(self):
        return TaskInfo('Example/model', 'audio-text-to-text', 'multimodal',
                        'transformers_model', 'transformers', 'a' * 40, 'unit',
                        runtime_profile_id='moss-transformers560', model_config={'model_type': 'moss_transcribe_diarize'})

    def fixture(self, root):
        plan = root / 'input_scale_plan.json'
        plan.write_text(json.dumps({"schema_version": 2, 'entries': [
            {'input_scale': 10, 'payload': {'text': 'large'}},
            {'input_scale': 1, 'payload': {'text': 'small'}},
        ]}))
        return dict(task_info=self.task(), image_info=ImageInfo(
            tag='sha256:' + 'b' * 64, runtime_environment={'build_fingerprint': 'build'},
        ), planned=SimpleNamespace(plan_file=str(plan)), cpu_list=[1, 4], mem_list=[2, 8],
                    gpu_list=['off', 'on'], output_dir=str(root), timeout_seconds=30)

    def test_both_devices_use_smallest_planned_payload_and_separate_containers(self):
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[:2] == ['docker', 'run']:
                assert ('--network') in (command)
                assert ('--memory=8g') in (command)
                mount = command[command.index('-v') + 1].split(':')[0]
                assert (json.loads(Path(mount).read_text())) == ({'text': 'small'})
                return subprocess.CompletedProcess(command, 0, stdout='ACPROF_RUNTIME_VALIDATION={"status":"ok"}\n', stderr='')
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

        with tempfile.TemporaryDirectory() as temporary, patch(
            'acprof.host.runtime_validation.run_command', side_effect=run,
        ), patch('acprof.host.container_state.inspect_container_state', return_value={}):
            root = Path(temporary)
            report = validate_runtime(**self.fixture(root))
            saved = json.loads((root / 'runtime_validation.json').read_text())
            assert (saved) == (report)
            assert (report['status']) == ('ok')
            assert (set(report['devices'])) == ({'off', 'on'})
            assert (report['input_scale']) == (1)
            assert (list(root.glob('*.csv'))) == ([])
        runs = [c for c in commands if c[:2] == ['docker', 'run']]
        removals = [c for c in commands if c[:3] == ['docker', 'rm', '-f']]
        assert (len(runs)) == (2)
        assert (runs[0][3]) != (runs[1][3])
        assert ({c[-1] for c in removals}) == ({'c' * 64})
        assert ('--gpus') not in (runs[0])
        assert ('--gpus') in (runs[1])
        assert (all('ACPROF_REQUEST_TIMEOUT_S=30' in command for command in runs))

    def test_same_process_residue_blocks_validation_and_records_preflight_failure(self):
        from acprof.host.container_lifecycle import (
            ContainerCleanupError,
            container_owner_labels,
            recover_abandoned_containers,
        )
        identifier = 'a' * 64
        owner = container_owner_labels()
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[1] == 'ps':
                return subprocess.CompletedProcess(command, 0, identifier, '')
            if command[1] == 'inspect':
                payload = [{'Id': identifier, 'Config': {'Labels': owner},
                            'State': {'Running': True, 'Status': 'running'}}]
                return subprocess.CompletedProcess(command, 0, json.dumps(payload), '')
            if command[1] == 'run':
                return subprocess.CompletedProcess(command, 0, 'ACPROF_RUNTIME_VALIDATION={"status":"ok"}', '')
            if command[1] == 'rm':
                return subprocess.CompletedProcess(command, 0, '', '')
            pytest.fail('unexpected Docker command')

        with tempfile.TemporaryDirectory() as directory, patch(
            'acprof.host.runtime_validation.recover_abandoned_containers', side_effect=recover_abandoned_containers,
        ), patch('acprof.host.runtime_validation.run_command', side_effect=run):
            root = Path(directory)
            with pytest.raises(ContainerCleanupError):
                validate_runtime(**self.fixture(root))
            report = json.loads((root / 'runtime_validation.json').read_text())
            assert (report['status']) == ('error')
            assert (report['cleanup_status']) == ('incomplete')
            assert (report['cleanup_error']['container_id']) == (identifier)
            assert (report['cleanup_error']['docker_state']['Running'])
            assert (report['devices']) == ({})
        assert ([command[1] for command in commands]) == (['ps', 'inspect'])

    def test_load_failure_keeps_stderr_and_stops_before_next_device(self):
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 1 if command[1] == 'run' else 0, stdout='', stderr='ImportError: wrong Transformers version')

        with tempfile.TemporaryDirectory() as temporary, patch(
            'acprof.host.runtime_validation.run_command', side_effect=run,
        ), patch('acprof.host.container_state.inspect_container_state', return_value={}):
            root = Path(temporary)
            with pytest.raises(RuntimeError, match='wrong Transformers'):
                validate_runtime(**self.fixture(root))
            assert ('wrong Transformers') in ((root / 'runtime_validation_off.log').read_text())
            assert (json.loads((root / 'runtime_validation.json').read_text())['status']) == ('error')
            assert (list(root.glob('*.csv'))) == ([])
        assert (sum(c[:2] == ['docker', 'run'] for c in commands)) == (1)

    def test_timeout_removes_owned_container(self):
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[:2] == ['docker', 'run']:
                raise subprocess.TimeoutExpired(command, 30, stderr=b'loading processor')
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

        with tempfile.TemporaryDirectory() as temporary, patch(
            'acprof.host.runtime_validation.run_command', side_effect=run,
        ), patch('acprof.host.container_state.inspect_container_state', return_value={}):
            with pytest.raises(RuntimeError, match='timeout'):
                validate_runtime(**self.fixture(Path(temporary)))
            assert ('loading processor') in ((Path(temporary) / 'runtime_validation_off.log').read_text())
        assert (commands[-1][:3]) == (['docker', 'rm', '-f'])

    def test_compatibility_timeout_is_inconclusive_with_phase_evidence(self):
        def run(command, **kwargs):
            if command[:2] == ['docker', 'run']:
                raise subprocess.TimeoutExpired(command, 60, output=(
                    b'ACPROF_RUNTIME_STAGE={"stage":"load","status":"verified"}\n'
                    b'ACPROF_RUNTIME_STAGE={"stage":"predict","status":"running"}\n'))
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
        with tempfile.TemporaryDirectory() as directory, patch(
            'acprof.host.runtime_validation.run_command', side_effect=run,
        ), patch('acprof.host.container_state.inspect_container_state', return_value={'Running': True}):
            root = Path(directory)
            with pytest.raises(RuntimeError):
                validate_runtime(**self.fixture(root))
            report = json.loads((root / 'runtime_validation.json').read_text())
        assert (report['status']) == ('inconclusive')
        failure = report['devices']['off']['failure']
        assert (failure['reason_code']) == ('compatibility_budget_exhausted')
        assert (failure['evidence']['request_phase']) == ('predict')
        assert (failure['evidence']['model_loaded']) is (True)
        assert (failure['evidence']['service_alive']) is (True)

        # A timeout returned by the container is distinct from the outer
        # compatibility budget, but is equally inconclusive for compatibility.
        from acprof.failures import Failure
        response = {"status": "error", "failed_stage": "completion", "stages": [
            {"stage": "load", "status": "verified"}, {"stage": "completion", "status": "error"}],
            "failure": Failure("completion", "request_timeout", "fixture", evidence={
                "timeout_seconds": 2.5, "timeout_scope": "completion_wait", "model_loaded": True}).to_dict()}
        with tempfile.TemporaryDirectory() as directory, patch(
            'acprof.host.runtime_validation.run_command', side_effect=lambda command, **kw: subprocess.CompletedProcess(
                command, 1 if command[1] == 'run' else 0, stdout='ACPROF_RUNTIME_VALIDATION=' + json.dumps(response), stderr=''),
        ), patch('acprof.host.container_state.inspect_container_state', return_value={'Running': False}):
            root = Path(directory)
            with pytest.raises(RuntimeError):
                validate_runtime(**self.fixture(root))
            report = json.loads((root / 'runtime_validation.json').read_text())
        assert (report['status']) == ('inconclusive')
        evidence = report['devices']['off']['failure']['evidence']
        assert (evidence['timeout_seconds']) == (2.5)
        assert (evidence['input_scale']) == (1)
        assert (evidence['request_id'].startswith('acprof-validate-'))
        assert (evidence['request_phase']) == ('completion')

    def test_invalid_validation_response_keeps_report_and_cleans_container(self):
        result = subprocess.CompletedProcess([], 0, stdout='ACPROF_RUNTIME_VALIDATION=[]\n', stderr='')
        with tempfile.TemporaryDirectory() as temporary, patch(
            'acprof.host.runtime_validation.run_command', return_value=result,
        ) as run, patch('acprof.host.container_state.inspect_container_state', return_value={}):
            root = Path(temporary)
            with pytest.raises(RuntimeError, match='validation response'):
                validate_runtime(**self.fixture(root))
            assert (json.loads((root / 'runtime_validation.json').read_text())['status']) == ('error')
        assert (run.call_args.args[0][:3]) == (['docker', 'rm', '-f'])

    def test_cgroup_oom_is_resource_limit_not_dependency_failure(self):
        result = subprocess.CompletedProcess([], 137, stdout='', stderr='Killed')
        with tempfile.TemporaryDirectory() as temporary, patch(
            'acprof.host.runtime_validation.run_command', side_effect=lambda command, **kw: result if command[1] == 'run' else subprocess.CompletedProcess(command, 0, '', ''),
        ), patch('acprof.host.container_state.inspect_container_state', return_value={'OOMKilled': True}):
            report = validate_runtime(**self.fixture(Path(temporary)))
        assert (report['status']) == ('resource_limited')
        assert (set(report['devices'])) == ({'off', 'on'})

    def test_validation_records_owner_and_cleans_only_cidfile_id(self):
        self.cid.stop()
        commands = []
        identifier = 'c' * 64
        def run(command, **kwargs):
            commands.append(command)
            if command[:2] == ['docker', 'run']:
                assert ('--cidfile') in (command)
                assert (any(value.startswith('org.acprof.owner.pid=') for value in command))
                Path(command[command.index('--cidfile') + 1]).write_text(identifier)
                return subprocess.CompletedProcess(command, 0, 'ACPROF_RUNTIME_VALIDATION={"status":"ok"}', '')
            return subprocess.CompletedProcess(command, 0, '', '')
        with tempfile.TemporaryDirectory() as directory, patch(
            'acprof.host.runtime_validation.run_command', side_effect=run,
        ), patch('acprof.host.container_state.inspect_container_state', return_value={}):
            validate_runtime(**{**self.fixture(Path(directory)), 'gpu_list': ['off']})
        assert ([c[-1] for c in commands if c[:2] == ['docker', 'rm']]) == ([identifier])

    def test_cleanup_failure_preserves_validation_error_and_stops_next_device(self):
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            if command[1] == 'run':
                return subprocess.CompletedProcess(command, 1, '', 'inference failed first')
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        with tempfile.TemporaryDirectory() as directory, patch(
            'acprof.host.runtime_validation.run_command', side_effect=run,
        ), patch('acprof.host.container_state.inspect_container_state', return_value={}):
            root = Path(directory)
            with pytest.raises(RuntimeError, match='cleanup unknown'):
                validate_runtime(**self.fixture(root))
            report = json.loads((root / 'runtime_validation.json').read_text())
            assert (report['status']) == ('error')
            device = report['devices']['off']
            assert ('inference failed first') in (device['error'])
            assert (device['cleanup_error']['final_state']) == ('unknown')
            assert ('inference failed first') in (device['cleanup_error']['run_error']['detail'])
            assert (sum(command[1] == 'run' for command in commands)) == (1)

    @pytest.mark.parametrize('mode', ('missing', 'malformed', 'permission_denied'))
    def test_timeout_with_unusable_cidfile_preserves_both_failures(self, mode):
        self.cid.stop()
        original_read = Path.read_text
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            if command[1] == 'run':
                if mode == 'malformed':
                    Path(command[command.index('--cidfile') + 1]).write_text('invalid id')
                raise subprocess.TimeoutExpired(command, 30, output=b'ACPROF_RUNTIME_STAGE={"stage":"load","status":"verified"}\n')
            return subprocess.CompletedProcess(command, 0, '', '')
        def read(path, *args, **kwargs):
            if mode == 'permission_denied' and path.suffix == '.cid':
                raise PermissionError('cidfile is not readable')
            return original_read(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch(
            'acprof.host.runtime_validation.run_command', side_effect=run,
        ), patch.object(Path, 'read_text', read), patch(
            'acprof.host.container_state.inspect_container_state', return_value={},
        ):
            root = Path(directory)
            with pytest.raises(RuntimeError):
                validate_runtime(**self.fixture(root))
            report = json.loads((root / 'runtime_validation.json').read_text())
            assert (report['status']) == ('error')
            device = report['devices']['off']
            assert (device['failure']['reason_code']) == ('compatibility_budget_exhausted')
            assert (device['cleanup_error']['final_state']) == ('unknown')
            assert (device['cleanup_error']['run_error']['exception_type']) == ('TimeoutExpired')
            assert (device['failure']['evidence']['model_loaded'])
            assert (sum(command[1] == 'run' for command in commands)) == (1)
            assert not (any(command[1] == 'rm' for command in commands))

    def test_managed_profilers_keep_image_adapter_code(self):
        command = profiler_container_command(
            task_info=self.task(), image_tag='sha256:' + 'b' * 64,
            cpu=1, mem=8, use_gpu=True, payload_file='/tmp/payload.json',
            profile_root='/tmp/profiles', tool_mount_roots=[],
        )
        assert not (any('/app/acprof' in item for item in command))
        assert ('--gpus') in (command)

    def test_cli_stops_before_matrix_when_runtime_validation_fails(self):
        import sys

        from acprof.cli import run
        from acprof.host.input_plan import PlannedInputScales

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', [
                'acprof run', '--model', self.task().model_id, '--cpus', '1', '--mems', '8',
                '--gpus', 'off,on', '--input-scales', '1', '--notify', 'none', '--output-dir', temporary,
            ]))
            for name in (
                'bootstrap_project_env', 'require_collection_host', 'require_native_docker',
                'require_packet_latency_prerequisites', 'require_cpu_energy_prerequisites',
                'require_mips_prerequisites',
            ):
                stack.enter_context(patch('acprof.cli.run.' + name))
            stack.enter_context(patch('acprof.cli.run.require_cgroup_prerequisites', return_value='v2'))
            stack.enter_context(patch('acprof.host.detect.detect_task', return_value=self.task()))
            stack.enter_context(patch('acprof.host.runtime_images.prepare_image', return_value=ImageInfo(
                tag='sha256:' + 'b' * 64, runtime_environment={'build_fingerprint': 'build'},
            )))
            stack.enter_context(patch('acprof.host.static_metadata.collect_static_meta', return_value=SimpleNamespace()))
            stack.enter_context(patch('acprof.host.static_metadata.enrich_static_meta', return_value=SimpleNamespace()))
            stack.enter_context(patch('acprof.host.static_metadata.enrich_static_meta_from_input_plan', return_value=SimpleNamespace()))
            stack.enter_context(patch('acprof.host.static_metadata.write_static_meta_json'))
            stack.enter_context(patch('acprof.host.input_plan.plan_input_scales', return_value=PlannedInputScales(
                scales=[1.0], source='manual', plan_file='',
            )))
            validation = stack.enter_context(patch(
                'acprof.host.runtime_validation.validate_runtime', side_effect=RuntimeError('adapter load failed'),
            ))
            matrix = stack.enter_context(patch('acprof.host.orchestrator.run_matrix'))
            compute = stack.enter_context(patch('acprof.host.compute_profile.collect_compute_profile_plan'))
            execution = stack.enter_context(patch('acprof.host.execution_profile.collect_execution_profile_plan'))
            with pytest.raises(SystemExit) as exited:
                run.main()
            assert (exited.value.code) == (1)
            validation.assert_called_once()
            matrix.assert_not_called()
            compute.assert_not_called()
            execution.assert_not_called()
            assert not (list(Path(temporary).rglob('*.csv')))
