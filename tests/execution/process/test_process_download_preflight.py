"""Process preparation must reach the same download decision before installers."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.runtimes.environment import prepare_process


def test_cancelled_process_download_confirmation_never_prepares_environment(tmp_path):
    task = SimpleNamespace(model_id='demo/model', model_revision='a' * 40)
    with (
        patch('acprof.host.model_store.store.plan_model', return_value={}),
        patch('acprof.host.runtimes.environment.prepare_environment',
              side_effect=AssertionError('installer ran before download confirmation')) as prepare,
        patch('acprof.host.preparation.network_preflight.preflight', side_effect=KeyboardInterrupt('cancelled')),
        patch('acprof.host.preparation.network_preflight.process_sources', return_value=([], {'kind': 'process'})),
        patch('acprof.host.runtimes.environment.select_runtime_profile', return_value=None),
        pytest.raises(KeyboardInterrupt, match='cancelled'),
    ):
        prepare_process(task, tmp_path, tmp_path)
    prepare.assert_not_called()


def test_process_environment_hit_verifies_contents_and_skips_network(tmp_path):
    from acprof.host.preparation.network_preflight import process_sources
    from acprof.runtime.dependency_locks import content_digest

    identity = {'artifacts': []}
    target = tmp_path / content_digest(identity)
    target.mkdir()
    (target / 'manifest.json').write_text(json.dumps({'identity': identity}))
    with (
        patch('acprof.host.runtimes.environment.environment_plan', return_value=(tmp_path, identity, None)),
        patch('acprof.host.runtimes.environment.verify_environment') as verify,
        patch('acprof.host.preparation.network_preflight.host_command.run_command',
              side_effect=AssertionError('cached environment queried installer')),
    ):
        sources, runtime = process_sources(None, tmp_path)
    assert sources == []
    assert runtime['environment_local']
    verify.assert_called_once_with(target, {'identity': identity})


def test_process_sources_preserve_uv_urls_and_do_not_create_environment(tmp_path, monkeypatch):
    from acprof.host.preparation.network_preflight import process_sources

    monkeypatch.setenv('ACPROF_PYPI_ARTIFACT_MIRROR', 'tencent')
    identity = {'artifacts': [{'url': 'https://files.pythonhosted.org/packages/test.whl', 'sha256': 'a' * 64}]}
    environment = tmp_path / 'environments'
    declaration = SimpleNamespace(requirements_lock='requirements.lock', platform=SimpleNamespace(python_version='3.12'))
    with (
        patch('acprof.host.runtimes.environment.environment_plan', return_value=(environment, identity, declaration)),
        patch('acprof.host.preparation.network_preflight.shutil.which', return_value='/bin/uv'),
        patch('acprof.host.preparation.network_preflight.host_command.run_command',
              return_value=SimpleNamespace(returncode=0)) as command,
    ):
        sources, runtime = process_sources(None, tmp_path)
    assert sources[0].url == identity['artifacts'][0]['url']
    assert sources[0].cache_status == 'unknown'
    assert not runtime['will_attempt_ghcr_pull']
    assert '--offline' in command.call_args.args[0]
    assert '--no-python-downloads' in command.call_args.args[0]
    assert not environment.exists()
