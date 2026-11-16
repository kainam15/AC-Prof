"""Release independent resources without masking the operation failure."""
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host.runtimes import process, validation
from acprof.host.runtimes.contracts import ProcessInfo
from acprof.system.cleanup import format_exception_notes


@pytest.fixture
def resources(tmp_path, monkeypatch):
    scope = tmp_path / 'scope'
    scope.mkdir()
    lease = SimpleNamespace(close=Mock(side_effect=OSError('lease release failed')), pass_fds=())
    limits = SimpleNamespace(close=scope.rmdir, path='', oom_killed=lambda: False)
    monkeypatch.setattr('acprof.host.model_store.store.acquire_mount', lambda *_args, **_kwargs: lease)
    monkeypatch.setattr('acprof.host.runtimes.resources.ProcessLimits', lambda *_args: limits)
    return SimpleNamespace(scope=scope, lease=lease, limits=limits)


def test_stop_releases_limits_even_if_lease_release_fails(resources):
    child = SimpleNamespace(close=Mock())
    session = SimpleNamespace(process=child, lease=resources.lease, limits=resources.limits)
    with pytest.raises(OSError, match='lease release failed'):
        process.ProcessRuntime().stop(session)
    child.close.assert_called_once_with()
    assert not resources.scope.exists()


@pytest.mark.parametrize('entrypoint', ['startup', 'validation'])
@pytest.mark.parametrize('error_type', [ValueError, KeyboardInterrupt])
def test_failed_operation_preserves_error_and_releases_limits(
    resources, tmp_path, monkeypatch, entrypoint, error_type,
):
    error = error_type('configuration rejected')
    owner = process if entrypoint == 'startup' else validation
    monkeypatch.setattr(owner, 'service_environment', Mock(side_effect=error))
    info = ProcessInfo('fixture-python', '', '', 'fixture', {})
    operation = (partial(process.ProcessRuntime().start, None, 1, 1, 'off', info, 'fixture', '')
                 if entrypoint == 'startup' else
                 partial(validation.run_validation, None, info, 'fixture', 'RESULT=', tmp_path))
    with pytest.raises(error_type) as caught:
        operation()
    assert caught.value is error
    assert 'lease release failed' in format_exception_notes(caught.value)
    assert not resources.scope.exists()


def test_validation_releases_limits_after_success_even_if_lease_release_fails(
    resources, tmp_path, monkeypatch,
):
    log = tmp_path / 'worker.log'
    log.write_text('RESULT={"status":"ok"}\n')
    child = SimpleNamespace(wait=lambda _timeout: 0, log_path=log, close=Mock())
    monkeypatch.setattr(validation, 'service_environment', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(validation, 'ManagedProcess', lambda *_args, **_kwargs: child)
    info = ProcessInfo('fixture-python', '', '', 'fixture', {})
    with pytest.raises(OSError, match='lease release failed'):
        validation.run_validation(None, info, 'fixture', 'RESULT=', tmp_path)
    child.close.assert_called_once_with()
    assert not resources.scope.exists()
