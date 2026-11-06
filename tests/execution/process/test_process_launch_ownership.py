"""A failed constructor must preserve cleanup debt across its callers."""
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.artifacts import atomic_write_json
from acprof.host.model_store.store import ModelStoreMount
from acprof.host.runtimes.contracts import RuntimeCleanupError
from acprof.host.runtimes.process import ProcessRuntime
from acprof.host.runtimes.process_control import (
    ManagedProcess,
    ProcessCleanupError,
    verify_process_cleanup,
)
from acprof.host.runtimes.validation import probe_interface, run_validation, validate_process
from acprof.process_identity import same_process
from acprof.system.cleanup import format_exception_notes
from acprof.system.exceptions import exception_chain
from tests.execution.process.test_process_service import service as service


def wait_for_workload(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            record = json.loads(path.read_text())
        except FileNotFoundError:
            record = {}
        if record.get('workload'):
            return record
        time.sleep(.01)
    pytest.fail('fixture supervisor did not publish its workload')


@pytest.fixture
def launch_fault(service, tmp_path, monkeypatch):
    task, info = service
    logs = Path(info.runtime_environment['logs_dir'])
    (Path(info.code_root) / 'acprof/container/worker.py').write_text('import time; time.sleep(90)\n')
    lease_path = tmp_path / 'model.lease'
    stream = lease_path.open('a+b')
    fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
    mount = ModelStoreMount([], stream)
    scope = tmp_path / 'resource-scope'
    scope.mkdir()
    state = SimpleNamespace(task=task, info=info, logs=logs, mount=mount, scope=scope,
        primary=OSError('owner publication failed'), cleanup=ProcessCleanupError('cleanup unconfirmed'),
        controller=None, workload=None, oom=False, diagnostic=None, lease_path=lease_path,
        started_publications=0, persistent_publication_error=False)
    close_process = ManagedProcess.close
    state.recover = lambda: close_process(state.controller)

    def publish(path, payload, **kwargs):
        if payload.get('launch_state') == 'started':
            state.started_publications += 1
            if state.started_publications == 1:
                state.workload = wait_for_workload(path.parent / payload['status_file'])['workload']
                raise state.primary
            if state.persistent_publication_error:
                raise OSError('recovery ledger publication failed')
        return atomic_write_json(path, payload, **kwargs)

    def close(child):
        state.controller = child
        if state.cleanup is not None:
            raise state.cleanup
        close_process(child)

    def oom_killed():
        if state.diagnostic is not None:
            raise state.diagnostic
        return state.oom

    limits = SimpleNamespace(path='', evidence=lambda: {}, verify=lambda _pid: None,
                             oom_killed=oom_killed, close=scope.rmdir)
    monkeypatch.setattr('acprof.host.runtimes.resources.ProcessLimits', lambda *_args: limits)
    monkeypatch.setattr('acprof.host.model_store.store.acquire_mount', lambda *_args, **_kwargs: mount)
    with patch('acprof.artifacts.atomic_write_json', publish), patch.object(ManagedProcess, 'close', close):
        try:
            yield state
        finally:
            try:
                if state.controller is not None:
                    close_process(state.controller)
            finally:
                mount.close()
                if scope.exists():
                    scope.rmdir()


def launch(state, entry):
    if entry == 'service':
        return ProcessRuntime().start(state.task, None, None, 'off', state.info, 'fixture', '[test]')
    if entry == 'validation':
        return run_validation(state.task, state.info, 'acprof.container.worker', 'RESULT=', state.logs.parent)
    return ManagedProcess([sys.executable, '-c', 'import time; time.sleep(90)'],
        python=sys.executable, env=dict(os.environ), directory=state.logs, name='fixture',
        pass_fds=state.mount.pass_fds)


@pytest.mark.parametrize('primary_type', [OSError, KeyboardInterrupt])
@pytest.mark.parametrize('cleanup_type', [OSError, ProcessCleanupError, KeyboardInterrupt])
def test_constructor_reports_unconfirmed_cleanup_with_original_failure(launch_fault, primary_type, cleanup_type):
    state = launch_fault
    state.primary = primary_type('owner publication failed')
    state.cleanup = cleanup_type('cleanup interrupted or failed')
    with pytest.raises((cleanup_type, ProcessCleanupError)) as caught:
        launch(state, 'direct')
    assert isinstance(caught.value, ProcessCleanupError)
    assert caught.value.run_error is state.primary
    assert state.cleanup in tuple(exception_chain(caught.value))
    assert same_process(state.workload)
    with pytest.raises(ProcessCleanupError):
        verify_process_cleanup(state.logs)


@pytest.mark.parametrize('entry', ['service', 'validation'])
def test_constructor_cleanup_debt_keeps_parent_lease_and_scope(launch_fault, entry):
    state = launch_fault
    with pytest.raises(RuntimeCleanupError):
        launch(state, entry)
    assert state.mount.pass_fds
    assert state.scope.is_dir()
    assert same_process(state.workload)
    # Even after explicit descendant recovery, the parent still owns its lease.
    state.recover()
    assert not same_process(state.workload)
    assert state.controller.state()['cleanup_complete']
    with pytest.raises(ProcessCleanupError):
        verify_process_cleanup(state.logs)
    with state.lease_path.open('a+b') as probe:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize('diagnostic', ['oom', 'cancel'])
def test_startup_diagnostics_cannot_erase_constructor_cleanup_debt(launch_fault, diagnostic):
    state = launch_fault
    state.oom = diagnostic == 'oom'
    if diagnostic == 'cancel':
        state.diagnostic = KeyboardInterrupt('OOM diagnostics interrupted')
    with pytest.raises((RuntimeCleanupError, KeyboardInterrupt)) as caught:
        launch(state, 'service')
    assert any(isinstance(error, RuntimeCleanupError) for error in exception_chain(caught.value))
    assert state.mount.pass_fds
    assert state.scope.is_dir()


@pytest.mark.parametrize('entry', ['direct', 'service', 'validation'])
@pytest.mark.parametrize('primary_type', [OSError, KeyboardInterrupt])
def test_confirmed_constructor_cleanup_preserves_original_failure(launch_fault, entry, primary_type):
    state = launch_fault
    state.primary = primary_type('owner publication failed')
    state.cleanup = None
    with pytest.raises(primary_type) as caught:
        launch(state, entry)
    assert caught.value is state.primary
    assert not same_process(state.workload)
    verify_process_cleanup(state.logs)
    ledger = json.loads(next(state.logs.glob('*.process.owner.json')).read_text())
    assert ledger['launch_state'] == 'started'
    assert ledger['owner_pid'] == state.controller.state()['owner']['pid']
    if entry != 'direct':
        assert state.mount.pass_fds == ()
        assert not state.scope.exists()


def test_cleanup_success_does_not_hide_persistently_missing_launch_evidence(launch_fault):
    state = launch_fault
    state.cleanup = None
    state.persistent_publication_error = True
    with pytest.raises(OSError, match='owner publication failed') as caught:
        launch(state, 'direct')
    assert caught.value is state.primary
    assert not same_process(state.workload)
    assert 'recovery ledger publication failed' in format_exception_notes(caught.value)
    with pytest.raises(ProcessCleanupError):
        verify_process_cleanup(state.logs)


@pytest.mark.parametrize('publication_error', [None, 'report', 'model-resolution'])
def test_process_validation_does_not_convert_cleanup_debt_to_compatibility_error(tmp_path, monkeypatch, publication_error):
    plan = tmp_path / 'input_scale_plan.json'
    plan.write_text(json.dumps({'schema_version': 2, 'entries': [{'input_scale': 1, 'payload': {}}]}))
    task = SimpleNamespace(model_resolution={'contract': {'fields': {}}})
    info = SimpleNamespace(environment_id='fixture-process')
    primary = OSError('owner publication failed')
    failure = ProcessCleanupError('cleanup unconfirmed')
    failure.run_error = primary
    calls = []

    def reject(*_args, **kwargs):
        calls.append(kwargs['gpu'])
        raise failure

    monkeypatch.setattr('acprof.host.runtimes.validation.run_validation', reject)
    if publication_error:
        def reject_publication(*_args, **_kwargs):
            raise OSError('validation evidence publication failed')

        target = ('acprof.host.runtimes.validation.atomic_write_json' if publication_error == 'report'
                  else 'acprof.models.contract.write_model_resolution')
        monkeypatch.setattr(target, reject_publication)
    with pytest.raises(RuntimeCleanupError) as caught:
        validate_process(task_info=task, image_info=info, planned=SimpleNamespace(plan_file=str(plan)),
            cpu_list=[None], mem_list=[None], gpu_list=['off', 'on'], output_dir=tmp_path)
    assert caught.value is failure
    assert calls == ['off']
    if publication_error:
        assert 'validation evidence publication failed' in format_exception_notes(caught.value)
    if publication_error == 'report':
        assert not (tmp_path / 'runtime_validation.json').exists()
        return
    report = json.loads((tmp_path / 'runtime_validation.json').read_text())
    assert report['status'] == 'error'
    assert report['cleanup_status'] == 'incomplete'
    assert report['devices']['off']['cleanup_error']['run_error'] == 'owner publication failed'
    recorded = task.model_resolution['runtime_validation']
    assert recorded['status'] == 'error'
    assert recorded['devices'] == report['devices']


def test_control_read_end_close_error_still_reaps_the_launched_supervisor(tmp_path):
    original_pipe, original_close, original_popen = os.pipe, os.close, subprocess.Popen
    controls, children = [], []
    pipe_identity = None
    failure = OSError('control read-end close failed')
    injected = False

    def pipe():
        nonlocal pipe_identity
        descriptors = original_pipe()
        if not controls:
            controls.extend(descriptors)
            metadata = os.fstat(descriptors[0])
            pipe_identity = (metadata.st_dev, metadata.st_ino)
        return descriptors

    def close(descriptor):
        nonlocal injected
        original_close(descriptor)
        if controls and descriptor == controls[0] and not injected:
            injected = True
            raise failure

    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    try:
        with patch('os.pipe', pipe), patch('os.close', close), patch('subprocess.Popen', popen):
            with pytest.raises(OSError, match='control read-end close failed') as caught:
                ManagedProcess([sys.executable, '-c', 'import time; time.sleep(90)'],
                    python=sys.executable, env=dict(os.environ), directory=tmp_path, name='pipe')
        assert caught.value is failure
        assert len(children) == 1
        assert children[0].poll() is not None
        verify_process_cleanup(tmp_path)
    finally:
        # Close only descriptors that still identify this fixture's own pipe.
        for descriptor in controls:
            try:
                metadata = os.fstat(descriptor)
            except OSError:
                continue
            if (metadata.st_dev, metadata.st_ino) == pipe_identity:
                original_close(descriptor)
        for child in children:
            child.wait(timeout=12)


@pytest.mark.parametrize('publication_error', [None, 'report', 'model-resolution'])
def test_interface_cleanup_debt_survives_evidence_publication_errors(tmp_path, monkeypatch, publication_error):
    task = SimpleNamespace(model_resolution={})
    failure = ProcessCleanupError('interface cleanup unconfirmed')
    failure.run_error = OSError('interface launch failed')

    def reject(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr('acprof.host.runtimes.validation.run_validation', reject)
    monkeypatch.setattr('acprof.host.preparation.source_bundle.source_bundle',
                        lambda _task: nullcontext(SimpleNamespace(root=tmp_path, identity='fixture-source')))
    if publication_error:
        def reject_publication(*_args, **_kwargs):
            raise OSError('interface evidence publication failed')

        target = ('acprof.host.runtimes.validation.atomic_write_json' if publication_error == 'report'
                  else 'acprof.models.contract.write_model_resolution')
        monkeypatch.setattr(target, reject_publication)
    with pytest.raises(RuntimeCleanupError) as caught:
        probe_interface(task, tmp_path, {'environment_id': 'fixture-env'}, tmp_path)
    assert caught.value is failure
    if publication_error:
        assert 'interface evidence publication failed' in format_exception_notes(caught.value)
    if publication_error == 'report':
        assert not (tmp_path / 'interface_validation.json').exists()
    else:
        report = json.loads((tmp_path / 'interface_validation.json').read_text())
        assert report['cleanup_status'] == 'incomplete'
        assert report['cleanup_error']['run_error'] == 'interface launch failed'
        assert task.model_resolution['interface_validation'] == report
