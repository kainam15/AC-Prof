"""Real process cleanup keeps its stop marker and never reuses a released fd."""
import fcntl
import json
import os
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.model_store.store import ModelStoreMount
from acprof.host.runtimes.contracts import RuntimeCleanupError
from acprof.host.runtimes.process_control import ManagedProcess, verify_process_cleanup
from acprof.host.runtimes.validation import ProcessValidationError, run_validation, validate_process
from acprof.process_identity import same_process
from acprof.system.cleanup import format_exception_notes
from acprof.system.exceptions import exception_chain
from tests.execution.process.test_process_service import service as service


@pytest.fixture
def owned_processes(tmp_path):
    children, pipes, replacements = [], {}, []
    real_pipe, real_close = os.pipe, os.close
    close_process = ManagedProcess.close

    def spawn(*args, **kwargs):
        control = []

        def pipe():
            descriptors = real_pipe()
            if not control:
                control.extend(descriptors)
            return descriptors

        with patch('os.pipe', pipe):
            child = ManagedProcess(*args, **kwargs)
        children.append(child)
        pipes[id(child)] = control[1]
        return child

    def reserve_released_fd(descriptor):
        # The injected close has just released this test-owned descriptor.
        stream = (tmp_path / f'replacement-{len(replacements)}').open('wb')
        replacements.append(stream)
        if stream.fileno() != descriptor:
            os.dup2(stream.fileno(), descriptor)
            replacements.append(os.fdopen(descriptor, 'wb'))
        return descriptor

    def protected_close(descriptor):
        # Failed-version teardown must not close the replacement as a stale pipe.
        for stream in replacements:
            if not stream.closed and stream.fileno() == descriptor:
                return
        real_close(descriptor)

    try:
        yield SimpleNamespace(spawn=spawn, children=children, pipes=pipes,
                              reserve=reserve_released_fd, replacements=replacements)
    finally:
        try:
            with patch('os.close', protected_close):
                for owned_child in children:
                    close_process(owned_child)
        finally:
            for replacement in replacements:
                # The old implementation may already have closed this fd.
                try:
                    replacement.close()
                except OSError as error:
                    if error.errno != 9:
                        raise


@contextmanager
def shutdown_faults(owned, child, *, pipe_error=None, wait_error=None, status_error=None):
    real_close, real_wait, real_open = os.close, subprocess.Popen.wait, os.open
    control = owned.pipes[id(child)]

    def close(descriptor):
        real_close(descriptor)
        if descriptor == control:
            owned.reserve(descriptor)
            raise pipe_error

    def wait(process, *args, **kwargs):
        if process is child.process:
            raise wait_error
        return real_wait(process, *args, **kwargs)

    def open_status(path, *args, **kwargs):
        if str(path) == child.status_path.name:
            raise status_error
        return real_open(path, *args, **kwargs)

    with ExitStack() as stack:
        if pipe_error is not None:
            stack.enter_context(patch('os.close', close))
        if wait_error is not None:
            stack.enter_context(patch('subprocess.Popen.wait', wait))
        if status_error is not None:
            stack.enter_context(patch('os.open', open_status))
        yield


def start_sleeping_child(owned, tmp_path):
    return owned.spawn([sys.executable, '-c', 'import time; time.sleep(90)'],
        python=sys.executable, env=dict(os.environ), directory=tmp_path, name='shutdown')


@pytest.mark.parametrize('fault', ['pipe', 'status', 'wait'])
def test_cleanup_errors_keep_the_runtime_stop_marker(owned_processes, tmp_path, fault):
    child = start_sleeping_child(owned_processes, tmp_path)
    error = KeyboardInterrupt('cleanup wait interrupted') if fault == 'wait' else OSError(f'{fault} cleanup failed')
    with shutdown_faults(owned_processes, child, **{f'{fault}_error': error}):
        with pytest.raises((RuntimeCleanupError, OSError, KeyboardInterrupt)) as caught:
            child.close()
    assert isinstance(caught.value, RuntimeCleanupError)
    assert error in tuple(exception_chain(caught.value))
    if fault == 'pipe':
        supervisor = child.process
        assert supervisor is not None
        assert supervisor.returncode is not None
        assert not same_process(child.identity)
    child.close()
    verify_process_cleanup(tmp_path)
    assert not same_process(child.identity)


def test_repeated_cleanup_does_not_close_a_reused_descriptor(owned_processes, tmp_path):
    child = start_sleeping_child(owned_processes, tmp_path)
    descriptor = owned_processes.pipes[id(child)]
    with shutdown_faults(owned_processes, child, pipe_error=OSError('pipe close failed')):
        with pytest.raises((RuntimeCleanupError, OSError), match='pipe close failed'):
            child.close()
    child.close()
    os.write(descriptor, b'replacement remains writable')
    assert any(path.read_bytes() == b'replacement remains writable' for path in tmp_path.glob('replacement-*'))
    verify_process_cleanup(tmp_path)


def test_pipe_failure_still_waits_and_preserves_secondary_cleanup_failure(owned_processes, tmp_path):
    child = start_sleeping_child(owned_processes, tmp_path)
    with shutdown_faults(owned_processes, child, pipe_error=OSError('pipe close failed'),
                         wait_error=subprocess.TimeoutExpired('fixture supervisor', 12)):
        with pytest.raises((RuntimeCleanupError, OSError), match='pipe close failed') as caught:
            child.close()
    assert isinstance(caught.value, RuntimeCleanupError)
    assert 'supervisor cleanup timed out' in format_exception_notes(caught.value)


@pytest.fixture
def validation_shutdown(service, owned_processes, tmp_path, monkeypatch):
    task, info = service
    task.model_resolution = {'contract': {'fields': {}}}
    worker = Path(info.code_root) / 'acprof/container/runtime_validate.py'
    worker.write_text('print(\'ACPROF_RUNTIME_VALIDATION={"status":"ok"}\', flush=True)\n')
    scope = tmp_path / 'resource-scope'
    scope.mkdir()
    lease_path = tmp_path / 'model.lease'
    lease_stream = lease_path.open('a+b')
    fcntl.flock(lease_stream.fileno(), fcntl.LOCK_SH)
    mount = ModelStoreMount([], lease_stream)
    monkeypatch.setattr('acprof.host.model_store.store.acquire_mount', lambda *_args, **_kwargs: mount)
    monkeypatch.setattr('acprof.host.runtimes.resources.ProcessLimits',
                        lambda *_args: SimpleNamespace(path='', close=scope.rmdir))
    close_process = ManagedProcess.close
    wait_process = ManagedProcess.wait
    state = SimpleNamespace(worker=worker, scope=scope, lease_path=lease_path, mount=mount,
                            task=task, info=info, cancel=None, output_dir=tmp_path,
                            cleanup_error=PermissionError('cleanup status denied'), children=owned_processes.children)

    def close(child):
        with shutdown_faults(owned_processes, child, status_error=state.cleanup_error):
            close_process(child)

    def wait(child, timeout):
        if state.cancel is not None:
            raise state.cancel
        return wait_process(child, timeout)

    try:
        with patch('acprof.host.runtimes.validation.ManagedProcess', owned_processes.spawn), \
                patch.object(ManagedProcess, 'close', close), patch.object(ManagedProcess, 'wait', wait):
            yield state
    finally:
        try:
            for child in owned_processes.children:
                close_process(child)
        finally:
            mount.close()
            if scope.exists():
                scope.rmdir()


@pytest.mark.parametrize('outcome', ['ok', 'validation-error', 'timeout', 'cancel'])
def test_validation_cleanup_retains_original_outcome_and_parent_lease(validation_shutdown, outcome):
    state = validation_shutdown
    if outcome == 'validation-error':
        state.worker.write_text('print(\'ACPROF_RUNTIME_VALIDATION={"status":"error","error":"fixture inference failed"}\', flush=True)\n')
    elif outcome in {'timeout', 'cancel'}:
        state.worker.write_text('import time; time.sleep(90)\n')
    if outcome == 'cancel':
        state.cancel = KeyboardInterrupt('validation cancelled')
    with pytest.raises(RuntimeCleanupError) as caught:
        run_validation(state.task, state.info, 'acprof.container.runtime_validate',
                       'ACPROF_RUNTIME_VALIDATION=', state.output_dir, timeout=.05 if outcome == 'timeout' else 5)
    failure = caught.value
    if outcome == 'validation-error':
        assert isinstance(failure.run_error, ProcessValidationError)
        assert str(failure.run_error) == 'fixture inference failed'
    elif outcome == 'timeout':
        assert isinstance(failure.run_error, subprocess.TimeoutExpired)
    elif outcome == 'cancel':
        assert failure.run_error is state.cancel
    else:
        assert failure.run_error is None
    assert state.cleanup_error in tuple(exception_chain(failure))
    assert state.mount.pass_fds
    assert state.scope.is_dir()
    assert not same_process(state.children[0].identity)
    with state.lease_path.open('a+b') as probe:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_runtime_validation_records_cleanup_failure_without_next_device(validation_shutdown):
    state = validation_shutdown
    state.worker.write_text('print(\'ACPROF_RUNTIME_VALIDATION={"status":"error","error":"fixture inference failed"}\', flush=True)\n')
    plan = state.output_dir / 'input_scale_plan.json'
    plan.write_text(json.dumps({'schema_version': 2, 'entries': [{'input_scale': 1, 'payload': {}}]}))
    with pytest.raises(RuntimeCleanupError):
        validate_process(task_info=state.task, image_info=state.info,
            planned=SimpleNamespace(plan_file=str(plan)), cpu_list=[None], mem_list=[None],
            gpu_list=['off', 'on'], output_dir=state.output_dir)
    assert len(state.children) == 1
    report = json.loads((state.output_dir / 'runtime_validation.json').read_text())
    assert report['cleanup_status'] == 'incomplete'
    assert report['devices']['off']['cleanup_error']['run_error'] == 'fixture inference failed'
    assert state.task.model_resolution['runtime_validation']['devices'] == report['devices']
