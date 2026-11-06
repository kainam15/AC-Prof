"""Real socket/ownership contract; the lightweight service is a protocol fixture."""
import asyncio
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.runtimes.contracts import ProcessInfo
from acprof.host.runtimes.process import ProcessRuntime
from acprof.host.runtimes.process_control import (
    ManagedProcess,
    ProcessCleanupError,
    verify_process_cleanup,
)
from acprof.process_identity import same_process
from acprof.system.cleanup import format_exception_notes


@pytest.fixture
def service(tmp_path):
    code = tmp_path / 'code'
    package = code / 'acprof/container'
    package.mkdir(parents=True)
    (code / 'acprof/__init__.py').write_text('')
    (package / '__init__.py').write_text('')
    (package / 'server.py').write_text('''
import json, os, socket
from http.server import HTTPServer, BaseHTTPRequestHandler
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(dict(status="ok", service_token=os.environ["ACPROF_SERVICE_TOKEN"], startup_timing={})).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps({"workload_contract": {"fixture_payload": payload}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *args): pass
server = HTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
server.socket.close()
server.socket = socket.fromfd(int(os.environ["ACPROF_SERVER_FD"]), socket.AF_INET, socket.SOCK_STREAM)
os.close(int(os.environ["ACPROF_SERVER_FD"]))
server.serve_forever(poll_interval=.01)
''')
    python = tmp_path / 'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve())
    info = ProcessInfo(str(python), str(code), str(tmp_path), 'fixture', {'logs_dir': str(tmp_path / 'logs')})
    task = SimpleNamespace(model_id='fixture/service', model_revision='1' * 40, pipeline_tag='tabular-regression',
                           task_family='structured', runtime_backend='onnxruntime', model_adapter='family-default',
                           runtime_profile_id='onnxruntime-cpu', model_spec={})
    return task, info


def test_process_service_owns_ephemeral_ports_and_no_docker(service, monkeypatch):
    task, info = service
    from acprof.system import command
    monkeypatch.setattr(command, 'run_command', lambda *a, **kw: pytest.fail('service must not execute Docker/host commands'))
    runtime = ProcessRuntime()
    sessions = []
    try:
        for _ in range(2):
            sessions.append(runtime.start(task, None, None, 'off', info, 'fixture', '[test]'))
        assert sessions[0].host_port != sessions[1].host_port
        for session in sessions:
            assert session.resource_plan['method'] == 'inherited'
            assert session.resource_plan['requested']['cpu_quota'] is None
            assert session.resource_plan['requested']['memory_gib'] is None
            assert same_process(session.process.identity)
    finally:
        for session in sessions:
            runtime.stop(session)
    assert all(not same_process(session.process.identity) for session in sessions)


def test_sigkill_exit_without_oom_counter_is_not_oom():
    session = SimpleNamespace(limits=SimpleNamespace(oom_killed=lambda: False),
                              process=SimpleNamespace(poll=lambda: -9, log_tail=lambda: 'killed'))
    assert ProcessRuntime().failure(session, None, 1).startswith('process_exit:')
    session.limits.oom_killed = lambda: True
    assert ProcessRuntime().failure(session, 1, 1).startswith('runtime_oom:')


@pytest.fixture
def failed_start(service, tmp_path, monkeypatch):
    from acprof.host.model_store import store
    from acprof.host.runtimes import resources

    task, info = service
    scope = tmp_path / 'owned-resource-scope'
    scope.mkdir()
    lease_stream = (tmp_path / 'lease').open('wb')
    lease = SimpleNamespace(pass_fds=(lease_stream.fileno(),), close=lease_stream.close)
    state = SimpleNamespace(
        primary=RuntimeError('startup verification failed'), diagnostic=None, oom=False,
        cleanup_error=None, process=None, scope=scope, lease=lease, lease_stream=lease_stream,
        retained=[], logs=Path(info.runtime_environment['logs_dir']), task=task, info=info,
    )

    def verify(_pid):
        raise state.primary

    def oom_killed():
        if state.diagnostic is not None:
            raise state.diagnostic
        return state.oom

    limits = SimpleNamespace(path='', evidence=lambda: {}, verify=verify,
                             oom_killed=oom_killed, close=scope.rmdir)
    monkeypatch.setattr(resources, 'ProcessLimits', lambda *_args: limits)
    monkeypatch.setattr(store, 'acquire_mount', lambda *_args, **_kwargs: lease)
    monkeypatch.setattr(store, 'retain_mount_for_cleanup_debt', state.retained.append)

    def cleanup_failed():
        raise state.cleanup_error

    def launch_process(*args, **kwargs):
        child = ManagedProcess(*args, **kwargs)
        state.process = child
        if state.cleanup_error is not None:
            monkeypatch.setattr(child, 'close', cleanup_failed)
        return child

    state.start = lambda: ProcessRuntime().start(task, None, None, 'off', info, 'fixture', '[test]')
    try:
        with patch('acprof.host.runtimes.process.ManagedProcess', launch_process):
            yield state
    finally:
        # The failing implementation can leave this real fixture service alive.
        try:
            if state.process is not None:
                ManagedProcess.close(state.process)
        finally:
            lease_stream.close()
            if scope.exists():
                scope.rmdir()


def assert_startup_resources_released(state):
    assert state.process is not None
    assert not same_process(state.process.identity)
    assert state.process.poll() is not None
    verify_process_cleanup(state.logs)
    assert state.lease_stream.closed
    assert not state.scope.exists()
    assert state.retained == []


@pytest.mark.parametrize('primary_type', [RuntimeError, KeyboardInterrupt, SystemExit, asyncio.CancelledError])
@pytest.mark.parametrize('diagnostic_type', [OSError, ValueError])
def test_startup_diagnostic_error_preserves_failure_and_releases_resources(failed_start, primary_type, diagnostic_type):
    state = failed_start
    state.primary = primary_type('original startup failure')
    state.diagnostic = diagnostic_type('memory.events read failed')
    with pytest.raises((primary_type, diagnostic_type)) as caught:
        state.start()
    assert_startup_resources_released(state)
    assert caught.value is state.primary
    assert 'memory.events read failed' in format_exception_notes(caught.value)
    assert 'runtime_oom' not in format_exception_notes(caught.value)


@pytest.mark.parametrize('primary_type', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('interruption_type', [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_interrupted_startup_diagnostics_still_release_resources(failed_start, primary_type, interruption_type):
    state = failed_start
    state.primary = primary_type('original startup failure')
    state.diagnostic = interruption_type('diagnostic interrupted')
    with pytest.raises((primary_type, interruption_type)) as caught:
        state.start()
    assert_startup_resources_released(state)
    if primary_type is RuntimeError:
        assert caught.value is state.diagnostic
        assert caught.value.__cause__ is state.primary
    else:
        assert caught.value is state.primary
    assert 'diagnostic interrupted' in format_exception_notes(caught.value)
    assert 'original startup failure' in format_exception_notes(caught.value)


@pytest.mark.parametrize('primary_type', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('oom', [False, True])
def test_startup_oom_classification_preserves_cancellation(failed_start, primary_type, oom):
    state = failed_start
    state.primary = primary_type('original startup failure')
    state.oom = oom
    with pytest.raises(primary_type) as caught:
        state.start()
    assert_startup_resources_released(state)
    if oom and primary_type is RuntimeError:
        assert str(caught.value).startswith('runtime_oom:')
        assert caught.value.__cause__ is state.primary
    else:
        assert caught.value is state.primary


@pytest.mark.parametrize('diagnostic_fails', [False, True])
def test_unconfirmed_startup_cleanup_keeps_lease_and_failure_evidence(failed_start, diagnostic_fails):
    state = failed_start
    state.cleanup_error = ProcessCleanupError('fixture cleanup unconfirmed')
    if diagnostic_fails:
        state.diagnostic = OSError('memory.events read failed')
    with pytest.raises(ProcessCleanupError) as caught:
        state.start()
    assert caught.value is state.cleanup_error
    startup_error = caught.value.run_error
    assert startup_error is not None
    assert startup_error is state.primary
    assert state.retained == [state.lease]
    assert not state.lease_stream.closed
    assert state.scope.exists()
    assert same_process(state.process.identity)
    if diagnostic_fails:
        assert 'memory.events read failed' in format_exception_notes(startup_error)


def test_process_startup_diagnostics_reach_error_rows_and_stderr(failed_start, tmp_path, capsys):
    from acprof.host.cases import CaseMeasurement, CasePlans, CaseRequest, CaseWorkload
    from acprof.host.orchestrator import run_single_case

    state = failed_start
    state.diagnostic = OSError('memory.events read failed')
    output = run_single_case(
        CaseRequest(task_info=state.task, cpu=None, mem=None, gpu='off', image_info=state.info,
                    output_dir=str(tmp_path / 'results'), project_dir=str(tmp_path)),
        workload=CaseWorkload(warmup=0, repeat=1),
        measurement=CaseMeasurement(profiling_mode='basic'),
        plans=CasePlans(input_scales='1'),
    )
    assert_startup_resources_released(state)
    with open(output, encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]['status'] == 'error'
    stderr = capsys.readouterr().err
    for message in ('process_start_failed: startup verification failed', 'memory.events read failed'):
        assert message in rows[0]['error']
        assert message in stderr
    assert 'runtime_oom' not in rows[0]['error']
