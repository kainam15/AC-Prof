"""Real socket/ownership contract; the lightweight service is a protocol fixture."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from acprof.host.runtimes.contracts import ProcessInfo
from acprof.host.runtimes.process import ProcessRuntime
from acprof.process_identity import same_process


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
