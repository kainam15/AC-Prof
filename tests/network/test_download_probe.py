"""Real local HTTP sockets; no model cache writes or external network."""
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import huggingface_hub as hub
import pytest

from acprof.host.model_store.store import probe_model_download
from acprof.hub.huggingface.transport import _httpx_factory, configure_hf_transport


@pytest.fixture
def probe_server(monkeypatch):
    visited = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def respond(self, body):
            mode = self.path.rsplit('/', 1)[-1]
            if '/broken-entry/' in self.path:
                mode = 'broken'
            visited.append((self.command, mode, self.headers.get('Range')))
            if mode == 'timeout' and body:
                time.sleep(0.2)
            if mode == 'head403' or (mode in {'403', '404', '429'} and body):
                self.send_response(403 if mode == 'head403' else int(mode))
                self.end_headers()
                return
            if mode in {'redirect', 'loop', 'untrusted'}:
                self.send_response(302)
                self.send_header('Location', '/data' if mode == 'redirect' else
                                 '/loop' if mode == 'loop' else 'https://untrusted.invalid/weights')
                self.send_header('Content-Length', '1000000000')
                self.end_headers()
                # A redirect body must never be drained, even without Range support.
                return
            ranged = body and self.headers.get('Range') and mode != 'ignore'
            length = int(self.headers['Range'].split('-')[1]) + 1 if ranged else 1024 * 1024
            self.send_response(206 if ranged else 200)
            self.send_header('Content-Length', str(length))
            if ranged:
                self.send_header('Content-Range', f'bytes 0-{length - 1}/1048576')
            self.end_headers()
            if not body:
                return
            try:
                if mode == 'broken':
                    self.wfile.write(b'x' * 1024)
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                else:
                    for _ in range(0, length, 1024):
                        self.wfile.write(b'x' * min(1024, length))
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_port}'
    for key in ('ACPROF_MAX_DOWNLOAD', 'ACPROF_MODEL_MAX_DOWNLOAD', 'HF_FALLBACK_ENDPOINTS'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('HF_ENDPOINT', endpoint)
    monkeypatch.setenv('HF_DOWNLOAD_MODE', 'mirror-only')
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    configure_hf_transport()
    hub.set_client_factory(_httpx_factory)
    yield endpoint, visited
    hub.close_session()
    server.shutdown()
    server.server_close()
    thread.join()


def plan_for(endpoint, filename='data'):
    return {'model_id': 'demo/model', 'model_revision': 'a' * 40, 'endpoint': endpoint,
            'selected_bytes': 1048576, 'files': [{'path': filename, 'size': 1048576}]}


@pytest.mark.parametrize('mode', ['data', 'ignore', 'redirect'])
def test_head_then_bounded_get_without_cache_writes(probe_server, tmp_path, mode):
    endpoint, visited = probe_server
    report = probe_model_download(plan_for(endpoint, mode), tmp_path)[-1]
    assert any(method == 'GET' for method, _, _ in visited)
    assert visited[0][0] == 'HEAD'
    assert report['status'] == 'passed'
    assert report['http_status'] == (200 if mode == 'ignore' else 206)
    assert 0 < report['received_bytes'] <= 256 * 1024
    assert report['final_host'] == '127.0.0.1'
    assert report['elapsed_seconds'] >= 0
    assert any(value == 'bytes=0-262143' for _, _, value in visited)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(('mode', 'reason'), [
    ('403', 'http-403'), ('404', 'http-404'), ('429', 'http-429'),
    ('broken', 'transfer-interrupted'), ('loop', 'redirect-loop'),
    ('untrusted', 'untrusted-host'),
])
def test_probe_returns_actionable_failures(probe_server, tmp_path, mode, reason):
    endpoint, _ = probe_server
    report = probe_model_download(plan_for(endpoint, mode), tmp_path)[-1]
    assert report['status'] == 'failed'
    assert report['reason'] == reason
    assert report['received_bytes'] <= 256 * 1024


def test_read_timeout_is_reported(probe_server, tmp_path):
    endpoint, _ = probe_server
    report = probe_model_download(plan_for(endpoint, 'timeout'), tmp_path, timeout=0.05)[-1]
    assert report['status'] == 'failed'
    assert report['reason'] == 'read-timeout'


@pytest.mark.parametrize('setting', ['ACPROF_MAX_DOWNLOAD', 'ACPROF_MODEL_MAX_DOWNLOAD'])
def test_zero_budget_never_sends_get(probe_server, tmp_path, monkeypatch, setting):
    endpoint, visited = probe_server
    monkeypatch.setenv(setting, '0')
    report = probe_model_download(plan_for(endpoint), tmp_path)[-1]
    assert all(method == 'HEAD' for method, _, _ in visited)
    assert report['status'] == 'unverified'
    assert report['reason'] == 'probe-budget'
    assert report['received_bytes'] == 0


def test_complete_cache_hit_uses_no_network(probe_server, tmp_path):
    endpoint, visited = probe_server
    with patch('acprof.host.model_store.store.cached_file', return_value=tmp_path / 'cached'):
        assert probe_model_download(plan_for(endpoint), tmp_path) == []
    assert visited == []


def test_failed_mirror_tries_existing_fallback_policy(probe_server, tmp_path, monkeypatch):
    endpoint, visited = probe_server
    monkeypatch.setenv('HF_DOWNLOAD_MODE', 'auto')
    monkeypatch.setenv('HF_FALLBACK_ENDPOINTS', endpoint)
    monkeypatch.setenv('HF_ENDPOINT', 'http://127.0.0.1:1')
    plan = plan_for('http://127.0.0.1:1')
    reports = probe_model_download(plan, tmp_path)
    assert reports[0]['status'] == 'failed'
    assert reports[-1]['status'] == 'passed'
    assert plan['endpoint'] == endpoint
    assert any(method == 'GET' for method, _, _ in visited)
    assert sum(item['received_bytes'] for item in reports) <= 256 * 1024


def test_dependency_sources_check_each_origin_once_and_skip_hits(probe_server):
    from acprof.host.network_preflight import probe_runtime_downloads
    from acprof.network_policy import DownloadSource

    endpoint, visited = probe_server
    sources = [DownloadSource('python', endpoint + '/redirect', 10),
               DownloadSource('python', endpoint + '/other', 20),
               DownloadSource('debian', endpoint + '/head403', 30),
               DownloadSource('python', 'https://cached.invalid/x', 0, cache_status='hit')]
    reports = probe_runtime_downloads(sources)
    assert [item['status'] for item in reports] == ['passed', 'failed']
    assert reports[1]['reason'] == 'http-403'
    assert reports[1]['final_host'] == '127.0.0.1'
    assert all(item['received_bytes'] == 0 for item in reports)
    assert all(method == 'HEAD' for method, _, _ in visited)
    assert [path for _, path, _ in visited] == ['redirect', 'data', 'head403']


def test_probe_budget_shared_by_repositories(probe_server, tmp_path):
    endpoint, _ = probe_server
    plan = plan_for(endpoint)
    plan['dependencies'] = [{'download': plan_for(endpoint, 'ignore')}]
    reports = probe_model_download(plan, tmp_path)
    assert sum(item['received_bytes'] for item in reports) <= 256 * 1024


def test_small_remaining_budget_does_not_prefetch_a_transport_buffer(probe_server, tmp_path):
    endpoint, visited = probe_server
    report = probe_model_download(plan_for(endpoint), tmp_path, max_bytes=1024)[-1]
    assert report['status'] == 'unverified'
    assert report['received_bytes'] == 0
    assert all(method == 'HEAD' for method, _, _ in visited)


def test_partial_failed_transfer_counts_against_fallback_budget(probe_server, tmp_path, monkeypatch):
    endpoint, _ = probe_server
    primary = endpoint + '/broken-entry'
    monkeypatch.setenv('HF_ENDPOINT', primary)
    monkeypatch.setenv('HF_DOWNLOAD_MODE', 'auto')
    monkeypatch.setenv('HF_FALLBACK_ENDPOINTS', endpoint)
    reports = probe_model_download(plan_for(primary), tmp_path)
    assert reports[0]['reason'] == 'transfer-interrupted'
    assert reports[0]['received_bytes'] == 1024
    assert reports[-1]['status'] == 'passed'
    assert sum(item['received_bytes'] for item in reports) <= 256 * 1024


@pytest.mark.parametrize(('error', 'reason'), [
    (socket.gaierror('name lookup failed'), 'dns'),
    (ssl.SSLCertVerificationError('certificate failed'), 'tls'),
    (httpx.ConnectTimeout('connect timed out'), 'connect-timeout'),
])
def test_transport_errors_keep_their_reason_without_leaking_credentials(probe_server, tmp_path, error, reason):
    import json

    endpoint, _ = probe_server

    def fail(request):
        raise httpx.ConnectError('credential=do-not-record', request=request) from error

    hub.set_client_factory(lambda: _httpx_factory(transport=httpx.MockTransport(fail)))
    reports = probe_model_download(plan_for(endpoint), tmp_path)
    assert reports[-1]['status'] == 'failed'
    assert reports[-1]['reason'] == reason
    assert 'do-not-record' not in json.dumps(reports)


def test_offline_mode_has_no_socket_requests(probe_server, tmp_path, monkeypatch):
    endpoint, visited = probe_server
    monkeypatch.setattr(hub.constants, 'HF_HUB_OFFLINE', True)
    reports = probe_model_download(plan_for(endpoint), tmp_path)
    assert reports[-1]['reason'] == 'offline'
    assert visited == []


@pytest.mark.parametrize('extra_budget', [0, 1024, 128 * 1024])
def test_preflight_reserves_full_download_before_get_and_passes_evidence_to_confirmation(
    probe_server, tmp_path, monkeypatch, extra_budget,
):
    from acprof.host.network_preflight import preflight

    endpoint, visited = probe_server
    monkeypatch.setenv('ACPROF_MAX_DOWNLOAD', str(1048576 + extra_budget))
    monkeypatch.setenv('ACPROF_INTERACTIVE_PREPARATION', '1')
    task = SimpleNamespace(model_id='demo/model', model_revision='a' * 40,
        task_family='nlp', runtime_backend='transformers_pipeline', model_adapter='family-default')
    with (
        patch('acprof.host.model_store.store.require_space', return_value={}),
        patch('acprof.host.collection_workflow.PreparationWorkflow.ask') as ask,
    ):
        report = preflight(task, None, tmp_path, plan_for(endpoint), root=tmp_path,
                           runtime_plan=([], {'kind': 'process'}))
    assert report['expected_download_bytes'] == 1048576
    assert report['probe_payload_bytes'] <= extra_budget
    assert any(method == 'GET' for method, _, _ in visited) == (extra_budget >= 65536)
    assert ask.call_args.kwargs['download_report']['network_probe'] == report['network_probe']
    assert list(tmp_path.iterdir()) == []
