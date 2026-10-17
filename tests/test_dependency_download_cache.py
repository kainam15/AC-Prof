import hashlib
import importlib.util
import json
import ssl
import sys
import tempfile
from contextlib import contextmanager
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import pytest

from acprof import dependency_locks, network_policy
from acprof.host.network_preflight import artifact_size


class TestDependencyDownloadCache:
    @contextmanager
    def artifact_server(self, payload):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):
                agent = self.headers.get("User-Agent", "")
                requests.append((self.command, self.path, agent))
                if not agent.startswith("acprof") or self.path == "/forbidden":
                    self.send_error(403)
                elif self.path == "/artifact":
                    self.send_response(302)
                    self.send_header("Location", "/payload")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    if self.command == "GET":
                        self.wfile.write(payload)

            do_GET = do_HEAD

            def log_message(self, format, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            thread.start()
            try:
                with patch.dict("os.environ", {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}):
                    yield f"http://127.0.0.1:{server.server_port}", requests
            finally:
                server.shutdown()
                thread.join()

    def test_download_identifies_client_through_redirect_and_publishes_verified_cache(self):
        helper = self.helper()
        payload = b"fixed dependency payload"
        with self.artifact_server(payload) as (base_url, requests), tempfile.TemporaryDirectory() as directory:
            entry = {"name": "example", "sha256": hashlib.sha256(payload).hexdigest(),
                     "url": base_url + "/artifact"}
            target = Path(directory) / "example.whl"
            with patch.object(helper, "_record_transfer") as recorded:
                assert (helper.cached_artifact(entry, target, "python")) == (target)
                assert (target.read_bytes()) == (payload)
                assert (recorded.call_args.args[2:]) == ((len(payload), "127.0.0.1", "miss"))
                assert (helper.cached_artifact(entry, target, "python")) == (target)
                assert (recorded.call_args.args[2:]) == ((0, "127.0.0.1", "hit"))
            assert ([(method, path) for method, path, _ in requests]) == ([("GET", "/artifact"), ("GET", "/payload")])
            assert (requests[0][2]) == (requests[1][2])
            assert (json.loads(target.with_suffix(".whl.source.json").read_text())) == ({"actual_source_host": "127.0.0.1"})
            assert not (target.with_suffix(".whl.part").exists())

    def test_preflight_identifies_client_through_redirect_without_getting_payload(self):
        payload = b"fixed dependency payload"
        with self.artifact_server(payload) as (base_url, requests):
            assert (artifact_size(base_url + "/artifact")) == (len(payload))
            assert (artifact_size(base_url + "/forbidden")) is None
            assert (artifact_size("invalid-url")) is None
        assert ([(method, path) for method, path, _ in requests]) == ([("HEAD", "/artifact"), ("HEAD", "/payload"), ("HEAD", "/forbidden")])
        assert (all(agent == requests[0][2] for _, _, agent in requests))

    def helper(self):
        path = Path(__file__).resolve().parents[1] / "dockerfiles/environment_tools.py"
        spec = importlib.util.spec_from_file_location("acprof_build_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependency_locks=dependency_locks, network_policy=network_policy):
            spec.loader.exec_module(module)
        return module

    def test_verified_cache_hit_does_not_open_network_and_reports_savings(self):
        helper = self.helper()
        data = b"fixed wheel payload"
        entry = {"name": "example", "sha256": hashlib.sha256(data).hexdigest(),
                 "url": "https://mirror.example/example.whl"}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "example.whl"
            target.write_bytes(data)
            with patch.object(helper, "_record_transfer") as recorded, patch("urllib.request.build_opener") as network:
                assert (helper.cached_artifact(entry, target, "python")) == (target)
                network.assert_not_called()
                assert (recorded.call_args.args[2:]) == ((0, None, "hit"))

    def test_download_is_verified_before_publishing_and_not_retried_on_hash_failure(self):
        helper = self.helper()
        entry = {"name": "example", "sha256": hashlib.sha256(b"correct").hexdigest(),
                 "url": "https://mirror.example/example.whl"}
        stream = BytesIO(b"corrupt")
        stream.geturl = lambda: entry["url"]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "example.whl"
            opener = SimpleNamespace(open=lambda *a, **kw: stream)
            with patch("urllib.request.build_opener", return_value=opener), patch.object(helper, "_record_transfer"):
                with pytest.raises(ValueError, match="SHA256"):
                    helper.cached_artifact(entry, target, "python")
            assert not (target.exists())
            assert (list(Path(directory).iterdir())) == ([])

    def test_wheel_and_debian_redirects_use_system_network_without_egress_guessing(self):
        helper = self.helper()
        import urllib.request
        request = urllib.request.Request("https://hf-mirror.com/artifact")
        with patch.dict("os.environ", {}, clear=True):
            redirected = helper.PolicyRedirectHandler().redirect_request(request, None, 302, "Found", {}, "https://files.pythonhosted.org/artifact")
            assert redirected.full_url == "https://files.pythonhosted.org/artifact"
            assert network_policy.source_route(redirected.full_url) == "direct-socket"

    @pytest.mark.parametrize('error', (
        URLError(ssl.SSLEOFError(8, 'unexpected EOF during handshake')),
        TimeoutError('connection timeout'), ConnectionResetError('connection reset'),
        IncompleteRead(b'partial', 100),
    ))
    def test_transient_get_failure_retries_same_source_before_publishing(self, tmp_path, error):
        helper = self.helper()
        payload = b'verified dependency payload'
        entry = {'name': 'example', 'sha256': hashlib.sha256(payload).hexdigest(),
                 'url': 'https://files.pythonhosted.org/example.whl'}
        stream = BytesIO(payload)
        stream.geturl = lambda: entry['url']
        opener = SimpleNamespace(open=Mock(side_effect=[error, stream]))
        target = tmp_path / 'example.whl'
        with patch('urllib.request.build_opener', return_value=opener), \
                patch('time.sleep') as sleep, patch.object(helper, '_record_transfer') as recorded:
            assert helper.cached_artifact(entry, target, 'python') == target
        assert target.read_bytes() == payload
        assert [call.args[0].full_url for call in opener.open.call_args_list] == [entry['url']] * 2
        assert all(call.args[0].get_header('User-agent') == network_policy.DEPENDENCY_USER_AGENT
                   for call in opener.open.call_args_list)
        assert sleep.call_count == 1
        recorded.assert_called_once_with(entry, 'python', len(payload), 'files.pythonhosted.org', 'miss')
        assert not target.with_suffix('.whl.part').exists()

    def test_partial_download_is_discarded_before_retry(self, tmp_path):
        helper = self.helper()
        payload = b'verified dependency payload'
        entry = {'name': 'example', 'sha256': hashlib.sha256(payload).hexdigest(),
                 'url': 'https://files.pythonhosted.org/example.whl'}
        broken, complete = BytesIO(), BytesIO(payload)
        broken.read = Mock(side_effect=[b'partial corrupt bytes', ssl.SSLEOFError(8, 'interrupted body')])
        complete.geturl = lambda: entry['url']
        opener = SimpleNamespace(open=Mock(side_effect=[broken, complete]))
        target = tmp_path / 'example.whl'
        with patch('urllib.request.build_opener', return_value=opener), \
                patch('time.sleep'), patch.object(helper, '_record_transfer') as recorded:
            helper.cached_artifact(entry, target, 'python')
        assert target.read_bytes() == payload
        assert broken.closed and complete.closed
        assert not target.with_suffix('.whl.part').exists()
        assert recorded.call_count == 1

    def test_transient_download_failure_exhausts_budget_and_leaves_no_cache(self, tmp_path):
        helper = self.helper()
        entry = {'name': 'example', 'sha256': 'a' * 64, 'url': 'https://files.pythonhosted.org/example.whl'}
        error = URLError(ssl.SSLEOFError(8, 'unexpected EOF'))
        opener = SimpleNamespace(open=Mock(side_effect=error))
        with patch('urllib.request.build_opener', return_value=opener), \
                patch('time.sleep') as sleep, patch.object(helper, '_record_transfer') as recorded:
            with pytest.raises(URLError) as caught:
                helper.cached_artifact(entry, tmp_path / 'example.whl', 'python')
        assert caught.value is error
        assert opener.open.call_count == 3
        assert [call.args[0] for call in sleep.call_args_list] == [1, 2]
        recorded.assert_not_called()
        assert list(tmp_path.iterdir()) == []

    def test_resume_after_interrupted_build_keeps_partial_and_verifies_complete_hash(self, tmp_path):
        helper = self.helper()
        payload = b"complete SHA256 must cover every byte of this wheel"
        prefix = payload[:17]
        entry = {"name": "example", "sha256": hashlib.sha256(payload).hexdigest(),
                 "url": "https://files.pythonhosted.org/example.whl"}
        target = tmp_path / "example.whl"
        first = BytesIO(prefix)
        first.geturl = lambda: entry["url"]
        first.status = 200
        first.headers = {"Content-Length": str(len(payload))}
        first_opener = SimpleNamespace(open=Mock(side_effect=[
            first, TimeoutError("network paused"), TimeoutError("network paused"),
        ]))
        with patch("urllib.request.build_opener", return_value=first_opener), \
                patch("time.sleep"), patch.object(helper, "_record_transfer"):
            with pytest.raises(TimeoutError, match="network paused"):
                helper.cached_artifact(entry, target, "python")
        partial = target.with_suffix(".whl.part")
        metadata = target.with_suffix(".whl.part.source.json")
        assert partial.read_bytes() == prefix
        assert json.loads(metadata.read_text())["sha256"] == entry["sha256"]
        assert not target.exists()

        rest = BytesIO(payload[len(prefix):])
        rest.status = 206
        rest.geturl = lambda: entry["url"]
        rest.headers = {
            "Content-Range": f"bytes {len(prefix)}-{len(payload) - 1}/{len(payload)}",
            "Content-Length": str(len(payload) - len(prefix)),
        }
        second_opener = SimpleNamespace(open=Mock(return_value=rest))
        with patch("urllib.request.build_opener", return_value=second_opener), \
                patch.object(helper, "_record_transfer") as recorded:
            assert helper.cached_artifact(entry, target, "python") == target
        assert second_opener.open.call_args.args[0].get_header("Range") == f"bytes={len(prefix)}-"
        assert target.read_bytes() == payload
        assert not partial.exists()
        assert not metadata.exists()
        recorded.assert_called_once_with(entry, "python", len(payload), "files.pythonhosted.org", "miss")

    def test_resume_restarts_safely_when_server_ignores_range(self, tmp_path):
        helper = self.helper()
        payload = b"a whole payload must not append twice"
        entry = {"name": "example", "sha256": hashlib.sha256(payload).hexdigest(),
                 "url": "https://files.pythonhosted.org/example.whl"}
        target = tmp_path / "example.whl"
        target.with_suffix(".whl.part").write_bytes(payload[:8])
        target.with_suffix(".whl.part.source.json").write_text(json.dumps({
            "url": entry["url"], "sha256": entry["sha256"],
            "actual_source_host": "files.pythonhosted.org",
        }))
        response = BytesIO(payload)
        response.status = 200
        response.headers = {"Content-Length": str(len(payload))}
        response.geturl = lambda: entry["url"]
        opener = SimpleNamespace(open=Mock(return_value=response))
        with patch("urllib.request.build_opener", return_value=opener), \
                patch.object(helper, "_record_transfer"):
            helper.cached_artifact(entry, target, "python")
        assert opener.open.call_args.args[0].get_header("Range") == "bytes=8-"
        assert target.read_bytes() == payload

    def test_bad_content_range_never_publishes_or_preserves_partial(self, tmp_path):
        helper = self.helper()
        payload = b"abcdefghijklmno"
        entry = {"name": "example", "sha256": hashlib.sha256(payload).hexdigest(),
                 "url": "https://files.pythonhosted.org/example.whl"}
        target = tmp_path / "example.whl"
        target.with_suffix(".whl.part").write_bytes(payload[:4])
        target.with_suffix(".whl.part.source.json").write_text(json.dumps({
            "url": entry["url"], "sha256": entry["sha256"],
            "actual_source_host": "files.pythonhosted.org",
        }))
        response = BytesIO(payload[4:])
        response.status = 206
        response.headers = {"Content-Range": f"bytes 3-14/{len(payload)}"}
        response.geturl = lambda: entry["url"]
        opener = SimpleNamespace(open=Mock(return_value=response))
        with patch("urllib.request.build_opener", return_value=opener), \
                patch.object(helper, "_record_transfer"):
            with pytest.raises(ValueError, match="Content-Range"):
                helper.cached_artifact(entry, target, "python")
        assert not target.exists()
        assert not target.with_suffix(".whl.part").exists()
        assert not target.with_suffix(".whl.part.source.json").exists()

    @pytest.mark.parametrize('error', (
        URLError(ssl.SSLCertVerificationError(1, 'certificate rejected')),
        URLError(ssl.SSLError(1, 'unsupported TLS version')),
        HTTPError('https://files.pythonhosted.org/example.whl', 403, 'Forbidden', {}, None),
        network_policy.DownloadPolicyError('source transition forbidden'),
        PermissionError('cache not writable'),
    ))
    def test_permanent_download_errors_are_not_retried(self, tmp_path, error):
        helper = self.helper()
        entry = {'name': 'example', 'sha256': 'a' * 64, 'url': 'https://files.pythonhosted.org/example.whl'}
        opener = SimpleNamespace(open=Mock(side_effect=error))
        with patch('urllib.request.build_opener', return_value=opener), \
                patch('time.sleep') as sleep, patch.object(helper, '_record_transfer') as recorded:
            with pytest.raises(type(error)) as caught:
                helper.cached_artifact(entry, tmp_path / 'example.whl', 'python')
        assert caught.value is error
        assert opener.open.call_count == 1
        sleep.assert_not_called()
        recorded.assert_not_called()
        assert list(tmp_path.iterdir()) == []


class TestDomesticPyPIArtifactMirror:
    """A selected direct mirror cannot silently fall back to a costly proxy."""

    def test_exact_route_classification_and_validation(self, monkeypatch):
        from acprof.network_policy import (
            DownloadPolicyError,
            DownloadSource,
            dependency_artifact_url,
            require_direct_artifact_mirror_redirect,
        )
        original = "https://files.pythonhosted.org/packages/a/b/locked.whl"
        opts = {"ACPROF_PYPI_ARTIFACT_MIRROR": "tencent", "HTTPS_PROXY": "http://jump:7897"}
        actual = dependency_artifact_url("python", original, opts)
        assert actual == "https://mirrors.cloud.tencent.com/pypi/packages/a/b/locked.whl"
        assert dependency_artifact_url("debian", original, opts) == original
        assert dependency_artifact_url("python", "https://download-r2.pytorch.org/whl/test.whl", opts).startswith(
            "https://download-r2.pytorch.org/"
        )
        monkeypatch.setenv("ACPROF_PYPI_ARTIFACT_MIRROR", "tencent")
        monkeypatch.setenv("HTTPS_PROXY", "http://charged-vpn.example:7897")
        assert DownloadSource("python", actual, 10).to_dict()["route"] == "direct-socket"
        assert dependency_artifact_url("python", original, {}) == original
        with pytest.raises(DownloadPolicyError, match="must be empty or tencent"):
            dependency_artifact_url("python", original, {"ACPROF_PYPI_ARTIFACT_MIRROR": "unknown"})
        with pytest.raises(DownloadPolicyError, match="unexpected"):
            dependency_artifact_url("python", "http://files.pythonhosted.org/packages/x.whl", opts)
        with pytest.raises(DownloadPolicyError, match="redirected outside"):
            require_direct_artifact_mirror_redirect(actual, original)
        require_direct_artifact_mirror_redirect(actual, actual)

    def test_direct_mirror_download_keeps_locked_digest_and_provenance(self, monkeypatch, tmp_path):
        helper = TestDependencyDownloadCache().helper()
        payload = b"verified mirror wheel"
        original = "https://files.pythonhosted.org/packages/abc/example.whl"
        mirror = "https://mirrors.cloud.tencent.com/pypi/packages/abc/example.whl"
        entry = {"name": "example", "sha256": hashlib.sha256(payload).hexdigest(), "url": original}
        monkeypatch.setenv("ACPROF_PYPI_ARTIFACT_MIRROR", "tencent")
        monkeypatch.setenv("HTTPS_PROXY", "http://charged-vpn.example:7897")
        response = BytesIO(payload)
        response.geturl = lambda: mirror
        response.status = 200
        response.headers = {"Content-Length": str(len(payload))}
        opener = SimpleNamespace(open=Mock(return_value=response))
        with patch("urllib.request.build_opener", return_value=opener) as build, \
                patch.object(helper, "_record_transfer") as record:
            target = tmp_path / "example.whl"
            assert helper.cached_artifact(entry, target, "python") == target
        assert target.read_bytes() == payload
        assert build.call_args.args[0].proxies == {}
        assert opener.open.call_args.args[0].full_url == mirror
        assert record.call_args.args == (entry, "python", len(payload), "mirrors.cloud.tencent.com", "miss")
        assert json.loads(target.with_suffix(".whl.source.json").read_text())["actual_source_host"] == (
            "mirrors.cloud.tencent.com"
        )

    def test_mirror_http_error_fails_closed_and_never_requests_official(self, monkeypatch, tmp_path):
        helper = TestDependencyDownloadCache().helper()
        monkeypatch.setenv("ACPROF_PYPI_ARTIFACT_MIRROR", "tencent")
        entry = {"name": "example", "sha256": "a" * 64,
                 "url": "https://files.pythonhosted.org/packages/example.whl"}
        error = HTTPError("https://mirrors.cloud.tencent.com/pypi/packages/example.whl", 403,
                          "Forbidden", {}, None)
        opener = SimpleNamespace(open=Mock(side_effect=error))
        with patch("urllib.request.build_opener", return_value=opener), \
                patch.object(helper, "_record_transfer"):
            with pytest.raises(HTTPError):
                helper.cached_artifact(entry, tmp_path / "file.whl", "python")
        assert opener.open.call_count == 1
        assert opener.open.call_args.args[0].full_url.startswith("https://mirrors.cloud.tencent.com/")

    def test_old_source_partial_is_invalidated_before_mirror_request(self, monkeypatch, tmp_path):
        helper = TestDependencyDownloadCache().helper()
        original = "https://files.pythonhosted.org/packages/abc/example.whl"
        mirror = "https://mirrors.cloud.tencent.com/pypi/packages/abc/example.whl"
        payload = b"complete mirror checksum"
        entry = {"name": "example", "sha256": hashlib.sha256(payload).hexdigest(), "url": original}
        target = tmp_path / "file.whl"
        target.with_suffix(".whl.part").write_bytes(b"old-source-")
        target.with_suffix(".whl.part.source.json").write_text(
            json.dumps({"url": original, "sha256": entry["sha256"],
                        "actual_source_host": "files.pythonhosted.org"})
        )
        monkeypatch.setenv("ACPROF_PYPI_ARTIFACT_MIRROR", "tencent")
        response = BytesIO(payload)
        response.geturl = lambda: mirror
        response.status = 200
        response.headers = {"Content-Length": str(len(payload))}
        opener = SimpleNamespace(open=Mock(return_value=response))
        with patch("urllib.request.build_opener", return_value=opener), \
                patch.object(helper, "_record_transfer"):
            helper.cached_artifact(entry, target, "python")
        assert target.read_bytes() == payload
        assert not opener.open.call_args.args[0].has_header("Range")

    def test_mirror_head_probe_does_not_use_environment_proxy(self, monkeypatch):
        from acprof.host.network_preflight import artifact_size
        monkeypatch.setenv("ACPROF_PYPI_ARTIFACT_MIRROR", "tencent")
        monkeypatch.setenv("HTTPS_PROXY", "http://charged-vpn.example:7897")
        url = "https://mirrors.cloud.tencent.com/pypi/packages/abc/sample.whl"
        response = BytesIO(b"")
        response.headers = {"Content-Length": "12345"}
        opener = SimpleNamespace(open=Mock(return_value=response))
        with patch("urllib.request.build_opener", return_value=opener) as build:
            assert artifact_size(url) == 12345
        assert build.call_args.args[0].proxies == {}
        assert opener.open.call_args.args[0].full_url == url
        assert opener.open.call_args.args[0].get_method() == "HEAD"

    def test_mirror_resume_uses_range_and_complete_lock_hash(self, monkeypatch, tmp_path):
        helper = TestDependencyDownloadCache().helper()
        original = "https://files.pythonhosted.org/packages/abc/example.whl"
        mirror = "https://mirrors.cloud.tencent.com/pypi/packages/abc/example.whl"
        payload = b"complete mirror download with resumable exact hashes"
        entry = {"name": "example", "sha256": hashlib.sha256(payload).hexdigest(), "url": original}
        target = tmp_path / "file.whl"
        offset = 13
        target.with_suffix(".whl.part").write_bytes(payload[:offset])
        target.with_suffix(".whl.part.source.json").write_text(json.dumps({
            "url": mirror, "sha256": entry["sha256"], "actual_source_host": "mirrors.cloud.tencent.com",
        }))
        monkeypatch.setenv("ACPROF_PYPI_ARTIFACT_MIRROR", "tencent")
        response = BytesIO(payload[offset:])
        response.status = 206
        response.geturl = lambda: mirror
        response.headers = {"Content-Range": f"bytes {offset}-{len(payload)-1}/{len(payload)}",
                            "Content-Length": str(len(payload)-offset)}
        opener = SimpleNamespace(open=Mock(return_value=response))
        with patch("urllib.request.build_opener", return_value=opener), \
                patch.object(helper, "_record_transfer"):
            helper.cached_artifact(entry, target, "python")
        assert opener.open.call_args.args[0].get_header("Range") == f"bytes={offset}-"
        assert target.read_bytes() == payload
        assert not target.with_suffix(".whl.part").exists()

    def test_mirror_hash_mismatch_rejects_file(self, monkeypatch, tmp_path):
        helper = TestDependencyDownloadCache().helper()
        original = "https://files.pythonhosted.org/packages/abc/example.whl"
        mirror = "https://mirrors.cloud.tencent.com/pypi/packages/abc/example.whl"
        entry = {"name": "example", "sha256": hashlib.sha256(b"trusted").hexdigest(), "url": original}
        response = BytesIO(b"tampered")
        response.status = 200
        response.geturl = lambda: mirror
        response.headers = {"Content-Length": str(len(b"tampered"))}
        monkeypatch.setenv("ACPROF_PYPI_ARTIFACT_MIRROR", "tencent")
        opener = SimpleNamespace(open=Mock(return_value=response))
        with patch("urllib.request.build_opener", return_value=opener), \
                patch.object(helper, "_record_transfer") as record:
            with pytest.raises(ValueError, match="SHA256 mismatch"):
                helper.cached_artifact(entry, tmp_path / "bad.whl", "python")
        assert not (tmp_path / "bad.whl").exists()
        record.assert_not_called()
