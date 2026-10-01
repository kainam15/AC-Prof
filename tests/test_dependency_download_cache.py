import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch

from acprof import dependency_locks, network_policy
from acprof.host.network_preflight import artifact_size


class DependencyDownloadCacheTests(unittest.TestCase):
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
                self.assertEqual(helper.cached_artifact(entry, target, "python"), target)
                self.assertEqual(target.read_bytes(), payload)
                self.assertEqual(recorded.call_args.args[2:], (len(payload), "127.0.0.1", "miss"))
                self.assertEqual(helper.cached_artifact(entry, target, "python"), target)
                self.assertEqual(recorded.call_args.args[2:], (0, "127.0.0.1", "hit"))
            self.assertEqual([(method, path) for method, path, _ in requests],
                             [("GET", "/artifact"), ("GET", "/payload")])
            self.assertEqual(requests[0][2], requests[1][2])
            self.assertEqual(json.loads(target.with_suffix(".whl.source.json").read_text()),
                             {"actual_source_host": "127.0.0.1"})
            self.assertFalse(target.with_suffix(".whl.part").exists())

    def test_preflight_identifies_client_through_redirect_without_getting_payload(self):
        payload = b"fixed dependency payload"
        with self.artifact_server(payload) as (base_url, requests):
            self.assertEqual(artifact_size(base_url + "/artifact"), len(payload))
            self.assertIsNone(artifact_size(base_url + "/forbidden"))
            self.assertIsNone(artifact_size("invalid-url"))
        self.assertEqual([(method, path) for method, path, _ in requests],
                         [("HEAD", "/artifact"), ("HEAD", "/payload"), ("HEAD", "/forbidden")])
        self.assertTrue(all(agent == requests[0][2] for _, _, agent in requests))

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
                self.assertEqual(helper.cached_artifact(entry, target, "python"), target)
                network.assert_not_called()
                self.assertEqual(recorded.call_args.args[2:], (0, None, "hit"))

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
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    helper.cached_artifact(entry, target, "python")
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_wheel_and_debian_redirects_cannot_silently_escalate_to_proxy(self):
        helper = self.helper()
        import urllib.request
        request = urllib.request.Request("https://hf-mirror.com/artifact")
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(network_policy.DownloadPolicyError):
            helper.PolicyRedirectHandler().redirect_request(request, None, 302, "Found", {}, "https://files.pythonhosted.org/artifact")
