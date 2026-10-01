import hashlib
import importlib.util
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof import dependency_locks, network_policy


class DependencyDownloadCacheTests(unittest.TestCase):
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
