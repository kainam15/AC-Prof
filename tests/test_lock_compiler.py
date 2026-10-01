"""锁检查保持只读，并拒绝与目标解释器不兼容的制品。"""
import hashlib
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from runtime_fixture import copy_dependency_tree

from scripts import compile_locks
from scripts.compile_locks import check_catalog
from scripts.compile_system_lock import resolve_in_container, snapshot_url


class LockCompilerTests(unittest.TestCase):
    def test_mirror_resolution_keeps_versions_hashes_and_target_platform(self):
        from acprof.dependency_locks import python_lock_text, read_python_lock
        from acprof.runtime_profiles import PLATFORMS
        digest = "a" * 64
        wheel_url = "https://mirror.example/packages/example-1.0-py3-none-any.whl"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "requirements.txt"
            output.write_text(python_lock_text([{"name": "example", "version": "1.0",
                "url": "https://files.pythonhosted.org/example-1.0-py3-none-any.whl", "sha256": digest}]))
            original = output.read_bytes()
            if sys.version_info < (3, 11):
                with patch("subprocess.run") as run, self.assertRaisesRegex(RuntimeError, r"Python 3\.11\+"):
                    compile_locks.resolve("uv", [root / "input.in"], output, PLATFORMS["cu124"], {"example": "1.0"},
                                          index_url="https://mirror.example/simple")
                run.assert_not_called()
                self.assertEqual(output.read_bytes(), original)
                self.assertFalse(output.with_suffix(".artifacts.json").exists())
                return

            def resolve(command, **kwargs):
                destination = Path(command[command.index("--output-file") + 1])
                destination.write_text('[[packages]]\nname="example"\nversion="1.0"\n'
                    '[[packages.wheels]]\nurl="' + wheel_url + '"\nsize=123\n'
                    '[packages.wheels.hashes]\nsha256="' + digest + '"\n')

            with patch("subprocess.run", side_effect=resolve) as run:
                compile_locks.resolve("uv", [root / "input.in"], output, PLATFORMS["cu124"], {"example": "1.0"},
                                      index_url="https://mirror.example/simple")
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--default-index") + 1], "https://mirror.example/simple")
            self.assertEqual(command[command.index("--python-platform") + 1], PLATFORMS["cu124"].python_target)
            self.assertEqual(read_python_lock(output)[0]["url"], wheel_url)
            self.assertEqual(read_python_lock(output)[0]["sha256"], digest)
            original = output.read_bytes()
            digest = "b" * 64
            with patch("subprocess.run", side_effect=resolve), self.assertRaisesRegex(ValueError, "artifact hash"):
                compile_locks.resolve("uv", [root / "input.in"], output, PLATFORMS["cu124"], {"example": "1.0"},
                                      index_url="https://mirror.example/simple")
            self.assertEqual(output.read_bytes(), original)
            with patch("subprocess.run", side_effect=resolve), self.assertRaisesRegex(ValueError, "artifact hash"):
                compile_locks.resolve("uv", [root / "input.in"], output, PLATFORMS["cu124"], {"example": "1.0"},
                                      torch_index_url="https://mirror.example/cu124")
            self.assertEqual(output.read_bytes(), original)

    @unittest.skipIf(sys.version_info < (3, 11), 'Host metadata checks require Python 3.11+')
    def test_host_check_rejects_metadata_drift_in_each_python_branch_without_resolving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyproject.toml').write_text('[project]\ndependencies = ["numpy>=2,<3"]\n')
            (root / 'requirements-host.in').write_text('numpy==2.2.6 ; python_version < "3.11"\n')
            lock = root / 'requirements.lock'
            lock.write_text('numpy==2.2.6 ; python_version < "3.11"\n'
                            'numpy==2.4.4 ; python_version >= "3.11"\n')
            with patch.object(compile_locks, 'ROOT', root), patch('subprocess.run', side_effect=AssertionError('resolver used')):
                self.assertEqual(compile_locks.main(['--host-only', '--check']), 0)
                lock.write_text('numpy==1.26.4 ; python_version < "3.11"\n'
                                'numpy==2.4.4 ; python_version >= "3.11"\n')
                with self.assertRaisesRegex(ValueError, 'numpy'):
                    compile_locks.main(['--host-only', '--check'])
                lock.write_text('numpy==2.2.6 ; python_version < "3.11"\n')
                with self.assertRaisesRegex(ValueError, 'numpy'):
                    compile_locks.main(['--host-only', '--check'])

    def test_check_is_read_only_and_does_not_use_docker_or_resolver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_dependency_tree(root)
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in root.rglob("*") if path.is_file()}
            with patch("subprocess.run", side_effect=AssertionError("check spawned process")), patch(
                "urllib.request.urlopen", side_effect=AssertionError("check accessed network"),
            ):
                result = check_catalog(root)
            self.assertEqual(result["profiles"], 40)
            self.assertEqual(len(result["environments"]), 27)
            onnx_environments = [environment for environment in result["environments"].values()
                                 if environment["environment_key"] == "onnxruntime-cpu"]
            self.assertEqual(len(onnx_environments), 1)
            self.assertEqual(onnx_environments[0]["platform_id"], "python-cpu")
            self.assertEqual(set(onnx_environments[0]["profiles"]),
                             {"onnxruntime-cpu", "onnxruntime-cv-cpu", "onnxruntime-nlp-cpu"})
            self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in root.rglob("*") if path.is_file()})

    def test_check_rejects_wheel_for_wrong_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_dependency_tree(root)
            path = root / "dockerfiles/locks/nlp-cpu.txt"
            text = path.read_text()
            self.assertIn("cp310-cp310", text)
            path.write_text("\n".join(line.replace("cp310-cp310", "cp311-cp311")
                                      if line.startswith("numpy @") else line for line in text.splitlines()) + "\n")
            with self.assertRaisesRegex(ValueError, "wheel.*平台|平台.*wheel"):
                check_catalog(root)

    def test_snapshot_uses_actual_timestamp_not_requested_cutoff(self):
        listing = b"20260912T203535Z 20260913T031122Z 20260901T000000Z"
        with patch("urllib.request.urlopen", return_value=BytesIO(listing)) as opened:
            url = snapshot_url("debian", "20260913T000000Z")
        self.assertEqual(url, "https://snapshot.debian.org/archive/debian/20260912T203535Z/")
        self.assertIn("?year=2026&month=9", opened.call_args.args[0])

    def test_system_resolver_cannot_mutate_host(self):
        with patch.dict("os.environ", {}, clear=True), patch("subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "容器"):
                resolve_in_container("20260913T000000Z", Path("unused"))
        run.assert_not_called()
