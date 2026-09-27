"""Resume and measurement exclusion must follow execution inputs, not UI or TMPDIR."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from acprof.host import run_state


class RunIdentityTests(unittest.TestCase):
    def test_changed_manifest_prevents_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "acprof/extensions/test/manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"max_new_tokens": 64}))
            with patch.object(run_state, "MeasurementLock",
                              side_effect=lambda: run_state.ResultDirectoryLock(root / "lock")):
                state = run_state.RunState(root / "result", {}, resume=False, project_dir=root)
                state.close()
                manifest.write_text(json.dumps({"max_new_tokens": 65}))
                with self.assertRaises(run_state.RunStateError):
                    resumed = run_state.RunState(root / "result", {}, resume=True, project_dir=root)
                    resumed.close()

    def test_input_asset_changes_identity_but_ui_changes_do_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("acprof/host/client.py", "acprof/tui/i18n.py", "assets/audio/input.wav"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"original")
            identity = run_state.host_identity(root)
            (root / "acprof/tui/i18n.py").write_bytes(b"new translation")
            self.assertEqual(identity, run_state.host_identity(root))
            (root / "assets/audio/input.wav").write_bytes(b"new samples")
            self.assertNotEqual(identity, run_state.host_identity(root))

    def test_different_tmpdirs_share_the_measurement_lock(self):
        # Inject the test's lock directory in code; production exposes no env override.
        code = """from pathlib import Path
import sys
from acprof.host import run_state
run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
try:
    with run_state.MeasurementLock():
        print('acquired', flush=True)
        sys.stdin.read(1)
except run_state.RunStateError:
    print('blocked', flush=True)
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("locks", "a", "b"):
                (root / name).mkdir()
            command = [sys.executable, "-c", code, str(root / "locks")]
            first = subprocess.Popen(command, env={**os.environ, "TMPDIR": str(root / "a")},
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(first.stdout.readline().strip(), "acquired")
                second = subprocess.run(command, env={**os.environ, "TMPDIR": str(root / "b")},
                                        input="\n", capture_output=True, text=True, timeout=15)
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertEqual(second.stdout.strip(), "blocked")
            finally:
                first.communicate("\n", timeout=15)
            released = subprocess.run(command, input="\n", capture_output=True, text=True, timeout=15)
            self.assertEqual(released.returncode, 0, released.stderr)
            self.assertEqual(released.stdout.strip(), "acquired")


if __name__ == "__main__":
    unittest.main()
