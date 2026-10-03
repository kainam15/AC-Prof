"""Resume and measurement exclusion must follow execution inputs, not UI or TMPDIR."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.host import run_state


def test_changed_manifest_prevents_resume():
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
            with pytest.raises(run_state.RunStateError):
                resumed = run_state.RunState(root / "result", {}, resume=True, project_dir=root)
                resumed.close()

def test_input_asset_changes_identity_but_ui_changes_do_not():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for relative in ("acprof/host/client.py", "acprof/tui/i18n.py", "acprof/cli/plot.py", "assets/audio/input.wav"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"original")
        identity = run_state.host_identity(root)
        (root / "acprof/tui/i18n.py").write_bytes(b"new translation")
        (root / "acprof/cli/plot.py").write_bytes(b"new presentation")
        assert (identity) == (run_state.host_identity(root))
        (root / "assets/audio/input.wav").write_bytes(b"new samples")
        assert (identity) != (run_state.host_identity(root))

def test_different_tmpdirs_share_the_measurement_lock():
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
            assert (first.stdout.readline().strip()) == ("acquired")
            second = subprocess.run(command, env={**os.environ, "TMPDIR": str(root / "b")},
                                    input="\n", capture_output=True, text=True, timeout=15)
            assert (second.returncode) == (0), second.stderr
            assert (second.stdout.strip()) == ("blocked")
        finally:
            first.communicate("\n", timeout=15)
        released = subprocess.run(command, input="\n", capture_output=True, text=True, timeout=15)
        assert (released.returncode) == (0), released.stderr
        assert (released.stdout.strip()) == ("acquired")
