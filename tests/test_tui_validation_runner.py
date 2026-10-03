"""验证辅助入口能挂载 TUI、执行子进程并退出，不启动真实采集。"""
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from acprof.experiment import RunConfig

ROOT = Path(__file__).resolve().parents[1]


class TestTuiValidationRunner:
    @pytest.fixture(autouse=True)
    def _isolate_child_measurement_lock(self, measurement_lock_root):
        self.measurement_root = measurement_lock_root

    def run_validation(self, child, *, returncode=0, expected="validation child finished", runner=None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = replace(RunConfig.smoke("test/model"), output_dir=str(output / "results"), notify="none")
            command = output / "command.json"
            command.write_text(json.dumps({"config": asdict(config), "command": child}))
            runner = runner or "import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
            guard = """import os, sys
def protect_production_lock(event, args):
    if event == "open" and args[0] == f"/tmp/acprof-measurement-{os.getuid()}.lock":
        raise AssertionError("offline child accessed the production measurement lock")
sys.addaudithook(protect_production_lock)
"""
            runner = (
                "from pathlib import Path\nfrom unittest.mock import patch\n"
                "with (\n"
                "    patch('acprof.tui.app.quick_preflight', return_value=[]),\n"
                f"    patch('acprof.host.run_state.MEASUREMENT_LOCK_ROOT', Path({str(self.measurement_root)!r})),\n"
                "):\n"
                f"    exec({runner!r})\n"
            )
            runner = guard + runner
            result = subprocess.run([
                sys.executable, *(["-c", runner] if runner else []),
                str(ROOT / "scripts/run_tui_validation.py"), str(command),
            ], cwd=ROOT, env={**os.environ, "ACPROF_WECOM_WEBHOOK_URL": ""},
                capture_output=True, text=True, timeout=30)
            assert (result.returncode) == (returncode), result.stderr
            assert ("DuplicateKey") not in (result.stderr)
            assert ((output / "tui-finished.svg").is_file()), result.stderr
            rendered = " ".join(ET.parse(output / "tui-finished.svg").getroot().itertext()).replace("\u00a0", " ")
            assert (expected) in (rendered)

    def test_headless_runner_mounts_once_and_records_completed_screen(self):
        self.run_validation([sys.executable, "-c", "print('validation child finished')"])

    @pytest.mark.parametrize('returncode', (0, 7))
    def test_completion_before_monitor_refresh_keeps_output_and_exit_code(self, returncode):
        # Complete in the launch callback, before Textual can render the monitor.
        runner = '''import runpy, sys
from unittest.mock import patch
from acprof.tui.app import AcprofTui

def finish_immediately(self, command, kind):
    self._consume_process_line("validation child finished", None, False)
    self._process_finished(kind, int(command[-1]), None, "")

sys.argv = sys.argv[1:]
with patch.object(AcprofTui, "_execute_command", finish_immediately):
    runpy.run_path(sys.argv[0], run_name="__main__")
'''
        self.run_validation(["unused", str(returncode)], returncode=returncode, runner=runner)

    def test_finished_snapshot_selects_monitor_after_pending_focus(self):
        runner = '''import runpy, sys
from unittest.mock import patch
from acprof.tui.app import AcprofTui

def finish_after_focus(self, command, kind):
    self._consume_process_line("validation child finished", None, False)

    def pending_focus():
        self._activate_tab("run-tab")
        self.call_after_refresh(self._process_finished, kind, 0, None, "")

    self.call_after_refresh(pending_focus)

sys.argv = sys.argv[1:]
with patch.object(AcprofTui, "_execute_command", finish_after_focus):
    runpy.run_path(sys.argv[0], run_name="__main__")
'''
        self.run_validation(["unused"], runner=runner)

    def test_launch_error_is_rendered_before_nonzero_exit(self):
        self.run_validation([str(ROOT / "missing-validation-child")], returncode=1,
                            expected="FileNotFoundError")

    def test_startup_check_failure_exits_without_launching_child(self):
        runner = '''import runpy, sys
from unittest.mock import patch

sys.argv = sys.argv[1:]
with patch("acprof.tui.app.quick_preflight", side_effect=OSError("host unavailable")), patch(
    "acprof.tui.app.AcprofTui._execute_command", side_effect=AssertionError("unexpected child"),
):
    runpy.run_path(sys.argv[0], run_name="__main__")
'''
        self.run_validation(["unused"], returncode=1, expected="host unavailable", runner=runner)
