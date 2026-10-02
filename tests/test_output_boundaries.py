"""Keep developer diagnostics separate from progress and user errors."""
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from acprof.host import client, orchestrator
from acprof.tui.process import ProcessLifecycle

ROOT = Path(__file__).resolve().parents[1]


class OutputBoundaryTests(unittest.TestCase):
    def test_library_imports_are_silent_and_do_not_configure_root_logging(self):
        modules = ["orchestrator", "posthoc.service", "preflight", "client",
                   "execution_profile", "input_plan", "largest_scale_probe", "profilers.ncu"]
        result = subprocess.run([sys.executable, "-c", (
            "import importlib, logging; before = list(logging.getLogger().handlers); "
            f"[importlib.import_module('acprof.host.' + m) for m in {modules!r}]; "
            "assert logging.getLogger().handlers == before"
        )], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "", ""))

    def test_client_startup_configuration_is_debug_only(self):
        config = Mock(pipeline_tag="fill-mask")
        with redirect_stdout(io.StringIO()) as stdout, patch.object(client, "ClientRunner"), patch.object(client, "_ensure_local_proxy_bypass"), self.assertLogs("acprof.host.client", level="DEBUG") as logs:
            client.main(config)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("pipeline_tag=fill-mask", logs.output[0])

    def test_idle_warning_goes_to_stderr(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            orchestrator._check_idle_power_values_stable(
                csv_path=str(Path(directory) / "case.csv"), metric_name="cpu_idle_power_w", idle_values=[1., 3.],
                invalid_rows=0, row_count=2, threshold=.05, remediation="retry")
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("[energy][WARN]", stderr.getvalue())

    def test_process_capture_preserves_events_and_user_stderr(self):
        owner = ProcessLifecycle()
        process = owner.start([sys.executable, "-c",
            "import sys; print('ACPROF_EVENT={}', flush=True); print('[WARN] detail', file=sys.stderr, flush=True)"], cwd=ROOT, env={})
        try:
            output, _ = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0)
            self.assertEqual(output.splitlines(), ["ACPROF_EVENT={}", "[WARN] detail"])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            owner.release(process)


if __name__ == "__main__":
    unittest.main()
