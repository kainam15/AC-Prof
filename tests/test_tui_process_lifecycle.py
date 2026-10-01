import asyncio
import fcntl
import io
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from textual.widgets import Button, Static

from acprof.tui.app import AcprofTui
from acprof.tui.process import ProcessLifecycle, StopResult


class TuiProcessFailureTests(unittest.TestCase):
    def test_callback_failure_keeps_live_process_busy_and_does_not_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            app = AcprofTui(settings_path=Path(directory) / "settings.json")
            process = Mock(pid=987654, stdout=io.StringIO("output\n"))
            process.poll.return_value = None
            process.wait.side_effect = subprocess.TimeoutExpired("collector", 5)
            app._process_started = Mock(side_effect=RuntimeError("callback failed"))
            app._process_finished = Mock()
            app.call_from_thread = lambda callback, *args: callback(*args)
            with (
                patch("subprocess.Popen", return_value=process),
                patch("os.killpg"),
                patch.object(app, "_watch_failed_process", create=True),
                patch.object(app, "_process_cleanup_incomplete", create=True),
            ):
                AcprofTui._execute_command.__wrapped__(app, ["collector"], "run")
            self.assertTrue(app._is_busy(), "live collector must remain owned")
            app._process_finished.assert_not_called()

    def test_ignoring_signals_retains_process_and_measurement_lock_until_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ProcessLifecycle(interrupt_timeout=0.05, terminate_timeout=0.05)
            script = '''
import signal
from pathlib import Path
import os
from acprof.host import run_state
run_state.MEASUREMENT_LOCK_ROOT = Path(os.environ["TMPDIR"])
MeasurementLock = run_state.MeasurementLock
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with MeasurementLock() as lock:
    print(lock.path, flush=True)
    while True:
        signal.pause()
'''
            process = manager.start([sys.executable, "-u", "-c", script],
                cwd=Path(__file__).resolve().parents[1], env={**os.environ, "TMPDIR": directory})
            try:
                lock_path = process.stdout.readline().strip()
                self.assertTrue(lock_path.startswith(directory), lock_path)
                with open(lock_path, "a+") as lock:
                    result = manager.stop()
                    self.assertFalse(result.complete)
                    self.assertIn("SIGTERM", result.error)
                    self.assertIs(manager.process, process)
                    self.assertFalse(manager.release(process))
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Force-kill only this test-owned fixture to exercise late reaping.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                    self.assertTrue(manager.release(process))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                process.stdout.close()

    def test_unmount_uses_cleanup_and_prevents_late_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            app = AcprofTui(settings_path=Path(directory) / "settings.json")
            asyncio.run(app.on_unmount())
            with self.assertRaisesRegex(RuntimeError, "closing"):
                app._lifecycle.start(["unused"], cwd=directory, env={})

    def test_missing_group_does_not_release_an_unreaped_process(self):
        manager = ProcessLifecycle(interrupt_timeout=0, terminate_timeout=0)
        process = Mock(pid=987654)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("collector", 0)
        with patch("subprocess.Popen", return_value=process), patch("os.killpg", side_effect=ProcessLookupError):
            manager.start(["unused"], cwd=".", env={})
            self.assertFalse(manager.stop().complete)
        self.assertIs(manager.process, process)


class TuiCleanupStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_timeout_keeps_stop_enabled_and_start_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            app = AcprofTui(settings_path=Path(directory) / "settings.json")
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                process = Mock(pid=12345)
                process.poll.return_value = None
                app._lifecycle.process = process
                app._process_kind = "run"
                app._process_cleanup_incomplete(StopResult(12345, None, "SIGTERM timeout"))
                self.assertTrue(app._is_busy())
                self.assertTrue(app.query_one("#start-run", Button).disabled)
                self.assertFalse(app.query_one("#stop-run", Button).disabled)
                self.assertIn("清理未完成", str(app.query_one("#status-stage", Static).render()))
                self.assertIn("12345", str(app.query_one("#status-detail", Static).render()))
                process.poll.return_value = 0
                app._lifecycle.release(process)


if __name__ == "__main__":
    unittest.main()
