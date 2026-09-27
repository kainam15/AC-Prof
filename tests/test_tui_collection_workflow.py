"""User decisions return to the same subprocess without another Start click."""
import io
import asyncio
from pathlib import Path
import tempfile
import sys
import textwrap
import unittest
from unittest.mock import Mock, patch

from textual.widgets import Button, Select, Static

from acprof.preparation_events import encode_event
from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig
from acprof.tui.progress import RunProgressTracker
from acprof.tui.process import ProcessLifecycle
from acprof.tui.log import SelectableLog


class PreparationProgressTests(unittest.TestCase):
    def test_runtime_failure_retains_successful_resolution(self):
        tracker = RunProgressTracker(structured=True)
        tracker.feed(encode_event("resolution", "passed"))
        state = tracker.feed(encode_event("runtime", "failed"))
        self.assertEqual(getattr(state, "interface_status", None), "passed")
        self.assertEqual(getattr(state, "runtime_status", None), "failed")
        self.assertEqual(getattr(state, "measurement_status", None), "not_started")
        self.assertFalse(state.measurement_active)


class TuiCollectionWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_button_keeps_real_child_alive_through_answer_and_retry(self):
        script = textwrap.dedent('''
            import os
            from acprof.host.collection_workflow import PreparationWorkflow
            workflow = PreparationWorkflow.from_environment("unused")
            reply = workflow.ask("resolution", "review", questions=[
                {"path": "task", "options": ["fill-mask", "text-generation"], "value": None}])
            assert reply["answers"] == {"task": "fill-mask"}
            workflow.emit("resolution", "passed")
            attempts = 0
            def validate():
                global attempts
                attempts += 1
                if attempts == 1:
                    raise FileNotFoundError("ultravox_config.py")
            workflow.run("runtime", validate)
            print("WORKFLOW_DONE", os.getpid(), flush=True)
        ''')
        with tempfile.TemporaryDirectory() as directory:
            app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=Path(directory, "settings.json"))
            original_start = ProcessLifecycle.start
            def launch_child(owner, _command, **kwargs):
                return original_start(owner, [sys.executable, "-c", script], **kwargs)
            with patch("acprof.tui.process.ProcessLifecycle.start", autospec=True, side_effect=launch_child) as launch:
                async with app.run_test(size=(80, 24)) as pilot:
                    async def wait_for_request(request_id):
                        async def wait():
                            while app._preparation_request is None or app._preparation_request[1] != request_id:
                                await pilot.pause()
                            await pilot.pause()
                        await asyncio.wait_for(wait(), timeout=15)

                    self.assertTrue(await pilot.click("#start-run"))
                    await pilot.pause()
                    self.assertTrue(await pilot.click("#confirm-yes"))
                    await wait_for_request(1)
                    pid = app._lifecycle.process.pid
                    app.screen.query_one("#preparation-answer-0", Select).value = "fill-mask"
                    self.assertTrue(await pilot.click("#preparation-continue"))
                    await wait_for_request(2)
                    self.assertEqual(app._lifecycle.process.pid, pid)
                    self.assertEqual(app._latest_snapshot.interface_status, "passed")
                    self.assertEqual(app._latest_snapshot.runtime_status, "failed")
                    self.assertEqual(app._latest_snapshot.measurement_status, "not_started")
                    self.assertTrue(await pilot.click("#preparation-continue"))
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    self.assertIsNone(app._lifecycle.process)
                    self.assertEqual(app._latest_snapshot.runtime_status, "passed")
                    self.assertIn(f"WORKFLOW_DONE {pid}", app.query_one("#run-log", SelectableLog).text)
                    self.assertNotIn("ACPROF_PREPARATION", app.query_one("#run-log", SelectableLog).text)
                    launch.assert_called_once()

    async def test_review_and_error_actions_reply_without_relaunch_at_all_sizes(self):
        for size in ((80, 24), (120, 30), (150, 45)):
            for language in ("zh", "en"):
                with self.subTest(size=size, language=language), tempfile.TemporaryDirectory() as directory:
                    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=Path(directory, "settings.json"))
                    process = Mock(stdin=io.StringIO())
                    process.poll.return_value = None
                    async with app.run_test(size=size) as pilot:
                        app.query_one("#ui-language", Select).value = language
                        await pilot.pause()
                        app._lifecycle.process = process
                        try:
                            with patch.object(app, "_launch") as launch:
                                handler = app._preparation_event
                                handler({"version": 1, "stage": "resolution", "status": "waiting", "request": {
                                    "id": 1, "kind": "review", "detail": "", "questions": [
                                        {"path": "task", "value": None, "options": ["fill-mask", "text-generation"], "reason": "two candidates"},
                                    ],
                                }})
                                await pilot.pause()
                                choice = app.screen.query_one("#preparation-answer-0", Select)
                                choice.value = "fill-mask"
                                self.assertTrue(await pilot.click("#preparation-continue"))
                                await pilot.pause()
                                self.assertIn('"action": "answer"', process.stdin.getvalue())
                                self.assertIn('"task": "fill-mask"', process.stdin.getvalue())
                                handler({"version": 1, "stage": "runtime", "status": "failed", "request": {
                                    "id": 2, "kind": "error", "detail": "FileNotFoundError: ultravox_config.py\n" + "diagnostic detail\n" * 40,
                                }})
                                await pilot.pause()
                                self.assertFalse(list(app.screen.query(Select)))
                                detail = app.screen.query_one("#preparation-detail", Static).content
                                assert isinstance(detail, str)
                                self.assertIn("ultravox_config.py", detail)
                                button = app.screen.query_one("#preparation-continue", Button)
                                self.assertEqual(str(button.label), "重新验证" if language == "zh" else "Retry validation")
                                self.assertTrue(await pilot.click(button))
                                await pilot.pause()
                                self.assertIn('"action": "retry"', process.stdin.getvalue())
                                launch.assert_not_called()
                        finally:
                            app._lifecycle.process = None
