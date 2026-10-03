"""User decisions return to the same subprocess without another Start click."""
import asyncio
import io
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Button, Select, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.preparation_events import encode_event
from acprof.tui.log import SelectableLog
from acprof.tui.process import ProcessLifecycle
from acprof.tui.progress import RunProgressTracker


def test_runtime_failure_retains_successful_resolution():
    tracker = RunProgressTracker(structured=True)
    tracker.feed(encode_event("resolution", "passed"))
    state = tracker.feed(encode_event("runtime", "failed"))
    assert (getattr(state, "interface_status", None)) == ("passed")
    assert (getattr(state, "runtime_status", None)) == ("failed")
    assert (getattr(state, "measurement_status", None)) == ("not_started")
    assert not (state.measurement_active)


async def test_start_button_keeps_real_child_alive_through_answer_and_retry():
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

                assert (await pilot.click("#start-run"))
                await pilot.pause()
                assert (await pilot.click("#confirm-yes"))
                await wait_for_request(1)
                pid = app._lifecycle.process.pid
                app.screen.query_one("#preparation-answer-0", Select).value = "fill-mask"
                assert (await pilot.click("#preparation-continue"))
                await wait_for_request(2)
                assert (app._lifecycle.process.pid) == (pid)
                assert (app._latest_snapshot.interface_status) == ("passed")
                assert (app._latest_snapshot.runtime_status) == ("failed")
                assert (app._latest_snapshot.measurement_status) == ("not_started")
                assert (await pilot.click("#preparation-continue"))
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert (app._lifecycle.process) is None
                assert (app._latest_snapshot.runtime_status) == ("passed")
                assert (f"WORKFLOW_DONE {pid}") in (app.query_one("#run-log", SelectableLog).text)
                assert ("ACPROF_PREPARATION") not in (app.query_one("#run-log", SelectableLog).text)
                launch.assert_called_once()

@pytest.mark.parametrize('language', ('zh', 'en'))
@pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
async def test_review_and_error_actions_reply_without_relaunch_at_all_sizes(language, size):
    with tempfile.TemporaryDirectory() as directory:
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
                    assert (await pilot.click("#preparation-continue"))
                    await pilot.pause()
                    assert ('"action": "answer"') in (process.stdin.getvalue())
                    assert ('"task": "fill-mask"') in (process.stdin.getvalue())
                    handler({"version": 1, "stage": "runtime", "status": "failed", "request": {
                        "id": 2, "kind": "error", "detail": "FileNotFoundError: ultravox_config.py\n" + "diagnostic detail\n" * 40,
                    }})
                    await pilot.pause()
                    assert not (list(app.screen.query(Select)))
                    detail = app.screen.query_one("#preparation-detail", Static).content
                    assert isinstance(detail, str)
                    assert ("ultravox_config.py") in (detail)
                    button = app.screen.query_one("#preparation-continue", Button)
                    assert (str(button.label)) == ("重新验证" if language == "zh" else "Retry validation")
                    assert (await pilot.click(button))
                    await pilot.pause()
                    assert ('"action": "retry"') in (process.stdin.getvalue())
                    launch.assert_not_called()
            finally:
                app._lifecycle.process = None
async def test_model_lookup_error_is_translated_in_preparation_dialog():
    from acprof.host.model_errors import ModelLookupError
    from acprof.tui.preparation import PreparationScreen

    error = ModelLookupError("asdf", "repository_unavailable", detail="RepositoryNotFoundError: fixture")
    event = {"stage": "resolution", "request": {"id": 1, "kind": "error", "detail": str(error),
                                               "model_error": error.to_dict()}}
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(RunConfig(model="asdf"), settings_path=Path(directory, "settings.json"))
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#ui-language", Select).value = "en"
            await pilot.pause()
            await app.push_screen(PreparationScreen(event, "pending"))
            await pilot.pause()
            detail = cast(str, app.screen.query_one("#preparation-detail", Static).content)
            assert ("not found") in (detail)
            assert ("asdf") in (detail)
            assert ("未找到") not in (detail)
            assert ("SystemExit") not in (detail)
