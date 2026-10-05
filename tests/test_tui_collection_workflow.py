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


async def test_confirmed_stop_reaps_child_releases_lock_and_removes_bundle(tmp_path):
    import os

    from textual.widgets import TabbedContent

    from acprof.host.run_state import MeasurementLock
    script = textwrap.dedent('''
        import json, sys, time
        from pathlib import Path
        from acprof.host.detect import TaskInfo
        from acprof.host import run_state
        from acprof.host.source_bundle import source_bundle
        from acprof.preparation_events import encode_event
        task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers", "transformers", "a" * 40, "fixture")
        run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
        with run_state.MeasurementLock(), source_bundle(task) as bundle:
            Path(sys.argv[1], "bundle.json").write_text(json.dumps(str(bundle.root)))
            print(encode_event("runtime", "running"), flush=True)
            time.sleep(120)
    ''')
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    original_start = ProcessLifecycle.start
    def launch_child(owner, _command, **kwargs):
        return original_start(owner, [sys.executable, "-c", script, str(tmp_path)], **kwargs)
    with patch("acprof.tui.process.ProcessLifecycle.start", autospec=True, side_effect=launch_child):
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#start-run")
            async def ready():
                while not (tmp_path / "bundle.json").exists():
                    await pilot.pause()
            await asyncio.wait_for(ready(), timeout=10)
            await pilot.pause()
            pid = app._lifecycle.process.pid
            assert app._preparation_screen is None
            assert await pilot.click("#stop-run")
            await pilot.pause()
            assert await pilot.click("#confirm-yes")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=15)
            await pilot.pause()
            assert app._lifecycle.process is None
            assert app.query_one(TabbedContent).active == "monitor-tab"
            assert "任务已由用户终止" in app.query_one("#run-log", SelectableLog).text
            assert not app.query_one("#start-run", Button).disabled
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    import json
    assert not Path(json.loads((tmp_path / "bundle.json").read_text())).exists()
    with patch("acprof.host.run_state.MEASUREMENT_LOCK_ROOT", tmp_path), MeasurementLock():
        pass


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
                await wait_for_request(1)
                pid = app._lifecycle.process.pid
                app.screen.query_one("#preparation-answer-0", Select).value = "fill-mask"
                assert (await pilot.click("#preparation-apply"))
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
                    assert (await pilot.click("#preparation-apply"))
                    await pilot.pause()
                    assert ('"action": "answer"') in (process.stdin.getvalue())
                    assert ('"task": "fill-mask"') in (process.stdin.getvalue())
                    handler({"version": 1, "stage": "runtime", "status": "failed", "request": {
                        "id": 2, "kind": "error", "detail": "FileNotFoundError: ultravox_config.py\n" + "diagnostic detail\n" * 40,
                    }})
                    await pilot.pause()
                    assert not (list(app.screen.query(Select)))
                    detail = app.screen.query_one("#preparation-traceback", Static).content
                    assert isinstance(detail, str)
                    assert ("ultravox_config.py") in (detail)
                    button = app.screen.query_one("#preparation-continue", Button)
                    assert (str(button.label)) == ("重试" if language == "zh" else "Retry")
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
            await app.push_screen(PreparationScreen(event))
            await pilot.pause()
            detail = cast(str, app.screen.query_one("#preparation-detail", Static).content)
            assert ("not found") in (detail)
            assert ("asdf") in (detail)
            assert ("未找到") not in (detail)
            assert ("SystemExit") not in (detail)


@pytest.mark.parametrize("size", ((80, 24), (120, 30)))
@pytest.mark.parametrize("completion", ("runtime", "process"))
async def test_preparation_completion_preserves_stop_confirmation(tmp_path, size, completion):
    from acprof.tui.views import ConfirmActionScreen

    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=size) as pilot:
        base = app.screen
        app._lifecycle.process = process
        app._process_kind = "run"
        try:
            app._set_busy(True)
            app._activate_tab("monitor-tab")
            stop = app.query_one("#stop-run", Button)
            stop.focus()
            app._show_preparation({"stage": "runtime", "status": "running"})
            await pilot.pause()
            preparation = app.screen
            await pilot.press("ctrl+x", "tab")
            await pilot.pause()
            confirmation = app.screen
            assert isinstance(confirmation, ConfirmActionScreen)
            focused = confirmation.focused
            assert focused is not None

            if completion == "runtime":
                app._preparation_event({"stage": "runtime", "status": "passed"})
            else:
                app._lifecycle.process = None
                app._process_finished("run", 1, None, "")
            await pilot.pause()

            assert app.screen is confirmation
            assert confirmation.focused is focused
            assert not app._stop_requested
            assert await pilot.click("#confirm-no")
            await pilot.pause()
            assert app.screen is base
            assert preparation not in app.screen_stack
            assert confirmation not in app.screen_stack
            assert app._preparation_screen is None
            assert app.screen.focused is not None
            assert app.screen.focused.screen is base
            if completion == "runtime":
                assert app.screen.focused is stop
                assert app._lifecycle.process is process
            else:
                assert not app.query_one("#start-run", Button).disabled
        finally:
            app._lifecycle.process = None
            app._process_kind = ""


async def test_completed_preparation_preserves_nested_dialog_replies_and_new_preparation(tmp_path):
    from acprof.tui.views import ConfirmActionScreen

    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    responses = []
    async with app.run_test(size=(80, 24)) as pilot:
        base = app.screen
        app._show_preparation({"stage": "runtime", "status": "running"})
        await pilot.pause()
        previous = app.screen
        first = ConfirmActionScreen("First", "Keep this decision", "Confirm")
        await app.push_screen(first, lambda answer: responses.append(("first", answer)))
        second = ConfirmActionScreen("Second", "Keep this decision too", "Confirm")
        await app.push_screen(second, lambda answer: responses.append(("second", answer)))
        await pilot.pause()
        app._close_preparation()
        app._close_preparation()
        await pilot.pause()
        assert app.screen is second
        assert responses == []

        # A completed screen can still have queued messages while covered.
        previous.update_event({"stage": "runtime", "status": "running"})
        previous.action_cancel()
        assert not app._stop_requested
        app._show_preparation({"stage": "resolution", "status": "running"})
        await pilot.pause()
        current = app.screen
        assert current is not previous
        app._close_preparation()
        await pilot.pause()
        assert app.screen is second
        assert await pilot.click("#confirm-yes")
        await pilot.pause()
        assert app.screen is first
        assert responses == [("second", True)]
        assert await pilot.click("#confirm-no")
        await pilot.pause()
        assert responses == [("second", True), ("first", False)]
        assert app.screen is base
        assert previous not in app.screen_stack
        assert current not in app.screen_stack
