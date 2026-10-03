"""One collection flow with model review, real stages, retry and cancellation."""
import io
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Button, Collapsible, Select, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.preparation import PreparationScreen


async def test_start_opens_waiting_dialog_without_manual_inspection(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        assert not list(app.query("#inspect-model"))
        with patch.object(app, "_execute_command"):
            assert await pilot.click("#start-run")
            await pilot.pause()
            assert isinstance(app.screen, PreparationScreen)
            assert app.screen.query_one("#preparation-title", Static).content == "正在检测模型"
            assert not list(app.screen.query("#confirm-yes"))
            assert await pilot.click("#preparation-cancel")


@pytest.mark.parametrize("language,size", [("zh", (80, 24)), ("en", (120, 30)), ("zh", (150, 45))])
async def test_review_is_readonly_for_resolved_fields_and_confirm_waits_for_resolver(tmp_path, language, size):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=size) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        app._lifecycle.process = process
        try:
            event = {"stage": "resolution", "status": "waiting", "request": {
                "id": 1, "kind": "review", "questions": [{"path": "task", "options": ["fill-mask", "text-generation"]}],
                "fields": {"Model": "demo/model", "Revision": "a" * 40, "Backend": "transformers_pipeline"},
                "summary": "cache_key: private\nprovenance: details", "resolved": False,
            }}
            app._preparation_event(event)
            await pilot.pause()
            screen = app.screen
            assert screen.query_one("#preparation-continue", Button).disabled
            assert screen.query_one("#preparation-fields", Static).content.startswith("✓ Model")
            assert not list(screen.query("#resolution-basic, #resolution-full"))
            choice = screen.query_one("#preparation-answer-0", Select)
            choice.value = "fill-mask"
            assert await pilot.click("#preparation-apply")
            await pilot.pause()
            assert '"action": "answer"' in process.stdin.getvalue()
            app._preparation_event({"stage": "resolution", "status": "waiting", "request": {
                "id": 2, "kind": "review", "questions": [], "resolved": True,
                "fields": {**event["request"]["fields"], "Task": "fill-mask"},
            }})
            await pilot.pause()
            assert app.screen is screen
            assert not screen.query_one("#preparation-continue", Button).disabled
            assert await pilot.click("#preparation-continue")
            await pilot.pause()
            assert '"action": "confirm"' in process.stdin.getvalue()
            app._preparation_event({"stage": "runtime", "status": "passed"})
            await pilot.pause()
            assert app.screen is not screen
        finally:
            app._lifecycle.process = None


async def test_runtime_error_replaces_wait_in_same_dialog_and_details_are_collapsed(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=(80, 24)) as pilot:
        app._lifecycle.process = process
        try:
            app._preparation_event({"stage": "runtime", "status": "running"})
            await pilot.pause()
            screen = app.screen
            app._preparation_event({"stage": "runtime", "status": "failed", "request": {
                "id": 1, "kind": "error", "failed_stage": "postprocess", "detail": "Traceback: invalid output",
            }})
            await pilot.pause()
            assert app.screen is screen
            assert screen.query_one("#preparation-title", Static).content == "模型无法正常运行"
            assert "输出处理" in screen.query_one("#preparation-detail", Static).content
            assert screen.query_one("#preparation-diagnostics", Collapsible).collapsed
            assert await pilot.click("#preparation-continue")
            await pilot.pause()
            assert app.screen is screen
            assert '"action": "retry"' in process.stdin.getvalue()
        finally:
            app._lifecycle.process = None


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "-Infinity", "1e999"])
async def test_nonfinite_json_answer_stays_editable_until_valid(tmp_path, invalid):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=(100, 30)) as pilot:
        app._lifecycle.process = process
        try:
            event = {"stage": "resolution", "status": "waiting", "request": {
                "id": 7, "kind": "review", "questions": [
                    {"path": "temperature", "kind": "json", "value": 1.0},
                ], "fields": {"Model": "demo/model"}, "resolved": False,
            }}
            app._preparation_event(event)
            await pilot.pause()
            field = app.screen.query_one("#preparation-answer-0", Input)
            field.value = invalid
            assert await pilot.click("#preparation-apply")
            await pilot.pause()
            assert process.stdin.getvalue() == ""
            assert app._preparation_request == (process, 7)
            assert app.screen is app._preparation_screen
            assert app.screen.query_one("#preparation-error", Static).content

            field.value = "0.5"
            app.screen.apply_answers()
            await pilot.pause()
            assert '"temperature": 0.5' in process.stdin.getvalue()
            assert app._preparation_request is None
        finally:
            app._lifecycle.process = None
