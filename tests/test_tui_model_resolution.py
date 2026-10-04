"""One collection flow with model review, real stages, retry and cancellation."""
import io
from unittest.mock import Mock, patch

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Collapsible, Select, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.downloads import download_summary
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.preparation import PreparationScreen


@pytest.mark.parametrize("summary", [None, "", " \n\t"])
async def test_review_omits_empty_advanced_details(tmp_path, summary):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    request = {"id": 1, "kind": "review", "questions": [], "resolved": True}
    if summary is not None:
        request["summary"] = summary
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(PreparationScreen({"stage": "resolution", "request": request}))
        await pilot.pause()
        assert not list(app.screen.query(Collapsible))


@pytest.mark.parametrize("language,size,download_size", [
    ("zh", (80, 24), 890_152_505),
    ("en", (120, 30), 890_152_505),
    ("zh", (150, 45), 890_152_505),
    ("en", (80, 24), None),
])
async def test_download_review_details_expand_and_confirmation_remains_reachable(tmp_path, language, size, download_size):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    replies = []
    report = {"expected_download_bytes": download_size,
              "disk": {"free_bytes": 41_370_132_480},
              "model": {"endpoint": "https://hf-mirror.com"},
              "runtime": {"detail": "long runtime detail " * 200}}
    event = {"stage": "image", "status": "waiting", "request": {
        "id": 1, "kind": "review", "resolved": True, "questions": [], "download_report": report,
    }}
    async with app.run_test(size=size) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        screen = PreparationScreen(event, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.query_one("#preparation-title", Static).content == (
            "下载确认" if language == "zh" else "Download confirmation"
        )
        assert not list(screen.query(Select))
        actions = screen.query("#preparation-actions Button")
        assert [str(button.label) for button in actions] == (
            ["取消", "确认下载"] if language == "zh" else ["Cancel", "Confirm download"]
        )
        assert not any(button.disabled for button in actions)
        assert screen.query_one("#preparation-download-note", Static).content == (
            "仅在准备阶段下载，正式测量离线。" if language == "zh"
            else "Downloads run only during preparation; measurement runs offline."
        )
        assert replies == []
        fields = screen.query_one("#preparation-fields", Static)
        assert ("预计下载" if language == "zh" else "Expected download") in fields.content
        assert ("Unknown" if download_size is None else "890 MB") in fields.content
        assert "Docker storage" not in fields.content
        assert not list(screen.query("#preparation-detail"))
        details = screen.query_one("#preparation-advanced-details", Collapsible)
        assert details.collapsed
        assert str(details.title) == ("高级详情" if language == "zh" else "Advanced details")
        assert await pilot.click("#preparation-advanced-details CollapsibleTitle")
        await pilot.pause()
        assert not details.collapsed
        content = details.query_one("Contents > Static", Static)
        assert content.content == download_summary(report, app.tr)
        assert content.styles.color == fields.styles.color
        scroll = screen.query_one("#preparation-scroll", VerticalScroll)
        scroll.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert scroll.scroll_y > 0
        assert content.region.bottom <= scroll.content_region.bottom
        for button in actions:
            assert button.region.width > 0 and button.region.height > 0
            assert app.get_widget_at(*button.region.center)[0] is button
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        title = details.query_one("CollapsibleTitle")
        title.focus()
        title.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert details.collapsed
        assert await pilot.click("#preparation-continue")
        await pilot.pause()
        assert replies == [{"action": "confirm"}]


@pytest.mark.parametrize("action", ["confirm", "cancel", "escape"])
async def test_download_confirmation_replies_to_waiting_process_once(tmp_path, action):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=(80, 24)) as pilot:
        app._lifecycle.process = process
        try:
            app._preparation_event({"stage": "image", "status": "waiting", "request": {
                "id": 3, "kind": "review", "resolved": True, "questions": [],
                "download_report": {"expected_download_bytes": 708_390_484},
            }})
            await pilot.pause()
            screen = app.screen
            assert str(screen.query_one("#preparation-continue", Button).label) == "确认下载"
            assert process.stdin.getvalue() == ""
            if action == "escape":
                await pilot.press("escape")
            elif action == "cancel":
                assert await pilot.click("#preparation-cancel")
            else:
                screen.query_one("#preparation-cancel", Button).focus()
                await pilot.press("tab")
                assert app.focused is screen.query_one("#preparation-continue", Button)
                await pilot.press("enter", "enter")
            await pilot.pause()
            expected = "confirm" if action == "confirm" else "cancel"
            assert process.stdin.getvalue() == f'{{"id": 3, "action": "{expected}"}}\n'
            assert app._preparation_request is None
            if action == "confirm":
                assert app.screen is screen
                assert not list(screen.query("#preparation-continue"))
                assert not app._stop_requested
            else:
                assert app.screen is not screen
                assert app._stop_requested
        finally:
            app._lifecycle.process = None


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
            assert not list(screen.query(Collapsible))
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
