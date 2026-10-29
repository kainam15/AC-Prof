"""One collection flow with model review, real stages, retry and cancellation."""
import io
from unittest.mock import Mock, patch

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Collapsible, Select, Static

from acprof.experiment import RunConfig
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.preparation import PreparationScreen
from tests.support.tui_fixtures import AcprofTui


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


@pytest.mark.parametrize(("language", "size", "download_size"), [
    ("zh", (80, 24), 890_152_505),
    ("en", (120, 30), 890_152_505),
    ("zh", (150, 45), 890_152_505),
    ("en", (80, 24), None),
])
async def test_download_review_is_direct_and_confirmation_remains_reachable(tmp_path, language, size, download_size):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    replies = []
    report = {"expected_download_bytes": download_size,
              "disk": {"free_bytes": 41_370_132_480},
              "model": {"endpoint": "https://hf-mirror.com"},
              "model_store_path": "/models/" + "long-path/" * 30,
              "runtime": {"source_mode": "build", "will_attempt_ghcr_pull": False},
              "network_probe": [{"category": "model", "status": "failed", "method": "HEAD+GET",
                                 "source": "https://hf-mirror.com", "final_host": "cdn.hf.co",
                                 "reason": "http-403", "http_status": 403, "received_bytes": 0}]}
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
            ["终止任务", "确认下载"] if language == "zh" else ["Stop task", "Confirm download"]
        )
        assert not any(button.disabled for button in actions)
        assert screen.query_one("#preparation-download-note", Static).content == (
            "仅在准备阶段下载，正式测量离线。" if language == "zh"
            else "Downloads run only during preparation; measurement runs offline."
        )
        assert replies == []
        sections = list(screen.query(".download-section"))
        assert [section.border_title for section in sections] == (
            ["下载概览", "运行环境与来源", "存储空间", "网络预检"] if language == "zh"
            else ["Download overview", "Runtime and sources", "Storage space", "Network preflight"]
        )
        fields = screen.query_one("#download-overview", Static)
        assert ("预计下载" if language == "zh" else "Expected download") in fields.content
        assert ("Unknown" if download_size is None else "890 MB") in fields.content
        assert not list(screen.query("#preparation-detail"))
        assert not list(screen.query(Collapsible))
        network = screen.query_one("#download-network", Static)
        assert "403" in network.content
        assert "cdn.hf.co" in network.content
        assert "VPN" not in network.content
        storage = screen.query_one("#download-storage", Static)
        assert report["model_store_path"] in storage.content
        assert "source_mode" not in screen.query_one("#download-runtime", Static).content
        scroll = screen.query_one("#preparation-scroll", VerticalScroll)
        scroll.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert scroll.scroll_y > 0
        for button in actions:
            assert button.region.width > 0
            assert button.region.height > 0
            assert app.get_widget_at(*button.region.center)[0] is button
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        scroll.scroll_home(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click("#preparation-continue")
        await pilot.pause()
        assert replies == [{"action": "confirm"}]


async def test_download_confirmation_replies_once_and_returns_to_monitor(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=(80, 24)) as pilot:
        base = app.screen
        app.lifecycle.process = process
        try:
            app.preflight.preparation_event({"stage": "image", "status": "waiting", "request": {
                "id": 3, "kind": "review", "resolved": True, "questions": [],
                "download_report": {"expected_download_bytes": 708_390_484},
            }})
            await pilot.pause()
            screen = app.screen
            assert str(screen.query_one("#preparation-continue", Button).label) == "确认下载"
            assert process.stdin.getvalue() == ""
            assert await pilot.click("#preparation-continue")
            await pilot.pause()
            assert process.stdin.getvalue() == '{"id": 3, "action": "confirm"}\n'
            assert app.preflight.preparation_request is None
            assert app.preflight.preparation_screen is None
            assert app.screen is base
            assert not app.stop_requested
        finally:
            app.lifecycle.process = None


@pytest.mark.parametrize("action", ["button", "escape"])
async def test_preparation_stop_rejection_keeps_pending_request_interactive(tmp_path, action):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=(80, 24)) as pilot:
        app.lifecycle.process = process
        app.process_kind = "run"
        try:
            app.preflight.preparation_event({"stage": "image", "status": "waiting", "request": {
                "id": 3, "kind": "review", "resolved": True, "questions": [],
                "download_report": {"expected_download_bytes": 708_390_484},
            }})
            await pilot.pause()
            screen = app.screen
            if action == "escape":
                await pilot.press("escape")
            else:
                assert await pilot.click("#preparation-cancel")
            await pilot.pause()
            assert process.stdin.getvalue() == ""
            assert app.preflight.preparation_request == (process, 3)
            assert not app.stop_requested
            assert list(app.screen.query("#confirm-yes"))
            assert await pilot.click("#confirm-no")
            await pilot.pause()
            assert app.screen is screen
            assert app.preflight.preparation_request == (process, 3)
            assert not app.stop_requested
            assert await pilot.click("#preparation-continue")
            await pilot.pause()
            assert process.stdin.getvalue() == '{"id": 3, "action": "confirm"}\n'
            assert app.preflight.preparation_request is None
        finally:
            app.lifecycle.process = None
            app.process_kind = ""


async def test_preparation_confirmed_stop_uses_graceful_stop_path(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=(80, 24)) as pilot:
        base = app.screen
        app.lifecycle.process = process
        app.process_kind = "run"
        try:
            app.preflight.preparation_event({"stage": "resolution", "status": "waiting", "request": {
                "id": 7, "kind": "review", "resolved": True, "questions": [],
            }})
            await pilot.pause()
            with patch.object(app, 'stop_process_gracefully') as stop:
                assert await pilot.click("#preparation-cancel")
                await pilot.pause()
                assert process.stdin.getvalue() == ""
                assert await pilot.click("#confirm-yes")
                await pilot.pause()
                assert app.stop_requested
                assert app.preflight.preparation_request is None
                assert app.preflight.preparation_screen is None
                assert app.screen is base
                stop.assert_called_once_with(app.process_token, process)
        finally:
            app.lifecycle.process = None
            app.process_kind = ""


async def test_start_stays_on_monitor_without_passive_preparation_modal(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        base = app.screen
        assert not list(app.query("#inspect-model"))
        with patch.object(app.process, 'execute_command'):
            assert await pilot.click("#start-run")
            await pilot.pause()
            assert app.screen is base
            assert app.preflight.preparation_screen is None
            assert app.query_one("#main-tabs").active == "monitor-tab"
            log = app.query_one("#run-log")
            assert log.display
            assert log.region.height > 0
            app.preflight.preparation_event({"stage": "model", "status": "running"})
            await pilot.pause()
            assert app.screen is base
            assert app.preflight.preparation_screen is None
            assert log.display
            assert log.region.height > 0


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30)), ("zh", (150, 45))])
async def test_review_is_readonly_for_resolved_fields_and_confirm_waits_for_resolver(tmp_path, language, size):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=size) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        app.lifecycle.process = process
        try:
            event = {"stage": "resolution", "status": "waiting", "request": {
                "id": 1, "kind": "review", "questions": [{"path": "task", "options": ["fill-mask", "text-generation"]}],
                "fields": {"Model": "demo/model", "Revision": "a" * 40, "Backend": "transformers_pipeline"},
                "summary": "cache_key: private\nprovenance: details", "resolved": False,
            }}
            app.preflight.preparation_event(event)
            await pilot.pause()
            first_screen = app.screen
            assert first_screen.query_one("#preparation-continue", Button).disabled
            assert first_screen.query_one("#preparation-fields", Static).content.startswith("✓ Model")
            assert not list(first_screen.query("#resolution-basic, #resolution-full"))
            choice = first_screen.query_one("#preparation-answer-0", Select)
            choice.value = "fill-mask"
            assert await pilot.click("#preparation-apply")
            await pilot.pause()
            assert '"action": "answer"' in process.stdin.getvalue()
            assert app.preflight.preparation_screen is None
            app.preflight.preparation_event({"stage": "resolution", "status": "waiting", "request": {
                "id": 2, "kind": "review", "questions": [], "resolved": True,
                "fields": {**event["request"]["fields"], "Task": "fill-mask"},
            }})
            await pilot.pause()
            second_screen = app.screen
            assert isinstance(second_screen, PreparationScreen)
            assert second_screen is not first_screen
            assert not list(second_screen.query(Collapsible))
            assert not second_screen.query_one("#preparation-continue", Button).disabled
            assert await pilot.click("#preparation-continue")
            await pilot.pause()
            assert '"action": "confirm"' in process.stdin.getvalue()
            assert app.preflight.preparation_screen is None
            assert app.screen is not second_screen
        finally:
            app.lifecycle.process = None


async def test_runtime_running_stays_in_monitor_and_error_opens_decision_dialog(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=(80, 24)) as pilot:
        base = app.screen
        app.lifecycle.process = process
        try:
            app.preflight.preparation_event({"stage": "runtime", "status": "running"})
            await pilot.pause()
            assert app.screen is base
            assert app.preflight.preparation_screen is None
            app.preflight.preparation_event({"stage": "runtime", "status": "failed", "request": {
                "id": 1, "kind": "error", "failed_stage": "postprocess", "detail": "Traceback: invalid output",
            }})
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, PreparationScreen)
            assert screen.query_one("#preparation-title", Static).content == "模型无法正常运行"
            assert "输出处理" in screen.query_one("#preparation-detail", Static).content
            assert not screen.query_one("#preparation-diagnostics", Collapsible).collapsed
            assert await pilot.click("#preparation-continue")
            await pilot.pause()
            assert app.screen is base
            assert app.preflight.preparation_screen is None
            assert '"action": "retry"' in process.stdin.getvalue()
        finally:
            app.lifecycle.process = None


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "-Infinity", "1e999"])
async def test_nonfinite_json_answer_stays_editable_until_valid(tmp_path, invalid):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=(100, 30)) as pilot:
        app.lifecycle.process = process
        try:
            event = {"stage": "resolution", "status": "waiting", "request": {
                "id": 7, "kind": "review", "questions": [
                    {"path": "temperature", "kind": "json", "value": 1.0},
                ], "fields": {"Model": "demo/model"}, "resolved": False,
            }}
            app.preflight.preparation_event(event)
            await pilot.pause()
            field = app.screen.query_one("#preparation-answer-0", Input)
            field.value = invalid
            assert await pilot.click("#preparation-apply")
            await pilot.pause()
            assert process.stdin.getvalue() == ""
            assert app.preflight.preparation_request == (process, 7)
            assert app.screen is app.preflight.preparation_screen
            assert app.screen.query_one("#preparation-error", Static).content

            field.value = "0.5"
            app.screen.apply_answers()
            await pilot.pause()
            assert '"temperature": 0.5' in process.stdin.getvalue()
            assert app.preflight.preparation_request is None
        finally:
            app.lifecycle.process = None
