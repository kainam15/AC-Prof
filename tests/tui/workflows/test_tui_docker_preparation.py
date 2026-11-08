"""Docker failures and progress remain actionable in narrow bilingual terminals."""
import os
from dataclasses import replace

import pytest
from rich.cells import cell_len
from textual.widgets import Button, Collapsible, Select, Static

from acprof.events.image import ImageProgress
from acprof.events.preparation import encode_event
from acprof.experiments.configuration import RunConfig
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run.preparation import PreparationScreen
from acprof.tui.run.progress import RunProgressTracker
from acprof.tui.ui.progress_widgets import DownloadProgressPanel, ImageProgressContent
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_docker_failure_actions_and_collapsed_logs(tmp_path, language, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    error = {"reason_code": "docker_content_incomplete", "stage": "environment", "step": "runtime.Dockerfile",
             "image": "acprof-platform-cu128:fixture", "detail": "content digest sha256:abc not found",
             "exit_code": 1, "log_path": "/tmp/实验/logs/preparation/attempt.log", "actions": ["retry", "repair", "rebuild"],
             "content": {"status": "incomplete", "present": 8, "required": 9, "engine_version": "29.4.0"}}
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        screen = PreparationScreen({"stage": "environment", "request": {
            "kind": "error", "id": 1, "docker_error": error, "detail": "\n".join(f"Docker log line {i}" for i in range(40)),
        }}, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        assert str(screen.query_one("#preparation-title").content) == (
            "运行环境准备失败" if language == "zh" else "Runtime preparation failed")
        assert "8/9" in str(screen.query_one("#preparation-detail", Static).content)
        diagnostics = screen.query_one(Collapsible)
        assert diagnostics.collapsed
        for size in ((80, 24), (120, 30)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            for button in screen.query("#preparation-actions Button"):
                assert screen.region.contains_region(button.region)
                assert cell_len(str(button.label)) <= button.content_region.width
                assert app.get_widget_at(*button.region.center)[0] is button
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        if destination := os.environ.get("ACPROF_DOCKER_EVIDENCE"):
            app.save_screenshot(filename=f"docker-failure-{language}.svg", path=destination)
        title = diagnostics.query_one("CollapsibleTitle")
        title.scroll_visible(immediate=True)
        await pilot.pause()
        assert await pilot.click(title)
        await pilot.pause()
        log = screen.query_one(SelectableLog)
        log.focus()
        await pilot.press("end")
        assert log.scroll_y == log.max_scroll_y
        button = screen.query_one("#preparation-repair", Button)
        button.focus()
        await pilot.press("enter")
        assert replies == [{"action": "repair"}]


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_docker_progress_shares_panel_and_stops_at_measurement(tmp_path, language, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#ui-language", Select).value = language
        app.activate_tab("monitor-tab")
        await pilot.pause()
        progress = ImageProgress("platform", "Platform · cu128", "pulling", "layer-a", current=40, total=100)
        tracker = RunProgressTracker()
        snapshot = tracker.feed(encode_event("environment", "running", image_progress=progress.to_dict()))
        app.execution.update_snapshot(snapshot)
        app.render_snapshot(snapshot)
        app.preflight.preparation_event({"stage": "environment", "status": "running", "image_progress": progress.to_dict()})
        await pilot.pause()
        panel = app.query_one(DownloadProgressPanel)
        assert panel.display
        assert app.query_one(ImageProgressContent).progress == progress
        if destination := os.environ.get("ACPROF_DOCKER_EVIDENCE"):
            app.save_screenshot(filename=f"docker-progress-{language}.svg", path=destination)
        app.execution.session.snapshot = replace(app.execution.session.snapshot, measurement_active=True)
        with pytest.raises(RuntimeError, match="inside measurement"):
            app.preflight.preparation_event({"stage": "environment", "status": "running", "image_progress": progress.to_dict()})


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_fallback_warning_survives_build_progress_without_error_modal(tmp_path, language, monkeypatch):
    from io import StringIO

    from rich.console import Console
    monkeypatch.delenv("NO_COLOR", raising=False)
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("monitor-tab")
        tracker = RunProgressTracker()
        for progress in (
            ImageProgress("platform", "ghcr.io/example/runtime:platform-fixture", "pulling"),
            ImageProgress("platform", "ghcr.io/example/runtime:platform-fixture", "fallback", reason_code="docker_platform_mismatch"),
            ImageProgress("platform", "base:fixture", "checking"),
            ImageProgress("platform", "acprof-platform-cpu:fixture", "building", "RUN install dependencies", current=40, total=100),
        ):
            line = encode_event("environment", "running", image_progress=progress.to_dict())
            snapshot = tracker.feed(line)
            app.execution.update_snapshot(snapshot)
            app.render_snapshot(snapshot)
            app.preflight.preparation_event({"stage": "environment", "status": "running", "image_progress": progress.to_dict()})
        await pilot.pause()
        widget = app.query_one(ImageProgressContent)
        stream = StringIO()
        Console(file=stream, width=76).print(widget.render())
        text = stream.getvalue()
        assert ("本地构建" if language == "zh" else "Building locally") in text
        assert "linux/amd64" in text
        assert "40%" in text
        assert ("警告" if language == "zh" else "Warning") in text
        assert app.preflight.preparation_screen is None
        assert not app.execution.session.snapshot.measurement_active
        assert "40%" in app.export_screenshot()
        if destination := os.environ.get("ACPROF_DOCKER_EVIDENCE"):
            app.save_screenshot(filename=f"docker-fallback-{language}-80x24.svg", path=destination)
        await pilot.resize_terminal(120, 30)
        await pilot.pause()
        if destination:
            app.save_screenshot(filename=f"docker-fallback-{language}-120x30.svg", path=destination)


@pytest.mark.parametrize(("code", "zh", "en"), [
    ("docker_manifest_missing", "标签不存在", "tag does not exist"),
    ("docker_platform_mismatch", "linux/amd64", "linux/amd64"),
    ("docker_network_failed", "连接失败", "connection failed"),
    ("docker_download_budget", "预算", "budget"),
])
@pytest.mark.parametrize("language", ["zh", "en"])
def test_pull_failure_summary_is_classified_and_raw_diagnostics_stay_collapsed(code, zh, en, language):
    from acprof.tui.i18n import translate
    from acprof.tui.run.preparation import failure_text
    detail = "RAW_DOCKER_ERROR " * 100
    error = {"reason_code": code, "step": "docker pull", "detail": detail, "log_path": "/tmp/preparation.log"}
    summary = failure_text({"docker_error": error}, "environment", lambda text: translate(text, language))
    assert (zh if language == "zh" else en) in summary
    assert "RAW_DOCKER_ERROR" not in summary
    assert "/tmp/preparation.log" in summary
