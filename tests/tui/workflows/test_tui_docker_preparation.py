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
