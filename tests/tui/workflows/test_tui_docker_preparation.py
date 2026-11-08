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
async def test_accounting_survives_terminal_events_without_false_percentage(tmp_path, language, monkeypatch):
    from io import StringIO

    from rich.console import Console

    monkeypatch.delenv("NO_COLOR", raising=False)
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    progress = ImageProgress("platform", "fixture:image", "pulling", "layer-a", current=50,
                             operation_id="op", transferred_bytes=150, known_bytes=200, unknown_items=2,
                             cached_bytes=20, cached_items=1, transfer_status="partial")
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("monitor-tab")
        tracker = RunProgressTracker()
        snapshot = tracker.feed(encode_event("environment", "running", image_progress=progress.to_dict()))
        app.execution.update_snapshot(snapshot)
        app.render_snapshot(snapshot)
        app.preflight.download_progress(progress)
        await pilot.pause()
        for size in ((80, 24), (120, 30)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            stream = StringIO()
            widget = app.query_one(ImageProgressContent)
            Console(file=stream, width=76).print(widget.render())
            rendered = stream.getvalue()
            assert "150 B" in rendered
            assert "200 B" in rendered
            assert "20 B" in rendered
            assert ("未知大小" if language == "zh" else "Unknown sizes") in rendered
            assert "%" not in rendered
            assert ("未知大小" if language == "zh" else "Unknown") in app.export_screenshot()
            panel = app.query_one(DownloadProgressPanel)
            panel.focus()
            await pilot.press("end")
            await pilot.pause()
            assert panel.scroll_y == panel.max_scroll_y
            await pilot.press("home")
            await pilot.pause()
            if destination := os.environ.get("ACPROF_DOCKER_EVIDENCE"):
                app.save_screenshot(filename=f"download-accounting-{language}-{size[0]}x{size[1]}.svg", path=destination)
        app.preflight.download_progress(ImageProgress("platform", "fixture:image", "verifying"))
        assert app.preflight.latest_download.transferred_bytes == 150
        app.preflight.finish_download(cancelled=True)
        assert app.preflight.latest_download.phase == "cancelled"
        assert app.preflight.latest_download.transferred_bytes == 150
        next_attempt = replace(progress, operation_id="retry", transferred_bytes=30)
        app.preflight.download_progress(next_attempt)
        app.preflight.download_progress(next_attempt)
        stream = StringIO()
        Console(file=stream, width=76).print(widget.render())
        assert "180 B" in stream.getvalue()
        app.preflight.reset_download()
        app.preflight.download_progress(next_attempt)
        stream = StringIO()
        Console(file=stream, width=76).print(widget.render())
        assert "180 B" not in stream.getvalue()


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
        with pytest.raises(RuntimeError, match="inside measurement"):
            app.preflight.network_preflight_report('[network-preflight] {}')


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


def _image_text(widget: ImageProgressContent, width: int = 76) -> str:
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    Console(file=stream, width=width).print(widget.render())
    return stream.getvalue()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_docker_byte_states_keep_unknown_distinct_from_zero(tmp_path, language, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    states = [
        ("waiting", None, "等待下载", "Waiting to download"),
        ("unavailable", None, "下载量暂不可统计", "Download bytes not yet measurable"),
        ("partial", 15, "已确认下载 15 B（部分统计）", "Confirmed download 15 B (partial)"),
        ("reported", 15, "已报告下载 15 B", "Reported download 15 B"),
        ("cached", 0, "0 B（缓存命中）", "0 B (cache hit)"),
        ("partial", 0, "已确认下载 0 B（部分统计）", "Confirmed download 0 B (partial)"),
    ]
    async with app.run_test(size=(80, 24)):
        widget = app.query_one(ImageProgressContent)
        for state, amount, zh, en in states:
            app.preflight.reset_download()
            app.preflight.download_progress(ImageProgress(
                "platform", "fixture:image", "pulling", operation_id="one",
                transfer_status=state, transferred_bytes=amount,
            ))
            text = _image_text(widget)
            assert (zh if language == "zh" else en) in text
            if amount is None:
                assert "0 B" not in text
            assert "%" not in text

        app.preflight.reset_download()
        unknown = ImageProgress("platform", "fixture:image", "pulling", operation_id="unobserved")
        app.preflight.download_progress(unknown)
        confirmed = replace(unknown, operation_id="reported", transferred_bytes=25, transfer_status="reported")
        app.preflight.download_progress(confirmed)
        app.preflight.download_progress(confirmed)
        text = _image_text(widget)
        assert ("已确认下载 25 B（部分统计）" if language == "zh" else "Confirmed download 25 B (partial)") in text
        assert "50 B" not in text

        app.preflight.download_progress(ImageProgress("platform", "fixture:image", "failed", detail="fresh failure"))
        assert app.preflight.latest_download.transferred_bytes == 25
        assert app.preflight.latest_download.detail == "fresh failure"
        assert "incomplete_stream" in app.preflight.latest_download.uncovered_sources
        assert "25 B" in _image_text(widget)


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_frontend_waiting_clock_updates_without_events_and_freezes_on_stop(tmp_path, language, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    monkeypatch.delenv("NO_COLOR", raising=False)
    clock = [100.0]
    monkeypatch.setattr("acprof.tui.ui.progress_widgets.time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr("acprof.tui.run.presenter.time", SimpleNamespace(monotonic=lambda: clock[0]))
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    progress = ImageProgress(
        "platform", "acprof-platform-cpu:fixture", "building",
        "resolve image config for docker-image://docker.io/docker/dockerfile:1",
        activity="frontend", operation_id="frontend", transfer_status="unavailable",
        source_image="docker.io/docker/dockerfile:1", source_registry="docker.io",
        source_domain="registry-1.docker.io", source_kind="official", source_category="docker_hub",
        network_owner="buildkit", elapsed_seconds=2, stall_after_seconds=5,
        uncovered_sources=("metadata", "protocol_overhead", "buildkit_layers"),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("monitor-tab")
        tracker = RunProgressTracker()
        snapshot = tracker.feed(encode_event("environment", "running", image_progress=progress.to_dict()))
        app.execution.update_snapshot(snapshot)
        app.render_snapshot(snapshot)
        app.preflight.download_progress(progress)
        app.execution.session.started_monotonic = 1
        await pilot.pause()
        widget = app.query_one(ImageProgressContent)
        panel = app.query_one(DownloadProgressPanel)
        with patch.object(app, "is_busy", return_value=True):
            clock[0] = 106.0
            app.tick_elapsed()
            text = _image_text(widget)
            assert "00:00:08" in text
            assert "00:00:06" in text
            assert ("长时间未收到进度" if language == "zh" else "No progress for a long time") in text
            assert ("不代表网络失败" if language == "zh" else "does not prove a network failure") in text
            assert ("下载量暂不可统计" if language == "zh" else "Download bytes not yet measurable") in text
            assert "registry-1.docker.io" in text
            assert "docker.io/docker/dockerfile:1" in text
            assert ("引用域名" if language == "zh" else "Reference host") in text
            assert "0 B" not in text
            assert "%" not in text

            for size in ((80, 24), (120, 30)):
                await pilot.resize_terminal(*size)
                await pilot.pause()
                assert panel.region.right <= size[0]
                assert panel.region.bottom <= size[1]
                panel.focus()
                await pilot.press("end")
                await pilot.pause()
                assert panel.scroll_y == panel.max_scroll_y
                await pilot.press("home")
                await pilot.pause()
                if destination := os.environ.get("ACPROF_DOCKER_EVIDENCE"):
                    app.save_screenshot(filename=f"docker-frontend-{language}-{size[0]}x{size[1]}.svg", path=destination)

            app.execution.session.snapshot = replace(snapshot, measurement_active=True)
            with patch.object(widget, "refresh") as refresh:
                clock[0] = 107.0
                app.tick_elapsed()
                refresh.assert_not_called()
            app.execution.session.snapshot = snapshot
            app.preflight.finish_download(cancelled=True)
            terminal = _image_text(widget)
            clock[0] = 125.0
            app.tick_elapsed()
            assert _image_text(widget) == terminal
            assert app.preflight.latest_download.phase == "cancelled"
            assert app.preflight.latest_download.source_domain == "registry-1.docker.io"
            assert "0 B" not in terminal
        app.execution.session.started_monotonic = 0
