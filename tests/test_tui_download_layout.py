"""Rendered download status remains readable without extra refresh work."""
from dataclasses import replace

import pytest
from rich.console import Console
from textual.strip import Strip
from textual.widgets import Select, Static
from tui_fixtures import AcprofTui

from acprof.download_events import DownloadProgress
from acprof.tui.log import SelectableLog
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.progress_widgets import DownloadProgressContent, DownloadProgressPanel


def rendered(widget):
    if isinstance(widget, DownloadProgressPanel):
        widget = widget.query_one(DownloadProgressContent)
    console = Console(width=widget.content_size.width, color_system="truecolor")
    segments = list(console.render(widget.render()))
    return "".join(segment.text for segment in segments), segments


@pytest.mark.parametrize(("size", "language", "theme"), [
    ((80, 24), "zh", "acprof-graphite"),
    ((80, 24), "en", "acprof-light"),
    ((120, 30), "en", "acprof-graphite"),
    ((150, 45), "zh", "acprof-light"),
])
async def test_download_metrics_and_colored_states_fit_monitor(tmp_path, size, language, theme):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        app.theme = theme
        app.activate_tab("monitor-tab")
        snapshot = ProgressSnapshot(interface_status="running", runtime_status="not_started")
        app.render_snapshot(snapshot)
        progress = DownloadProgress("BAAI/bge-m3", "huggingface", "downloading",
                                    4_590_000_000, 49_400_000, 238_000_000,
                                    178_000_000, 13_500_000, 28, 30)
        app.preflight.download_progress(progress)
        await pilot.pause()
        panel = app.query_one("#download-progress", DownloadProgressPanel)
        phases = app.query_one("#status-preparation", Static)
        text, segments = rendered(panel)
        for value in ("BAAI/bge-m3", "238 MB", "4.59 GB", "5%", "28/30", "49.4 MB", "178 MB", "13.5 MB/s"):
            assert value in text
        assert ("剩余时间" if language == "zh" else "ETA") in text
        assert len({segment.style.color for segment in segments if segment.style and segment.style.color}) >= 3
        phase_text, phase_segments = rendered(phases)
        assert ("接口解析" if language == "zh" else "Interface") in phase_text
        assert len({segment.style.color for segment in phase_segments if segment.style and segment.style.color}) >= 2
        assert phases.region.bottom <= panel.region.y
        assert panel.region.bottom <= app.query_one("#run-log", SelectableLog).region.y
        assert app.query_one("#run-log", SelectableLog).content_region.height >= 3
        stop = app.query_one("#stop-run")
        assert app.get_widget_at(*stop.region.center)[0] is stop
        assert stop.region.bottom <= size[1]

        colors = []
        for state, label in (("downloading", "下载中" if language == "zh" else "Downloading"),
                             ("complete", "模型文件已校验" if language == "zh" else "Model files verified"),
                             ("failed", "下载或校验失败" if language == "zh" else "Download or verification failed")):
            app.preflight.download_progress(replace(progress, state=state))
            await pilot.pause()
            state_text, state_segments = rendered(panel)
            assert label in state_text
            state_segment = next(segment for segment in state_segments if label in segment.text)
            colors.append(state_segment.style.color)
            rgb = state_segment.style.color.get_truecolor()
            assert f"38;2;{rgb.red};{rgb.green};{rgb.blue}" in Strip([state_segment]).render(app.console)
        assert len(set(colors)) == 3
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert "13.5 MB/s" in rendered(panel)[0]
        assert app.get_widget_at(*stop.region.center)[0] is stop


async def test_unknown_progress_and_external_text_are_preserved(tmp_path):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("monitor-tab")
        progress = DownloadProgress("demo/[red]model-" + "long-name-" * 30, "huggingface", "downloading",
                                    1_000, 0, None, None, None, 0, 2,
                                    detail="retry_cache_baseline_unknown")
        app.preflight.download_progress(progress)
        await pilot.pause()
        panel = app.query_one("#download-progress", DownloadProgressPanel)
        text, _ = rendered(panel)
        assert "demo/[red]model" in text
        assert "未知" in text
        assert "0%" not in text
        assert "重试后的缓存基线" in text
        assert panel.auto_refresh is None
        panel.focus()
        await pilot.press("end")
        await pilot.pause()
        assert panel.scroll_y == panel.max_scroll_y > 0
        visible = "\n".join("".join(segment.text for segment in strip)
                            for strip in app.screen._compositor.render_strips()[panel.region.y:panel.region.bottom])
        assert "校验阶段恢复" in visible
        stop = app.query_one("#stop-run")
        assert app.get_widget_at(*stop.region.center)[0] is stop
        assert stop.region.bottom <= 24
        await pilot.press("home")
        await pilot.pause()
        assert panel.scroll_y == 0
        app.preflight.reset_download()
        assert not panel.display


async def test_progress_refreshes_language_and_theme_without_translating_model(tmp_path):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("monitor-tab")
        app.consume_process_line("", ProgressSnapshot(interface_status="passed"), True)
        progress = DownloadProgress("待确认", "huggingface", "complete", 1_000, 0, 1_000, 1_000, None, 2, 2)
        app.preflight.download_progress(progress)
        await pilot.pause()
        panel = app.query_one("#download-progress", DownloadProgressPanel)
        _, before = rendered(panel)
        old_color = next(segment.style.color for segment in before if "模型文件已校验" in segment.text)
        app.query_one("#ui-language", Select).value = "en"
        app.query_one("#ui-theme", Select).value = "acprof-light"
        await pilot.pause()
        assert app.theme == "acprof-light"
        text, after = rendered(panel)
        assert "待确认" in text
        assert "Model files verified" in text
        assert "100%" in text
        assert "Interface:" in rendered(app.query_one("#status-preparation", Static))[0]
        assert old_color != next(segment.style.color for segment in after if "Model files verified" in segment.text)
        assert app.preflight.latest_download == progress
