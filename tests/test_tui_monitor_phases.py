"""Phase changes reclaim space and keep failures visible above ordinary output."""
from dataclasses import replace
from unittest.mock import patch

import pytest
from textual.widgets import Static
from tui_fixtures import AcprofTui

from acprof.download_events import DownloadProgress
from acprof.tui.log import SelectableLog
from acprof.tui.process import StopResult
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.stages import Stage


def screen_text(app):
    return "\n".join("".join(segment.text for segment in strip)
                     for strip in app.screen._compositor.render_strips())


@pytest.mark.parametrize(("size", "language"), [
    ((80, 24), "zh"), ((80, 24), "en"), ((120, 30), "en"), ((150, 45), "zh"),
])
async def test_monitor_shows_only_the_current_phase(tmp_path, size, language):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        app.activate_tab("monitor-tab")
        app.consume_process_line("", ProgressSnapshot(stage=Stage.PREPARING_MODEL), True)
        progress = DownloadProgress("demo/model", "huggingface", "downloading",
                                    1000, 100, 500, 400, 50, 1, 2)
        app.preflight.download_progress(progress)
        await pilot.pause()
        text = screen_text(app)
        assert "50%" in text
        assert "Case" not in text
        assert "CPU=" not in text
        assert "0/0" not in text
        assert "未开始" not in text
        assert "Not started" not in text
        assert "尝试 1" not in text
        assert "Attempt 1" not in text

        app.preflight.download_progress(replace(progress, state="complete", completed_bytes=1000))
        snapshot = ProgressSnapshot(stage=Stage.RUNTIME_VALIDATION,
                                    interface_status="passed", runtime_status="running")
        app.consume_process_line("", snapshot, True)
        await pilot.pause()
        text = screen_text(app)
        assert ("运行验证" if language == "zh" else "Runtime") in text
        assert "Case" not in text
        assert "CPU=" not in text
        assert not app.query_one("#download-progress").display
        assert "模型文件已校验" not in text
        assert "Model files verified" not in text
        app.consume_process_line("", replace(snapshot, runtime_status="passed"), True)
        await pilot.pause()
        assert ("通过" if language == "zh" else "Passed") in screen_text(app)

        snapshot = replace(snapshot, stage=Stage.MEASURING, current_case=1, total_cases=4,
                           cpu="2", mem="4", gpu="on", runtime_status="passed",
                           measurement_status="running", measurement_active=True)
        app.consume_process_line("", snapshot, True)
        await pilot.pause()
        text = screen_text(app)
        for value in ("Case", "CPU=2", "MEM=4GB", "GPU=on"):
            assert value in text
        assert "0/4" in text
        assert not app.query_one("#download-progress").display
        assert "运行验证：" not in text
        assert "Runtime:" not in text
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.query_one("#run-log", SelectableLog).content_region.height >= 3
        stop = app.query_one("#stop-run")
        assert app.get_widget_at(*stop.region.center)[0] is stop
        app.consume_process_line("", replace(snapshot, stage=Stage.CLEANING_CASE, measurement_active=False), True)
        # A following model must not inherit the previous model's case fields.
        app.consume_process_line("", replace(snapshot, stage=Stage.PREPARING_MODEL, measurement_active=False), True)
        await pilot.pause()
        assert "Case" not in screen_text(app)
        assert "CPU=" not in screen_text(app)
        assert not app.query_one("#download-progress").display


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_failure_reason_and_traceback_survive_info_output(tmp_path, language):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("monitor-tab")
        log = app.query_one("#run-log", SelectableLog)
        log.max_lines = 20
        failure = {"reason_code": "model_load_failed", "detail": "Missing model weights", "stage": "runtime"}
        snapshot = ProgressSnapshot(stage=Stage.RUNTIME_VALIDATION, runtime_status="failed",
                                    detail="Missing model weights", errors=1, failure=failure)
        app.consume_process_line('[ERROR] load failed\nTraceback (most recent call last):\n'
                                 '  File "模型.py", line 2\nValueError: weights missing', snapshot, True)
        app.logs.flush()
        app.logs.append("\n".join(f"[INFO] cleanup {i}" for i in range(50)))
        app.logs.flush()
        await pilot.pause()
        error = app.query_one("#monitor-error")
        assert error.display
        assert "Missing model weights" in screen_text(app)
        assert "ValueError: weights missing" in str(app.query_one("#monitor-error-log", Static).content)
        assert "cleanup 49" not in str(app.query_one("#monitor-error-log", Static).content)
        assert error.region.bottom <= log.region.y
        assert log.content_region.height >= 3
        stop = app.query_one("#stop-run")
        assert app.get_widget_at(*stop.region.center)[0] is stop
        # Changing filters must not hide or translate the pinned diagnosis.
        log.set_view(level="INFO", query="cleanup")
        await pilot.pause()
        assert "Missing model weights" in screen_text(app)
        app.consume_process_line("", ProgressSnapshot(stage=Stage.STARTING), True)
        await pilot.pause()
        assert not error.display


async def test_error_log_is_pinned_without_structured_failure_and_can_be_cleared(tmp_path):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("monitor-tab")
        app.logs.append("[INFO] mentions [ERROR] literally")
        app.logs.flush()
        await pilot.pause()
        assert not app.query_one("#monitor-error").display
        app.logs.append("[ERROR] Missing dependency")
        app.logs.flush()
        await pilot.pause()
        assert "Missing dependency" in screen_text(app)
        assert await pilot.click("#clear-log")
        await pilot.pause()
        assert not app.query_one("#monitor-error").display


async def test_error_render_is_deferred_until_measurement_ends(tmp_path):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("monitor-tab")
        app.consume_process_line("", ProgressSnapshot(stage=Stage.MEASURING, measurement_active=True), True)
        await pilot.pause()
        before = screen_text(app)
        with patch.object(app, "set_timer", side_effect=AssertionError("extra measurement timer")):
            app.logs.append("[ERROR] deferred diagnostic")
            await pilot.pause()
            assert screen_text(app) == before
        app.consume_process_line("", ProgressSnapshot(stage=Stage.FAILED, detail="Workload failed"), True)
        app.logs.flush()
        await pilot.pause()
        assert "Workload failed" in screen_text(app)
        assert "deferred diagnostic" in str(app.query_one("#monitor-error-log", Static).content)


async def test_cleanup_failure_shows_reason_without_replaying_measurement_logs(tmp_path):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("monitor-tab")
        app.consume_process_line("", ProgressSnapshot(stage=Stage.MEASURING, measurement_active=True), True)
        app.logs.append("[ERROR] buffered diagnostic")
        with patch.object(app.logs.log.records.error_context, "snapshot", side_effect=AssertionError("read during measurement")):
            app.process_cleanup_incomplete(StopResult(12345, None, "SIGTERM timeout"))
            await pilot.pause()
        assert "SIGTERM timeout" in screen_text(app)
        assert "buffered diagnostic" not in screen_text(app)
        assert app.logs.log.suspended
        assert app.logs.timer is None


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_long_error_is_scrollable_and_keeps_log_actions_reachable(tmp_path, language):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("monitor-tab")
        reason = "\n".join(f"模型诊断 / diagnostic {i}" for i in range(30)) + "\nROOT-CAUSE-END"
        app.consume_process_line("", ProgressSnapshot(stage=Stage.FAILED, detail=reason), True)
        await pilot.pause()
        panel = app.query_one("#monitor-error")
        panel.focus()
        await pilot.press("end")
        await pilot.pause()
        assert panel.scroll_y == panel.max_scroll_y > 0
        assert "ROOT-CAUSE-END" in screen_text(app)
        await pilot.press("f8")
        await pilot.pause()
        assert panel.region.height > 0
        for selector in ("#stop-run", "#restore-log", "#copy-log"):
            button = app.query_one(selector)
            assert app.get_widget_at(*button.region.center)[0] is button
        await pilot.press("escape")
        await pilot.pause()
        assert panel.region.bottom <= app.query_one("#run-log").region.y
