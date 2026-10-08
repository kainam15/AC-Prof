"""Fixed rendering fixtures; run only in the pinned snapshot environment."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import pytest_textual_snapshot
from textual.widgets import Button, Static, TabbedContent

from acprof.experiment import RunConfig
from acprof.platform import Environment
from acprof.preparation_events import encode_event
from acprof.tui.app import AcprofTui
from acprof.tui.log import SelectableLog
from acprof.tui.preparation import PreparationScreen
from acprof.tui.process import StopResult
from acprof.tui.progress import ProgressSnapshot, RunProgressTracker
from acprof.tui.reports import ReportRow, ReportView
from acprof.tui.table import ResizableDataTable
from acprof.tui.views import ConfirmActionScreen

pytestmark = pytest.mark.visual


@pytest.mark.parametrize("language,size,scene", [
    ("zh", (80, 24), "form"),
    ("en", (120, 30), "form"),
    ("zh", (150, 45), "form"),
    ("zh", (80, 24), "modal"),
    ("en", (120, 30), "resized-table"),
    ("zh", (120, 30), "measuring"),
    ("en", (80, 24), "cleanup-incomplete"),
    ("zh", (80, 24), "wsl-preparation"),
    ("en", (120, 30), "wsl-preparation"),
    ("zh", (80, 24), "wsl-review"),
    ("en", (120, 30), "wsl-review"),
])
def test_fixed_scenes(snap_compare, tmp_path, monkeypatch, language, size, scene):
    normalize = pytest_textual_snapshot.normalize_svg
    # Rich emits whitespace-only CSS lines. Keep baseline serialization stable
    # under the repository's whitespace hooks without changing rendered text.
    monkeypatch.setattr(pytest_textual_snapshot, "normalize_svg",
                        lambda svg: "\n".join(line.rstrip() for line in normalize(svg).splitlines()) + "\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("acprof.tui.app.PROJECT_DIR", Path("/workspace"))
    monkeypatch.setattr("acprof.tui.app.PYTHON_EXECUTABLE", Path("/usr/bin/python"))
    environment = Environment("wsl2" if scene in {"wsl-preparation", "wsl-review"} else "native_linux")
    monkeypatch.setattr("acprof.tui.app.detect_environment", lambda: environment)
    monkeypatch.setattr("acprof.tui.diagnostics.detect_environment", lambda: environment)
    monkeypatch.setattr("acprof.tui.app.quick_preflight", lambda *args, **kwargs: [])
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    monkeypatch.setattr(app, "_execute_command", lambda *args: None)
    app.ui_preferences = replace(app.ui_preferences, theme="acprof-graphite", language=language)

    async def prepare(pilot):
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.clear_notifications()
        app.set_input_cursor_blink_enabled(False)
        if scene in {"wsl-preparation", "wsl-review"}:
            monkeypatch.setattr(app, "_format_elapsed", lambda _seconds: "00:00:01")
            await pilot.press("f5")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._process_kind == "run"
            assert app._active_run_config.profiling_mode == "basic"
            process = Mock(pid=12345, returncode=None)
            process.poll.return_value = None
            monkeypatch.setattr(app._lifecycle, "process", process)
            monkeypatch.setattr(app._lifecycle, "stop", lambda **kwargs: StopResult(12345, 0))
            app._process_started(process.pid, "run")
            tracker = RunProgressTracker(structured=True)
            snapshot = tracker.feed(encode_event("interface", "running"))
            app._consume_process_line("Checking model interface...", snapshot, True)
            app._preparation_event({"stage": "interface", "status": "running"})
            await pilot.pause()
            assert app.query_one("#main-tabs", TabbedContent).active == "monitor-tab"
            assert app.screen.id == "_default"
            assert app._preparation_screen is None
            assert app.query_one("#status-stage", Static).content == app.tr("正在确认模型接口…")
            assert app._latest_snapshot.interface_status == "running"
            assert app._latest_snapshot.runtime_status == "not_started"
            assert not app._latest_snapshot.measurement_active
            log = app.query_one("#run-log", SelectableLog)
            assert "Checking model interface..." in log.text
            assert log.region.width > 0 and log.region.height > 0
            assert app.get_widget_at(log.content_region.x + 1, log.content_region.y + 1)[0] is log
            assert not app.query_one("#stop-run", Button).disabled
            if scene == "wsl-review":
                # Only an explicit user decision covers the passive monitor.
                request = {"id": 1, "kind": "review", "resolved": True, "questions": [],
                           "fields": {"model": "fixture/model", "revision": "a" * 40,
                                      "task": "fill-mask", "backend": "transformers_pipeline"}}
                snapshot = tracker.feed(encode_event("resolution", "waiting", request=request))
                app._consume_process_line("Model fields require confirmation.", snapshot, True)
                app._preparation_event({"stage": "resolution", "status": "waiting", "request": request})
                await pilot.pause()
                assert isinstance(app.screen, PreparationScreen)
                assert app._preparation_request == (process, 1)
                assert app._latest_snapshot.interface_status == "waiting"
                assert not app._latest_snapshot.measurement_active
                assert "fixture/model" in app.screen.query_one("#preparation-fields", Static).content
                confirm = app.screen.query_one("#preparation-continue", Button)
                assert str(confirm.label) == ("确定" if language == "zh" else "Confirm")
                for identifier in ("preparation-cancel", "preparation-continue"):
                    button = app.screen.query_one(f"#{identifier}", Button)
                    assert not button.disabled
                    assert button.region.width > 0 and button.region.height > 0
                    assert app.get_widget_at(button.region.x + 1, button.region.y + 1)[0] is button
                await pilot.press("tab")
                assert app.focused.id in {"preparation-cancel", "preparation-continue"}
        elif scene == "modal":
            app.push_screen(ConfirmActionScreen("终止当前任务？", "确认后将请求采集进程安全停止。", "终止任务"))
        elif scene == "resized-table":
            app._report_view = ReportView(Path("fixture.json"), "Latency report", ("Metric", "Mean", "Status"),
                (ReportRow(("Application latency", "25 ms", "ok"), "Fixed fixture"),), "Fixed fixture")
            app._render_report_view()
            app._activate_tab("reports-tab")
            await pilot.pause()
            table = app.query_one("#report-table", ResizableDataTable)
            # This scene verifies the drag result, without a timing-dependent
            # help tooltip covering its cells during slower test runs.
            table.tooltip = None
            # Drag the visible first header separator with actual mouse events.
            boundary = next(table._header_boundaries())[0]
            origin = table.content_region.offset - table.region.offset
            start = (origin.x + boundary, origin.y)
            await pilot.mouse_down(table, offset=start)
            end = (table.region.x + start[0] + 5, table.region.y + start[1])
            await pilot.hover(offset=end)
            await pilot.mouse_up(offset=end)
        elif scene in {"measuring", "cleanup-incomplete"}:
            process = Mock(pid=12345, returncode=None)
            process.poll.return_value = None
            monkeypatch.setattr(app._lifecycle, "process", process)
            monkeypatch.setattr(app._lifecycle, "stop", lambda **kwargs: StopResult(12345, 0))
            app._process_kind = "run"
            app._active_command = ("acprof", "run")
            app._pending_launch = None
            snapshot = ProgressSnapshot(
                stage="正式测量", detail="Fixed measurement window",
                current_case=1, total_cases=4, cpu="2", mem="4", gpu="off", measurement_active=True,
                interface_status="passed", runtime_status="passed", measurement_status="running",
            )
            app._set_busy(True)
            await pilot.pause()
            app._activate_tab("monitor-tab")
            await pilot.pause()
            assert app.query_one("#main-tabs", TabbedContent).active == "monitor-tab"
            app._consume_process_line("", snapshot, True)
            if scene == "cleanup-incomplete":
                app._process_cleanup_incomplete(StopResult(12345, None, "SIGTERM timeout"))
            assert not app.query_one("#stop-run", Button).disabled
            app.clear_notifications()
        await pilot.pause()
        app.clear_notifications()
        await pilot.pause()
        # No focused input avoids blinking or platform cursor differences.
        app.set_focus(None)

    try:
        assert snap_compare(app, terminal_size=size, run_before=prepare)
    finally:
        app.set_input_cursor_blink_enabled(True)
