"""Fixed rendering fixtures; run only in the pinned snapshot environment."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import pytest_textual_snapshot
from textual.widgets import Button, TabbedContent

from acprof.experiment import RunConfig
from acprof.platform import Environment
from acprof.tui.app import AcprofTui
from acprof.tui.process import StopResult
from acprof.tui.progress import ProgressSnapshot
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
    environment = Environment("wsl2" if scene == "wsl-preparation" else "native_linux")
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
        if scene == "wsl-preparation":
            await pilot.press("f5")
            await pilot.pause()
            from acprof.tui.preparation import PreparationScreen
            assert isinstance(app.screen, PreparationScreen)
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
