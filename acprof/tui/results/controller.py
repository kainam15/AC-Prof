"""Read-only result/report workflows, cancellable readers, and presentation.

The parent App owns background-reader tokens, process mutual exclusion and
lifecycle; read-only work is never initiated from a measurement window.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Event, Lock
from typing import TYPE_CHECKING

from rich.text import Text
from textual.timer import Timer
from textual.widgets import Button, Collapsible, ContentSwitcher, DataTable, Static, TabbedContent
from textual.worker import Worker

from acprof.experiments.configuration import RunConfigError
from acprof.messages import join_messages
from acprof.tui.commands import (
    PendingLaunch,
    prepare_comparison,
    prepare_plot,
    prepare_stats,
    resolve_result_path,
)
from acprof.tui.experiments.catalog import ExperimentSourceDisplay
from acprof.tui.i18n import error_message, message_id
from acprof.tui.results.data_status import data_status_text, inspect_experiment_data
from acprof.tui.results.html_controller import HtmlReportController
from acprof.tui.results.overview import overview_view, read_experiment_overview, selection_path
from acprof.tui.results.overview_panel import ExperimentOverviewPanel
from acprof.tui.results.report_actions import report_action_state
from acprof.tui.results.reports import ReportView, read_report
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.structured import StatusBanner

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class _OverlappingOfflineReads:
    """Share one measurement exclusion lease across concurrent read-only jobs.

    Slow, cancelled readers must not block newer selections. Holding the lease
    until the last reader exits still excludes the collector, including when
    the latest reader finishes before an obsolete one.
    """

    def __init__(self) -> None:
        self._guard = Lock()
        self._readers = 0
        self._lease: AbstractContextManager[object] | None = None

    @contextmanager
    def read(self) -> Iterator[None]:
        from acprof.host.run_state import offline_analysis_lock

        with self._guard:
            if self._readers == 0:
                lease = offline_analysis_lock()
                lease.__enter__()
                self._lease = lease
            self._readers += 1
        try:
            yield
        finally:
            with self._guard:
                self._readers -= 1
                if self._readers == 0:
                    held_lease = self._lease
                    self._lease = None
                    assert held_lease is not None
                    held_lease.__exit__(None, None, None)


class ResultController:
    """Own results state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.html = HtmlReportController(app)
        self.offline_reads = _OverlappingOfflineReads()
        self.overview_request: Event | None = None
        self.data_request: Event | None = None
        self.source_path: Path | None = None
        self.overview = None
        self.data_status = None
        self.data_error = ""
        self.data_cancelled = False
        self.selection_timer: Timer | None = None
        self.report_request: Event | None = None
        self.report_path: Path | None = None
        self.report_view: ReportView | None = None
        self.analysis_view: ReportView | None = None
        self.analysis_source = app.project_dir
        self.analysis_selection: tuple[str, ...] = ()
        self.analysis_kind = "stats"
        self.analysis_html_path = ""
        self.source_display: ExperimentSourceDisplay | None = None
        self.unavailable_remembered_source = ""
        self.formal_ok_rows: int | None = None

    def clear_log_button(self) -> None:
        self.clear_log()

    def clear_log(self) -> None:
        self.app.logs.clear()

    def restore_recent_result(self) -> None:
        """Keep unavailable recent paths in settings, not in the active selection."""
        source = self.app.configuration.input('experiment-source').strip()
        if source:
            path = selection_path(resolve_result_path(source, self.app.project_dir))
            if not path.exists():
                self.unavailable_remembered_source = source
                with self.app.prevent(Input.Changed):
                    self.app.query_one('#experiment-source', Input).value = ''
                self.refresh_source_display()
            else:
                self.select_result(str(path))
                return
        self.app.localization.set_text(self.app.query_one('#report-empty', Static), self.statistics_empty_hint())
        self.show_overview_hint(self.source_summary_hint())

    def show_overview_hint(self, source: str, *, level: str = "neutral") -> None:
        self.app.query_one(ExperimentOverviewPanel).show_hint(source, level=level)

    def render_overview(self) -> None:
        if self.overview is None or self.app.execution.session.snapshot.measurement_active:
            return
        self.app.query_one(ExperimentOverviewPanel).update_view(
            overview_view(self.overview, self.data_status, data_error=self.data_error,
                          data_cancelled=self.data_cancelled))

    def overview_action(self, action_id: str) -> None:
        if action_id == "overview-reason":
            self.app.query_one(ExperimentOverviewPanel).show_reason()
        elif action_id == "overview-statistics":
            self.show_result_section("statistics-view")
        elif action_id == "overview-resume" and self.overview is not None:
            # Existing recovery re-reads evidence and retains launch/preflight gates.
            self.app.catalog.resume_experiment(self.overview.record)

    def source_summary_hint(self):
        if self.unavailable_remembered_source:
            return message_id('results.summary.previous_unavailable')
        display = self.source_display
        if display is None:
            return message_id('results.summary.empty')
        if display.availability == 'no_data':
            return message_id('results.summary.no_data')
        if display.availability == 'missing':
            return message_id('results.summary.path_missing')
        if display.availability == 'unsupported':
            return message_id('results.summary.unsupported_source')
        return message_id('results.selection.changed')

    def statistics_empty_hint(self):
        if self.unavailable_remembered_source:
            return message_id('results.statistics.previous_unavailable')
        display = self.source_display
        if display is None:
            return message_id('results.statistics.no_selection')
        if display.availability == 'no_data':
            return message_id('results.statistics.no_data')
        if display.availability == 'missing':
            return message_id('results.statistics.path_missing')
        if display.availability == 'unsupported':
            return message_id('results.statistics.unsupported_source')
        return message_id('results.statistics.empty_hint')

    def result_source_changed(self, *, force: bool = False) -> None:
        if not self.app.form_ready:
            return
        source = self.app.configuration.input("experiment-source").strip()
        path = resolve_result_path(source, self.app.project_dir) if source else None
        if not force and path == self.source_path:
            return
        self.refresh_experiment(source, load=False)
        if source:
            self.show_overview_hint(message_id("results.overview.loading"), level="info")
            self.app.profiles.reset_overview(message_id("results.overview.profiles_checking"), phase="checking")
            self.selection_timer = self.app.set_timer(0.2, self.start_experiment_reads)

    def select_result(self, source: str, *, load: bool = True) -> None:
        """Retain the experiment directory as the shared, persistent selection."""
        if source:
            source = str(selection_path(resolve_result_path(source, self.app.project_dir)))
        with self.app.prevent(Input.Changed):
            self.app.query_one("#experiment-source", Input).value = source
        self.refresh_experiment(source, load=load)

    def show_result_section(self, section: str) -> None:
        if section == "exports-view":
            section = "statistics-view"
        self.app.activate_tab("results-tab")
        views = self.app.query_one("#result-views", TabbedContent)
        if views.get_tab(section).display:
            views.active = section

    def refresh_experiment(self, source: str | None = None, *, load: bool = True) -> None:
        source = self.app.configuration.input("experiment-source").strip() if source is None else source
        if source != self.app.configuration.input("experiment-source"):
            self.select_result(source, load=load)
            return
        self.invalidate_reads()
        self.source_path = resolve_result_path(source, self.app.project_dir) if source else None
        self.overview = self.data_status = self.source_display = None
        self.data_error = ""
        self.data_cancelled = False
        self.formal_ok_rows = None
        self.unavailable_remembered_source = ""
        self.app.profiles.reset_overview()
        self.report_source_changed()
        for selector in ("#result-data-status", "#result-html-status"):
            self.app.localization.set_text(self.app.query_one(selector, Static), "")
        self.show_overview_hint(self.source_summary_hint())
        if not source or not load:
            return
        self.start_experiment_reads()

    def start_experiment_reads(self) -> None:
        """Start a settled selection without clearing independently edited reports."""
        self.selection_timer = None
        if self.source_path is None or not self.app.is_running or self.app.ui_closing:
            return
        if not self.app.operation_state().allows("summary"):
            self.show_overview_hint("请等待当前任务完成")
            return
        self.source_path = selection_path(self.source_path)
        with self.app.prevent(Input.Changed):
            self.app.query_one("#experiment-source", Input).value = str(self.source_path)
        self.refresh_source_display()
        self.show_overview_hint(message_id("results.overview.loading"), level="info")
        self.app.localization.set_text(self.app.query_one("#result-data-status", Static), message_id("results.overview.checking"))
        self.app.profiles.reset_overview(message_id("results.overview.profiles_checking"), phase="checking")
        self.html.show_discovered(None, "checking")
        self.data_cancelled = False
        self.overview_request, self.data_request = Event(), Event()
        for kind, token in (("overview", self.overview_request), ("data", self.data_request)):
            self.app.read_jobs.add(token)
            self.app.run_worker(
                partial(self._read_experiment, self.source_path, token, kind),
                name="read_experiment_" + kind, thread=True, group="experiment-" + kind, exit_on_error=False,
            )
        self.app.set_busy(True)

    def refresh_data_status(self) -> None:
        """Posthoc completion refreshes data/profiler/report state, retaining metadata."""
        if self.source_path is None or not self.app.operation_state().allows("summary"):
            return
        if self.data_request is not None:
            self.data_request.set()
        token = self.data_request = Event()
        self.data_status = None
        self.data_error = ""
        self.data_cancelled = False
        self.render_overview()
        self.app.profiles.reset_overview(message_id("results.overview.profiles_checking"), phase="checking")
        self.html.show_discovered(None, "checking")
        self.app.localization.set_text(self.app.query_one("#result-data-status", Static), message_id("results.overview.checking"))
        self.app.read_jobs.add(token)
        self.app.run_worker(partial(self._read_experiment, self.source_path, token, "data"),
            name="read_experiment_data", thread=True, group="experiment-data", exit_on_error=False)
        self.app.set_busy(True)

    def _read_experiment(self, path: Path, token: Event, kind: str) -> None:
        from acprof.cancellation import check_cancelled

        try:
            with self.offline_reads.read():
                check_cancelled(token.is_set)
                result = (read_experiment_overview(path) if kind == "overview"
                          else inspect_experiment_data(path, cancelled=token.is_set))
                check_cancelled(token.is_set)
            error = ""
        except Exception as exc:
            result, error = None, error_message(exc)
        self.app.process.safe_process_callback(self.experiment_read_finished, path, token, kind, result, error)

    def experiment_read_finished(self, path: Path, token: Event, kind: str, result, error: str) -> None:
        self.app.read_jobs.discard(token)
        if not self.app.is_running or self.app.ui_closing:
            return
        self.app.set_busy(self.app.is_busy())
        if not self.app.read_jobs and not self.app.preflight.check_completed and not self.app.preflight.check_running:
            self.app.call_after_refresh(self.app.preflight.quick_check)
        current = self.overview_request if kind == "overview" else self.data_request
        if token is not current or token.is_set() or path != self.source_path:
            return
        if kind == "overview":
            self.overview = result
            if result is None:
                self.source_display = ExperimentSourceDisplay("", None, "", "missing")
                self.show_overview_hint(message_id("results.summary.read_failed", error), level="error")
            else:
                self.source_display = ExperimentSourceDisplay(result.record.model_id, result.record.run_id,
                    result.record.status, "selected" if result.data_state == "exists" else "no_data")
                with self.app.prevent(Input.Changed):
                    self.app.query_one("#experiment-source", Input).value = str(path)
                self.refresh_source_display()
                self.app.configuration.remember_last_used(result_csv=str(path))
        else:
            self.data_status = result
            self.data_error = error
            self.formal_ok_rows = result.formal_ok_rows if result else 0
            self.app.localization.set_text(self.app.query_one("#result-data-status", Static),
                data_status_text(result) if result else message_id("results.summary.read_failed", error))
            self.app.profiles.show_overview(result.profiler_overview if result else None,
                                            result.profiler_error if result else error)
            if result:
                self.html.show_discovered(result.html_path, result.html_state)
            else:
                self.html.show_discovered(None, "unverified")
            if result is not None and self.formal_ok_rows == 0:
                self.app.localization.set_text(self.app.query_one("#report-empty", Static), message_id("results.statistics.no_successful_windows"))
        self.render_overview()
        if self.source_display and not self.source_display.can_read:
            self.app.localization.set_text(self.app.query_one("#report-empty", Static), self.statistics_empty_hint())
        self.update_report_actions()

    def invalidate_reads(self) -> None:
        self.app.profiles.cancel_validation()
        if self.selection_timer is not None:
            self.selection_timer.stop()
            self.selection_timer = None
        for token in (self.overview_request, self.data_request):
            if token is not None:
                token.set()
        self.overview_request = self.data_request = None

    def cancel_result_reads(self) -> None:
        if self.app.recovery.recovery_request is not None:
            self.app.recovery.recovery_request.set()
        self.app.recovery.recovery_after_check = None
        self.data_cancelled = self.data_status is None
        self.invalidate_reads()
        if self.report_request is not None:
            self.report_request.set()
        self.report_request = None
        if self.overview is None:
            self.show_overview_hint(message_id("results.read.cancelling"))
        else:
            self.render_overview()
        if self.html.freshness == "checking":
            self.html.show_discovered(None, "unverified")
        self.app.localization.set_text(self.app.query_one("#result-data-status", Static), "读取已取消；等待后台任务释放资源。")
        self.app.profiles.reset_overview("读取已取消；等待后台任务释放资源。", phase="cancelled")
        self.clear_report("读取已取消；等待后台任务释放资源。")
        self.app.set_busy(self.app.is_busy())

    def plot_results_button(self) -> None:
        self.launch_plot()

    def launch_plot(self, path: str | None = None) -> None:
        if not self.app.allow_operation("plot"):
            return
        result_csv = path or self.app.configuration.input("experiment-source")
        if not result_csv:
            self.app.notify("请填写结果 CSV 路径", severity="warning")
            return
        try:
            pending = prepare_plot(result_csv, project_dir=self.app.project_dir, python_executable=self.app.python_executable)
        except RunConfigError as exc:
            self.app.notify(str(exc), severity="error")
            return
        self.app.launch(pending)

    def refresh_source_display(self) -> None:
        """Refresh source availability for summary hints and analysis actions."""
        value = self.app.query_one('#experiment-source', Input).value
        source = value.strip()
        self.app.query_one("#experiment-source", Input).tooltip = source or None
        with self.app.prevent(Input.Changed):
            self.app.query_one("#backfill-source", Input).value = value

    def update_report_actions(self) -> None:
        """Apply pure action rules to the existing widgets and operation lock."""
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        from acprof.platform import can_open_local_browser

        actions = report_action_state(
            self.app.operation_state(),
            source_ready=bool(self.source_display and self.source_display.can_read and (self.data_status is not None or self.data_request in self.app.read_jobs)),
            formal_ok_rows=self.formal_ok_rows,
            analysis_payload=self.report_view.payload if self.report_view is not None else {},
            report_json_path=self.app.configuration.input("report-json-path"),
            html_path=self.app.configuration.input("html-report-path"),
            execution_kind=self.app.execution.session.kind,
            browser_available=can_open_local_browser(),
        )
        if not self.app.execution.session.snapshot.measurement_active:
            self.app.query_one("#overview-banner", StatusBanner).set_action_disabled(
                "overview-resume", not self.app.operation_state().allows("run"))
        cancel_analysis = self.app.query_one("#results-cancel", Button)
        reading = any(token in self.app.read_jobs for token in (
            self.overview_request, self.data_request, self.app.profiles.validation_request))
        cancel_analysis.display = actions.cancel_analysis_visible or reading
        cancel_analysis.disabled = not (actions.cancel_analysis_enabled or reading)
        self.app.query_one("#result-actions").set_class(cancel_analysis.display, "cancellable")
        cancel_backfill = self.app.query_one("#backfill-cancel", Button)
        cancel_backfill.display = reading
        cancel_backfill.disabled = not cancel_backfill.display
        for format, enabled in (
            ("csv", actions.analysis_export_csv_enabled),
            ("json", actions.analysis_export_json_enabled),
        ):
            self.app.query_one("#analysis-export-" + format, Button).disabled = not enabled
        for selector, enabled in (
            ("#refresh-experiment-state", bool(self.source_path) and self.app.operation_state().allows("summary")),
            ("#refresh-backfill-state", bool(self.source_path) and self.app.operation_state().allows("summary")),
            ("#plot-results", actions.plot_enabled),
            ("#report-calculate", actions.html_generate_enabled),
            ("#report-window-stats", actions.calculate_enabled),
            ("#html-report-generate", actions.html_generate_enabled),
            ("#report-open", actions.report_open_enabled),
            ("#html-report-copy", actions.html_copy_enabled),
            ("#html-export-toggle", actions.html_export_toggle_enabled),
            ("#html-report-export", actions.html_export_enabled),
        ):
            self.app.query_one(selector, Button).disabled = not enabled
        self.app.query_one("#html-report-file").display = actions.html_file_visible
        self.app.query_one("#html-report-panel").set_class(actions.html_file_visible, "has-report")
        self.app.query_one("#html-report-filename", Static).update(actions.html_filename)
        location = self.app.query_one("#html-report-location", Static)
        location.update(actions.html_location)
        location.tooltip = actions.html_location or None
        if actions.hide_html_export_fields:
            self.app.query_one("#html-export-fields").display = False
        self.app.query_one("#html-remote-hint").display = actions.html_remote_hint_visible
        open_html = self.app.query_one("#html-report-open", Button)
        open_html.display = actions.html_open_visible
        open_html.disabled = not actions.html_open_enabled or self.html.freshness in {"stale", "damaged"}
        generate_html = self.app.query_one("#html-report-generate", Button)
        generate_html.display = actions.html_generate_visible
        cancel = self.app.query_one("#html-report-cancel", Button)
        cancel.display = actions.html_cancel_visible
        cancel.disabled = not actions.html_cancel_enabled

    def clear_report(self, status: str, *, phase: str = 'empty', empty_hint=None) -> None:
        self.report_view = None
        self.analysis_view = None
        self.app.query_one("#report-table", DataTable).clear(columns=True)
        self.app.localization.set_text(self.app.query_one("#report-status", Static), status)
        self.app.localization.set_text(
            self.app.query_one("#report-detail", Static),
            "表格可滚动；选择一行查看口径与数据来源。",
        )
        target = {"empty": "report-empty", "loading": "report-loading", "error": "report-error"}[phase]
        if phase != "empty":
            self.app.localization.set_text(self.app.query_one("#" + target, Static), status)
        else:
            self.app.localization.set_text(self.app.query_one('#report-empty', Static), empty_hint or message_id('results.statistics.empty_hint'))
        self.app.query_one("#report-content", ContentSwitcher).current = target
        self.update_report_actions()

    def report_source_changed(self) -> None:
        if not self.app.form_ready:
            return
        self.refresh_source_display()
        if self.report_request is not None:
            self.report_request.set()
        self.report_request = None
        with self.app.prevent(Input.Changed):
            self.app.query_one("#report-json-path", Input).value = ""
            self.app.query_one("#html-report-path", Input).value = ""
        self.app.localization.set_text(self.app.query_one("#analysis-export-status", Static), "")
        self.clear_report(message_id('results.statistics.source_hint') if self.source_display and self.source_display.can_read
                          else '', empty_hint=self.statistics_empty_hint())
        self.update_report_actions()

    def report_json_changed(self) -> None:
        if not self.app.form_ready:
            return
        source = self.app.configuration.input("report-json-path").strip()
        if (self.report_request is not None and
                resolve_result_path(source, self.app.project_dir) == self.report_path):
            return
        if self.report_request is not None:
            self.report_request.set()
        self.report_request = None
        self.clear_report(message_id("results.import.path_changed") if source else message_id("results.statistics.source_hint"))
        self.update_report_actions()

    def open_report_button(self) -> None:
        self.open_report()

    def open_report(self, path: str | None = None) -> None:
        if not self.app.allow_operation("report"):
            return
        self.show_result_section("statistics-view")
        self.app.query_one("#analysis-details", Collapsible).collapsed = False
        self.app.query_one("#report-panel").scroll_visible(animate=False)
        source = path if path is not None else self.app.configuration.input("report-json-path")
        if not source.strip():
            self.clear_report(message_id("results.import.json_required"), phase="error")
            self.show_result_section("exports-view")
            self.app.query_one("#report-file-options", Collapsible).collapsed = False
            return
        report_path = resolve_result_path(source, self.app.project_dir)
        if report_path.suffix.lower() != ".json":
            self.clear_report("请选择 JSON 报告；实验目录或 CSV 请点击“计算统计”。", phase="error")
            self.show_result_section("exports-view")
            self.app.query_one("#report-file-options", Collapsible).collapsed = False
            return
        with self.app.prevent(Input.Changed):
            self.app.query_one("#report-json-path", Input).value = str(report_path)
        self.update_report_actions()
        self.clear_report(message_id("results.import.reading"), phase="loading")
        if self.report_request is not None:
            self.report_request.set()
        token = self.report_request = Event()
        self.report_path = report_path
        self.app.read_jobs.add(token)
        self.app.set_busy(True)
        self.execute_report_read(report_path, token)

    def execute_report_read(self, path: Path, token: Event) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_report_read_worker, path, token),
            name="execute_report_read", thread=True, group='report', exit_on_error=False,
        )

    def _execute_report_read_worker(self, path: Path, token: Event) -> None:
        try:
            with self.offline_reads.read():
                view, error = read_report(path, cancelled=token.is_set), ""
        except Exception as exc:
            view, error = None, error_message(exc)
        self.app.process.safe_process_callback(self.show_report, view, error, token)

    def show_report(self, view: ReportView | None, error: str, token: object) -> None:
        self.app.read_jobs.discard(token)
        if not self.app.is_running or self.app.ui_closing:
            return
        if (token is not self.report_request or
                resolve_result_path(self.app.configuration.input("report-json-path"), self.app.project_dir) != self.report_path):
            self.app.set_busy(self.app.is_busy())
            return
        self.app.set_busy(self.app.is_busy())
        if view is None:
            self.clear_report(message_id('results.import.read_failed', error), phase="error")
            return
        self.report_view = replace(view, note=join_messages("\n", (
            message_id("results.import.historical_notice", view.source), view.note)))
        self.render_report_view()
        self.update_report_actions()

    def render_report_view(self) -> None:
        view = self.report_view
        if view is None:
            return
        self.app.query_one("#analysis-details", Collapsible).collapsed = False
        table = self.app.query_one("#report-table", DataTable)
        cursor, scroll = table.cursor_coordinate, table.scroll_offset
        table.clear(columns=True)
        if not view.rows:
            self.app.localization.set_text(self.app.query_one("#report-status", Static), view.title)
            self.app.localization.set_text(
                self.app.query_one("#report-empty", Static), view.note or message_id("results.statistics.no_rows"),
            )
            self.app.query_one("#report-content", ContentSwitcher).current = "report-empty"
            return
        for column in view.columns:
            table.add_column(Text(self.app.localization.tr(column)), key=str(column))
        for index, row in enumerate(view.rows):
            table.add_row(*(Text(self.app.localization.tr(cell)) for cell in row.cells), key=str(index))
        table.move_cursor(row=min(cursor.row, max(0, len(view.rows) - 1)), column=cursor.column, scroll=False)
        table.scroll_to(scroll.x, scroll.y, animate=False, force=True)
        self.app.localization.set_text(self.app.query_one("#report-status", Static), view.title)
        self.app.query_one("#report-content", ContentSwitcher).current = "report-results"
        self.show_report_row(table.cursor_row)

    def report_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.show_report_row(event.cursor_row)

    def show_report_row(self, index: int) -> None:
        view = self.report_view
        if view is None:
            return
        detail = view.rows[index].detail if 0 <= index < len(view.rows) else message_id('results.statistics.no_successful_windows')
        self.app.localization.set_text(self.app.query_one("#report-detail", Static), join_messages("\n", (detail, view.note)))

    def calculate_report_button(self) -> None:
        self.html.launch()

    def compare_experiments_button(self) -> None:
        if not self.app.allow_operation("compare"):
            return
        try:
            pending = prepare_comparison(self.app.configuration.input("comparison-left"), self.app.configuration.input("comparison-right"),
                baseline=self.app.configuration.select("comparison-baseline"), purpose=self.app.configuration.select("comparison-purpose"),
                project_dir=self.app.project_dir, python_executable=self.app.python_executable,
                html_report=True)
        except (RunConfigError, ValueError, OSError) as exc:
            self.app.localization.set_text(self.app.query_one("#comparison-status", Static), error_message(exc))
            self.app.notify(error_message(exc), severity="error")
            return
        self.app.launch(pending)

    def launch_stats(self, path: str | None = None) -> None:
        if not self.app.allow_operation("stats"):
            return
        self.show_result_section("statistics-view")
        self.app.query_one("#analysis-details", Collapsible).collapsed = False
        self.app.query_one("#report-panel").scroll_visible(animate=False)
        source = path if path is not None else self.app.configuration.input("experiment-source")
        if not source.strip():
            self.clear_report(message_id("results.statistics.input_required"), phase="error")
            return
        try:
            pending = prepare_stats(source, project_dir=self.app.project_dir, python_executable=self.app.python_executable,
                                    analysis_args=())
        except (OSError, ValueError) as exc:
            self.clear_report(error_message(exc), phase="error")
            return
        if pending.result_csv != self.app.configuration.input("experiment-source"):
            self.select_result(pending.result_csv, load=False)
        self.app.launch(pending)

    def selection_identity(self) -> tuple[str, ...]:
        return tuple(self.app.configuration.input(name) for name in
                     ("experiment-source", "comparison-left", "comparison-right")) + (
                         self.app.configuration.select("comparison-baseline"),
                         self.app.configuration.select("comparison-purpose"),
                     )

    def begin_analysis(self, pending: PendingLaunch) -> None:
        self.analysis_kind = pending.kind
        self.analysis_html_path = pending.report_path
        self.clear_report(message_id("results.statistics.calculating"), phase="loading")
        if pending.kind == "compare":
            self.app.localization.set_text(self.app.query_one("#comparison-status", Static), message_id("results.analysis.comparing"))
        self.analysis_selection = self.selection_identity()
        source = pending.result_csv if pending.kind == "stats" else pending.command[pending.command.index("--left") + 1]
        self.analysis_source = Path(source)
        with self.app.prevent(Input.Changed):
            self.app.query_one("#report-json-path", Input).value = ""

    def accept_analysis(self, view: ReportView) -> None:
        if self.selection_identity() == self.analysis_selection:
            self.analysis_view = view

    def finish_analysis(self, *, error: str, cancelled: bool) -> None:
        view, self.analysis_view = self.analysis_view, None
        if cancelled or error or view is None:
            status = error or message_id("results.statistics.cancelled" if cancelled else "results.statistics.invalid_analysis_output")
            self.clear_report(status, phase="error")
            section = "comparison-view" if self.analysis_kind == "compare" else "statistics-view"
            self.show_result_section(section)
            if self.analysis_kind == "compare":
                self.app.localization.set_text(self.app.query_one("#comparison-status", Static), status)
        else:
            self.show_result_section("statistics-view")
            self.report_view = view
            self.render_report_view()
            html_path = view.payload.get("html_report_path")
            if html_path and html_path == self.analysis_html_path:
                self.app.query_one("#html-report-path", Input).value = html_path
                self.app.configuration.remember_last_used(html_report=html_path)
                self.html.set_status(message_id("results.comparison.created_notice"))
                self.app.query_one("#analysis-details", Collapsible).collapsed = True
                self.app.query_one("#html-report-panel").scroll_visible(animate=False)
        self.update_report_actions()

    def export_analysis(self, format: str) -> None:
        if not self.app.allow_operation("analysis-export"):
            return
        view = self.report_view
        destination = self.app.configuration.input("analysis-export-path").strip()
        if view is None or not view.payload or not destination:
            self.app.notify(message_id("results.export.statistics_required"), severity="warning")
            return
        path = resolve_result_path(destination, self.app.project_dir)
        token = object()
        self.app.read_jobs.add(token)
        self.app.set_busy(True)
        self.app.run_worker(partial(self._export_analysis_worker, token, view.payload, path, format),
                            name="export_analysis", thread=True, group="analysis-export", exit_on_error=False)

    def _export_analysis_worker(self, token: object, payload: dict, path: Path, format: str) -> None:
        from acprof.analysis.export import export_analysis
        from acprof.host.run_state import offline_analysis_lock
        try:
            with offline_analysis_lock():
                export_analysis(payload, path, format=format)
            status = message_id("results.export.completed", path)
        except Exception as exc:
            status = message_id("results.export.failed", error_message(exc))
        self.app.process.safe_process_callback(self.analysis_export_finished, token, status)

    def analysis_export_finished(self, token: object, status: str) -> None:
        self.app.read_jobs.discard(token)
        self.app.set_busy(self.app.is_busy())
        self.app.localization.set_text(self.app.query_one("#analysis-export-status", Static), status)
