"""Read-only result/report workflows, cancellable readers, and presentation.

The parent App owns background-reader tokens, process mutual exclusion and
lifecycle; read-only work is never initiated from a measurement window.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from textual.timer import Timer
from textual.widgets import Button, Static, TabbedContent
from textual.worker import Worker

from acprof.experiments.configuration import RunConfigError
from acprof.tui.commands import PendingLaunch, prepare_plot, resolve_result_path
from acprof.tui.experiments.catalog import ExperimentSourceDisplay
from acprof.tui.i18n import error_message, message_id
from acprof.tui.results.analysis_controller import AnalysisReportController
from acprof.tui.results.data_status import data_status_text, inspect_experiment_data
from acprof.tui.results.html_controller import HtmlReportController
from acprof.tui.results.offline_reads import OverlappingOfflineReads
from acprof.tui.results.overview import overview_view, read_experiment_overview, selection_path
from acprof.tui.results.overview_panel import ExperimentOverviewPanel
from acprof.tui.results.report_actions import report_action_state
from acprof.tui.results.reports import ReportView, read_report
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.structured import StatusBanner

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class ResultController:
    """Own results state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.html = HtmlReportController(app)
        self.offline_reads = OverlappingOfflineReads()
        self.overview_request: Event | None = None
        self.data_request: Event | None = None
        self.source_path: Path | None = None
        self.overview = None
        self.data_status = None
        self.data_error = ""
        self.data_cancelled = False
        self.selection_timer: Timer | None = None
        self.analysis = AnalysisReportController(self)
        self.source_display: ExperimentSourceDisplay | None = None
        self.unavailable_remembered_source = ""
        self.formal_ok_rows: int | None = None

    @property
    def report_request(self) -> Event | None:
        return self.analysis.report_request

    @report_request.setter
    def report_request(self, value: Event | None) -> None:
        self.analysis.report_request = value

    @property
    def report_view(self) -> ReportView | None:
        return self.analysis.report_view

    @report_view.setter
    def report_view(self, value: ReportView | None) -> None:
        self.analysis.report_view = value

    @property
    def analysis_view(self) -> ReportView | None:
        return self.analysis.analysis_view

    @property
    def analysis_source(self) -> Path:
        return self.analysis.analysis_source

    @property
    def analysis_html_path(self) -> str:
        return self.analysis.analysis_html_path

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
        self.app.query_one(ExperimentOverviewPanel).show_hint(source, level=level, has_source=self.source_path is not None)

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
        # Textual stops the app before pruning widgets and calling on_unmount.
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
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
        self.sync_result_actions_for_view(views.active)

    def sync_result_actions_for_view(self, section: str) -> None:
        if not self.app.is_running or self.app.ui_closing:
            return
        backfill = section == "backfill-view"
        self.app.query_one("#result-actions").set_class(section in {"overview-view", "statistics-view"}, "source-actions")
        self.app.query_one("#open-result-profiling", Button).display = backfill
        self.app.query_one("#report-calculate", Button).display = not backfill
        self.app.query_one("#plot-results", Button).display = not backfill

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
        self.html.set_status(message_id("results.overview.html_unverified" if source else "results.html.select_experiment"),
                             level="neutral")
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
                read = read_experiment_overview if kind == "overview" else inspect_experiment_data
                result = read(path, cancelled=token.is_set)
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
        footer = self.app.query_one("#result-actions")
        footer.set_class(cancel_analysis.display, "cancellable")
        footer.set_class(self.source_path is None and not cancel_analysis.display, "no-source")
        for selector, enabled in (
            ("#analysis-export-csv", actions.analysis_export_csv_enabled),
            ("#analysis-export-json", actions.analysis_export_json_enabled),
            ("#refresh-experiment-state", bool(self.source_path) and self.app.operation_state().allows("summary")),
            ("#plot-results", actions.plot_enabled),
            ("#report-calculate", actions.html_generate_enabled),
            ("#report-window-stats", actions.calculate_enabled),
            ("#report-open", actions.report_open_enabled),
        ):
            self.app.query_one(selector, Button).disabled = not enabled
        self.html.apply_action_state(actions)

    def clear_report(self, status: str, *, phase: str = "empty", empty_hint=None) -> None:
        self.analysis.clear_report(status, phase=phase, empty_hint=empty_hint)

    def report_source_changed(self) -> None:
        self.analysis.report_source_changed()

    def report_json_changed(self) -> None:
        self.analysis.report_json_changed()

    def open_report_button(self) -> None:
        self.open_report()

    def open_report(self, path: str | None = None) -> None:
        self.analysis.open_report(path)

    def execute_report_read(self, path: Path, token: Event) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_report_read_worker, path, token),
            name="execute_report_read", thread=True, group="report", exit_on_error=False,
        )

    def _execute_report_read_worker(self, path: Path, token: Event) -> None:
        try:
            with self.offline_reads.read():
                view, error = read_report(path, cancelled=token.is_set), ""
        except Exception as exc:
            view, error = None, error_message(exc)
        self.app.process.safe_process_callback(self.show_report, view, error, token)

    def show_report(self, view: ReportView | None, error: str, token: object) -> None:
        self.analysis.show_report(view, error, token)

    def render_report_view(self) -> None:
        self.analysis.render_report_view()

    def report_row_highlighted(self, event) -> None:
        self.show_report_row(event.cursor_row)

    def show_report_row(self, index: int) -> None:
        self.analysis.show_report_row(index)

    def calculate_report_button(self) -> None:
        self.html.launch()

    def compare_experiments_button(self) -> None:
        self.analysis.compare_experiments_button()

    def launch_stats(self, path: str | None = None) -> None:
        self.analysis.launch_stats(path)

    def selection_identity(self) -> tuple[str, ...]:
        return self.analysis.selection_identity()

    def begin_analysis(self, pending: PendingLaunch) -> None:
        self.analysis.begin_analysis(pending)

    def accept_analysis(self, view: ReportView) -> None:
        self.analysis.accept_analysis(view)

    def finish_analysis(self, *, error: str, cancelled: bool) -> None:
        self.analysis.finish_analysis(error=error, cancelled=cancelled)

    def export_analysis(self, format: str) -> None:
        self.analysis.export_analysis(format)

    def analysis_export_finished(self, token: object, status: str) -> None:
        self.analysis.analysis_export_finished(token, status)
