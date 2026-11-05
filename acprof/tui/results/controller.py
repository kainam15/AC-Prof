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
from textual.widgets import Button, Collapsible, ContentSwitcher, DataTable, Static, TabbedContent
from textual.worker import Worker

from acprof.experiments.configuration import RunConfigError
from acprof.results.layout import experiment_result_file, resolve_single_result_source
from acprof.messages import join_messages
from acprof.tui.commands import (
    PendingLaunch,
    prepare_comparison,
    prepare_plot,
    prepare_stats,
    resolve_result_path,
)
from acprof.tui.run.diagnostics import summarize_result_csv
from acprof.tui.experiments.catalog import ExperimentSourceDisplay, experiment_source_display
from acprof.tui.i18n import error_message, message_id
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.presentation import CALCULATING
from acprof.tui.results.html_controller import HtmlReportController
from acprof.tui.results.report_actions import report_action_state
from acprof.tui.results.reports import ReportView, read_report

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
        self.summary_request: Event | None = None
        self.summary_path: Path | None = None
        self.report_request: Event | None = None
        self.report_path: Path | None = None
        self.report_view: ReportView | None = None
        self.analysis_view: ReportView | None = None
        self.analysis_source = app.project_dir
        self.analysis_selection: tuple[str, ...] = ()
        self.source_display: ExperimentSourceDisplay | None = None
        self.unavailable_remembered_source = ""
        self.formal_ok_rows: int | None = None
        self.summary_read_failed = False

    def clear_log_button(self) -> None:
        self.clear_log()

    def clear_log(self) -> None:
        self.app.logs.clear()

    def restore_recent_result(self) -> None:
        """Keep unavailable recent paths in settings, not in the active selection."""
        source = self.app.configuration.input('result-csv').strip()
        if source and source == self.app.configuration.saved_settings.last_result_csv:
            path = resolve_result_path(source, self.app.project_dir)
            display = experiment_source_display(path)
            if not display.can_read:
                fallback = (experiment_result_file(path.parent)
                            if path.name == 'result_layers.json' and not path.exists() else path)
                if fallback != path and experiment_source_display(fallback).can_read:
                    self.app.configuration.remember_last_used(result_csv=str(fallback))
                else:
                    self.unavailable_remembered_source = source
                    with self.app.prevent(Input.Changed):
                        self.app.query_one('#result-csv', Input).value = ''
        # On mount, refresh labels only: resetting the report source would erase the
        # independently remembered HTML path before the user has selected anything.
        self.update_report_source_label()
        self.app.localization.set_text(self.app.query_one('#report-empty', Static), self.statistics_empty_hint())
        self.app.localization.set_text(self.app.query_one('#result-summary', Static), self.source_summary_hint())

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
        if self.app.form_ready:
            # A queued Changed event can arrive after a relative path was
            # normalized and its reader started. Do not reset its verified state.
            if (not force and self.summary_request is not None
                    and resolve_result_path(self.app.configuration.input("result-csv"), self.app.project_dir) == self.summary_path):
                return
            self.formal_ok_rows = None
            self.summary_read_failed = False
            # Initial Input.Changed may arrive after restoring a missing recent path.
            # A deliberate picker selection (including Clear) supersedes the stale
            # notice; a delayed empty startup event does not.
            if force or self.app.configuration.input('result-csv').strip():
                self.unavailable_remembered_source = ''
            if self.summary_request is not None:
                self.summary_request.set()
            self.summary_request = None
            self.app.profiles.reset_overview()
            self.report_source_changed()
            self.app.localization.set_text(self.app.query_one('#result-summary', Static), self.source_summary_hint())

    def select_result(self, source: str, *, summarize: bool = True) -> None:
        """One selection for all consumers; picker selection loads only the light summary."""
        with self.app.prevent(Input.Changed):
            self.app.query_one("#result-csv", Input).value = source
        self.result_source_changed(force=True)
        if summarize and source:
            self.update_result_summary(source, notify=False)

    def show_result_section(self, section: str) -> None:
        self.app.activate_tab("results-tab")
        self.app.query_one("#result-views", TabbedContent).active = section

    def summarize_results_button(self) -> None:
        self.show_result_section("overview-view")
        self.update_result_summary(self.app.configuration.input("result-csv"))

    def update_result_summary(self, result_csv: str, *, notify: bool = True) -> None:
        if not self.app.allow_operation("summary"):
            return
        if not result_csv:
            if notify:
                self.app.notify("请填写结果 CSV 路径", severity="warning")
            return
        path = resolve_result_path(result_csv, self.app.project_dir)
        if path.is_dir():
            try:
                path = resolve_single_result_source(path)
            except (OSError, ValueError):
                pass  # The reader reports an invalid/ambiguous directory.
            if path.is_dir():
                path = experiment_result_file(path)
        if self.app.configuration.input("result-csv") != str(path):
            self.select_result(str(path), summarize=False)
        if self.summary_request is not None:
            self.summary_request.set()
        token = self.summary_request = Event()
        self.summary_path = path
        self.app.profiles.reset_overview()
        self.app.read_jobs.add(token)
        self.app.localization.set_text(self.app.query_one("#result-summary", Static), message_id('results.read.loading_summary', CALCULATING))
        self.app.set_busy(True)
        self.execute_summary_read(path, token, notify)

    def execute_summary_read(self, path: Path, token: Event, notify: bool) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_summary_read_worker, path, token, notify),
            name="execute_summary_read", thread=True, group='summary', exit_on_error=False,
        )

    def _execute_summary_read_worker(self, path: Path, token: Event, notify: bool) -> None:
        from concurrent.futures import CancelledError

        try:
            with self.offline_reads.read():
                if token.is_set():
                    raise CancelledError()
                summary, error = summarize_result_csv(path, cancelled=token.is_set), ""
        except Exception as exc:
            summary, error = None, error_message(exc)
        self.app.process.safe_process_callback(
            self.show_result_summary, path, token, summary, error, notify,
            summary.profiler_overview if summary is not None else None,
            summary.profiler_error if summary is not None else "")

    def show_result_summary(
        self, path, token, summary, error: str, notify: bool, overview=None, profile_error: str = "",
    ) -> None:
        self.app.read_jobs.discard(token)
        if not self.app.is_running or self.app.ui_closing:
            return
        self.app.set_busy(self.app.is_busy())
        if (token is not self.summary_request or
                resolve_result_path(self.app.configuration.input("result-csv"), self.app.project_dir) != path):
            return
        if summary is None:
            self.formal_ok_rows = 0
            self.summary_read_failed = True
            self.update_report_source_label()
            self.update_report_actions()
            self.app.profiles.reset_overview()
            self.app.localization.set_text(self.app.query_one("#result-summary", Static), message_id('results.summary.read_failed', error))
            if notify:
                self.app.notify(error, severity="error")
            return
        from acprof.tui.run.diagnostics import result_summary_text
        self.formal_ok_rows = summary.formal_ok_rows
        self.summary_read_failed = False
        self.update_report_source_label()
        self.update_report_actions()
        self.app.configuration.remember_last_used(result_csv=str(path))
        self.app.localization.set_text(self.app.query_one("#result-summary", Static), result_summary_text(summary, path))
        self.app.profiles.show_overview(overview, profile_error)
        if self.formal_ok_rows == 0 and self.app.query_one('#report-content', ContentSwitcher).current == 'report-empty':
            self.app.localization.set_text(self.app.query_one('#report-empty', Static),
                                           message_id('results.statistics.no_successful_windows'))
        if notify:
            self.app.notify("结果摘要已更新", timeout=3)

    def cancel_result_reads(self) -> None:
        if self.app.recovery.recovery_request is not None:
            self.app.recovery.recovery_request.set()
        self.app.recovery.recovery_after_check = None
        if self.summary_request is not None:
            self.summary_request.set()
        if self.report_request is not None:
            self.report_request.set()
        self.summary_request = self.report_request = None
        self.app.localization.set_text(self.app.query_one("#result-summary", Static), "读取已取消；等待后台任务释放资源。")
        self.app.profiles.reset_overview()
        self.clear_report("读取已取消；等待后台任务释放资源。")
        self.app.set_busy(self.app.is_busy())

    def plot_results_button(self) -> None:
        self.launch_plot()

    def launch_plot(self, path: str | None = None) -> None:
        if not self.app.allow_operation("plot"):
            return
        result_csv = path or self.app.configuration.input("result-csv")
        if not result_csv:
            self.app.notify("请填写结果 CSV 路径", severity="warning")
            return
        try:
            pending = prepare_plot(result_csv, project_dir=self.app.project_dir, python_executable=self.app.python_executable)
        except RunConfigError as exc:
            self.app.notify(str(exc), severity="error")
            return
        self.app.launch(pending)

    def update_report_source_label(self) -> None:
        source = self.app.configuration.input('result-csv').strip()
        label = self.app.query_one('#report-experiment', Static)
        self.source_display = (experiment_source_display(resolve_result_path(source, self.app.project_dir))
                               if source else None)
        display = self.source_display
        if display is None:
            heading = (message_id('results.selection.previous_unavailable')
                       if self.unavailable_remembered_source else message_id('results.selection.no_results'))
        elif display.availability == 'selected':
            if self.summary_read_failed:
                heading = message_id('results.selection.read_failed')
            elif self.formal_ok_rows == 0:
                heading = (message_id('results.selection.no_valid_windows', display.title) if display.title
                           else message_id('results.selection.unnamed_no_data'))
            elif self.formal_ok_rows is not None and display.title:
                if display.recorded_status == 'complete':
                    heading = message_id('results.selection.selected_complete_windows', display.title, self.formal_ok_rows)
                elif display.recorded_status:
                    heading = message_id('results.selection.selected_incomplete_windows', display.title, self.formal_ok_rows)
                else:
                    heading = message_id('results.selection.selected_windows', display.title, self.formal_ok_rows)
            elif not display.title:
                heading = message_id('results.selection.unnamed_selected')
            elif display.recorded_status == 'complete':
                heading = message_id('results.selection.selected_complete', display.title)
            elif display.recorded_status:
                heading = message_id('results.selection.selected_incomplete', display.title)
            else:
                heading = message_id('results.selection.selected', display.title)
        elif display.availability == 'no_data':
            heading = (message_id('results.selection.no_data', display.title) if display.title
                       else message_id('results.selection.unnamed_no_data'))
        elif display.availability == 'missing':
            heading = message_id('results.selection.path_missing')
        else:
            heading = message_id('results.selection.unsupported_source')
        self.app.localization.set_text(label, heading)
        tooltip_source = source or self.unavailable_remembered_source
        label.tooltip = (f'{tooltip_source}\nRun ID: {display.run_id}'
                         if display is not None and display.run_id else tooltip_source or None)

    def update_report_actions(self) -> None:
        """Apply pure action rules to the existing widgets and operation lock."""
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        from acprof.platform import can_open_local_browser

        actions = report_action_state(
            self.app.operation_state(),
            source_ready=bool(self.source_display and self.source_display.can_read),
            formal_ok_rows=self.formal_ok_rows,
            analysis_payload=self.report_view.payload if self.report_view is not None else {},
            report_json_path=self.app.configuration.input("report-json-path"),
            html_path=self.app.configuration.input("html-report-path"),
            execution_kind=self.app.execution.session.kind,
            browser_available=can_open_local_browser(),
        )
        cancel_analysis = self.app.query_one("#results-cancel", Button)
        cancel_analysis.display = actions.cancel_analysis_visible
        cancel_analysis.disabled = not actions.cancel_analysis_enabled
        for format, enabled in (
            ("csv", actions.analysis_export_csv_enabled),
            ("json", actions.analysis_export_json_enabled),
        ):
            self.app.query_one("#analysis-export-" + format, Button).disabled = not enabled
        for selector, enabled in (
            ("#summarize-results", actions.summary_enabled),
            ("#plot-results", actions.plot_enabled),
            ("#report-calculate", actions.calculate_enabled),
            ("#html-report-generate", actions.html_generate_enabled),
            ("#report-open", actions.report_open_enabled),
            ("#html-report-copy", actions.html_copy_enabled),
            ("#html-export-toggle", actions.html_export_toggle_enabled),
            ("#html-report-export", actions.html_export_enabled),
        ):
            self.app.query_one(selector, Button).disabled = not enabled
        self.app.query_one("#html-report-file").display = actions.html_file_visible
        self.app.query_one("#html-report-filename", Static).update(actions.html_filename)
        location = self.app.query_one("#html-report-location", Static)
        location.update(actions.html_location)
        location.tooltip = actions.html_location or None
        if actions.hide_html_export_fields:
            self.app.query_one("#html-export-fields").display = False
        self.app.query_one("#html-remote-hint").display = actions.html_remote_hint_visible
        open_html = self.app.query_one("#html-report-open", Button)
        open_html.display = actions.html_open_visible
        open_html.disabled = not actions.html_open_enabled
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
        self.update_report_source_label()
        if self.report_request is not None:
            self.report_request.set()
        self.report_request = None
        with self.app.prevent(Input.Changed):
            self.app.query_one("#report-json-path", Input).value = ""
            self.app.query_one("#html-report-path", Input).value = ""
        self.app.localization.set_text(self.app.query_one("#analysis-export-status", Static), "")
        # Existing HTML remains available even when the source CSV has been removed or damaged.
        # Do not conflate discovery of a historical report with permission to generate a new one.
        if (self.app.operation_state().allows('html-report')
                and self.app.configuration.input('result-csv').strip()):
            self.html.find()
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
        self.launch_stats()

    def compare_experiments_button(self) -> None:
        if not self.app.allow_operation("compare"):
            return
        try:
            pending = prepare_comparison(self.app.configuration.input("comparison-left"), self.app.configuration.input("comparison-right"),
                baseline=self.app.configuration.select("comparison-baseline"), purpose=self.app.configuration.select("comparison-purpose"),
                project_dir=self.app.project_dir, python_executable=self.app.python_executable)
        except (RunConfigError, ValueError, OSError) as exc:
            self.app.notify(error_message(exc), severity="error")
            return
        self.app.launch(pending)

    def launch_stats(self, path: str | None = None) -> None:
        if not self.app.allow_operation("stats"):
            return
        self.show_result_section("statistics-view")
        source = path if path is not None else self.app.configuration.input("result-csv")
        if not source.strip():
            self.clear_report(message_id("results.statistics.input_required"), phase="error")
            return
        try:
            pending = prepare_stats(source, project_dir=self.app.project_dir, python_executable=self.app.python_executable)
        except (OSError, ValueError) as exc:
            self.clear_report(error_message(exc), phase="error")
            return
        if pending.result_csv != self.app.configuration.input("result-csv"):
            self.select_result(pending.result_csv, summarize=False)
        self.app.launch(pending)

    def selection_identity(self) -> tuple[str, ...]:
        return tuple(self.app.configuration.input(name) for name in
                     ("result-csv", "comparison-left", "comparison-right")) + (
                         self.app.configuration.select("comparison-baseline"),
                         self.app.configuration.select("comparison-purpose"),
                     )

    def begin_analysis(self, pending: PendingLaunch) -> None:
        self.clear_report(message_id("results.statistics.calculating"), phase="loading")
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
        self.show_result_section("statistics-view")
        if cancelled or error or view is None:
            self.clear_report(error or message_id("results.statistics.cancelled" if cancelled else "results.statistics.invalid_analysis_output"), phase="error")
        else:
            self.report_view = view
            self.render_report_view()
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
