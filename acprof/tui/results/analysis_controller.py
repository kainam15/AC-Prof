"""Statistics, JSON report inspection, comparison, and analysis export coordination."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import Collapsible, ContentSwitcher, DataTable, Static

from acprof.experiments.configuration import RunConfigError
from acprof.messages import join_messages
from acprof.tui.commands import PendingLaunch, prepare_comparison, prepare_stats, resolve_result_path
from acprof.tui.i18n import error_message, message_id
from acprof.tui.results.reports import ReportView
from acprof.tui.ui.input import BarCursorInput as Input

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui
    from acprof.tui.results.controller import ResultController


class AnalysisReportController:
    """Manage JSON statistics reports, stats/compare launches, and exports."""

    def __init__(self, owner: ResultController) -> None:
        self.owner = owner
        self.report_request: Event | None = None
        self.report_path: Path | None = None
        self.report_view: ReportView | None = None
        self.analysis_view: ReportView | None = None
        self.analysis_source = owner.app.project_dir
        self.analysis_selection: tuple[str, ...] = ()
        self.analysis_kind = "stats"
        self.analysis_html_path = ""

    @property
    def app(self) -> AcprofTui:
        return self.owner.app

    def clear_report(self, status: str, *, phase: str = "empty", empty_hint=None) -> None:
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
            self.app.localization.set_text(
                self.app.query_one("#report-empty", Static),
                empty_hint or message_id("results.statistics.empty_hint"),
            )
        self.app.query_one("#report-content", ContentSwitcher).current = target
        self.owner.update_report_actions()

    def report_source_changed(self) -> None:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        self.owner.refresh_source_display()
        if self.report_request is not None:
            self.report_request.set()
        self.report_request = None
        with self.app.prevent(Input.Changed):
            self.app.query_one("#report-json-path", Input).value = ""
            self.app.query_one("#html-report-path", Input).value = ""
        self.app.localization.set_text(self.app.query_one("#analysis-export-status", Static), "")
        source_hint = (
            message_id("results.statistics.source_hint")
            if self.owner.source_display and self.owner.source_display.can_read
            else ""
        )
        self.clear_report(source_hint, empty_hint=self.owner.statistics_empty_hint())
        self.owner.update_report_actions()

    def report_json_changed(self) -> None:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        source = self.app.configuration.input("report-json-path").strip()
        if (self.report_request is not None
                and resolve_result_path(source, self.app.project_dir) == self.report_path):
            return
        if self.report_request is not None:
            self.report_request.set()
        self.report_request = None
        self.clear_report(
            message_id("results.import.path_changed") if source else message_id("results.statistics.source_hint")
        )
        self.owner.update_report_actions()

    def open_report(self, path: str | None = None) -> None:
        if not self.app.allow_operation("report"):
            return
        self.owner.show_result_section("statistics-view")
        self.app.query_one("#analysis-details", Collapsible).collapsed = False
        self.app.query_one("#report-panel").scroll_visible(animate=False)
        source = path if path is not None else self.app.configuration.input("report-json-path")
        if not source.strip():
            self.clear_report(message_id("results.import.json_required"), phase="error")
            self.owner.show_result_section("exports-view")
            self.app.query_one("#report-file-options", Collapsible).collapsed = False
            return
        report_path = resolve_result_path(source, self.app.project_dir)
        if report_path.suffix.lower() != ".json":
            self.clear_report("请选择 JSON 报告；实验目录或 CSV 请点击“计算统计”。", phase="error")
            self.owner.show_result_section("exports-view")
            self.app.query_one("#report-file-options", Collapsible).collapsed = False
            return
        with self.app.prevent(Input.Changed):
            self.app.query_one("#report-json-path", Input).value = str(report_path)
        self.owner.update_report_actions()
        self.clear_report(message_id("results.import.reading"), phase="loading")
        if self.report_request is not None:
            self.report_request.set()
        token = self.report_request = Event()
        self.report_path = report_path
        self.app.read_jobs.add(token)
        self.app.set_busy(True)
        self.owner.execute_report_read(report_path, token)

    def show_report(self, view: ReportView | None, error: str, token: object) -> None:
        self.app.read_jobs.discard(token)
        if not self.app.is_running or self.app.ui_closing:
            return
        if (token is not self.report_request
                or resolve_result_path(self.app.configuration.input("report-json-path"), self.app.project_dir) != self.report_path):
            self.app.set_busy(self.app.is_busy())
            return
        self.app.set_busy(self.app.is_busy())
        if view is None:
            self.clear_report(message_id("results.import.read_failed", error), phase="error")
            return
        self.report_view = replace(view, note=join_messages("\n", (
            message_id("results.import.historical_notice", view.source), view.note)))
        self.render_report_view()
        self.owner.update_report_actions()

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

    def show_report_row(self, index: int) -> None:
        view = self.report_view
        if view is None:
            return
        detail = view.rows[index].detail if 0 <= index < len(view.rows) else message_id("results.statistics.no_successful_windows")
        self.app.localization.set_text(self.app.query_one("#report-detail", Static), join_messages("\n", (detail, view.note)))

    def compare_experiments_button(self) -> None:
        if not self.app.allow_operation("compare"):
            return
        try:
            pending = prepare_comparison(
                self.app.configuration.input("comparison-left"),
                self.app.configuration.input("comparison-right"),
                baseline=self.app.configuration.select("comparison-baseline"),
                purpose=self.app.configuration.select("comparison-purpose"),
                project_dir=self.app.project_dir,
                python_executable=self.app.python_executable,
                html_report=True,
            )
        except (RunConfigError, ValueError, OSError) as exc:
            self.app.localization.set_text(self.app.query_one("#comparison-status", Static), error_message(exc))
            self.app.notify(error_message(exc), severity="error")
            return
        self.app.launch(pending)

    def launch_stats(self, path: str | None = None) -> None:
        if not self.app.allow_operation("stats"):
            return
        self.owner.show_result_section("statistics-view")
        self.app.query_one("#analysis-details", Collapsible).collapsed = False
        self.app.query_one("#report-panel").scroll_visible(animate=False)
        source = path if path is not None else self.app.configuration.input("experiment-source")
        if not source.strip():
            self.clear_report(message_id("results.statistics.input_required"), phase="error")
            return
        try:
            pending = prepare_stats(
                source, project_dir=self.app.project_dir, python_executable=self.app.python_executable,
                analysis_args=(),
            )
        except (OSError, ValueError) as exc:
            self.clear_report(error_message(exc), phase="error")
            return
        if pending.result_csv != self.app.configuration.input("experiment-source"):
            self.owner.select_result(pending.result_csv, load=False)
        self.app.launch(pending)

    def selection_identity(self) -> tuple[str, ...]:
        return tuple(self.app.configuration.input(name) for name in (
            "experiment-source", "comparison-left", "comparison-right",
        )) + (
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
            self.owner.show_result_section(section)
            if self.analysis_kind == "compare":
                self.app.localization.set_text(self.app.query_one("#comparison-status", Static), status)
        else:
            self.owner.show_result_section("statistics-view")
            self.report_view = view
            self.render_report_view()
            html_path = view.payload.get("html_report_path")
            if html_path and html_path == self.analysis_html_path:
                self.app.query_one("#html-report-path", Input).value = html_path
                self.app.configuration.remember_last_used(html_report=html_path)
                self.owner.html.set_status(message_id("results.comparison.created_notice"), level="success")
                self.app.query_one("#analysis-details", Collapsible).collapsed = True
                self.app.query_one("#html-report-panel").scroll_visible(animate=False)
        self.owner.update_report_actions()

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
        self.app.run_worker(
            partial(self._export_analysis_worker, token, view.payload, path, format),
            name="export_analysis", thread=True, group="analysis-export", exit_on_error=False,
        )

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
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        self.app.set_busy(self.app.is_busy())
        self.app.localization.set_text(self.app.query_one("#analysis-export-status", Static), status)
