"""Read-only result/report workflows, cancellable readers, and presentation.

The parent App owns background-reader tokens, process mutual exclusion and
lifecycle; read-only work is never initiated from a measurement window.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import DataTable, Static
from textual.worker import Worker

from acprof.experiment import RunConfigError
from acprof.messages import join_messages
from acprof.tui.commands import (
    prepare_comparison,
    prepare_plot,
    prepare_stats,
    resolve_result_path,
)
from acprof.tui.diagnostics import summarize_result_csv
from acprof.tui.i18n import error_message, message_id
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.presentation import CALCULATING
from acprof.tui.reports import ReportView, read_report

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

class ResultController:
    """Own results state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.summary_request: Event | None = None
        self.summary_path: Path | None = None
        self.report_request: Event | None = None
        self.report_path: Path | None = None
        self.report_view: ReportView | None = None
        self.stats_report_path: Path | None = None
        self.comparison_report_path: Path | None = None
        self.stats_report_reused = False

    def clear_log_button(self) -> None:
        self.clear_log()

    def clear_log(self) -> None:
        self.app.logs.clear()

    def result_source_changed(self) -> None:
        if self.app.form_ready:
            if (self.summary_request is not None and
                    resolve_result_path(self.app.configuration.input("result-csv"), self.app.project_dir) == self.summary_path):
                return
            if self.summary_request is not None:
                self.summary_request.set()
            self.summary_request = None
            self.app.localization.set_text(self.app.query_one("#result-summary", Static), "结果选择已更改，请读取摘要。")

    def summarize_results_button(self) -> None:
        self.update_result_summary(self.app.configuration.input("result-csv"))

    def update_result_summary(self, result_csv: str, *, notify: bool = True) -> None:
        if not self.app.allow_operation("summary"):
            return
        if not result_csv:
            if notify:
                self.app.notify("请填写结果 CSV 路径", severity="warning")
            return
        path = resolve_result_path(result_csv, self.app.project_dir)
        with self.app.prevent(Input.Changed):
            self.app.query_one("#result-csv", Input).value = str(path)
        if self.summary_request is not None:
            self.summary_request.set()
        token = self.summary_request = Event()
        self.summary_path = path
        self.app.read_jobs.add(token)
        self.app.localization.set_text(self.app.query_one("#result-summary", Static), message_id('ui.0199', CALCULATING))
        self.app.set_busy(True)
        self.execute_summary_read(path, token, notify)

    def execute_summary_read(self, path: Path, token: Event, notify: bool) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_summary_read_worker, path, token, notify),
            name="execute_summary_read", thread=True, group='summary', exit_on_error=False,
        )

    def _execute_summary_read_worker(self, path: Path, token: Event, notify: bool) -> None:
        try:
            summary, error = summarize_result_csv(path, cancelled=token.is_set), ""
        except Exception as exc:
            summary, error = None, error_message(exc)
        self.app.process.safe_process_callback(self.show_result_summary, path, token, summary, error, notify)

    def show_result_summary(self, path, token, summary, error: str, notify: bool) -> None:
        self.app.read_jobs.discard(token)
        if not self.app.is_running or self.app.ui_closing:
            return
        self.app.set_busy(self.app.is_busy())
        if (token is not self.summary_request or
                resolve_result_path(self.app.configuration.input("result-csv"), self.app.project_dir) != path):
            return
        if summary is None:
            self.app.localization.set_text(self.app.query_one("#result-summary", Static), message_id('ui.0811', error))
            if notify:
                self.app.notify(error, severity="error")
            return
        from acprof.tui.diagnostics import result_summary_text
        self.app.configuration.remember_last_used(result_csv=str(path))
        self.app.localization.set_text(self.app.query_one("#result-summary", Static), result_summary_text(summary, path))
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

    def clear_report(self, status: str) -> None:
        self.report_view = None
        self.app.query_one("#report-table", DataTable).clear(columns=True)
        self.app.localization.set_text(self.app.query_one("#report-status", Static), status)
        self.app.localization.set_text(self.app.query_one("#report-detail", Static), "表格可滚动；选择一行查看口径与数据来源。")

    def report_source_changed(self) -> None:
        if self.app.form_ready:
            if (self.report_request is not None and
                    resolve_result_path(self.app.configuration.input("report-source"), self.app.project_dir) == self.report_path):
                return
            if self.report_request is not None:
                self.report_request.set()
            self.report_request = None
            self.clear_report("CSV / 目录：计算统计；JSON：查看报告。采集结束后操作。")

    def open_report_button(self) -> None:
        self.open_report()

    def open_report(self, path: str | None = None) -> None:
        if not self.app.allow_operation("report"):
            return
        self.app.activate_tab("reports-tab")
        source = path if path is not None else self.app.configuration.input("report-source")
        if not source.strip():
            self.clear_report("请填写报告 JSON 路径，或使用当前结果计算统计。")
            return
        report_path = resolve_result_path(source, self.app.project_dir)
        if report_path.suffix.lower() != ".json":
            self.clear_report("请选择 JSON 报告；实验目录或 CSV 请点击“计算统计”。")
            return
        with self.app.prevent(Input.Changed):
            self.app.query_one("#report-source", Input).value = str(report_path)
        self.clear_report(message_id('ui.0577', CALCULATING))
        self.app.screen.set_focus(self.app.query_one("#report-table"), scroll_visible=False)
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
            view, error = read_report(path, cancelled=token.is_set), ""
        except Exception as exc:
            view, error = None, error_message(exc)
        self.app.process.safe_process_callback(self.show_report, view, error, token)

    def show_report(self, view: ReportView | None, error: str, token: object) -> None:
        self.app.read_jobs.discard(token)
        if not self.app.is_running or self.app.ui_closing:
            return
        if (token is not self.report_request or
                resolve_result_path(self.app.configuration.input("report-source"), self.app.project_dir) != self.report_path):
            self.app.set_busy(self.app.is_busy())
            return
        self.app.set_busy(self.app.is_busy())
        if view is None:
            self.clear_report(message_id('ui.0578', error))
            return
        self.report_view = view
        self.render_report_view()

    def render_report_view(self) -> None:
        view = self.report_view
        if view is None:
            return
        table = self.app.query_one("#report-table", DataTable)
        cursor, scroll = table.cursor_coordinate, table.scroll_offset
        table.clear(columns=True)
        for column in view.columns:
            table.add_column(Text(self.app.localization.tr(column)), key=str(column))
        for index, row in enumerate(view.rows):
            table.add_row(*(Text(self.app.localization.tr(cell)) for cell in row.cells), key=str(index))
        table.move_cursor(row=min(cursor.row, max(0, len(view.rows) - 1)), column=cursor.column, scroll=False)
        table.scroll_to(scroll.x, scroll.y, animate=False, force=True)
        self.app.localization.set_text(self.app.query_one("#report-status", Static), view.title)
        self.show_report_row(table.cursor_row)

    def report_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.show_report_row(event.cursor_row)

    def show_report_row(self, index: int) -> None:
        view = self.report_view
        if view is None:
            return
        detail = view.rows[index].detail if 0 <= index < len(view.rows) else message_id('ui.0579')
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
        self.app.activate_tab("reports-tab")
        source = path if path is not None else self.app.configuration.input("report-source")
        if not source.strip():
            self.clear_report("请填写实验目录或结果 CSV 路径。")
            return
        try:
            pending = prepare_stats(source, project_dir=self.app.project_dir, python_executable=self.app.python_executable)
        except (OSError, ValueError) as exc:
            self.clear_report(str(exc))
            return
        self.stats_report_path = None
        self.stats_report_reused = False
        self.clear_report(message_id('ui.0582', CALCULATING))
        self.app.launch(pending)
