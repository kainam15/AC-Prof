"""Read-only result/report workflows, cancellable readers, and presentation.

The parent App owns background-reader tokens, process mutual exclusion and
lifecycle; read-only work is never initiated from a measurement window.
"""
from __future__ import annotations

from pathlib import Path
from threading import Event

from rich.text import Text
from textual import on, work
from textual.message_pump import MessagePump
from textual.widgets import Button, DataTable, Static

from acprof.experiment import RunConfigError
from acprof.messages import join_messages, message
from acprof.tui.commands import prepare_comparison, prepare_plot, prepare_stats, resolve_result_path
from acprof.tui.i18n import error_message
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog
from acprof.tui.presentation import CALCULATING
from acprof.tui.reports import ReportView


class ResultActions(MessagePump):
    @on(Button.Pressed, "#clear-log")
    def clear_log_button(self) -> None:
        self.action_clear_log()

    def action_clear_log(self) -> None:
        self.query_one("#run-log", SelectableLog).clear()

    @on(Input.Changed, "#result-csv")
    def result_source_changed(self) -> None:
        if self._form_ready:
            if (self._summary_request is not None and
                    resolve_result_path(self._input("result-csv"), self._project_dir) == self._summary_path):
                return
            if self._summary_request is not None:
                self._summary_request.set()
            self._summary_request = None
            self._set_text(self.query_one("#result-summary", Static), "结果选择已更改，请读取摘要。")

    @on(Button.Pressed, "#summarize-results")
    def summarize_results_button(self) -> None:
        self._update_result_summary(self._input("result-csv"))

    def _update_result_summary(self, result_csv: str, *, notify: bool = True) -> None:
        if not self._allow_operation("summary"):
            return
        if not result_csv:
            if notify:
                self.notify("请填写结果 CSV 路径", severity="warning")
            return
        path = resolve_result_path(result_csv, self._project_dir)
        with self.prevent(Input.Changed):
            self.query_one("#result-csv", Input).value = str(path)
        if self._summary_request is not None:
            self._summary_request.set()
        token = self._summary_request = Event()
        self._summary_path = path
        self._read_jobs.add(token)
        self._set_text(self.query_one("#result-summary", Static), message("{0} 正在读取结果摘要", CALCULATING))
        self._set_busy(True)
        self._execute_summary_read(path, token, notify)

    @work(thread=True, group="summary", exit_on_error=False)
    def _execute_summary_read(self, path: Path, token: Event, notify: bool) -> None:
        try:
            summary, error = self._summarize_result_csv(path, cancelled=token.is_set), ""
        except Exception as exc:
            summary, error = None, error_message(exc)
        self._safe_process_callback(self._show_result_summary, path, token, summary, error, notify)

    def _show_result_summary(self, path, token, summary, error: str, notify: bool) -> None:
        self._read_jobs.discard(token)
        if not self.is_running or self._ui_closing:
            return
        self._set_busy(self._is_busy())
        if (token is not self._summary_request or
                resolve_result_path(self._input("result-csv"), self._project_dir) != path):
            return
        if summary is None:
            self._set_text(self.query_one("#result-summary", Static), message("无法读取结果：{0}", error))
            if notify:
                self.notify(error, severity="error")
            return
        from acprof.tui.diagnostics import result_summary_text
        self._remember_last_used(result_csv=str(path))
        self._set_text(self.query_one("#result-summary", Static), result_summary_text(summary, path))
        if notify:
            self.notify("结果摘要已更新", timeout=3)

    def _cancel_result_reads(self) -> None:
        if self._recovery_request is not None:
            self._recovery_request.set()
        self._recovery_after_check = None
        if self._summary_request is not None:
            self._summary_request.set()
        if self._report_request is not None:
            self._report_request.set()
        self._summary_request = self._report_request = None
        self._set_text(self.query_one("#result-summary", Static), "读取已取消；等待后台任务释放资源。")
        self._clear_report("读取已取消；等待后台任务释放资源。")
        self._set_busy(self._is_busy())

    @on(Button.Pressed, "#plot-results")
    def plot_results_button(self) -> None:
        self._launch_plot()

    def _launch_plot(self, path: str | None = None) -> None:
        if not self._allow_operation("plot"):
            return
        result_csv = path or self._input("result-csv")
        if not result_csv:
            self.notify("请填写结果 CSV 路径", severity="warning")
            return
        try:
            pending = prepare_plot(result_csv, project_dir=self._project_dir, python_executable=self._python_executable)
        except RunConfigError as exc:
            self.notify(str(exc), severity="error")
            return
        self._launch(pending)

    def _clear_report(self, status: str) -> None:
        self._report_view = None
        self.query_one("#report-table", DataTable).clear(columns=True)
        self._set_text(self.query_one("#report-status", Static), status)
        self._set_text(self.query_one("#report-detail", Static), "表格可滚动；选择一行查看口径与数据来源。")

    @on(Input.Changed, "#report-source")
    def report_source_changed(self) -> None:
        if self._form_ready:
            if (self._report_request is not None and
                    resolve_result_path(self._input("report-source"), self._project_dir) == self._report_path):
                return
            if self._report_request is not None:
                self._report_request.set()
            self._report_request = None
            self._clear_report("CSV / 目录：计算统计；JSON：查看报告。采集结束后操作。")

    @on(Button.Pressed, "#report-open")
    def open_report_button(self) -> None:
        self._open_report()

    def _open_report(self, path: str | None = None) -> None:
        if not self._allow_operation("report"):
            return
        self._activate_tab("reports-tab")
        source = path if path is not None else self._input("report-source")
        if not source.strip():
            self._clear_report("请填写报告 JSON 路径，或使用当前结果计算统计。")
            return
        report_path = resolve_result_path(source, self._project_dir)
        if report_path.suffix.lower() != ".json":
            self._clear_report("请选择 JSON 报告；实验目录或 CSV 请点击“计算统计”。")
            return
        with self.prevent(Input.Changed):
            self.query_one("#report-source", Input).value = str(report_path)
        self._clear_report(message("{0} 正在读取报告", CALCULATING))
        self.screen.set_focus(self.query_one("#report-table"), scroll_visible=False)
        if self._report_request is not None:
            self._report_request.set()
        token = self._report_request = Event()
        self._report_path = report_path
        self._read_jobs.add(token)
        self._set_busy(True)
        self._execute_report_read(report_path, token)

    @work(thread=True, group="report", exit_on_error=False)
    def _execute_report_read(self, path: Path, token: Event) -> None:
        try:
            view, error = self._read_report(path, cancelled=token.is_set), ""
        except Exception as exc:
            view, error = None, error_message(exc)
        self._safe_process_callback(self._show_report, view, error, token)

    def _show_report(self, view: ReportView | None, error: str, token: object) -> None:
        self._read_jobs.discard(token)
        if not self.is_running or self._ui_closing:
            return
        if (token is not self._report_request or
                resolve_result_path(self._input("report-source"), self._project_dir) != self._report_path):
            self._set_busy(self._is_busy())
            return
        self._set_busy(self._is_busy())
        if view is None:
            self._clear_report(message("报告读取失败：{0}", error))
            return
        self._report_view = view
        self._render_report_view()

    def _render_report_view(self) -> None:
        view = self._report_view
        if view is None:
            return
        table = self.query_one("#report-table", DataTable)
        cursor, scroll = table.cursor_coordinate, table.scroll_offset
        table.clear(columns=True)
        for column in view.columns:
            table.add_column(Text(self.tr(column)), key=str(column))
        for index, row in enumerate(view.rows):
            table.add_row(*(Text(self.tr(cell)) for cell in row.cells), key=str(index))
        table.move_cursor(row=min(cursor.row, max(0, len(view.rows) - 1)), column=cursor.column, scroll=False)
        table.scroll_to(scroll.x, scroll.y, animate=False, force=True)
        self._set_text(self.query_one("#report-status", Static), view.title)
        self._show_report_row(table.cursor_row)

    @on(DataTable.RowHighlighted, "#report-table")
    def report_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._show_report_row(event.cursor_row)

    def _show_report_row(self, index: int) -> None:
        view = self._report_view
        if view is None:
            return
        detail = view.rows[index].detail if 0 <= index < len(view.rows) else message("没有可统计的正式成功窗口。")
        self._set_text(self.query_one("#report-detail", Static), join_messages("\n", (detail, view.note)))

    @on(Button.Pressed, "#report-calculate")
    def calculate_report_button(self) -> None:
        self._launch_stats()

    @on(Button.Pressed, "#report-compare")
    def compare_experiments_button(self) -> None:
        if not self._allow_operation("compare"):
            return
        try:
            pending = prepare_comparison(self._input("comparison-left"), self._input("comparison-right"),
                baseline=self._select("comparison-baseline"), purpose=self._select("comparison-purpose"),
                project_dir=self._project_dir, python_executable=self._python_executable)
        except (RunConfigError, ValueError, OSError) as exc:
            self.notify(error_message(exc), severity="error")
            return
        self._launch(pending)

    def _launch_stats(self, path: str | None = None) -> None:
        if not self._allow_operation("stats"):
            return
        self._activate_tab("reports-tab")
        source = path if path is not None else self._input("report-source")
        if not source.strip():
            self._clear_report("请填写实验目录或结果 CSV 路径。")
            return
        try:
            pending = prepare_stats(source, project_dir=self._project_dir, python_executable=self._python_executable)
        except (OSError, ValueError) as exc:
            self._clear_report(str(exc))
            return
        self._stats_report_path = None
        self._stats_report_reused = False
        self._clear_report(message("{0} 正在计算窗口统计，完成后自动显示报告。", CALCULATING))
        self._launch(pending)
