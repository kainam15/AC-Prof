"""Read-only result/report workflows, cancellable readers, and presentation.

The parent App owns background-reader tokens, process mutual exclusion and
lifecycle; read-only work is never initiated from a measurement window.
"""
from __future__ import annotations

import csv
import io
import json
from collections import deque
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Event, Lock
from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import Button, Collapsible, ContentSwitcher, DataTable, Static, TabbedContent
from textual.worker import Worker

from acprof.experiment import RunConfigError
from acprof.messages import join_messages
from acprof.tui.commands import (
    PendingLaunch,
    html_report_inputs,
    prepare_comparison,
    prepare_html_report,
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
        self.summary_lock = Lock()
        self.summary_request: Event | None = None
        self.summary_path: Path | None = None
        self.report_request: Event | None = None
        self.report_path: Path | None = None
        self.report_view: ReportView | None = None
        self.analysis_view: ReportView | None = None
        self.analysis_source = app.project_dir
        self.analysis_selection: tuple[str, ...] = ()
        self.html_output: Path | None = None
        self.html_log: deque[str] = deque(maxlen=20)
        self.html_summary = ""

    def clear_log_button(self) -> None:
        self.clear_log()

    def clear_log(self) -> None:
        self.app.logs.clear()

    def result_source_changed(self, *, force: bool = False) -> None:
        if self.app.form_ready:
            # A queued Changed event can arrive after a relative path was
            # normalized and its reader started. Do not cancel that same source.
            if (not force and self.summary_request is not None
                    and resolve_result_path(self.app.configuration.input("result-csv"), self.app.project_dir) == self.summary_path):
                return
            if self.summary_request is not None:
                self.summary_request.set()
            self.summary_request = None
            self.app.localization.set_text(self.app.query_one("#result-summary", Static), "结果选择已更改，请读取摘要。")
            self.app.profiles.reset_overview()
            self.report_source_changed()

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

        from acprof.host.run_state import offline_analysis_lock
        try:
            with self.summary_lock:
                if token.is_set():
                    raise CancelledError()
                with offline_analysis_lock():
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
            self.app.profiles.reset_overview()
            self.app.localization.set_text(self.app.query_one("#result-summary", Static), message_id('results.summary.read_failed', error))
            if notify:
                self.app.notify(error, severity="error")
            return
        from acprof.tui.diagnostics import result_summary_text
        self.app.configuration.remember_last_used(result_csv=str(path))
        self.app.localization.set_text(self.app.query_one("#result-summary", Static), result_summary_text(summary, path))
        self.app.profiles.show_overview(overview, profile_error)
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
        source = self.app.configuration.input("result-csv").strip()
        title: str
        if not source:
            title = message_id("results.selection.none")
        else:
            path = Path(source.rstrip("/"))
            title = path.parent.name if path.name == "result_layers.json" else path.name
        label = self.app.query_one("#report-experiment", Static)
        self.app.localization.set_text(label, message_id("results.selection.current_experiment", title))
        label.tooltip = source or None

    def update_report_actions(self) -> None:
        """Keep actions bound to the correct input and the existing operation lock."""
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        state = self.app.operation_state()
        source = self.app.configuration.input("result-csv").strip()
        cancel_analysis = self.app.query_one("#results-cancel", Button)
        cancel_analysis.display = self.app.process_kind in {"stats", "compare"}
        cancel_analysis.disabled = not state.allows("stop")
        payload = self.report_view.payload if self.report_view is not None else {}
        for format in ("csv", "json"):
            available = bool(payload) and (format == "json" or bool(payload.get("groups")))
            self.app.query_one("#analysis-export-" + format, Button).disabled = not (
                available and state.allows("analysis-export")
            )
        for selector, operation in (("#summarize-results", "summary"), ("#plot-results", "plot")):
            self.app.query_one(selector, Button).disabled = not (source and state.allows(operation))
        json_path = self.app.configuration.input("report-json-path").strip()
        html_path = self.app.configuration.input("html-report-path").strip()
        self.app.query_one("#report-calculate", Button).disabled = not (
            source and state.allows("stats")
        )
        self.app.query_one("#html-report-generate", Button).disabled = not (
            source and state.allows("html-report")
        )
        self.app.query_one("#report-open", Button).disabled = not (
            json_path and state.allows("report")
        )
        open_html = self.app.query_one("#html-report-open", Button)
        open_html.display = bool(html_path)
        open_html.disabled = not (html_path and state.allows("html-open"))
        export_html = self.app.query_one("#html-report-export", Button)
        export_html.disabled = not (html_path and state.allows("html-export"))
        cancel = self.app.query_one("#html-report-cancel", Button)
        cancel.display = self.app.process_kind == "html-report"
        cancel.disabled = not (
            self.app.process_kind == "html-report" and state.allows("stop")
        )

    def clear_report(self, status: str, *, phase: str = "empty") -> None:
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
            self.app.localization.set_text(self.app.query_one("#report-empty", Static), message_id("results.statistics.empty_hint"))
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
        if (self.app.operation_state().allows("html-report")
                and self.app.configuration.input("result-csv").strip()):
            self.find_html_report()
        self.clear_report(message_id("results.statistics.source_hint"))
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
        from acprof.host.run_state import offline_analysis_lock
        try:
            with offline_analysis_lock():
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

    def set_html_status(self, text: str, *, expand: bool = True) -> None:
        if expand:
            self.app.query_one("#html-report-panel", Collapsible).collapsed = False
        self.app.localization.set_text(self.app.query_one("#html-report-status", Static), text)
        self.update_report_actions()

    def html_source(self, *, include_extra: bool = False) -> str:
        # The unified selector is ONE literal path; extra HTML sources are explicit.
        sources = [self.app.configuration.input("result-csv").strip()]
        if not sources[0]:
            return ""
        if include_extra:
            extra = self.app.configuration.input("html-extra-sources").strip()
            if extra:
                sources.extend(next(csv.reader([extra], delimiter=";", skipinitialspace=True, strict=True)))
        output = io.StringIO()
        csv.writer(output, delimiter=";", lineterminator="").writerow(sources)
        return output.getvalue()

    def find_html_report(self) -> None:
        """Small directory lookup only; all result parsing stays in the CLI."""
        from acprof.tui.report_files import existing_html_report
        source = self.app.configuration.input("result-csv")
        # Viewing a JSON statistics report must not replace the HTML selection.
        if source.endswith(".json") and not source.endswith("result_layers.json"):
            return
        try:
            paths = html_report_inputs(self.html_source(), project_dir=self.app.project_dir)
            existing = existing_html_report(paths)
        except (OSError, ValueError) as exc:
            self.set_html_status(error_message(exc), expand=False)
            self.app.query_one("#html-report-path", Input).value = ""
            return
        self.app.query_one("#html-report-path", Input).value = str(existing) if existing else ""
        self.set_html_status(message_id('results.html.existing_report', existing) if existing else message_id('results.html.generation_note'), expand=False)

    def launch_html_report(self) -> None:
        if not self.app.allow_operation("html-report"):
            return
        self.show_result_section("exports-view")
        try:
            pending = prepare_html_report(self.html_source(include_extra=True),
                project_dir=self.app.project_dir, python_executable=self.app.python_executable)
        except (OSError, ValueError, csv.Error) as exc:
            self.set_html_status(error_message(exc))
            return
        self.app.launch(pending)

    def html_report_started(self, pending: PendingLaunch) -> None:
        self.html_output = Path(pending.report_path)
        self.html_log.clear()
        self.html_summary = ""
        self.set_html_status(message_id('results.html.generating', self.html_output))

    def consume_html_line(self, line: str) -> None:
        if not line.startswith("ACPROF_REPORT "):
            self.html_log.append(line)
            return
        receipt = json.loads(line.removeprefix("ACPROF_REPORT "))
        if (not isinstance(receipt, dict) or receipt.get("report_path") != str(self.html_output)
                or any(type(receipt.get(key)) is not int or receipt[key] < 0
                       for key in ("config_count", "valid_windows", "excluded_rows"))
                or any(not isinstance(receipt.get(key), list) or not receipt[key]
                       or any(not isinstance(value, str) for value in receipt[key])
                       for key in ("quality_status", "measurement_status"))):
            raise ValueError(message_id('results.html.invalid_receipt'))
        self.html_summary = message_id('results.html.receipt_summary', receipt["config_count"], receipt["valid_windows"],
            receipt["excluded_rows"], ", ".join(receipt["quality_status"]), ", ".join(receipt["measurement_status"]))
        if receipt["valid_windows"] == 0:
            self.html_summary = join_messages("\n", (self.html_summary, message_id('results.html.no_valid_windows')))

    def html_report_error(self, returncode: int, error: str) -> str:
        if error:
            return error
        if returncode:
            return "\n".join(self.html_log) or message_id('run.completion.failed_exit_code', returncode)
        if self.html_output is None or not self.html_output.is_file() or not self.html_summary:
            return message_id('results.html.output_missing')
        return ""

    def html_report_finished(self, *, error: str, cancelled: bool) -> None:
        output, self.html_output = self.html_output, None
        if cancelled:
            self.set_html_status(message_id('results.html.cancelled'))
        elif error:
            self.set_html_status(message_id('results.html.generation_failed', error))
        elif output is not None:
            self.app.query_one("#html-report-path", Input).value = str(output)
            self.app.configuration.remember_last_used(html_report=str(output))
            self.set_html_status(join_messages("\n", (message_id('results.html.generated', output), self.html_summary)))
        self.html_log.clear()
        self.html_summary = ""

    def open_html_report(self) -> None:
        from acprof.platform import can_open_local_browser
        from acprof.tui.report_files import require_html_file
        if not self.app.allow_operation("html-open"):
            return
        source = self.app.configuration.input("html-report-path")
        if not source:
            self.set_html_status(message_id('results.html.open_requires_report'))
            return
        path = resolve_result_path(source, self.app.project_dir).resolve()
        try:
            require_html_file(path)
        except (OSError, ValueError) as exc:
            self.set_html_status(error_message(exc))
            return
        if not can_open_local_browser():
            self.set_html_status(message_id('results.html.remote_open_instructions', path))
            return
        self.run_html_file_action("open", path)

    def export_html_report(self) -> None:
        if not self.app.allow_operation("html-export"):
            return
        source = self.app.configuration.input("html-report-path")
        destination = self.app.configuration.input("html-export-path")
        if not source or not destination:
            self.set_html_status(message_id('results.html.export_requires_paths'))
            return
        self.run_html_file_action("export", resolve_result_path(source, self.app.project_dir),
                                 resolve_result_path(destination, self.app.project_dir))

    def run_html_file_action(self, action: str, source: Path, destination: Path | None = None) -> None:
        token = object()
        self.app.read_jobs.add(token)
        self.app.set_busy(True)
        self.app.run_worker(partial(self._html_file_worker, token, action, source, destination),
                            name="html_file_action", thread=True, group="html-file", exit_on_error=False)

    def _html_file_worker(self, token: object, action: str, source: Path, destination: Path | None) -> None:
        from acprof.tui.report_files import export_html_file, open_html_file
        try:
            if action == "export" and destination is not None:
                export_html_file(source, destination)
                status = message_id('results.html.exported', destination)
            else:
                open_html_file(source)
                status = message_id('results.html.browser_requested', source)
        except Exception as exc:
            status = message_id('results.html.file_action_failed', error_message(exc), source)
        self.app.process.safe_process_callback(self.html_file_finished, token, status)

    def html_file_finished(self, token: object, status: str) -> None:
        self.app.read_jobs.discard(token)
        self.app.set_busy(self.app.is_busy())
        self.set_html_status(status)
