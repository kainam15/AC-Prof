"""HTML report state and file operations; the App owns workers and processes."""
from __future__ import annotations

import csv
import io
import json
from collections import deque
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Button, Input, Static

from acprof.messages import join_messages
from acprof.tui.commands import (
    PendingLaunch,
    html_report_inputs,
    prepare_html_report,
    resolve_result_path,
)
from acprof.tui.i18n import error_message, message_id

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class HtmlReportController:
    """Coordinate HTML reports without owning execution or worker lifecycle."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.output: Path | None = None
        self.log: deque[str] = deque(maxlen=20)
        self.summary = ""

    def set_status(self, text: str) -> None:
        self.app.localization.set_text(self.app.query_one("#html-report-status", Static), text)
        self.app.results.update_report_actions()

    def update_extra_button(self) -> None:
        has_extra = bool(self.app.configuration.input("html-extra-sources").strip())
        key = "results.html.edit_inputs" if has_extra else "results.html.add_inputs"
        self.app.localization.set_text(
            self.app.query_one("#html-extra-toggle", Button), message_id(key), attribute="label",
        )

    def toggle_extra_sources(self) -> None:
        field = self.app.query_one("#html-extra-sources", Input)
        field.display = not field.display
        if field.display:
            field.focus()

    def toggle_export_fields(self) -> None:
        field = self.app.query_one("#html-export-fields")
        field.display = not field.display
        if field.display:
            self.app.query_one("#html-export-path", Input).focus()

    def toggle_existing_path(self) -> None:
        field = self.app.query_one("#html-report-path", Input)
        field.display = not field.display
        if field.display:
            field.focus()

    def copy_path(self) -> None:
        if not self.app.allow_operation("html-open"):
            return
        source = self.app.configuration.input("html-report-path").strip()
        if source:
            path = resolve_result_path(source, self.app.project_dir).resolve()
            self.app.copy_to_clipboard(str(path))
            self.set_status(message_id("results.html.path_copied", path))

    def source(self, *, include_extra: bool = False) -> str:
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

    def find(self) -> None:
        """Small directory lookup only; all result parsing stays in the CLI."""
        from acprof.tui.results.report_files import existing_html_report

        source = self.app.configuration.input("result-csv")
        # Viewing a JSON statistics report must not replace the HTML selection.
        if source.endswith(".json") and not source.endswith("result_layers.json"):
            return
        try:
            paths = html_report_inputs(self.source(), project_dir=self.app.project_dir)
            existing = existing_html_report(paths)
        except (OSError, ValueError) as exc:
            self.set_status(error_message(exc))
            self.app.query_one("#html-report-path", Input).value = ""
            return
        self.app.query_one("#html-report-path", Input).value = str(existing) if existing else ""
        self.set_status(message_id("results.html.existing_report") if existing else message_id("results.html.generation_note"))

    def launch(self) -> None:
        if not self.app.allow_operation("html-report"):
            return
        self.app.results.show_result_section("statistics-view")
        try:
            pending = prepare_html_report(
                self.source(include_extra=True),
                project_dir=self.app.project_dir,
                python_executable=self.app.python_executable,
                analysis_args=("--confidence", "0.95", "--resamples", "5000", "--seed", "0", "--block-size", "1"),
            )
        except (OSError, ValueError, csv.Error) as exc:
            self.set_status(error_message(exc))
            return
        self.app.launch(pending)

    def started(self, pending: PendingLaunch) -> None:
        self.output = Path(pending.report_path)
        self.log.clear()
        self.summary = ""
        self.set_status(message_id("results.html.generating", self.output))

    def consume_line(self, line: str) -> None:
        if not line.startswith("ACPROF_REPORT "):
            self.log.append(line)
            return
        receipt = json.loads(line.removeprefix("ACPROF_REPORT "))
        if (not isinstance(receipt, dict) or receipt.get("report_path") != str(self.output)
                or any(type(receipt.get(key)) is not int or receipt[key] < 0
                       for key in ("config_count", "valid_windows", "excluded_rows"))
                or any(not isinstance(receipt.get(key), list) or not receipt[key]
                       or any(not isinstance(value, str) for value in receipt[key])
                       for key in ("quality_status", "measurement_status"))):
            raise ValueError(message_id("results.html.invalid_receipt"))
        self.summary = message_id(
            "results.html.receipt_summary", receipt["config_count"], receipt["valid_windows"],
            receipt["excluded_rows"], ", ".join(receipt["quality_status"]),
            ", ".join(receipt["measurement_status"]),
        )
        if receipt["valid_windows"] == 0:
            self.summary = join_messages("\n", (self.summary, message_id("results.html.no_valid_windows")))

    def completion_error(self, returncode: int, error: str) -> str:
        if error:
            return error
        if returncode:
            return "\n".join(self.log) or message_id("run.completion.failed_exit_code", returncode)
        if self.output is None or not self.output.is_file() or not self.summary:
            return message_id("results.html.output_missing")
        return ""

    def finished(self, *, error: str, cancelled: bool) -> None:
        output, self.output = self.output, None
        if cancelled:
            self.set_status(message_id("results.html.cancelled"))
        elif error:
            self.set_status(message_id("results.html.generation_failed", error))
        elif output is not None:
            self.app.query_one("#html-report-path", Input).value = str(output)
            self.app.configuration.remember_last_used(html_report=str(output))
            self.set_status(join_messages("\n", (message_id("results.html.generated_short"), self.summary)))
        self.log.clear()
        self.summary = ""

    def open(self) -> None:
        from acprof.platform import can_open_local_browser
        from acprof.tui.results.report_files import require_html_file

        if not self.app.allow_operation("html-open"):
            return
        source = self.app.configuration.input("html-report-path")
        if not source:
            self.set_status(message_id("results.html.open_requires_report"))
            return
        path = resolve_result_path(source, self.app.project_dir).resolve()
        try:
            require_html_file(path)
        except (OSError, ValueError) as exc:
            self.set_status(error_message(exc))
            return
        if not can_open_local_browser():
            self.set_status(message_id("results.html.remote_open_instructions", path))
            return
        self.run_file_action("open", path)

    def export(self) -> None:
        if not self.app.allow_operation("html-export"):
            return
        source = self.app.configuration.input("html-report-path")
        destination = self.app.configuration.input("html-export-path")
        if not source or not destination:
            self.set_status(message_id("results.html.export_requires_paths"))
            return
        self.run_file_action(
            "export", resolve_result_path(source, self.app.project_dir),
            resolve_result_path(destination, self.app.project_dir),
        )

    def run_file_action(self, action: str, source: Path, destination: Path | None = None) -> None:
        token = object()
        self.app.read_jobs.add(token)
        self.app.set_busy(True)
        self.app.run_worker(
            partial(self._file_worker, token, action, source, destination),
            name="html_file_action", thread=True, group="html-file", exit_on_error=False,
        )

    def _file_worker(self, token: object, action: str, source: Path, destination: Path | None) -> None:
        from acprof.tui.results.report_files import export_html_file, open_html_file

        try:
            if action == "export" and destination is not None:
                export_html_file(source, destination)
                status = message_id("results.html.exported", destination)
            else:
                open_html_file(source)
                status = message_id("results.html.browser_requested", source)
        except Exception as exc:
            status = message_id("results.html.file_action_failed", error_message(exc), source)
        self.app.process.safe_process_callback(self.file_finished, token, status)

    def file_finished(self, token: object, status: str) -> None:
        self.app.read_jobs.discard(token)
        self.app.set_busy(self.app.is_busy())
        self.set_status(status)
