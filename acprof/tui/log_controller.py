"""Log presentation, bounded batches and explicit export of retained text."""
from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from textual.timer import Timer
from textual.widgets import Select, Static

from acprof.tui.diagnostic_buffer import BoundedLogBuffer
from acprof.tui.i18n import message_id
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog
from acprof.tui.log_export import (
    default_log_export_dir,
    default_log_export_filename,
    list_export_directories,
    log_export_error,
    write_log_export,
)
from acprof.tui.log_export_screen import ExportDirectoryScreen, LogExportScreen
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.stages import Stage

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class LogController:
    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.pending = BoundedLogBuffer(max_entries=5000, max_bytes=8 * 1024 * 1024,
                                        max_entry_bytes=1024 * 1024)
        self.timer: Timer | None = None
        self.stage: Stage | None = None
        self.stage_started = time.monotonic()
        self.tools_open = False
        self.last_export_path: Path | None = None

    @property
    def pending_size(self) -> int:
        return self.pending.byte_size

    @property
    def log(self) -> SelectableLog:
        return self.app.query_one("#run-log", SelectableLog)

    def append(self, line: str) -> None:
        self.pending.append(line)
        if self.log.suspended:
            return
        if len(self.pending) >= 128 or self.pending_size >= 64 * 1024:
            self.flush()
        elif self.timer is None:
            self.timer = self.app.set_timer(0.25, self.flush)

    def stage_changed(self, snapshot: ProgressSnapshot) -> None:
        if snapshot.stage == self.stage:
            return
        now = time.monotonic()
        elapsed = f"{now - self.stage_started:.1f}"
        previous = self.stage
        self.stage, self.stage_started = snapshot.stage, now
        if snapshot.stage is Stage.STARTING:
            self.log.records.continuation = ""
        if previous is not None and previous is not Stage.WAITING and not snapshot.measurement_active:
            level = ("SUCCESS" if snapshot.stage in {Stage.COMPLETED, Stage.PROBE_COMPLETED} else
                     "ERROR" if snapshot.stage in {Stage.FAILED, Stage.PROBE_FAILED, Stage.CLEANUP_INCOMPLETE} else
                     "WARN" if snapshot.stage is Stage.PARTIAL else "INFO")
            self.append(self.app.localization.tr(message_id('ui.0947', level, snapshot.stage.label, elapsed)))

    def flush(self) -> None:
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        if not self.app.is_running or self.app.ui_closing or self.log.suspended:
            return
        if self.pending:
            batch = self.pending.snapshot()
            entries = list(batch.entries)
            if batch.discarded_entries:
                entries.insert(0, self.app.localization.tr(message_id(
                    "ui.2143", batch.discarded_entries, batch.oversized_entries,
                )))
            if entries:
                self.log.write("\n".join(entries))
            self.pending.clear()

    def suspend(self, measuring: bool) -> None:
        if measuring and self.timer is not None:
            self.timer.stop()
            self.timer = None
        self.log.set_suspended(measuring)
        for widget in self.app.query(".log-control"):
            widget.disabled = measuring
        if not measuring:
            self.flush()

    def clear(self) -> None:
        if self.app.latest_snapshot.measurement_active:
            return
        if self.timer:
            self.timer.stop()
            self.timer = None
        self.pending.clear()
        self.log.clear_log()

    def view_changed(self) -> None:
        if not self.app.form_ready or self.app.latest_snapshot.measurement_active:
            return
        self.flush()
        self.log.set_view(level=str(self.app.query_one("#log-level", Select).value),
                          query=self.app.query_one("#log-search", Input).value)

    def toggle_tools(self) -> None:
        if self.app.latest_snapshot.measurement_active:
            return
        self.tools_open = not self.tools_open
        self.app.query_one("#log-controls").display = self.tools_open
        self.refresh_status()
        if self.tools_open:
            self.app.query_one("#log-search", Input).focus()

    def find(self) -> None:
        if not self.app.latest_snapshot.measurement_active:
            self.log.find_match()

    def refresh_status(self) -> None:
        if (not self.app.is_running or self.app.ui_closing or not self.app.form_ready
                or self.app.latest_snapshot.measurement_active):
            return
        log = self.log
        self.app.query_one("#log-status", Static).display = self.tools_open or log.discarded_lines > 0
        self.app.localization.set_text(self.app.query_one("#log-status", Static), message_id(
            'ui.0934',
            len(log.records.records), log.document.line_count if log.text else 0,
            log.match_count, log.discarded_lines,
        ))

    def copy_all(self) -> None:
        if self.app.latest_snapshot.measurement_active:
            return
        self.flush()
        if self.log.copy_all():
            self.app.notify(self.app.localization.tr('ui.0935'))

    def export(self) -> None:
        if not self.app.operation_state().allows('log-export'):
            self.app.notify(self.app.localization.tr(message_id('ui.0807')), severity="warning")
            return
        self.flush()
        if not self.log.raw_text:
            return
        token = self.begin_export_job()
        previous = self.app.configuration.saved_settings.last_log_export_dir
        self.app.push_screen(LogExportScreen(
            Path(previous) if previous else default_log_export_dir(),
            default_log_export_filename(), self.load_export_directories,
        ), partial(self.export_chosen, token))

    def begin_export_job(self) -> object:
        # Keep all reads and writes owned until the actual thread returns,
        # including directory reads whose dialog has already been cancelled.
        token = object()
        self.app.read_jobs.add(token)
        self.app.set_busy(True)
        return token

    def finish_export_job(self, token: object) -> None:
        self.app.read_jobs.discard(token)
        self.app.set_busy(self.app.is_busy())

    def load_export_directories(self, screen: ExportDirectoryScreen, path: Path, nearest_parent: bool) -> None:
        token = self.begin_export_job()

        def read() -> None:
            directory, error = path, ""
            entries: tuple[Path, ...] = ()
            try:
                directory, entries = list_export_directories(
                    path, nearest_parent=nearest_parent, cancelled=screen.cancelled,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                error = log_export_error(exc)
            self.app.process.safe_process_callback(finished, directory, entries, error)

        def finished(directory: Path, entries: tuple[Path, ...], error: str) -> None:
            self.finish_export_job(token)
            screen.loaded(directory, entries, error)

        self.app.run_worker(read, thread=True, group="log-export-directory", exit_on_error=False)

    def export_chosen(self, token: object, path: Path | None) -> None:
        if path is None:
            self.finish_export_job(token)
            return
        self.flush()
        text = self.log.raw_text

        def write() -> None:
            destination = path
            error = ""
            try:
                destination = write_log_export(path, text)
            except (OSError, ValueError, UnicodeError) as exc:
                error = log_export_error(exc)
            self.app.process.safe_process_callback(self.export_finished, token, destination, error)

        self.app.run_worker(write, thread=True, group="log-export", exit_on_error=False)

    def export_finished(self, token: object, path: Path, error: str) -> None:
        if not self.app.is_running or self.app.ui_closing:
            self.app.read_jobs.discard(token)
            return
        if error:
            self.finish_export_job(token)
            self.app.notify(self.app.localization.tr(message_id('ui.0936', error)), severity="error", timeout=8)
            return
        self.last_export_path = path
        self.app.configuration.remember_last_used(log_export_dir=str(path.parent))
        self.app.query_one('#log-export-result').display = True
        self.finish_export_job(token)
        self.app.notify(self.app.localization.tr(message_id('ui.0937')))

    def copy_export_path(self) -> None:
        if self.last_export_path is not None and not self.app.latest_snapshot.measurement_active:
            self.app.copy_to_clipboard(str(self.last_export_path))
            self.app.notify(self.app.localization.tr(message_id('ui.2229')))

    def view_export_path(self) -> None:
        if self.last_export_path is None or self.app.latest_snapshot.measurement_active:
            return
        from acprof.tui.views import ConfirmActionScreen

        def closed(copy: bool | None) -> None:
            if copy:
                self.copy_export_path()

        self.app.push_screen(ConfirmActionScreen(
            message_id('ui.2228'), str(self.last_export_path), message_id('ui.2227'),
        ), closed)
