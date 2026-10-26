"""Log presentation, bounded batches and explicit export of retained text."""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from textual.timer import Timer
from textual.widgets import Checkbox, Select, Static

from acprof.tui.diagnostic_buffer import BoundedLogBuffer
from acprof.tui.i18n import message_id
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog
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
                          query=self.app.query_one("#log-search", Input).value,
                          details=self.app.query_one("#log-details", Checkbox).value)

    def toggle_tools(self) -> None:
        if self.app.latest_snapshot.measurement_active:
            return
        self.tools_open = not self.tools_open
        self.app.query_one("#log-controls").display = self.tools_open
        self.refresh_status()
        if self.tools_open:
            self.app.query_one("#log-search", Input).focus()

    def find(self, *, previous: bool = False) -> None:
        if not self.app.latest_snapshot.measurement_active:
            self.log.find_match(previous=previous)

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
        # Export may do disk I/O. Require the entire operation to be idle, and
        # hold a read-job token until completion so a measurement cannot start.
        if self.app.is_busy():
            self.app.notify(self.app.localization.tr('ui.0807'), severity="warning")
            return
        self.flush()
        text = self.log.raw_text
        if not text:
            return
        token = object()
        self.app.read_jobs.add(token)
        self.app.set_busy(True)
        path = self.app.configuration.settings_path.parent / "logs" / f"retained-{time.time_ns()}.log"

        def write() -> None:
            error = ""
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8", newline="") as stream:
                    stream.write(text)
            except OSError as exc:
                error = str(exc)
            self.app.process.safe_process_callback(self.export_finished, token, path, error)

        self.app.run_worker(write, thread=True, group="log-export", exit_on_error=False)

    def export_finished(self, token: object, path: Path, error: str) -> None:
        self.app.read_jobs.discard(token)
        if not self.app.is_running or self.app.ui_closing:
            return
        self.app.set_busy(self.app.is_busy())
        self.app.notify(self.app.localization.tr(message_id(
            'ui.0936' if error else 'ui.0937', error or path,
        )), severity="error" if error else "information", timeout=8)
