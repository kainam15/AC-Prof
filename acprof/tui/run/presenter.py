"""Run-page presentation; session identity and subprocesses belong to ExecutionCoordinator."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Button, Checkbox, Static

from acprof.messages import message
from acprof.tui.commands import PendingLaunch, format_command
from acprof.tui.i18n import message_id
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run.monitor import MonitorPhase, monitor_phase
from acprof.tui.run.process import StopResult
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.stages import Stage
from acprof.tui.ui.presentation import NOT_APPLICABLE, UNKNOWN

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class RunPresenter:
    """Apply execution state to the existing Textual widgets; no process ownership."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app


    def launch(self, pending: PendingLaunch) -> None:
        if pending.kind == "run" and not self.app.preflight.allow_collection():
            return
        if not self.app.allow_operation("run"):
            return
        self.app.logs.flush()
        if pending.kind in {"run", "probe"}:
            with self.app.prevent(Checkbox.Changed):
                self.app.query_one("#rebuild-image", Checkbox).value = False
        model = pending.config.model if pending.config is not None else ""
        # Intended output is not evidence of a result. Keep the selected/history
        # paths until a matching run attempt has actually published artifacts.
        if pending.kind == "run":
            self.app.configuration.planned_input = None
            self.app.configuration.planned_input_identity = None
            self.app.configuration.remember_last_used(model=model)
            self.app.results.summary_request = None
            self.app.profiles.reset_overview()
            self.app.localization.set_text(self.app.query_one("#result-summary", Static),
                           "本次尚未产生结果；历史结果可通过原路径查看。")
        else:
            self.app.configuration.remember_last_used(model=model, result_dir=pending.result_dir, result_csv=pending.result_csv)
        self.app.execution.begin(pending)
        if pending.kind in {"stats", "compare"}:
            self.app.results.begin_analysis(pending)
        if pending.kind == "html-report":
            self.app.results.html_report_started(pending)
        self.app.preflight.preparation_cancelled = False
        self.app.preflight.reset_download()
        self.app.set_busy(True)
        if pending.kind in {"stats", "compare"}:
            self.app.results.show_result_section("statistics-view")
            self.app.query_one("#results-cancel", Button).focus()
        else:
            self.app.activate_tab("monitor-tab")
            # Moving focus away from inputs stops the native cursor's blink timer.
            self.app.query_one("#stop-run", Button).focus()
        # Keep the small elapsed-time display active during every run phase.
        # Logs and other periodic redraws remain gated during measurement.
        if self.app.elapsed_timer is not None:
            self.app.elapsed_timer.stop()
        self.app.elapsed_timer = self.app.set_interval(1.0, self.app.tick_elapsed)
        log = self.app.query_one("#run-log", SelectableLog)
        log.write(f"$ {format_command(pending.command)}")
        log.write(self.app.localization.tr(message_id('logs.capture.pipe_mode')))
        self.app.render_snapshot(self.app.execution.session.snapshot)
        self.app.process.execute_command(list(pending.command), pending.kind, config=pending.config)

    def tick_elapsed(self) -> None:
        """Update only the elapsed-time widget, even during measurement."""
        # Textual stops the app before removing widgets, ahead of on_unmount.
        if not self.app.is_running or self.app.ui_closing:
            return
        if not self.app.execution.session.started_monotonic or not self.app.is_busy():
            return
        elapsed = self.app.format_elapsed(time.monotonic() - self.app.execution.session.started_monotonic)
        widget = self.app.query_one('#status-elapsed', Static)
        if str(widget.content) != elapsed:
            # Fixed-size clock cell: avoid a full layout pass for each tick.
            self.app.localization.set_text(widget, elapsed, layout=False)

    def process_cleanup_incomplete(self, result: StopResult) -> None:
        detail = message_id('run.process.cleanup_incomplete', result.pid, result.error)
        self.app.execution.session.snapshot = replace(self.app.execution.session.snapshot, stage=Stage.CLEANUP_INCOMPLETE, detail=detail)
        self.app.render_snapshot(self.app.execution.session.snapshot)
        self.app.write_log(detail)
        self.app.set_busy(True)
        self.app.notify(detail, severity="error", timeout=10)

    def process_started(self, pid: int, kind: str) -> None:
        self.app.set_busy(True)
        self.app.query_one("#run-log", SelectableLog).write(
            self.app.localization.tr(message_id('logs.process.started', kind, pid))
        )

    def show_suppressed_count(self, count: int) -> None:
        self.app.query_one("#run-log", SelectableLog).write(
            self.app.localization.tr(message_id('logs.measurement.suppressed_count', count))
        )

    def show_deferred_lines(self, lines: tuple[str, ...], discarded_entries: int = 0,
                            oversized_entries: int = 0) -> None:
        log = self.app.query_one("#run-log", SelectableLog)
        log.write(self.app.localization.tr(message_id('logs.measurement.deferred_heading')))
        if discarded_entries:
            log.write(self.app.localization.tr(message_id("logs.retention.omitted_records", discarded_entries, oversized_entries)))
        if lines:
            log.write("\n".join(lines))

    def write_log(self, line: str) -> None:
        self.app.query_one("#run-log", SelectableLog).write(self.app.localization.tr(line))

    def consume_process_line(
        self,
        line: str,
        snapshot: ProgressSnapshot | None,
        state_changed: bool,
    ) -> None:
        if self.app.execution.session.kind == "html-report" and line:
            self.app.results.consume_html_line(line)
        if snapshot is not None:
            if self.app.execution.session.kind == "run" and snapshot.stage is Stage.COMPLETED:
                snapshot = replace(snapshot, stage=Stage.VERIFYING_ARTIFACT, detail="进程已报告完成，等待本次产物核验。")
            if (monitor_phase(snapshot) is MonitorPhase.PREPARATION
                    and monitor_phase(self.app.execution.session.snapshot) is not MonitorPhase.PREPARATION):
                self.app.preflight.reset_download()
            crossed_boundary = self.app.execution.update_snapshot(snapshot)
            if crossed_boundary:
                self.app.logs.suspend(snapshot.measurement_active)
            self.app.set_input_cursor_blink_enabled(not snapshot.measurement_active)
            self.app.images.sync_image_refresh_timer()
            if state_changed:
                self.app.render_snapshot(snapshot)
        if line:
            self.app.logs.append(line)

    @staticmethod
    def format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def render_snapshot(self, snapshot: ProgressSnapshot) -> None:
        self.app.logs.stage_changed(snapshot)
        from acprof.tui.ui.progress_widgets import PreparationStatus
        self.app.query_one("#status-preparation", PreparationStatus).update_snapshot(snapshot)
        phase = monitor_phase(snapshot)
        for selector, visible in (
            (".case-status", phase is MonitorPhase.MEASUREMENT and snapshot.total_cases > 0),
            (".resource-status", phase is MonitorPhase.MEASUREMENT and snapshot.current_case > 0),
            (".elapsed-status", bool(self.app.execution.session.started_monotonic)),
            (".error-count", bool(snapshot.errors or snapshot.warnings)),
            (".detail-status", bool(snapshot.detail) and (
                snapshot.stage is Stage.WAITING or snapshot.detail != message_id('run.monitor.not_started')
            )),
        ):
            for widget in self.app.query(selector):
                widget.display = visible
        self.app.query_one("#download-progress").display = (
            phase is MonitorPhase.PREPARATION and self.app.preflight.latest_download is not None
        )
        elapsed = (
            self.app.format_elapsed(time.monotonic() - self.app.execution.session.started_monotonic)
            if self.app.execution.session.started_monotonic
            else NOT_APPLICABLE
        )
        # Stage text with visual category coloring.
        stage_widget = self.app.query_one("#status-stage", Static)
        self.app.localization.set_text(stage_widget, snapshot.stage.label)
        stage_widget.set_classes(
            self.app._STAGE_CSS_CLASS.get(snapshot.stage, "stage-running")
        )
        self.app.localization.set_text(self.app.query_one('#status-elapsed', Static), elapsed)
        self.app.localization.set_text(self.app.query_one("#status-case", Static), message_id(
            'run.monitor.case_progress', snapshot.current_case or NOT_APPLICABLE,
            snapshot.completed_cases, snapshot.total_cases,
        ))
        missing = UNKNOWN if snapshot.current_case else NOT_APPLICABLE
        cpu = message_id('run.resources.inherited_limits') if snapshot.cpu == "inherited" else snapshot.cpu if snapshot.cpu != "-" else missing
        mem = message_id('run.resources.inherited_limits') if snapshot.mem == "inherited" else f"{snapshot.mem}GB" if snapshot.mem != "-" else missing
        gpu = snapshot.gpu if snapshot.gpu != "-" else missing
        self.app.localization.set_text(self.app.query_one("#status-resource", Static), message("CPU={0}  MEM={1}  GPU={2}", cpu, mem, gpu))
        self.app.localization.set_text(self.app.query_one('#status-errors', Static), f'{snapshot.warnings} / {snapshot.errors}')
        self.app.localization.set_text(self.app.query_one('#status-detail', Static), snapshot.detail)

    def process_finished(
        self,
        kind: str,
        returncode: int,
        snapshot: ProgressSnapshot | None,
        launch_error: str,
    ) -> None:
        # The child has exited; flush preparation output before its final outcome.
        self.app.logs.suspend(False)
        cancelled = self.app.execution.session.stop_requested
        if kind == "html-report" and not cancelled:
            launch_error = self.app.results.html_report_error(returncode, launch_error)
        completion = self.app.execution.finish(
            kind, returncode, snapshot, launch_error,
            analysis_ready=self.app.results.analysis_view is not None,
        )
        self.app.preflight.close_preparation()
        # Stop the elapsed-time ticker.
        if self.app.elapsed_timer is not None:
            self.app.elapsed_timer.stop()
            self.app.elapsed_timer = None
        # Show the final elapsed time.
        if self.app.execution.session.started_monotonic:
            final_elapsed = self.app.format_elapsed(
                time.monotonic() - self.app.execution.session.started_monotonic
            )
            self.app.localization.set_text(self.app.query_one('#status-elapsed', Static), final_elapsed)
        if kind in {"stats", "compare", "html-report"}:
            # Move focus before disabling the monitor page's focused Stop button;
            # its queued focus event could otherwise reactivate that old page.
            self.app.activate_tab("results-tab")
        log = self.app.query_one("#run-log", SelectableLog)
        for message_line in completion.log_messages:
            log.write(self.app.localization.tr(message_line))
        if completion.primary_notice is not None:
            self.app.notify(completion.primary_notice.message, **completion.primary_notice.kwargs())
        self.app.render_snapshot(self.app.execution.session.snapshot)
        if completion.current_csv:
            self.app.configuration.remember_last_used(
                result_csv=completion.current_csv,
                result_dir=str(Path(completion.current_csv).parent),
            )
        elif completion.summary_text:
            self.app.localization.set_text(
                self.app.query_one("#result-summary", Static),
                completion.summary_text,
            )
        if completion.run_notice is not None:
            self.app.notify(completion.run_notice.message, **completion.run_notice.kwargs())

        self.app.preflight.finish_download(cancelled=cancelled)
        self.app.logs.flush()
        self.app.set_busy(self.app.is_busy())
        if self.app.preflight.preparation_cancelled and not self.app.is_busy():
            self.app.activate_tab("run-tab")
            self.app.preflight.preparation_cancelled = False
            if self.app.preflight.pending_source_change is not None:
                from uuid import uuid4
                change, self.app.preflight.pending_source_change = self.app.preflight.pending_source_change, None
                config = self.app.execution.session.config or self.app.configuration.initial_config
                self.app.configuration.apply_config(replace(config, model=change["model_id"], revision=change["revision"],
                    model_source="modelscope", download_mode="auto", task="", task_family="", backend="",
                    model_spec="", resume=False, rebuild_image=False,
                    output_dir=str(Path(config.output_dir) / ("modelscope-" + uuid4().hex[:8]))))
        if completion.current_csv:
            self.app.results.update_result_summary(completion.current_csv, notify=False)
        elif kind == "profile" and self.app.results.summary_path is not None:
            # Posthoc can publish profiler layers without touching base rows.
            self.app.results.update_result_summary(str(self.app.results.summary_path), notify=False)
        self.app.images.sync_image_refresh_timer()
        if kind in {"stats", "compare"}:
            self.app.results.finish_analysis(error=completion.launch_error, cancelled=cancelled)
        elif kind == "html-report":
            self.app.results.html_report_finished(error=completion.launch_error, cancelled=cancelled)
