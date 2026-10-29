"""Subprocess reading, UI callbacks and late reaping; no measurement writes."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from functools import partial
from typing import TYPE_CHECKING

from textual.worker import Worker

from acprof.tui.diagnostic_buffer import BoundedLogBuffer
from acprof.tui.progress import RunProgressTracker
from acprof.tui.run_results import RunArtifacts, RunResult, inspect_probe_result, inspect_run_result

if TYPE_CHECKING:
    from acprof.experiment import RunConfig
    from acprof.tui.app import AcprofTui


class ProcessController:
    """Own process state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app

    def execute_command(self, command: list[str], kind: str, *, config: RunConfig | None) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_command_worker, command, kind, config=config),
            name="execute_command", thread=True, group='process', exclusive=True, exit_on_error=False,
        )

    def _execute_command_worker(self, command: list[str], kind: str, *, config: RunConfig | None) -> None:
        token = self.app.process_token
        def deliver(callback, *args):
            self.app.call_from_thread(self.deliver_process_callback, token, callback, *args)
        run_before = None
        if kind == "run" and config is not None:
            try:
                run_before = RunArtifacts.read(config.result_dir(self.app.project_dir))
            except (OSError, ValueError, RuntimeError):
                pass  # CLI reports invalid history; it cannot become current results.
        from acprof.download_events import parse_event as parse_download_event
        from acprof.preparation_events import PREFIX, parse_event as parse_preparation_event
        tracker = RunProgressTracker() if kind in {"run", "probe"} else None
        suppressed_lines = 0
        deferred_important_lines = BoundedLogBuffer()
        process: subprocess.Popen[str] | None = None
        launch_error = ""
        returncode = 1
        started_at_ns = time.time_ns()
        diagnostic_continuation = False
        analysis = None
        if kind in {"stats", "compare"}:
            from acprof.tui.analysis_output import AnalysisOutput
            analysis = AnalysisOutput()
        try:
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["MPLBACKEND"] = "Agg"
            child_env["ACPROF_TUI"] = "1"
            child_env.pop("ACPROF_INTERACTIVE_PREPARATION", None)
            if kind == "run":
                child_env["ACPROF_INTERACTIVE_PREPARATION"] = "1"
            # acprof run otherwise captures the entire full-screen pane, including
            # ANSI redraws, when the TUI itself is launched inside tmux.
            child_env.pop("TMUX", None)
            child_env.pop("TMUX_PANE", None)
            process = self.app.lifecycle.start(command, cwd=self.app.project_dir, env=child_env)
            if self.app.stop_requested:
                self.app.lifecycle.stop()
            deliver(self.app.process_started, process.pid, kind)
            assert process.stdout is not None
            output = iter(partial(process.stdout.read, 65536), "") if analysis is not None else process.stdout
            for raw_line in output:
                if analysis is not None:
                    analysis.append(raw_line)
                    continue
                line = raw_line.rstrip("\r\n")
                download = parse_download_event(line)
                if download is not None:
                    if tracker is not None and tracker.snapshot.measurement_active:
                        raise ValueError("download progress inside measurement window")
                    deliver(self.app.preflight.download_progress, download)
                    continue
                before = tracker.snapshot if tracker is not None else None
                snapshot = tracker.feed(line) if tracker is not None else None
                state_changed = snapshot != before if snapshot is not None else False
                important = (
                    "[ERROR]" in line
                    or "[WARN]" in line
                    or line.startswith("Traceback")
                )
                # perf and tracebacks emit untagged continuation lines. Retain
                # diagnostic context until the next structured log entry, within bounds.
                if line.startswith("["):
                    diagnostic_continuation = important
                elif important:
                    diagnostic_continuation = True
                important = important or diagnostic_continuation
                if (
                    before is not None
                    and before.measurement_active
                    and snapshot is not None
                    and snapshot.measurement_active
                ):
                    if important:
                        deferred_important_lines.append(line)
                    else:
                        suppressed_lines += 1
                    continue
                if (
                    before is not None
                    and before.measurement_active
                    and snapshot is not None
                    and not snapshot.measurement_active
                ):
                    # Release the log rendering gate before replaying diagnostics.
                    deliver(self.app.consume_process_line, "", snapshot, state_changed)
                    state_changed = False
                    if deferred_important_lines:
                        batch = deferred_important_lines.snapshot()
                        deliver(self.app.show_deferred_lines, batch.entries,
                                batch.discarded_entries, batch.oversized_entries)
                        deferred_important_lines.clear()
                    if suppressed_lines:
                        deliver(
                            self.app.show_suppressed_count,
                            suppressed_lines,
                        )
                        suppressed_lines = 0
                deliver(
                    self.app.consume_process_line,
                    "" if line.startswith(("ACPROF_EVENT ", PREFIX)) else line,
                    snapshot,
                    state_changed,
                )
                preparation = parse_preparation_event(line)
                if line.startswith("[network-preflight] "):
                    deliver(self.app.preflight.network_preflight_report, line)
                elif line.startswith("[network-download] "):
                    deliver(self.app.preflight.network_download_report, line)
                if preparation is not None:
                    deliver(self.app.preflight.preparation_event, preparation)
            returncode = process.wait()
            process.stdout.close()
            if analysis is not None and not self.app.stop_requested:
                from acprof.host.run_state import offline_analysis_lock
                with offline_analysis_lock():
                    view = analysis.finish(kind, returncode, self.app.results.analysis_source)
                deliver(self.app.results.accept_analysis, view)
        except Exception as exc:  # process errors must become visible in the UI
            launch_error = f"{type(exc).__name__}: {exc}"
            if process is not None:
                result = self.app.lifecycle.stop()
                if not result.complete:
                    self.safe_process_callback(self.app.process_cleanup_incomplete, result)
        finally:
            final_snapshot = tracker.snapshot if tracker is not None else None
            if process is not None and process.poll() is None:
                # Keep both ownership and the output reader until exit. A failed
                # UI callback must not orphan a collector or block its pipe.
                self.watch_failed_process(process, kind, final_snapshot, launch_error, run_before, token,
                                          deferred_important_lines, suppressed_lines)
            else:
                if process is not None:
                    if process.stdout is not None:
                        process.stdout.close()
                    self.app.lifecycle.release(process)
                    returncode = process.returncode
                if kind == "run" and process is not None and run_before is not None:
                    self.inspect_finished_run(run_before, process.pid, token)
                elif kind == "probe" and process is not None and returncode == 0 and not launch_error:
                    try:
                        inspect_probe_result(final_snapshot, config,
                                             self.app.project_dir, process.pid, started_at_ns)
                    except (OSError, ValueError) as exc:
                        launch_error = str(exc)
                if suppressed_lines:
                    self.safe_process_callback(self.app.show_suppressed_count, suppressed_lines)
                if deferred_important_lines:
                    batch = deferred_important_lines.drain()
                    self.safe_process_callback(self.app.show_deferred_lines, batch.entries,
                                               batch.discarded_entries, batch.oversized_entries)
                self.safe_process_callback(
                    self.app.process_finished, kind, returncode, final_snapshot, launch_error,
                )

    def deliver_process_callback(self, token, callback, *args) -> None:
        if self.app.is_running and self.app.form_ready and not self.app.ui_closing and token is self.app.process_token:
            callback(*args)

    def accept_run_result(self, token, result: RunResult) -> None:
        if token is self.app.process_token:
            self.app.run_result = result

    def inspect_finished_run(self, before: RunArtifacts, pid: int, token: object) -> None:
        self.safe_process_callback(self.deliver_process_callback, token, self.app.set_busy, True)
        try:
            result = inspect_run_result(before, pid)
        except Exception as exc:
            result = RunResult(detail=f"{type(exc).__name__}: {exc}")
        self.safe_process_callback(self.accept_run_result, token, result)

    def safe_process_callback(self, callback, *args) -> None:
        if self.app.ui_closing or not self.app.is_running:
            return
        try:
            self.app.call_from_thread(self.deliver_ui_callback, callback, *args)
        except Exception as exc:
            # Widgets may already have been pruned during shutdown. Process
            # cleanup and ownership must not depend on their availability.
            print(f"[TUI] callback failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def deliver_ui_callback(self, callback, *args) -> None:
        if self.app.is_running and self.app.form_ready and not self.app.ui_closing:
            callback(*args)

    def watch_failed_process(self, process, kind, snapshot, error, before, token,
                             deferred: BoundedLogBuffer | None = None, suppressed_lines: int = 0) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._watch_failed_process_worker, process, kind, snapshot, error, before, token,
                    deferred, suppressed_lines),
            name="watch_failed_process", thread=True, group='process-reap', exit_on_error=False,
        )

    def _watch_failed_process_worker(self, process, kind, snapshot, error, before, token,
                                     deferred: BoundedLogBuffer | None = None, suppressed_lines: int = 0) -> None:
        if deferred is None:
            deferred = BoundedLogBuffer()
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    deferred.append(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if self.app.lifecycle.release(process):
            if kind == "run" and before is not None:
                self.inspect_finished_run(before, process.pid, token)
            if suppressed_lines:
                self.safe_process_callback(self.deliver_process_callback, token,
                                           self.app.show_suppressed_count, suppressed_lines)
            if deferred:
                batch = deferred.drain()
                self.safe_process_callback(self.deliver_process_callback, token, self.app.show_deferred_lines,
                                           batch.entries, batch.discarded_entries, batch.oversized_entries)
            self.safe_process_callback(
                self.deliver_process_callback, token,
                self.app.process_finished, kind, process.returncode, snapshot, error,
            )
