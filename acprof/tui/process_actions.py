"""Subprocess reading, UI callbacks and late reaping; no measurement writes."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from textual import work
from textual.message_pump import MessagePump

from acprof.tui.progress import RunProgressTracker
from acprof.tui.run_results import RunArtifacts, RunResult, inspect_probe_result, inspect_run_result

if TYPE_CHECKING:
    from acprof.experiment import RunConfig
    from acprof.tui.app import AcprofTui


class ProcessWorkerActions(MessagePump):
    @work(thread=True, group="process", exclusive=True, exit_on_error=False)
    def _execute_command(self, command: list[str], kind: str, *, config: RunConfig | None) -> None:
        if TYPE_CHECKING:
            assert isinstance(self, AcprofTui)
        token = self._process_token
        def deliver(callback, *args):
            self.call_from_thread(self._deliver_process_callback, token, callback, *args)
        run_before = None
        if kind == "run" and config is not None:
            try:
                run_before = RunArtifacts.read(config.result_dir(self._project_dir))
            except (OSError, ValueError, RuntimeError):
                pass  # CLI reports invalid history; it cannot become current results.
        from acprof.preparation_events import PREFIX, parse_event as parse_preparation_event
        tracker = RunProgressTracker() if kind in {"run", "probe"} else None
        suppressed_lines = 0
        deferred_important_lines: list[str] = []
        process: subprocess.Popen[str] | None = None
        launch_error = ""
        returncode = 1
        started_at_ns = time.time_ns()
        diagnostic_continuation = False
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
            process = self._lifecycle.start(command, cwd=self._project_dir, env=child_env)
            if self._stop_requested:
                self._lifecycle.stop()
            deliver(self._process_started, process.pid, kind)
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                before = tracker.snapshot if tracker is not None else None
                snapshot = tracker.feed(line) if tracker is not None else None
                state_changed = snapshot != before if snapshot is not None else False
                important = (
                    "[ERROR]" in line
                    or "[WARN]" in line
                    or line.startswith("Traceback")
                )
                # perf and tracebacks emit untagged continuation lines. Retain
                # the complete diagnostic until the next structured log entry.
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
                    if deferred_important_lines:
                        deliver(
                            self._show_deferred_lines,
                            tuple(deferred_important_lines),
                        )
                        deferred_important_lines.clear()
                    if suppressed_lines:
                        deliver(
                            self._show_suppressed_count,
                            suppressed_lines,
                        )
                        suppressed_lines = 0
                deliver(
                    self._consume_process_line,
                    "" if line.startswith(("ACPROF_EVENT ", PREFIX)) else line,
                    snapshot,
                    state_changed,
                )
                preparation = parse_preparation_event(line)
                if line.startswith("[network-preflight] "):
                    deliver(self._network_preflight_report, line)
                elif line.startswith("[network-download] "):
                    deliver(self._network_download_report, line)
                if preparation is not None:
                    deliver(self._preparation_event, preparation)
            returncode = process.wait()
            process.stdout.close()
        except Exception as exc:  # process errors must become visible in the UI
            launch_error = f"{type(exc).__name__}: {exc}"
            if process is not None:
                result = self._lifecycle.stop()
                if not result.complete:
                    self._safe_process_callback(self._process_cleanup_incomplete, result)
        finally:
            final_snapshot = tracker.snapshot if tracker is not None else None
            if process is not None and process.poll() is None:
                # Keep both ownership and the output reader until exit. A failed
                # UI callback must not orphan a collector or block its pipe.
                self._watch_failed_process(process, kind, final_snapshot, launch_error, run_before, token)
            else:
                if process is not None:
                    if process.stdout is not None:
                        process.stdout.close()
                    self._lifecycle.release(process)
                    returncode = process.returncode
                if kind == "run" and process is not None and run_before is not None:
                    self._inspect_finished_run(run_before, process.pid, token)
                elif kind == "probe" and process is not None and returncode == 0 and not launch_error:
                    try:
                        inspect_probe_result(final_snapshot, config,
                                             self._project_dir, process.pid, started_at_ns)
                    except (OSError, ValueError) as exc:
                        launch_error = str(exc)
                if suppressed_lines:
                    self._safe_process_callback(self._show_suppressed_count, suppressed_lines)
                if deferred_important_lines:
                    self._safe_process_callback(self._show_deferred_lines, tuple(deferred_important_lines))
                self._safe_process_callback(
                    self._process_finished, kind, returncode, final_snapshot, launch_error,
                )

    def _deliver_process_callback(self, token, callback, *args) -> None:
        if TYPE_CHECKING:
            assert isinstance(self, AcprofTui)
        if self.is_running and self._form_ready and not self._ui_closing and token is self._process_token:
            callback(*args)

    def _accept_run_result(self, token, result: RunResult) -> None:
        if TYPE_CHECKING:
            assert isinstance(self, AcprofTui)
        if token is self._process_token:
            self._run_result = result

    def _inspect_finished_run(self, before: RunArtifacts, pid: int, token: object) -> None:
        if TYPE_CHECKING:
            assert isinstance(self, AcprofTui)
        self._safe_process_callback(self._deliver_process_callback, token, self._set_busy, True)
        try:
            result = inspect_run_result(before, pid)
        except Exception as exc:
            result = RunResult(detail=f"{type(exc).__name__}: {exc}")
        self._safe_process_callback(self._accept_run_result, token, result)

    def _safe_process_callback(self, callback, *args) -> None:
        if TYPE_CHECKING:
            assert isinstance(self, AcprofTui)
        if self._ui_closing or not self.is_running:
            return
        try:
            self.call_from_thread(self._deliver_ui_callback, callback, *args)
        except Exception as exc:
            # Widgets may already have been pruned during shutdown. Process
            # cleanup and ownership must not depend on their availability.
            print(f"[TUI] callback failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def _deliver_ui_callback(self, callback, *args) -> None:
        if TYPE_CHECKING:
            assert isinstance(self, AcprofTui)
        if self.is_running and self._form_ready and not self._ui_closing:
            callback(*args)

    @work(thread=True, group="process-reap", exit_on_error=False)
    def _watch_failed_process(self, process, kind, snapshot, error, before, token) -> None:
        if TYPE_CHECKING:
            assert isinstance(self, AcprofTui)
        try:
            if process.stdout is not None:
                for _ in process.stdout:
                    pass
        except (OSError, ValueError):
            pass
        process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if self._lifecycle.release(process):
            if kind == "run" and before is not None:
                self._inspect_finished_run(before, process.pid, token)
            self._safe_process_callback(
                self._process_finished, kind, process.returncode, snapshot, error,
            )
