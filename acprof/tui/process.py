"""Owned subprocess lifecycle, independent of Textual callbacks.

SIGKILL is deliberately not automatic: killing the orchestrator can bypass its
Docker cleanup. A timeout keeps ownership and permits a later stop/reap attempt.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class StopResult:
    pid: int | None
    returncode: int | None
    error: str = ""

    @property
    def complete(self) -> bool:
        return self.pid is None or self.returncode is not None


class ProcessLifecycle:
    def __init__(self, *, interrupt_timeout: float = 30, terminate_timeout: float = 5):
        self.process: subprocess.Popen[str] | None = None
        self.cleanup_error = ""
        self.interrupt_timeout = interrupt_timeout
        self.terminate_timeout = terminate_timeout
        self._lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._closing = False

    def start(self, command: list[str], *, cwd, env) -> subprocess.Popen[str]:
        with self._lock:
            if self._closing or self.process is not None:
                raise RuntimeError("subprocess lifecycle is closing or already occupied")
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                stdin=subprocess.PIPE if env.get("ACPROF_INTERACTIVE_PREPARATION") == "1" else subprocess.DEVNULL,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1, start_new_session=(os.name == "posix"),
            )
            self.process = process
            self.cleanup_error = ""
            return process

    def reply(self, process: subprocess.Popen[str], line: str) -> None:
        """A reply belongs to exactly one live child, never a later launch."""
        with self._lock:
            if self.process is not process or process.poll() is not None or process.stdin is None:
                raise RuntimeError("preparation process is no longer waiting")
            process.stdin.write(line)
            process.stdin.flush()

    def release(self, process: subprocess.Popen[str]) -> bool:
        """Release only the matching process, after observing its exit."""
        with self._lock:
            if self.process is not process or process.poll() is None:
                return False
            if process.stdin is not None:
                process.stdin.close()
            self.process = None
            self.cleanup_error = ""
            return True

    def stop(self, *, closing: bool = False, expected_process: subprocess.Popen[str] | None = None) -> StopResult:
        with self._lock:
            self._closing = self._closing or closing
        # Stop, exception cleanup and unmount can arrive concurrently.
        with self._stop_lock:
            with self._lock:
                process = self.process
                if expected_process is not None and process is not expected_process:
                    return StopResult(None, None)
            if process is None:
                return StopResult(None, None)
            for sig, timeout in (
                (signal.SIGINT, self.interrupt_timeout),
                (signal.SIGTERM, self.terminate_timeout),
            ):
                if process.poll() is not None:
                    break
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, sig)
                    else:  # pragma: no cover - formal collection requires Linux
                        process.send_signal(sig)
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.cleanup_error = f"{sig.name} wait timed out after {timeout:g}s"
                except ProcessLookupError:
                    # A missing process group is not proof that Popen was reaped.
                    try:
                        process.wait(timeout=timeout)
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        self.cleanup_error = str(exc)
                except OSError as exc:
                    self.cleanup_error = f"{type(exc).__name__}: {exc}"
            code = process.poll()
            if code is not None:
                self.cleanup_error = ""
            return StopResult(process.pid, code, self.cleanup_error)
