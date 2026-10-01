"""Shared Docker PID lookup and fixed-cadence sampling; metric reduction stays local."""
from __future__ import annotations

import threading
import time
from typing import Callable

from acprof.host.command import run_command


def docker_container_pid(container_name: str, *, error_type: type[RuntimeError] = RuntimeError) -> int:
    result = run_command(
        ["docker", "inspect", "--format", "{{.State.Pid}}", container_name],
        capture_output=True, text=True, check=False, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise error_type(result.stderr.strip() or f"docker inspect failed for {container_name}")
    try:
        pid = int(result.stdout.strip())
    except ValueError as exc:
        raise error_type(f"invalid container pid: {result.stdout.strip()!r}") from exc
    if pid <= 0:
        raise error_type(f"container is not running: {container_name}")
    return pid


def sample_periodically(stop: threading.Event, started_at: float | None, interval: float,
                        append_sample: Callable[[float], None]) -> None:
    """Keep absolute deadlines, including the existing first-interval and stop semantics."""
    next_t = (started_at if started_at is not None else time.perf_counter()) + interval
    while not stop.is_set():
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0 and stop.wait(sleep_s):
            break
        if stop.is_set():
            break
        timestamp = time.perf_counter()
        if started_at is not None and timestamp >= started_at:
            append_sample(timestamp)
        next_t += interval
