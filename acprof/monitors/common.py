"""Shared Docker PID lookup and fixed-cadence sampling; metric reduction stays local."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

from acprof.host.command import DOCKER_INSPECT_TIMEOUT_S, run_command
from acprof.monitors.sampling import (
    SamplingDiagnostics,
    SamplingPolicy,
    next_deadline,
    sampling_policy,
)


def docker_container_pid(container_name: str, *, error_type: type[RuntimeError] = RuntimeError) -> int:
    result = run_command(
        ["docker", "inspect", "--format", "{{.State.Pid}}", container_name],
        capture_output=True, text=True, check=False, encoding="utf-8", errors="replace",
        timeout=DOCKER_INSPECT_TIMEOUT_S,
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
                        append_sample: Callable[[float], None], *,
                        policy: SamplingPolicy = "burst",
                        diagnostics: SamplingDiagnostics | None = None) -> None:
    """Keep the first-interval/stop semantics and select missed-tick handling."""
    sampling_policy(policy)
    next_t = (started_at if started_at is not None else time.perf_counter()) + interval
    while not stop.is_set():
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0 and stop.wait(sleep_s):
            break
        if stop.is_set():
            break
        timestamp = time.perf_counter()
        if started_at is not None and timestamp >= started_at:
            failed = False
            try:
                append_sample(timestamp)
            except BaseException:
                failed = True
                raise
            finally:
                finished = time.perf_counter() if diagnostics is not None or policy == "skip" else timestamp
                if diagnostics is not None:
                    diagnostics.observe(next_t, timestamp, finished, failed=failed)
        else:
            finished = timestamp
        next_t, skipped = next_deadline(next_t, finished, interval, policy)
        if diagnostics is not None:
            diagnostics.skipped_deadlines += skipped
