"""Explicit periodic scheduling policy and constant-space timing diagnostics.

Timing describes the callback, including its sensor reads. It is not a claim
about the instant at which a device measured a value. No per-tick trace, I/O or
logging is performed by the diagnostic accumulator.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, cast

SamplingPolicy = Literal["burst", "skip"]
SAMPLING_PROTOCOL_VERSION = "periodic-v1"
DEFAULT_RAW_SAMPLE_LIMIT = 100_000


def sampling_policy(value: str) -> SamplingPolicy:
    if value not in {"burst", "skip"}:
        raise ValueError("sampling_policy must be burst or skip")
    return cast(SamplingPolicy, value)


@dataclass
class SamplingDiagnostics:
    policy: SamplingPolicy = "burst"
    interval_s: float = 0.05
    callback_count: int = 0
    late_count: int = 0
    overrun_count: int = 0
    skipped_deadlines: int = 0
    callback_errors: int = 0
    total_lateness_s: float = 0.0
    max_lateness_s: float = 0.0
    total_callback_s: float = 0.0
    max_callback_s: float = 0.0
    first_scheduled_s: float | None = None
    first_actual_s: float | None = None
    last_scheduled_s: float | None = None
    last_actual_s: float | None = None

    def __post_init__(self) -> None:
        sampling_policy(self.policy)
        if not math.isfinite(self.interval_s) or self.interval_s <= 0.0:
            raise ValueError("sampling interval must be finite and positive")

    def observe(self, scheduled: float, actual: float, finished: float, *, failed: bool = False) -> None:
        lateness = max(0.0, actual - scheduled)
        duration = max(0.0, finished - actual)
        self.callback_count += 1
        self.late_count += int(lateness > 0.0)
        self.overrun_count += int(duration > self.interval_s)
        self.callback_errors += int(failed)
        self.total_lateness_s += lateness
        self.max_lateness_s = max(self.max_lateness_s, lateness)
        self.total_callback_s += duration
        self.max_callback_s = max(self.max_callback_s, duration)
        if self.first_scheduled_s is None:
            self.first_scheduled_s, self.first_actual_s = scheduled, actual
        self.last_scheduled_s, self.last_actual_s = scheduled, actual

    def snapshot(self) -> dict[str, object]:
        return {
            "protocol_version": SAMPLING_PROTOCOL_VERSION,
            **asdict(self),
            "clock": "time.perf_counter",
            "scope": "periodic_callbacks_only; excludes boundary samples",
            "diagnostic_storage": "constant_space_aggregate",
        }


def next_deadline(previous: float, finished: float, interval: float,
                  policy: SamplingPolicy) -> tuple[float, int]:
    """Advance on the original phase; Skip discards strictly expired ticks."""
    candidate = previous + interval
    if policy == "skip" and candidate < finished:
        skipped = int(math.ceil((finished - candidate) / interval))
        return candidate + skipped * interval, skipped
    return candidate, 0
