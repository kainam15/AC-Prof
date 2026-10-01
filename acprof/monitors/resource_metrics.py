"""Pure resource sample reduction and window counter metric derivation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

BYTES_PER_GIB = 1024 ** 3


@dataclass
class ResourceUsageSample:
    timestamp: float
    container_cpu_s: Optional[float]
    container_mem_usage_bytes: Optional[int]
    gpu_util_pct: Optional[float]
    gpu_mem_used_bytes: Optional[int]
    gpu_mem_total_bytes: Optional[int]
    cpu_freq_avg_hz: Optional[float] = None
    cpu_freq_peak_hz: Optional[float] = None
    gpu_sm_clock_mhz: Optional[float] = None
    gpu_memory_clock_mhz: Optional[float] = None
    gpu_pstate: Optional[str] = None
    gpu_temp_c: Optional[float] = None
    container_swap_usage_bytes: Optional[int] = None


@dataclass
class ResourceUsageResult:
    resource_usage_iters: int
    container_cpu_util_avg_pct: float
    container_cpu_util_peak_pct: float
    cpu_freq_avg_hz: float
    cpu_freq_peak_hz: float
    container_mem_usage_avg_bytes: float
    container_mem_usage_peak_bytes: float
    container_mem_util_avg_pct: float
    container_mem_util_peak_pct: float
    gpu_util_avg_pct: float
    gpu_util_peak_pct: float
    gpu_sm_clock_mhz: float
    gpu_memory_clock_mhz: float
    gpu_pstate: str
    gpu_temp_c: float
    gpu_mem_used_avg_bytes: float
    gpu_mem_used_peak_bytes: float
    gpu_mem_util_avg_pct: float
    gpu_mem_util_peak_pct: float
    gpu_mem_total_bytes: float
    container_swap_limit_bytes: float = float("nan")
    container_swap_usage_avg_bytes: float = float("nan")
    container_swap_usage_peak_bytes: float = float("nan")
    container_io_read_bytes: float = float("nan")
    container_io_write_bytes: float = float("nan")
    container_cpu_nr_periods_delta: float = float("nan")
    container_cpu_nr_throttled_delta: float = float("nan")
    container_cpu_throttled_period_ratio_pct: float = float("nan")
    container_cpu_throttled_time_s: float = float("nan")
    container_cpu_pressure_some_stall_pct: float = float("nan")
    container_cpu_pressure_full_stall_pct: float = float("nan")
    container_mem_high_events_delta: float = float("nan")
    container_mem_max_events_delta: float = float("nan")
    container_mem_oom_events_delta: float = float("nan")
    container_mem_oom_kill_events_delta: float = float("nan")
    container_mem_pressure_some_stall_pct: float = float("nan")
    container_mem_pressure_full_stall_pct: float = float("nan")
    container_mem_peak_cgroup_bytes: float = float("nan")
    container_mem_anon_bytes_end: float = float("nan")
    container_mem_file_bytes_end: float = float("nan")
    container_mem_slab_bytes_end: float = float("nan")
    container_mem_pgfault_delta: float = float("nan")
    container_mem_pgmajfault_delta: float = float("nan")
    container_mem_workingset_refault_delta: float = float("nan")
    container_io_read_ops: float = float("nan")
    container_io_write_ops: float = float("nan")
    container_io_pressure_some_stall_pct: float = float("nan")
    container_io_pressure_full_stall_pct: float = float("nan")
    container_pids_current_end: float = float("nan")
    container_pids_peak_cgroup: float = float("nan")
    container_pids_max_events_delta: float = float("nan")


@dataclass
class _WindowCounterSnapshots:
    cpu_throttle: Optional[Dict[str, float]] = None
    memory_events: Optional[Dict[str, float]] = None
    cpu_pressure: Optional[Dict[str, float]] = None
    memory_pressure: Optional[Dict[str, float]] = None
    io_pressure: Optional[Dict[str, float]] = None
    memory_peak: Optional[Dict[str, float]] = None
    memory_stat: Optional[Dict[str, float]] = None
    io_operations: Optional[Dict[str, float]] = None
    pids: Optional[Dict[str, float]] = None


def _nan_result(resource_usage_iters: int = 0) -> ResourceUsageResult:
    nan = float("nan")
    return ResourceUsageResult(
        resource_usage_iters=resource_usage_iters,
        container_cpu_util_avg_pct=nan,
        container_cpu_util_peak_pct=nan,
        cpu_freq_avg_hz=nan,
        cpu_freq_peak_hz=nan,
        container_mem_usage_avg_bytes=nan,
        container_mem_usage_peak_bytes=nan,
        container_mem_util_avg_pct=nan,
        container_mem_util_peak_pct=nan,
        gpu_util_avg_pct=nan,
        gpu_util_peak_pct=nan,
        gpu_sm_clock_mhz=nan,
        gpu_memory_clock_mhz=nan,
        gpu_pstate="nan",
        gpu_temp_c=nan,
        gpu_mem_used_avg_bytes=nan,
        gpu_mem_used_peak_bytes=nan,
        gpu_mem_util_avg_pct=nan,
        gpu_mem_util_peak_pct=nan,
        gpu_mem_total_bytes=nan,
        container_swap_limit_bytes=nan,
        container_swap_usage_avg_bytes=nan,
        container_swap_usage_peak_bytes=nan,
        container_io_read_bytes=nan,
        container_io_write_bytes=nan,
    )


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _normalize_pstate(value: object) -> Optional[str]:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized.startswith("P") and normalized[1:].isdigit():
            numeric = int(normalized[1:])
            return normalized if 0 <= numeric <= 15 else None
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return f"P{numeric}" if 0 <= numeric <= 15 else None


def _dominant_pstate(values: List[str]) -> str:
    normalized = [state for value in values if (state := _normalize_pstate(value))]
    if not normalized:
        return "nan"
    counts = Counter(normalized)
    highest_count = max(counts.values())
    candidates = [state for state, count in counts.items() if count == highest_count]
    return min(candidates, key=lambda state: int(state[1:]))


def _counter_delta(
    start: Optional[Dict[str, float]],
    end: Optional[Dict[str, float]],
    key: str,
) -> float:
    if start is None or end is None:
        return float("nan")
    start_value = float(start.get(key, float("nan")))
    end_value = float(end.get(key, float("nan")))
    if (
        start_value != start_value
        or end_value != end_value
        or end_value < start_value
    ):
        return float("nan")
    return end_value - start_value


def _snapshot_value(snapshot: Optional[Dict[str, float]], key: str) -> float:
    if snapshot is None:
        return float("nan")
    try:
        return float(snapshot.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _pressure_stall_pct(
    start: Optional[Dict[str, float]],
    end: Optional[Dict[str, float]],
    scope: str,
    elapsed_s: float,
) -> float:
    delta_usec = _counter_delta(start, end, scope)
    if delta_usec != delta_usec or elapsed_s <= 0.0:
        return float("nan")
    return (delta_usec / (elapsed_s * 1_000_000.0)) * 100.0


def _apply_window_counter_metrics(
    result: ResourceUsageResult,
    start: _WindowCounterSnapshots,
    end: _WindowCounterSnapshots,
    elapsed_s: float,
) -> None:
    nr_periods = _counter_delta(
        start.cpu_throttle,
        end.cpu_throttle,
        "nr_periods",
    )
    nr_throttled = _counter_delta(
        start.cpu_throttle,
        end.cpu_throttle,
        "nr_throttled",
    )
    throttled_usec = _counter_delta(
        start.cpu_throttle,
        end.cpu_throttle,
        "throttled_usec",
    )
    result.container_cpu_nr_periods_delta = nr_periods
    result.container_cpu_nr_throttled_delta = nr_throttled
    if nr_periods == nr_periods and nr_periods > 0.0 and nr_throttled == nr_throttled:
        result.container_cpu_throttled_period_ratio_pct = (
            nr_throttled / nr_periods
        ) * 100.0
    if throttled_usec == throttled_usec:
        result.container_cpu_throttled_time_s = throttled_usec / 1_000_000.0

    result.container_cpu_pressure_some_stall_pct = _pressure_stall_pct(
        start.cpu_pressure,
        end.cpu_pressure,
        "some",
        elapsed_s,
    )
    result.container_cpu_pressure_full_stall_pct = _pressure_stall_pct(
        start.cpu_pressure,
        end.cpu_pressure,
        "full",
        elapsed_s,
    )

    memory_event_fields = {
        "high": "container_mem_high_events_delta",
        "max": "container_mem_max_events_delta",
        "oom": "container_mem_oom_events_delta",
        "oom_kill": "container_mem_oom_kill_events_delta",
    }
    for event, field in memory_event_fields.items():
        setattr(
            result,
            field,
            _counter_delta(start.memory_events, end.memory_events, event),
        )

    result.container_mem_pressure_some_stall_pct = _pressure_stall_pct(
        start.memory_pressure,
        end.memory_pressure,
        "some",
        elapsed_s,
    )
    result.container_mem_pressure_full_stall_pct = _pressure_stall_pct(
        start.memory_pressure,
        end.memory_pressure,
        "full",
        elapsed_s,
    )

    result.container_mem_peak_cgroup_bytes = _snapshot_value(
        end.memory_peak,
        "peak",
    )
    result.container_mem_anon_bytes_end = _snapshot_value(
        end.memory_stat,
        "anon",
    )
    result.container_mem_file_bytes_end = _snapshot_value(
        end.memory_stat,
        "file",
    )
    result.container_mem_slab_bytes_end = _snapshot_value(
        end.memory_stat,
        "slab",
    )
    result.container_mem_pgfault_delta = _counter_delta(
        start.memory_stat,
        end.memory_stat,
        "pgfault",
    )
    result.container_mem_pgmajfault_delta = _counter_delta(
        start.memory_stat,
        end.memory_stat,
        "pgmajfault",
    )
    result.container_mem_workingset_refault_delta = _counter_delta(
        start.memory_stat,
        end.memory_stat,
        "workingset_refault",
    )

    result.container_io_read_ops = _counter_delta(
        start.io_operations,
        end.io_operations,
        "read_ops",
    )
    result.container_io_write_ops = _counter_delta(
        start.io_operations,
        end.io_operations,
        "write_ops",
    )
    result.container_io_pressure_some_stall_pct = _pressure_stall_pct(
        start.io_pressure,
        end.io_pressure,
        "some",
        elapsed_s,
    )
    result.container_io_pressure_full_stall_pct = _pressure_stall_pct(
        start.io_pressure,
        end.io_pressure,
        "full",
        elapsed_s,
    )
    result.container_pids_current_end = _snapshot_value(end.pids, "current")
    result.container_pids_peak_cgroup = _snapshot_value(end.pids, "peak")
    result.container_pids_max_events_delta = _counter_delta(
        start.pids,
        end.pids,
        "max_events",
    )


def _result_from_samples(
    samples: List[ResourceUsageSample],
    cpu_cores: float,
    mem_limit_bytes: float,
    min_cpu_interval_s: float = 0.0,
) -> ResourceUsageResult:
    if not samples:
        return _nan_result(0)

    samples = sorted(samples, key=lambda sample: sample.timestamp)
    result = _nan_result(len(samples))

    cpu_interval_utils: List[float] = []
    cpu_elapsed_s = 0.0
    cpu_delta_s = 0.0
    if cpu_cores > 0:
        cpu_samples = [
            (sample.timestamp, sample.container_cpu_s)
            for sample in samples
            if sample.container_cpu_s is not None
        ]
        for idx in range(1, len(cpu_samples)):
            prev_t, prev_cpu = cpu_samples[idx - 1]
            curr_t, curr_cpu = cpu_samples[idx]
            dt = curr_t - prev_t
            delta = float(curr_cpu) - float(prev_cpu)
            if dt <= 0 or delta < 0:
                continue
            cpu_elapsed_s += dt
            cpu_delta_s += delta
            if dt >= min_cpu_interval_s:
                util = (delta / (dt * cpu_cores)) * 100.0
                cpu_interval_utils.append(util)

    if cpu_elapsed_s > 0:
        result.container_cpu_util_avg_pct = (cpu_delta_s / (cpu_elapsed_s * cpu_cores)) * 100.0
        if cpu_interval_utils:
            result.container_cpu_util_peak_pct = max(cpu_interval_utils)

    cpu_freq_avgs = [
        float(sample.cpu_freq_avg_hz)
        for sample in samples
        if sample.cpu_freq_avg_hz is not None
    ]
    cpu_freq_peaks = [
        float(sample.cpu_freq_peak_hz)
        for sample in samples
        if sample.cpu_freq_peak_hz is not None
    ]
    if cpu_freq_avgs:
        result.cpu_freq_avg_hz = _mean(cpu_freq_avgs)
    if cpu_freq_peaks:
        result.cpu_freq_peak_hz = max(cpu_freq_peaks)

    mem_values = [
        float(sample.container_mem_usage_bytes)
        for sample in samples
        if sample.container_mem_usage_bytes is not None
    ]
    if mem_values:
        result.container_mem_usage_avg_bytes = _mean(mem_values)
        result.container_mem_usage_peak_bytes = max(mem_values)
        if mem_limit_bytes > 0:
            result.container_mem_util_avg_pct = (
                result.container_mem_usage_avg_bytes / mem_limit_bytes
            ) * 100.0
            result.container_mem_util_peak_pct = (
                result.container_mem_usage_peak_bytes / mem_limit_bytes
            ) * 100.0

    swap_values = [
        float(sample.container_swap_usage_bytes)
        for sample in samples
        if sample.container_swap_usage_bytes is not None
    ]
    if swap_values:
        result.container_swap_usage_avg_bytes = _mean(swap_values)
        result.container_swap_usage_peak_bytes = max(swap_values)

    gpu_utils = [
        float(sample.gpu_util_pct)
        for sample in samples
        if sample.gpu_util_pct is not None
    ]
    if gpu_utils:
        result.gpu_util_avg_pct = _mean(gpu_utils)
        result.gpu_util_peak_pct = max(gpu_utils)

    gpu_sm_clocks = [
        float(sample.gpu_sm_clock_mhz)
        for sample in samples
        if sample.gpu_sm_clock_mhz is not None
    ]
    gpu_memory_clocks = [
        float(sample.gpu_memory_clock_mhz)
        for sample in samples
        if sample.gpu_memory_clock_mhz is not None
    ]
    gpu_pstates = [
        sample.gpu_pstate
        for sample in samples
        if sample.gpu_pstate is not None
    ]
    gpu_temperatures = [
        float(sample.gpu_temp_c)
        for sample in samples
        if sample.gpu_temp_c is not None
    ]
    if gpu_sm_clocks:
        result.gpu_sm_clock_mhz = _mean(gpu_sm_clocks)
    if gpu_memory_clocks:
        result.gpu_memory_clock_mhz = _mean(gpu_memory_clocks)
    result.gpu_pstate = _dominant_pstate(gpu_pstates)
    if gpu_temperatures:
        result.gpu_temp_c = _mean(gpu_temperatures)

    gpu_mem_used = [
        float(sample.gpu_mem_used_bytes)
        for sample in samples
        if sample.gpu_mem_used_bytes is not None
    ]
    gpu_mem_totals = [
        float(sample.gpu_mem_total_bytes)
        for sample in samples
        if sample.gpu_mem_total_bytes is not None and sample.gpu_mem_total_bytes > 0
    ]
    if gpu_mem_used:
        result.gpu_mem_used_avg_bytes = _mean(gpu_mem_used)
        result.gpu_mem_used_peak_bytes = max(gpu_mem_used)
    if gpu_mem_totals:
        result.gpu_mem_total_bytes = max(gpu_mem_totals)

    gpu_mem_utils = [
        (float(sample.gpu_mem_used_bytes) / float(sample.gpu_mem_total_bytes)) * 100.0
        for sample in samples
        if sample.gpu_mem_used_bytes is not None
        and sample.gpu_mem_total_bytes is not None
        and sample.gpu_mem_total_bytes > 0
    ]
    if gpu_mem_utils:
        result.gpu_mem_util_avg_pct = _mean(gpu_mem_utils)
        result.gpu_mem_util_peak_pct = max(gpu_mem_utils)

    return result
