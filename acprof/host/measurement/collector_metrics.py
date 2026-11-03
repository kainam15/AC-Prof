"""Pure projections from finalized collectors to window metric groups."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.measurement.numbers import per_positive_denominator, to_float_or_nan
from acprof.results.metrics import row_group_fields

if TYPE_CHECKING:
    from acprof.monitors.energy_cpu import CPUEnergyResult, DRAMEnergyResult
    from acprof.monitors.gpu_metrics import EnergyResult
    from acprof.monitors.perf_mips import MIPSResult
    from acprof.monitors.resource_metrics import ResourceUsageResult


def _eff_negative_warnings(
    *,
    avg_power_eff_w: float,
    peak_power_eff_w: float,
    energy_eff_j: float,
) -> list[str]:
    metrics = [
        ("gpu_avg_power_eff_w", avg_power_eff_w),
        ("gpu_peak_power_eff_w", peak_power_eff_w),
        ("gpu_energy_eff_j", energy_eff_j),
    ]
    return [f"{name}<0" for name, value in metrics if value == value and value < 0.0]


def _named_negative_warnings(metrics: dict[str, float]) -> list[str]:
    return [f"{name}<0" for name, value in metrics.items() if value == value and value < 0.0]


def window_monitor_warnings(
    *, resource_error: str, gpu_error: str, cpu_error: str,
    dram_energy: str, gpu_metrics: dict[str, float], cpu_metrics: dict[str, float],
) -> list[str]:
    """Derive warnings after monitor shutdown, preserving existing row semantics."""
    warnings = [
        f"{name} monitor: {error}"
        for name, error in (
            ("resource", resource_error),
            ("gpu", gpu_error),
            ("cpu", cpu_error),
        )
        if error and not (
            name == "cpu"
            and dram_energy != "required"
            and error in {
                "RAPL unavailable",
                "RAPL powercap energy_uj unavailable",
            }
        )
    ]
    warnings.extend(_eff_negative_warnings(
        avg_power_eff_w=gpu_metrics["gpu_avg_power_eff_w"],
        peak_power_eff_w=gpu_metrics["gpu_peak_power_eff_w"],
        energy_eff_j=gpu_metrics["gpu_energy_eff_j"],
    ))
    warnings.extend(_named_negative_warnings({
        "cpu_avg_power_eff_w": cpu_metrics["cpu_avg_power_eff_w"],
        "cpu_peak_power_eff_w": cpu_metrics["cpu_peak_power_eff_w"],
        "cpu_energy_eff_j": cpu_metrics["cpu_energy_eff_j"],
        "vcpu_avg_power_eff_w": cpu_metrics["vcpu_avg_power_eff_w"],
        "vcpu_peak_power_eff_w": cpu_metrics["vcpu_peak_power_eff_w"],
        "vcpu_energy_eff_j": cpu_metrics["vcpu_energy_eff_j"],
    }))
    return warnings


def estimate_cpu_cycles(
    latency_s: float,
    cpu_freq_avg_hz: float,
    cpu_cores: float,
    container_cpu_util_avg_pct: float,
) -> float:
    latency = to_float_or_nan(latency_s)
    freq = to_float_or_nan(cpu_freq_avg_hz)
    cores = to_float_or_nan(cpu_cores)
    cpu_util_pct = to_float_or_nan(container_cpu_util_avg_pct)
    if (
        latency == latency
        and freq == freq
        and cores == cores
        and cpu_util_pct == cpu_util_pct
        and latency > 0.0
        and freq > 0.0
        and cores > 0.0
        and cpu_util_pct >= 0.0
    ):
        return latency * freq * cores * (cpu_util_pct / 100.0)
    return float("nan")


DRAM_METRIC_FIELDS = row_group_fields("dram_energy")


def _divide_if_number(value: float, divisor: float) -> float:
    value = to_float_or_nan(value)
    if value == value:
        return value / divisor
    return value


def gpu_metrics_from_result(result: EnergyResult, repeat_in_window: int) -> dict[str, float]:
    return {
        "gpu_idle_power_w": to_float_or_nan(result.idle_power_w),
        "gpu_energy_iters": float(result.energy_iters),
        "gpu_avg_power_total_w": to_float_or_nan(result.avg_power_total_w),
        "gpu_peak_power_total_w": to_float_or_nan(result.peak_power_total_w),
        "gpu_energy_total_j": _divide_if_number(result.energy_total_j, float(repeat_in_window)),
        "gpu_avg_power_eff_w": to_float_or_nan(result.avg_power_eff_w),
        "gpu_peak_power_eff_w": to_float_or_nan(result.peak_power_eff_w),
        "gpu_energy_eff_j": _divide_if_number(result.energy_eff_j, float(repeat_in_window)),
    }


def cpu_metrics_from_result(result: CPUEnergyResult, repeat_in_window: int) -> dict[str, float]:
    return {
        "cpu_idle_power_w": to_float_or_nan(result.cpu_idle_power_w),
        "cpu_energy_iters": float(result.cpu_energy_iters),
        "cpu_avg_power_total_w": to_float_or_nan(result.cpu_avg_power_total_w),
        "cpu_peak_power_total_w": to_float_or_nan(result.cpu_peak_power_total_w),
        "cpu_energy_total_j": _divide_if_number(result.cpu_energy_total_j, float(repeat_in_window)),
        "cpu_avg_power_eff_w": to_float_or_nan(result.cpu_avg_power_eff_w),
        "cpu_peak_power_eff_w": to_float_or_nan(result.cpu_peak_power_eff_w),
        "cpu_energy_eff_j": _divide_if_number(result.cpu_energy_eff_j, float(repeat_in_window)),
        "vcpu_cpu_share": to_float_or_nan(result.vcpu_cpu_share),
        "vcpu_cpu_time_s": _divide_if_number(result.vcpu_cpu_time_s, float(repeat_in_window)),
        "vcpu_avg_power_total_w": to_float_or_nan(result.vcpu_avg_power_total_w),
        "vcpu_peak_power_total_w": to_float_or_nan(result.vcpu_peak_power_total_w),
        "vcpu_energy_total_j": _divide_if_number(result.vcpu_energy_total_j, float(repeat_in_window)),
        "vcpu_avg_power_eff_w": to_float_or_nan(result.vcpu_avg_power_eff_w),
        "vcpu_peak_power_eff_w": to_float_or_nan(result.vcpu_peak_power_eff_w),
        "vcpu_energy_eff_j": _divide_if_number(result.vcpu_energy_eff_j, float(repeat_in_window)),
        **_dram_metrics_from_result(getattr(result, "dram", None), repeat_in_window),
    }


def _dram_metrics_from_result(result: DRAMEnergyResult | None, count: int) -> dict[str, float]:
    metrics = {name: to_float_or_nan(getattr(result, name.removeprefix("dram_"), None))
               for name in DRAM_METRIC_FIELDS}
    metrics["dram_energy_per_request_j"] = per_positive_denominator(metrics["dram_window_energy_j"], count)
    metrics["dram_effective_energy_per_request_j"] = per_positive_denominator(
        metrics["dram_window_effective_energy_j"], float(count))
    return metrics


def derived_efficiency_metrics(
    *,
    gpu_mode: str,
    batch_size: int,
    latency_app_s: float,
    output_token_count_avg: float,
    gpu_energy_eff_j: float,
    vcpu_energy_eff_j: float,
    input_units_per_request: float = float("nan"),
) -> dict[str, float]:
    latency = to_float_or_nan(latency_app_s)
    output_tokens = to_float_or_nan(output_token_count_avg)
    gpu_energy = to_float_or_nan(gpu_energy_eff_j)
    vcpu_energy = to_float_or_nan(vcpu_energy_eff_j)

    required_energy = [vcpu_energy]
    if normalize_gpu_mode(gpu_mode) == "on":
        required_energy.append(gpu_energy)
    attributed_energy = (
        sum(required_energy)
        if all(math.isfinite(value) and value >= 0.0 for value in required_energy)
        else float("nan")
    )

    samples_per_j = (
        float(batch_size) / attributed_energy
        if math.isfinite(attributed_energy)
        and attributed_energy > 0.0
        and batch_size > 0
        else float("nan")
    )
    edp_app = (
        attributed_energy * latency
        if math.isfinite(attributed_energy)
        and attributed_energy >= 0.0
        and math.isfinite(latency)
        and latency > 0.0
        else float("nan")
    )
    output_tokens_per_s = (
        output_tokens / latency
        if math.isfinite(output_tokens)
        and output_tokens > 0.0
        and math.isfinite(latency)
        and latency > 0.0
        else float("nan")
    )
    joules_per_output_token = (
        attributed_energy / output_tokens
        if math.isfinite(attributed_energy)
        and attributed_energy >= 0.0
        and math.isfinite(output_tokens)
        and output_tokens > 0.0
        else float("nan")
    )
    joules_per_input_unit = per_positive_denominator(
        attributed_energy,
        input_units_per_request,
    )
    return {
        "container_attributed_energy_eff_j": attributed_energy,
        "container_attributed_samples_per_j": samples_per_j,
        "container_attributed_edp_app_js": edp_app,
        "output_tokens_per_s_app": output_tokens_per_s,
        "container_attributed_j_per_output_token": joules_per_output_token,
        "container_attributed_j_per_input_unit": joules_per_input_unit,
    }


def resource_usage_metrics_from_result(
    result: ResourceUsageResult,
    repeat_in_window: int,
) -> dict[str, float]:
    io_read_bytes = to_float_or_nan(
        getattr(result, "container_io_read_bytes", float("nan"))
    )
    io_write_bytes = to_float_or_nan(
        getattr(result, "container_io_write_bytes", float("nan"))
    )
    io_read_ops = to_float_or_nan(
        getattr(result, "container_io_read_ops", float("nan"))
    )
    io_write_ops = to_float_or_nan(
        getattr(result, "container_io_write_ops", float("nan"))
    )
    if repeat_in_window > 0:
        io_read_bytes /= float(repeat_in_window)
        io_write_bytes /= float(repeat_in_window)
        io_read_ops /= float(repeat_in_window)
        io_write_ops /= float(repeat_in_window)
    else:
        io_read_bytes = float("nan")
        io_write_bytes = float("nan")
        io_read_ops = float("nan")
        io_write_ops = float("nan")
    return {
        "resource_usage_iters": float(result.resource_usage_iters),
        "container_cpu_util_avg_pct": to_float_or_nan(result.container_cpu_util_avg_pct),
        "container_cpu_util_peak_pct": to_float_or_nan(result.container_cpu_util_peak_pct),
        "container_cpu_nr_periods_delta": to_float_or_nan(
            getattr(result, "container_cpu_nr_periods_delta", float("nan"))
        ),
        "container_cpu_nr_throttled_delta": to_float_or_nan(
            getattr(result, "container_cpu_nr_throttled_delta", float("nan"))
        ),
        "container_cpu_throttled_period_ratio_pct": to_float_or_nan(
            getattr(
                result,
                "container_cpu_throttled_period_ratio_pct",
                float("nan"),
            )
        ),
        "container_cpu_throttled_time_s_per_request": (
            _divide_if_number(
                getattr(result, "container_cpu_throttled_time_s", float("nan")),
                float(repeat_in_window),
            )
            if repeat_in_window > 0
            else float("nan")
        ),
        "container_cpu_pressure_some_stall_pct": to_float_or_nan(
            getattr(
                result,
                "container_cpu_pressure_some_stall_pct",
                float("nan"),
            )
        ),
        "container_cpu_pressure_full_stall_pct": to_float_or_nan(
            getattr(
                result,
                "container_cpu_pressure_full_stall_pct",
                float("nan"),
            )
        ),
        "cpu_freq_avg_hz": to_float_or_nan(
            getattr(result, "cpu_freq_avg_hz", float("nan"))
        ),
        "cpu_freq_peak_hz": to_float_or_nan(
            getattr(result, "cpu_freq_peak_hz", float("nan"))
        ),
        "container_mem_usage_avg_bytes": to_float_or_nan(result.container_mem_usage_avg_bytes),
        "container_mem_usage_peak_bytes": to_float_or_nan(result.container_mem_usage_peak_bytes),
        "container_mem_util_avg_pct": to_float_or_nan(result.container_mem_util_avg_pct),
        "container_mem_util_peak_pct": to_float_or_nan(result.container_mem_util_peak_pct),
        "container_mem_peak_cgroup_bytes": to_float_or_nan(
            getattr(result, "container_mem_peak_cgroup_bytes", float("nan"))
        ),
        "container_mem_anon_bytes_end": to_float_or_nan(
            getattr(result, "container_mem_anon_bytes_end", float("nan"))
        ),
        "container_mem_file_bytes_end": to_float_or_nan(
            getattr(result, "container_mem_file_bytes_end", float("nan"))
        ),
        "container_mem_slab_bytes_end": to_float_or_nan(
            getattr(result, "container_mem_slab_bytes_end", float("nan"))
        ),
        "container_mem_pgfault_delta": to_float_or_nan(
            getattr(result, "container_mem_pgfault_delta", float("nan"))
        ),
        "container_mem_pgmajfault_delta": to_float_or_nan(
            getattr(result, "container_mem_pgmajfault_delta", float("nan"))
        ),
        "container_mem_workingset_refault_delta": to_float_or_nan(
            getattr(
                result,
                "container_mem_workingset_refault_delta",
                float("nan"),
            )
        ),
        "container_mem_high_events_delta": to_float_or_nan(
            getattr(result, "container_mem_high_events_delta", float("nan"))
        ),
        "container_mem_max_events_delta": to_float_or_nan(
            getattr(result, "container_mem_max_events_delta", float("nan"))
        ),
        "container_mem_oom_events_delta": to_float_or_nan(
            getattr(result, "container_mem_oom_events_delta", float("nan"))
        ),
        "container_mem_oom_kill_events_delta": to_float_or_nan(
            getattr(result, "container_mem_oom_kill_events_delta", float("nan"))
        ),
        "container_mem_pressure_some_stall_pct": to_float_or_nan(
            getattr(
                result,
                "container_mem_pressure_some_stall_pct",
                float("nan"),
            )
        ),
        "container_mem_pressure_full_stall_pct": to_float_or_nan(
            getattr(
                result,
                "container_mem_pressure_full_stall_pct",
                float("nan"),
            )
        ),
        "container_swap_limit_bytes": to_float_or_nan(
            getattr(result, "container_swap_limit_bytes", float("nan"))
        ),
        "container_swap_usage_avg_bytes": to_float_or_nan(
            getattr(result, "container_swap_usage_avg_bytes", float("nan"))
        ),
        "container_swap_usage_peak_bytes": to_float_or_nan(
            getattr(result, "container_swap_usage_peak_bytes", float("nan"))
        ),
        "container_io_read_bytes_per_request": io_read_bytes,
        "container_io_write_bytes_per_request": io_write_bytes,
        "container_io_read_ops_per_request": io_read_ops,
        "container_io_write_ops_per_request": io_write_ops,
        "container_io_pressure_some_stall_pct": to_float_or_nan(
            getattr(
                result,
                "container_io_pressure_some_stall_pct",
                float("nan"),
            )
        ),
        "container_io_pressure_full_stall_pct": to_float_or_nan(
            getattr(
                result,
                "container_io_pressure_full_stall_pct",
                float("nan"),
            )
        ),
        "container_pids_current_end": to_float_or_nan(
            getattr(result, "container_pids_current_end", float("nan"))
        ),
        "container_pids_peak_cgroup": to_float_or_nan(
            getattr(result, "container_pids_peak_cgroup", float("nan"))
        ),
        "container_pids_max_events_delta": to_float_or_nan(
            getattr(result, "container_pids_max_events_delta", float("nan"))
        ),
        "gpu_util_avg_pct": to_float_or_nan(result.gpu_util_avg_pct),
        "gpu_util_peak_pct": to_float_or_nan(result.gpu_util_peak_pct),
        "gpu_mem_used_avg_bytes": to_float_or_nan(result.gpu_mem_used_avg_bytes),
        "gpu_mem_used_peak_bytes": to_float_or_nan(result.gpu_mem_used_peak_bytes),
        "gpu_mem_util_avg_pct": to_float_or_nan(result.gpu_mem_util_avg_pct),
        "gpu_mem_util_peak_pct": to_float_or_nan(result.gpu_mem_util_peak_pct),
    }


def gpu_runtime_metrics_from_result(resource_usage_result: ResourceUsageResult | None) -> dict[str, Any]:
    pstate = str(
        getattr(resource_usage_result, "gpu_pstate", "nan")
        if resource_usage_result is not None
        else "nan"
    ).strip().upper()
    if not (
        pstate.startswith("P")
        and pstate[1:].isdigit()
        and 0 <= int(pstate[1:]) <= 15
    ):
        pstate = "nan"
    return {
        "gpu_sm_clock_mhz": to_float_or_nan(
            getattr(resource_usage_result, "gpu_sm_clock_mhz", float("nan"))
            if resource_usage_result is not None
            else float("nan")
        ),
        "gpu_memory_clock_mhz": to_float_or_nan(
            getattr(resource_usage_result, "gpu_memory_clock_mhz", float("nan"))
            if resource_usage_result is not None
            else float("nan")
        ),
        "gpu_pstate": pstate,
        "gpu_temp_c": to_float_or_nan(
            getattr(resource_usage_result, "gpu_temp_c", float("nan"))
            if resource_usage_result is not None
            else float("nan")
        ),
    }


def mips_metrics_from_result(result: MIPSResult) -> dict[str, float]:
    return {
        "cpu_cycles_per_request": to_float_or_nan(getattr(result, "cycles_per_request", float("nan"))),
        "cpu_ref_cycles_per_request": to_float_or_nan(getattr(result, "ref_cycles_per_request", float("nan"))),
        "cpu_ipc": to_float_or_nan(getattr(result, "ipc", float("nan"))),
        "cpu_perf_running_pct": to_float_or_nan(getattr(result, "running_pct", float("nan"))),
        "cpu_instructions_per_request": to_float_or_nan(
            result.instructions_per_request
        ),
        "cpu_mips_app": to_float_or_nan(result.cpu_mips_app),
        "cpu_mips_packet": float("nan"),
        "cpu_perf_elapsed_s": to_float_or_nan(result.perf_elapsed_s),
        "cpu_cache_references_per_request": to_float_or_nan(
            getattr(result, "cache_references_per_request", float("nan"))
        ),
        "cpu_cache_misses_per_request": to_float_or_nan(
            getattr(result, "cache_misses_per_request", float("nan"))
        ),
        "cpu_cache_miss_rate_pct": to_float_or_nan(
            getattr(result, "cache_miss_rate_pct", float("nan"))
        ),
        "cpu_dtlb_loads_per_request": to_float_or_nan(
            getattr(result, "dtlb_loads_per_request", float("nan"))
        ),
        "cpu_dtlb_load_misses_per_request": to_float_or_nan(
            getattr(result, "dtlb_load_misses_per_request", float("nan"))
        ),
        "cpu_dtlb_load_miss_rate_pct": to_float_or_nan(
            getattr(result, "dtlb_load_miss_rate_pct", float("nan"))
        ),
    }
