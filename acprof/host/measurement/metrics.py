"""测量结果的纯计算与格式化，不初始化客户端或监测器。"""
from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from acprof.capabilities import measurement_requested
from acprof.host.measurement.collector_metrics import (
    cpu_metrics_from_result,
    derived_efficiency_metrics,
    estimate_cpu_cycles,
    gpu_metrics_from_result,
    gpu_runtime_metrics_from_result,
    mips_metrics_from_result,
    resource_usage_metrics_from_result,
    window_monitor_warnings,
)
from acprof.host.measurement.errors import EnergyAbort
from acprof.host.measurement.numbers import (
    finite_positive,
    format_float,
    per_positive_denominator,
    resource_text,
    to_float_or_nan,
)
from acprof.host.profiling.compute_plan import (
    NCU_ERROR_FIELD,
    NCU_KERNEL_COUNT_FIELD,
    NCU_KERNEL_TIME_FIELD,
    NCU_SCALAR_MFLOP_FIELD,
    NCU_TENSOR_MFLOP_FIELD,
    NCU_TENSOR_SHARE_FIELD,
    NCU_TOTAL_MFLOP_FIELD,
    TORCH_ERROR_FIELD,
    TORCH_LOGICAL_MFLOP_FIELD,
    compute_mflops as _compute_mflops,
    find_compute_profile_entry,
)
from acprof.pixel_metrics import pixel_counts_from_metadata, pixel_rate_metrics
from acprof.results.metrics import GPU_RUNTIME_STATE_FIELDS, METRICS, row_group_fields, tool_fields
from acprof.workloads.contract import summarize_workload_contracts

if TYPE_CHECKING:
    from acprof.host.measurement.config import ClientConfig
    from acprof.host.measurement.requests import WindowRequestProgress
    from acprof.host.measurement.types import WindowObservations
    from acprof.host.profiling.execution_plan import ExecutionProfileResolver


def _input_units_per_request(input_scale: Any, batch_size: Any) -> float:
    scale = to_float_or_nan(input_scale)
    batch = to_float_or_nan(batch_size)
    if math.isfinite(scale) and scale > 0.0 and math.isfinite(batch) and batch > 0.0:
        return scale * batch
    return float("nan")


def _compute_profile_row_metrics(
    profile: dict[str, Any],
    latency_app_s: Any,
) -> dict[str, str]:
    """Format independent Torch-eager and NCU metrics for one live row."""
    logical_mflop = to_float_or_nan(profile.get(TORCH_LOGICAL_MFLOP_FIELD))
    logical_mflops_app = _compute_mflops(logical_mflop, latency_app_s)

    ncu_total_mflop = to_float_or_nan(profile.get(NCU_TOTAL_MFLOP_FIELD))
    ncu_mflops_app = _compute_mflops(ncu_total_mflop, latency_app_s)
    return {
        # Explicit Torch eager logical FLOPs.
        TORCH_LOGICAL_MFLOP_FIELD: format_float(logical_mflop),
        "model_logical_mflops_app_torch_profiler_eager": format_float(
            logical_mflops_app
        ),
        # Packet latency is not known until merge_packet_latency runs, so use
        # the application-denominator value until the packet merge completes.
        "model_logical_mflops_packet_torch_profiler_eager": format_float(
            logical_mflops_app
        ),
        TORCH_ERROR_FIELD: str(profile.get(TORCH_ERROR_FIELD) or ""),
        # Explicit NCU GPU-executed FLOPs and launch metadata.
        NCU_TOTAL_MFLOP_FIELD: format_float(ncu_total_mflop),
        NCU_TENSOR_MFLOP_FIELD: format_float(
            to_float_or_nan(profile.get(NCU_TENSOR_MFLOP_FIELD))
        ),
        NCU_SCALAR_MFLOP_FIELD: format_float(
            to_float_or_nan(profile.get(NCU_SCALAR_MFLOP_FIELD))
        ),
        NCU_TENSOR_SHARE_FIELD: format_float(
            to_float_or_nan(profile.get(NCU_TENSOR_SHARE_FIELD))
        ),
        "gpu_executed_mflops_app_ncu": format_float(ncu_mflops_app),
        "gpu_executed_mflops_packet_ncu": format_float(ncu_mflops_app),
        NCU_KERNEL_COUNT_FIELD: format_float(
            to_float_or_nan(profile.get(NCU_KERNEL_COUNT_FIELD))
        ),
        NCU_KERNEL_TIME_FIELD: format_float(
            to_float_or_nan(profile.get(NCU_KERNEL_TIME_FIELD))
        ),
        NCU_ERROR_FIELD: str(profile.get(NCU_ERROR_FIELD) or ""),
    }


_EXECUTION_PROFILE_FIELDS = (*tool_fields("massif"), *tool_fields("nsys"))
EXECUTION_PROFILE_NUMERIC_FIELDS = tuple(name for name in _EXECUTION_PROFILE_FIELDS if METRICS[name].kind == "number")


EXECUTION_PROFILE_ERROR_FIELDS = tuple(name for name in _EXECUTION_PROFILE_FIELDS if METRICS[name].kind == "text")


def _execution_profile_row_metrics(
    profile: dict[str, Any],
) -> dict[str, str]:
    """Format intrusive profiler summaries without changing their semantics."""
    result = {
        field: format_float(to_float_or_nan(profile.get(field)))
        for field in EXECUTION_PROFILE_NUMERIC_FIELDS
    }
    result.update({
        field: str(profile.get(field) or "")
        for field in EXECUTION_PROFILE_ERROR_FIELDS
    })
    return result


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _mean_finite(xs: list[float]) -> float:
    values = [
        value
        for value in (to_float_or_nan(item) for item in xs)
        if math.isfinite(value)
    ]
    return _mean(values)


def _input_num_samples(input_metadata: Any) -> float:
    if not isinstance(input_metadata, dict):
        return float("nan")
    for key in ("input_num_samples", "num_samples", "actual_num_samples"):
        value = to_float_or_nan(input_metadata.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


def _percentile_nearest_rank(xs: list[float], percentile: float) -> float:
    values = sorted(
        value
        for value in (to_float_or_nan(item) for item in xs)
        if math.isfinite(value)
    )
    if not values:
        return float("nan")
    rank = int(math.ceil((float(percentile) / 100.0) * len(values)))
    index = min(max(rank - 1, 0), len(values) - 1)
    return values[index]


def _slow_ratio(xs: list[float], *, slow_latency_threshold_s: float | None) -> float:
    values = [
        value
        for value in (to_float_or_nan(item) for item in xs)
        if math.isfinite(value)
    ]
    if (not values or slow_latency_threshold_s is None
            or not math.isfinite(slow_latency_threshold_s) or slow_latency_threshold_s <= 0):
        return float("nan")
    return sum(value > slow_latency_threshold_s for value in values) / float(len(values))


def _latency_distribution_metrics(
    prefix: str,
    latencies: list[float],
    *,
    slow_latency_threshold_s: float | None = None,
) -> dict[str, float]:
    values = [
        value
        for value in (to_float_or_nan(item) for item in latencies)
        if math.isfinite(value)
    ]
    mean = _mean(values)
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        if len(values) >= 2
        else float("nan")
    )
    p50 = _percentile_nearest_rank(values, 50.0)
    p95 = _percentile_nearest_rank(values, 95.0)
    return {
        f"{prefix}_request_count": float(len(values)),
        f"{prefix}_p50_s": p50,
        f"{prefix}_p90_s": _percentile_nearest_rank(values, 90.0),
        f"{prefix}_p95_s": p95,
        f"{prefix}_tail_ratio": p95 / p50 if p50 > 0 and math.isfinite(p95) else float("nan"),
        f"{prefix}_std_s": std,
        f"{prefix}_cv": (
            std / mean
            if math.isfinite(std) and math.isfinite(mean) and mean > 0.0
            else float("nan")
        ),
        f"{prefix}_iqr_s": (
            _percentile_nearest_rank(values, 75.0)
            - _percentile_nearest_rank(values, 25.0)
            if len(values) >= 2
            else float("nan")
        ),
        f"{prefix}_max_s": max(values) if values else float("nan"),
        f"{prefix}_slow_ratio": _slow_ratio(
            values, slow_latency_threshold_s=slow_latency_threshold_s
        ),
    }


GPU_METRIC_FIELDS = row_group_fields("gpu_energy")


DRAM_METRIC_FIELDS = row_group_fields("dram_energy")

CPU_METRIC_FIELDS = [*row_group_fields("cpu_energy"), *row_group_fields("cpu_attribution"), *DRAM_METRIC_FIELDS]


RESOURCE_USAGE_METRIC_FIELDS = row_group_fields("resource_usage")


LATENCY_PACKET_DISTRIBUTION_FIELDS = row_group_fields("latency_packet_distribution")


LATENCY_APP_DISTRIBUTION_FIELDS = row_group_fields("latency_app_distribution")


EFFICIENCY_METRIC_FIELDS = row_group_fields("efficiency")


MIPS_METRIC_FIELDS = row_group_fields("cpu_pmu")


def _nan_metrics(fields: list[str]) -> dict[str, float]:
    return {field: float("nan") for field in fields}


def idle_power_debug_stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_idle_valid_count": 0,
            f"{prefix}_idle_min_w": float("nan"),
            f"{prefix}_idle_max_w": float("nan"),
            f"{prefix}_idle_mean_w": float("nan"),
            f"{prefix}_idle_rel_range_so_far": float("nan"),
        }

    mean_idle = sum(values) / len(values)
    relative_range = (
        (max(values) - min(values)) / mean_idle
        if mean_idle > 0.0
        else float("nan")
    )
    return {
        f"{prefix}_idle_valid_count": len(values),
        f"{prefix}_idle_min_w": min(values),
        f"{prefix}_idle_max_w": max(values),
        f"{prefix}_idle_mean_w": mean_idle,
        f"{prefix}_idle_rel_range_so_far": relative_range,
    }


def idle_debug_stats(values: list[float]) -> dict[str, float]:
    return idle_power_debug_stats(values, "cpu")

class WindowMetrics:
    """Reduce completed collector data and assemble a window's CSV values.

    Reduction updates fields in protocol order so an ordinary failure retains
    already-computed values. Construction and row assembly do no measurement I/O.
    """

    def __init__(
        self, config: ClientConfig, scale_entry: dict[str, Any], *,
        use_energy: bool, slow_latency_threshold_s: float | None = None,
    ) -> None:
        self.config = config
        self.scale_entry = scale_entry
        self.use_energy = use_energy
        self.slow_latency_threshold_s = slow_latency_threshold_s
        self.input_num_samples = _input_num_samples(scale_entry.get("input_metadata"))
        self.resolved_input_scale = float(scale_entry["input_scale"])
        self.input_units_per_request = _input_units_per_request(
            self.resolved_input_scale, config.batch_size,
        )
        self.latency_app_s = float("nan")
        self.latency_app_s_per_input_unit = float("nan")
        self.request_payload_bytes = float("nan")
        self.output_length_avg = float("nan")
        self.output_token_count_avg = float("nan")
        self.throughput = float("nan")
        self.throughput_per_cpu_core = float("nan")
        self.status = "ok"
        self.error = ""
        self.gpu = _nan_metrics(GPU_METRIC_FIELDS)
        self.cpu = _nan_metrics(CPU_METRIC_FIELDS)
        self.efficiency = _nan_metrics(EFFICIENCY_METRIC_FIELDS)
        self.resource = _nan_metrics(RESOURCE_USAGE_METRIC_FIELDS)
        self.process = _nan_metrics(row_group_fields("process_usage"))
        self.gpu_runtime: dict[str, Any] = {
            **_nan_metrics(GPU_RUNTIME_STATE_FIELDS), "gpu_pstate": "nan",
        }
        self.mips = _nan_metrics(MIPS_METRIC_FIELDS)
        self.packet_distribution = _nan_metrics(LATENCY_PACKET_DISTRIBUTION_FIELDS)
        self.app_distribution = _nan_metrics(LATENCY_APP_DISTRIBUTION_FIELDS)

    def reduce(
        self, progress: WindowRequestProgress, observations: WindowObservations,
    ) -> None:
        """Consume finalized samples, preserving partial results on failure."""
        self.latency_app_s = _mean(progress.latencies)
        self.request_payload_bytes = _mean_finite(progress.payload_sizes)
        self.output_length_avg = _mean_finite(progress.output_lengths)
        self.output_token_count_avg = _mean_finite(progress.output_token_counts)
        self.app_distribution = _latency_distribution_metrics(
            "latency_app", progress.latencies,
            slow_latency_threshold_s=self.slow_latency_threshold_s,
        )
        if observations.gpu_result is not None:
            self.gpu = gpu_metrics_from_result(observations.gpu_result, progress.count)
        if observations.cpu_result is not None:
            self.cpu = cpu_metrics_from_result(observations.cpu_result, progress.count)
        if self.config.dram_energy == "required" and not all(math.isfinite(self.cpu[field]) for field in (
            "dram_window_energy_j", "dram_energy_per_request_j", "dram_window_effective_energy_j"
        )):
            raise EnergyAbort("required DRAM RAPL measurement unavailable or incomplete")
        if observations.resource_result is not None:
            if self.config.profiling_mode == "basic" and self.config.runtime_kind == "docker":
                if observations.resource_error:
                    raise RuntimeError(f"required CPU/memory measurement failed: {observations.resource_error}")
                if not all(math.isfinite(value) for value in (
                    observations.resource_result.container_cpu_util_avg_pct,
                    observations.resource_result.container_mem_usage_avg_bytes,
                )):
                    raise RuntimeError("required CPU/memory measurement unavailable")
            self.process = {name: to_float_or_nan(getattr(observations.resource_result, name, None))
                            for name in self.process}
            self.resource = resource_usage_metrics_from_result(
                observations.resource_result, progress.count,
            )
        self.resolved_input_scale = (
            progress.effective_input_scale
            if progress.effective_input_scale is not None
            else float(self.scale_entry["input_scale"])
        )
        self.input_units_per_request = _input_units_per_request(
            self.resolved_input_scale, self.config.batch_size,
        )
        self.efficiency = derived_efficiency_metrics(
            gpu_mode=self.config.gpu_mode,
            batch_size=self.config.batch_size,
            latency_app_s=self.latency_app_s,
            output_token_count_avg=self.output_token_count_avg,
            gpu_energy_eff_j=self.gpu["gpu_energy_eff_j"],
            vcpu_energy_eff_j=self.cpu["vcpu_energy_eff_j"],
            input_units_per_request=self.input_units_per_request,
        )
        self.gpu_runtime = gpu_runtime_metrics_from_result(observations.resource_result)
        if observations.mips_result is not None:
            self.mips = mips_metrics_from_result(observations.mips_result)
        warnings = window_monitor_warnings(
            resource_error=observations.resource_error if self.config.runtime_kind == "docker" else "",
            gpu_error=observations.gpu_error,
            cpu_error=observations.cpu_error,
            dram_energy=self.config.dram_energy,
            gpu_metrics=self.gpu,
            cpu_metrics=self.cpu,
        )
        if warnings:
            self.status = "warn"
            self.error = "; ".join(warnings)
        if self.latency_app_s == self.latency_app_s and self.latency_app_s > 0:
            self.throughput = float(self.config.batch_size) / float(self.latency_app_s)
        else:
            self.throughput = float("nan")
        self.latency_app_s_per_input_unit = per_positive_denominator(
            self.latency_app_s, self.input_units_per_request,
        )
        self.throughput_per_cpu_core = per_positive_denominator(
            self.throughput, self.config.cpu_cores,
        )

    def fail(self, exc: Exception) -> None:
        """Mark an ordinary window failure without discarding prior reduction."""
        self.status = "error"
        self.error = repr(exc)
        self.throughput = float("nan")
        self.throughput_per_cpu_core = float("nan")

    def record_idle_history(
        self, cpu_idle_history: list[float], gpu_idle_history: list[float],
    ) -> None:
        if not self.config.idle_debug:
            return
        if finite_positive(self.gpu["gpu_idle_power_w"]):
            gpu_idle_history.append(to_float_or_nan(self.gpu["gpu_idle_power_w"]))
        if finite_positive(self.cpu["cpu_idle_power_w"]):
            cpu_idle_history.append(to_float_or_nan(self.cpu["cpu_idle_power_w"]))

    def build_row(
        self, progress: WindowRequestProgress, observations: WindowObservations, *,
        warmup: int, repeat_idx: int, environment_class: str,
        cold_start: dict[str, Any], compute_profile_plan: dict[str, Any],
        execution_profiles: ExecutionProfileResolver,
        cpu_idle_history: list[float], gpu_idle_history: list[float],
    ) -> dict[str, Any]:
        """Join formatting and profile metadata only after the sampling window."""
        from acprof.host.measurement.requests import canonical_task_param

        gpu_idle_stats = idle_power_debug_stats(gpu_idle_history, "gpu")
        cpu_idle_stats = idle_debug_stats(cpu_idle_history)
        cpu_cycles = estimate_cpu_cycles(
            self.latency_app_s,
            self.resource["cpu_freq_avg_hz"],
            to_float_or_nan(self.config.cpu_cores),
            self.resource["container_cpu_util_avg_pct"],
        )
        gpu_power_requested = measurement_requested(
            self.config.profiling_mode, "gpu_power", gpu=self.use_energy,
        )
        plan = self.config.resource_plan
        gpu_evidence = observations.collector_evidence.get("resource", {}).get("gpu", {})
        def gpu_status(name: str, fields: tuple[str, ...]) -> str:
            if not self.use_energy:
                return "not_requested"
            if all(math.isfinite(to_float_or_nan(getattr(observations.resource_result, field, None))) for field in fields):
                return "verified"
            status = str(gpu_evidence.get(name + "_status", gpu_evidence.get("status", "unavailable")))
            return "unavailable" if status in {"available", "verified"} else status
        row: dict[str, Any] = {
            "runtime_kind": self.config.runtime_kind, "runtime_id": self.config.image_tag,
            "resource_limit_source": plan.get("method", "docker_cgroup"),
            "resource_scope": "observed_process_tree" if self.config.runtime_kind == "process" else "container_cgroup",
            "environment_sharing": "unknown", "gpu_metric_scope": "device" if self.use_energy else "not_requested",
            "process_startup_s": self.config.process_startup_s,
            "process_resource_error": observations.resource_error if self.config.runtime_kind == "process" else "",
            "process_resource_status": getattr(observations.resource_result, "process_resource_status", "not_requested"),
            "gpu_utilization_status": gpu_status("utilization", ("gpu_util_avg_pct", "gpu_util_peak_pct")),
            "gpu_memory_status": gpu_status("memory", ("gpu_mem_used_avg_bytes", "gpu_mem_used_peak_bytes")),
            "cpu_cores": resource_text(self.config.cpu_cores),
            "mem_cap_gb": resource_text(self.config.mem_cap_gb),
            "gpu_mode": self.config.gpu_mode,
            "packet_request_wire_bytes_per_request": "nan",
            "packet_response_wire_bytes_per_request": "nan",
            "packet_total_wire_bytes_per_request": "nan",
            "packet_tcp_payload_bytes_per_request": "nan",
            "packet_protocol_overhead_bytes_per_request": "nan",
            "packet_protocol_overhead_ratio": "nan",
            **cold_start,
            "result_origin": "formal_measurement",
            "environment_class": environment_class,
            "gpu_device_uuid": self.config.gpu_device_uuid if self.use_energy else "nan",
            "gpu_energy_source": getattr(observations.gpu_result, "energy_source", "unavailable") if gpu_power_requested else "not_requested",
            "gpu_energy_fallback_reason": getattr(observations.gpu_result, "energy_fallback_reason", ""),
            "gpu_idle_energy_source": getattr(observations.gpu_monitor, "idle_energy_source", "unavailable") if gpu_power_requested else "not_requested",
            "input_scale": str(self.resolved_input_scale),
            "input_units_per_request": format_float(self.input_units_per_request),
            "input_num_samples": format_float(self.input_num_samples),
            "request_payload_bytes": format_float(self.request_payload_bytes),
            "task_param": (
                progress.executed_task_param
                if progress.executed_task_param is not None
                else canonical_task_param(self.scale_entry.get("payload"))
            ),
            "workload_contract": json.dumps(
                summarize_workload_contracts(progress.workload_contracts),
                ensure_ascii=False, allow_nan=False, separators=(",", ":"),
            ),
            "output_length_avg": format_float(self.output_length_avg),
            "output_token_count_avg": format_float(self.output_token_count_avg),
            "repeat_idx": str(repeat_idx),
            "warmup": str(warmup),
            "repeat_in_window": str(progress.count),
            "latency_s": "nan",  # Filled by merge_packet_latency.
            "latency_s_per_input_unit": "nan",
            "latency_app_s": format_float(self.latency_app_s),
            "latency_app_s_per_input_unit": format_float(self.latency_app_s_per_input_unit),
            "throughput_samples_per_s": format_float(self.throughput),
            "throughput_samples_per_s_per_cpu_core": format_float(self.throughput_per_cpu_core),
            "gpu_idle_measured_at": observations.gpu_idle_measured_at if self.config.idle_debug else "nan",
            "gpu_idle_rel_range_so_far": (
                format_float(gpu_idle_stats["gpu_idle_rel_range_so_far"])
                if self.config.idle_debug else "nan"
            ),
            "dram_energy_status": (
                "not_requested" if self.config.profiling_mode == "basic" or self.config.dram_energy == "off" else
                getattr(getattr(observations.cpu_result, "dram", None), "status", "unavailable")
            ),
            "dram_energy_error": getattr(getattr(observations.cpu_result, "dram", None), "error", ""),
            "cpu_idle_measured_at": observations.cpu_idle_measured_at if self.config.idle_debug else "nan",
            "cpu_idle_rel_range_so_far": (
                format_float(cpu_idle_stats["cpu_idle_rel_range_so_far"])
                if self.config.idle_debug else "nan"
            ),
            "cpu_cycles_est_app": format_float(cpu_cycles),
            "cpu_cycles_est_packet": "nan",
            "status": self.status,
            "error": self.error,
        }
        for metrics in (
            self.gpu, self.cpu, self.efficiency, self.resource, self.process,
            self.gpu_runtime, self.mips, self.packet_distribution, self.app_distribution,
        ):
            row.update({name: str(value) if name == "gpu_pstate" else format_float(value)
                        for name, value in metrics.items()})
        pixel_counts = pixel_counts_from_metadata(
            self.scale_entry.get("input_metadata")
            if self.resolved_input_scale == float(self.scale_entry["input_scale"]) else None,
            self.config.batch_size,
            task_family=self.config.task_family,
            pipeline_tag=self.config.pipeline_tag,
            workload=self.scale_entry.get("workload"),
        )
        row.update({field: format_float(value) for field, value in pixel_counts.items()})
        row.update({field: format_float(value) for field, value in pixel_rate_metrics(row).items()})
        row_scale = to_float_or_nan(row["input_scale"])
        compute_profile = find_compute_profile_entry(
            compute_profile_plan, self.config.gpu_mode, row_scale,
        )
        row.update(_compute_profile_row_metrics(compute_profile, self.latency_app_s))
        execution_profile = execution_profiles.resolve(
            self.config.cpu_cores, self.config.mem_cap_gb, self.config.gpu_mode, row_scale,
        )
        row.update(_execution_profile_row_metrics(execution_profile))
        if environment_class == "wsl2":
            from acprof.platform import native_only_metric
            for name, metric in METRICS.items():
                if native_only_metric(metric):
                    row[name] = "nan" if metric.kind == "number" else "unavailable"
        return row
