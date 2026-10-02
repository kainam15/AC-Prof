"""AC-Prof Universal Client - workload generation, latency measurement, energy monitoring.

Runs on HOST (not inside container). Generalized from example-code/client.py.
"""
from __future__ import annotations

import csv
import datetime
import json
import logging
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import requests

from acprof.artifact_layout import ArtifactLayout, case_sidecar
from acprof.capabilities import measurement_requested
from acprof.config import (
    CLIENT_REQUEST_TIMEOUT_EXIT_CODE,
    CSV_FIELDS,
    GPU_RUNTIME_STATE_FIELDS,
    IDLE_DIAG_DIRNAME,
)
from acprof.host import client_diagnostics, client_metrics as _client_metrics, client_publication
from acprof.host.client_config import ClientConfig
from acprof.host.client_metrics import (
    CPU_METRIC_FIELDS,
    EFFICIENCY_METRIC_FIELDS,
    GPU_METRIC_FIELDS,
    LATENCY_APP_DISTRIBUTION_FIELDS,
    LATENCY_PACKET_DISTRIBUTION_FIELDS,
    MIPS_METRIC_FIELDS,
    RESOURCE_USAGE_METRIC_FIELDS,
    _compute_profile_row_metrics,
    _cpu_metrics_from_result,
    _derived_efficiency_metrics,
    _eff_negative_warnings,
    _estimate_cpu_cycles,
    _execution_profile_row_metrics,
    _finite_positive,
    _fmt_float,
    _gpu_metrics_from_result,
    _gpu_runtime_metrics_from_result,
    _idle_debug_stats,
    _idle_power_debug_stats,
    _input_num_samples,
    _input_units_per_request,
    _mean,
    _mean_finite,
    _mips_metrics_from_result,
    _named_negative_warnings,
    _nan_metrics,
    _per_positive_denominator,
    _prepared_body_size_bytes,
    _resource_usage_metrics_from_result,
    _to_float_or_nan,
)
from acprof.host.compute_profile_plan import (
    find_compute_profile_entry as _find_compute_profile_entry,
    load_compute_profile_plan as _load_compute_profile_plan,
)
from acprof.host.execution_profile_plan import (
    find_execution_profile_entry as _find_execution_profile_entry,
    load_execution_profile_plan as _load_execution_profile_plan,
)
from acprof.host.measurement_window import (
    MonitorCleanupError,
    MonitorGroup,
    run_matched_control_window,
)
from acprof.pixel_metrics import pixel_counts_from_metadata, pixel_rate_metrics
from acprof.workloads.contract import summarize_workload_contracts


def _ensure_local_proxy_bypass() -> None:
    local_hosts = ("localhost", "127.0.0.1", "::1")
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [part.strip() for part in current.split(",") if part.strip()]
        known = {part.lower() for part in parts}
        missing = [host for host in local_hosts if host.lower() not in known]
        if missing:
            os.environ[key] = ",".join(parts + missing)


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _is_file_empty(path: str) -> bool:
    try:
        return (not os.path.exists(path)) or os.path.getsize(path) == 0
    except Exception:
        return True

def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

class EnergyAbort(RuntimeError):
    """Raised when energy measurement prerequisites are not stable enough."""

class MIPSAbort(RuntimeError):
    """Raised when perf MIPS profiling cannot continue."""

class RequestTimeoutAbort(RuntimeError):
    """Raised when the current resource case cannot finish inference in time."""

    def __init__(
        self,
        message: str,
        *,
        input_scale: Optional[float] = None,
        request_id: str = "",
        timeout_s: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.input_scale = input_scale
        self.request_id = request_id
        self.timeout_s = timeout_s

def _request_phase_context(request_id: str) -> Dict[str, Any]:
    auto_match = re.search(r"_auto_warmup(?P<request_idx>\d+)$", request_id)
    if auto_match:
        return {
            "request_phase": "auto_repeat_window_warmup",
            "request_index_in_window": int(auto_match.group("request_idx")),
        }

    measurement_match = re.search(
        r"_(?P<phase>[wr])(?P<repeat_idx>\d+):(?P<request_idx>\d+)$",
        request_id,
    )
    if measurement_match:
        phase = measurement_match.group("phase")
        return {
            "request_phase": (
                "measurement_warmup" if phase == "w" else "measurement_repeat"
            ),
            "measurement_repeat_idx": int(measurement_match.group("repeat_idx")),
            "request_index_in_window": int(measurement_match.group("request_idx")),
        }

    return {"request_phase": "unknown"}

def _canonical_task_param(payload: Optional[Dict[str, Any]]) -> str:
    """Serialize the parameters that the server actually receives."""
    params: Any = {}
    if isinstance(payload, dict):
        nested_params = payload.get("params")
        if isinstance(nested_params, dict):
            params = dict(nested_params)
        elif nested_params is not None:
            params = {"params": nested_params}

        # Time-series handlers consume prediction_length at the top level,
        # rather than through the common nested params object.
        if "prediction_length" in payload:
            params["prediction_length"] = payload["prediction_length"]
    return json.dumps(
        params,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )

def _latency_distribution_metrics(
    prefix: str, latencies: List[float], *, slow_latency_threshold_s: float | None = None,
) -> Dict[str, float]:
    return _client_metrics._latency_distribution_metrics(
        prefix, latencies, slow_latency_threshold_s=slow_latency_threshold_s,
    )

def _parse_effective_input_scale(resp: Dict[str, Any]) -> Optional[float]:
    if not isinstance(resp, dict):
        return None
    value = resp.get("effective_input_scale")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _merge_effective_input_scale(
    current: Optional[float],
    candidate: Optional[float],
    requested_scale: float,
) -> float:
    resolved = float(requested_scale) if candidate is None else float(candidate)
    if current is None:
        return resolved
    if not math.isclose(current, resolved, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"inconsistent effective_input_scale for requested_scale={requested_scale}: "
            f"{current} vs {resolved}"
        )
    return current

def _generic_scale_label(scale_value: float) -> str:
    return f"scale{float(scale_value):g}"


class ClientRunner:
    """Own one configuration, request state and monitor dependencies per execution."""

    def __init__(self, config: ClientConfig):
        self.config = config
        from acprof.platform import detect_environment
        self.collection_environment = detect_environment()
        self.first_predict_app_s = float("nan")
        self.input_scale_entries: List[Dict[str, Any]] = []
        self.use_energy = config.gpu_mode == "on"
        self.energy_mod = self._monitor_module("energy_nvml") if self.use_energy else None
        self.cpu_energy_mod = self._monitor_module("energy_cpu")
        self.resource_usage_mod = self._monitor_module("resource_usage")
        self.perf_mips_mod = self._monitor_module("perf_mips")

    @staticmethod
    def _monitor_module(name):
        from importlib import import_module
        try:
            return import_module(f"acprof.monitors.{name}")
        except ImportError as exc:
            print(f"[WARN] monitor {name} unavailable: {exc}", file=sys.stderr)
            return None

    def _sniff_groups_path(self, csv_path: str) -> str:
        return self.config.sniff_groups_path or str(case_sidecar(csv_path, "sniff_groups"))

    def _idle_diag_path(self, csv_path: str) -> str:
        if self.config.idle_diag_path:
            return self.config.idle_diag_path
        csv_dir = os.path.dirname(csv_path)
        csv_name = os.path.basename(csv_path)
        return os.path.join(
            csv_dir,
            IDLE_DIAG_DIRNAME,
            f"{csv_name}.idle_diag.jsonl",
        )

    def _cold_start_row_metrics(self) -> Dict[str, str]:
        return {
            "cold_start_started_at": self.config.cold_start_started_at or "nan",
            "cold_start_ready_at": self.config.cold_start_ready_at or "nan",
            "cold_start_container_launch_s": self.config.cold_start_container_launch_s or "nan",
            "cold_start_server_setup_s": self.config.cold_start_server_setup_s or "nan",
            "cold_start_cuda_init_s": self.config.cold_start_cuda_init_s or "nan",
            "cold_start_model_load_s": self.config.cold_start_model_load_s or "nan",
            "cold_start_ready_wait_s": self.config.cold_start_ready_wait_s or "nan",
            "cold_start_first_predict_app_s": _fmt_float(self.first_predict_app_s),
            "cold_start_s": self.config.cold_start_s or "nan",
        }

    def _write_client_error_sidecar(self, exc: RequestTimeoutAbort) -> None:
        if not self.config.client_error_path:
            return

        payload = {
            "schema_version": 1,
            "error_type": "client_request_timeout",
            "message": str(exc),
            "input_scale": exc.input_scale,
            "request_id": exc.request_id,
            "request_timeout_s": exc.timeout_s,
            "measurement_completed": False,
            "timeout_semantics": "connect_or_read_inactivity",
            **_request_phase_context(exc.request_id),
        }
        os.makedirs(os.path.dirname(self.config.client_error_path) or ".", exist_ok=True)
        tmp_path = f"{self.config.client_error_path}.tmp-{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.config.client_error_path)

    def _prepare_repeat_window(self,
        scale_value: float,
        scale_label: str,
        payload_override: Optional[Dict[str, Any]],
    ) -> int:
        if self.config.repeat_in_window > 0:
            return self.config.repeat_in_window
        if self.config.repeat_window_seconds <= 0.0:
            raise EnergyAbort(
                f"invalid REPEAT_WINDOW_SECONDS={self.config.repeat_window_seconds!r}; expected a positive value"
            )
        if self.config.auto_warmup_requests < 0:
            raise EnergyAbort(
                "invalid AUTO_WARMUP_REQUESTS="
                f"{self.config.auto_warmup_requests!r}; expected >= 0"
            )

        for idx in range(self.config.auto_warmup_requests):
            req_id = f"{self.config.case_name}_{scale_label}_auto_warmup{idx}"
            self._one_request(scale_value, req_id=req_id, payload_override=payload_override)

        print(
            "[client] auto repeat-window "
            f"scale={scale_value:g} warmup_requests={self.config.auto_warmup_requests} "
            f"target_window_s={self.config.repeat_window_seconds:.3f}",
            flush=True,
        )
        return 1

    def _should_send_window_request(self,
        completed_requests: int,
        latency_sum_s: float,
        repeat_request_limit: int,
    ) -> bool:
        if self.config.repeat_in_window > 0:
            return completed_requests < repeat_request_limit
        if completed_requests <= 0:
            return True
        return latency_sum_s < self.config.repeat_window_seconds

    def _one_request(self, scale_value: float, req_id: str, payload_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if payload_override is None:
            raise RuntimeError("input scale plan entry has no payload; the plan must be self-contained")
        payload = payload_override

        headers = {
            "Connection": "close",
            "X-Req-Id": req_id,
        }

        t0 = time.perf_counter()
        try:
            r = requests.post(
                self.config.base_url + self.config.endpoint,
                json=payload,
                headers=headers,
                timeout=(self.config.request_timeout_seconds, self.config.request_timeout_seconds),
            )
        except requests.exceptions.Timeout as exc:
            raise RequestTimeoutAbort(
                "inference request connect/read inactivity timeout: "
                f"{self.config.request_timeout_seconds:g}s per phase (not a total deadline) "
                f"(input_scale={scale_value:g}, req_id={req_id})",
                input_scale=float(scale_value),
                request_id=req_id,
                timeout_s=float(self.config.request_timeout_seconds),
            ) from exc
        t1 = time.perf_counter()
        if r.status_code >= 400:
            try:
                detail = r.json().get("error", "")
            except Exception:
                detail = r.text[:500]
            raise RuntimeError(f"HTTP {r.status_code}: {detail or r.reason}")
        resp = r.json()
        request_latency_s = t1 - t0
        if not math.isfinite(self.first_predict_app_s):
            self.first_predict_app_s = request_latency_s
        return {
            "latency_app_s": request_latency_s,
            "resp": resp,
            "effective_input_scale": _parse_effective_input_scale(resp),
            "request_payload_bytes": _prepared_body_size_bytes(r),
            "output_length": _to_float_or_nan(resp.get("output_length")),
            "output_token_count": _to_float_or_nan(resp.get("output_token_count")),
            "task_param": _canonical_task_param(payload),
            "workload_contract": resp.get("workload_contract"),
        }

    def _load_input_scale_entries(self) -> List[Dict[str, Any]]:
        if self.config.input_scale_plan_file:
            with open(self.config.input_scale_plan_file, "r", encoding="utf-8") as f:
                plan = json.load(f)

            from acprof.artifacts import require_schema_version
            require_schema_version(plan, 2, "input_scale_plan.json")

            entries = plan.get("entries")
            if not isinstance(entries, list) or not entries:
                raise RuntimeError(f"invalid input scale plan file: {self.config.input_scale_plan_file}")

            loaded_entries: List[Dict[str, Any]] = []
            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise RuntimeError(
                        f"invalid input scale plan entry at index {idx}: {entry!r}"
                    )

                raw_scale = entry.get("input_scale")
                payload = entry.get("payload")
                if raw_scale is None or not isinstance(payload, dict):
                    raise RuntimeError(
                        f"input scale plan entry missing input_scale/payload at index {idx}"
                    )

                scale_value = float(raw_scale)
                scale_label = str(
                    entry.get("scale_label") or _generic_scale_label(scale_value)
                )
                input_metadata = entry.get("input_metadata", {})
                if not isinstance(input_metadata, dict):
                    raise ValueError(f"invalid input_metadata at input scale plan entry {idx}")
                loaded_entries.append({
                    "input_scale": scale_value,
                    "scale_label": scale_label,
                    "payload": payload,
                    "input_metadata": input_metadata,
                    "workload": plan.get("workload", {}),
                })

            scale_order = self.config.input_scale_order
            if scale_order:
                order = json.loads(scale_order)
                available = {entry["input_scale"]: entry for entry in loaded_entries}
                if (not isinstance(order, list) or len(order) != len(available)
                        or len(set(order)) != len(order) or set(order) != set(available)):
                    raise ValueError("frozen matrix input-scale order does not match input plan")
                loaded_entries = [available[scale] for scale in order]
            return loaded_entries

        raise ValueError("INPUT_SCALE_PLAN_FILE is required; generate a schema v2 input plan first")

    def _is_mips_error(self, exc: Exception) -> bool:
        if isinstance(exc, MIPSAbort):
            return True
        if self.perf_mips_mod is None:
            return False
        mips_error_cls = getattr(self.perf_mips_mod, "MIPSProfilingError", None)
        return bool(mips_error_cls is not None and isinstance(exc, mips_error_cls))

    def _sleep_before_idle_baseline(self) -> None:
        if self.config.idle_cooldown_seconds > 0.0:
            time.sleep(self.config.idle_cooldown_seconds)

    def _collect_gpu_idle_debug_snapshot(self, device_index: int | None = None) -> Dict[str, Any]:
        device_index = self.config.device_index if device_index is None else device_index
        snapshot: Dict[str, Any] = {"gpu_snapshot_scope": "after_gpu_idle"}
        try:
            snapshot["nvidia_smi_gpu"] = client_diagnostics._collect_nvidia_smi_gpu_snapshot(device_index)
        except Exception as exc:
            snapshot["nvidia_smi_gpu_error"] = repr(exc)

        try:
            snapshot["nvidia_smi_pmon"] = client_diagnostics._collect_nvidia_smi_pmon(device_index)
        except Exception as exc:
            snapshot["nvidia_smi_pmon_error"] = repr(exc)

        try:
            snapshot["nvidia_smi_compute_apps"] = client_diagnostics._collect_nvidia_smi_compute_apps(device_index)
        except Exception as exc:
            snapshot["nvidia_smi_compute_apps_error"] = repr(exc)

        return snapshot

    def _build_result_row(self, values, *, metric_groups, scale_entry, resolved_input_scale,
                          latency_app_s, compute_profile_plan, execution_profile_plan):
        """Assemble CSV values after sampling; profile and pixel joins stay out of the window."""
        row = {
            "cpu_cores": self.config.cpu_cores,
            "mem_cap_gb": self.config.mem_cap_gb,
            "gpu_mode": self.config.gpu_mode,
            "packet_request_wire_bytes_per_request": "nan",
            "packet_response_wire_bytes_per_request": "nan",
            "packet_total_wire_bytes_per_request": "nan",
            "packet_tcp_payload_bytes_per_request": "nan",
            "packet_protocol_overhead_bytes_per_request": "nan",
            "packet_protocol_overhead_ratio": "nan",
            **self._cold_start_row_metrics(),
            "result_origin": "formal_measurement",
            **values,
        }
        for metrics in metric_groups:
            row.update({name: str(value) if name == "gpu_pstate" else _fmt_float(value)
                        for name, value in metrics.items()})
        pixel_counts = pixel_counts_from_metadata(
            scale_entry.get("input_metadata") if resolved_input_scale == float(scale_entry["input_scale"]) else None,
            self.config.batch_size,
            task_family=self.config.task_family,
            pipeline_tag=self.config.pipeline_tag,
            workload=scale_entry.get("workload"),
        )
        row.update({field: _fmt_float(value) for field, value in pixel_counts.items()})
        row.update({field: _fmt_float(value) for field, value in pixel_rate_metrics(row).items()})
        row_scale = _to_float_or_nan(row["input_scale"])
        compute_profile = _find_compute_profile_entry(
            compute_profile_plan,
            self.config.gpu_mode,
            row_scale,
        )
        row.update(
            _compute_profile_row_metrics(
                compute_profile,
                latency_app_s,
            )
        )
        execution_profile = _find_execution_profile_entry(
            execution_profile_plan,
            self.config.cpu_cores,
            self.config.mem_cap_gb,
            self.config.gpu_mode,
            row_scale,
        )
        row.update(
            _execution_profile_row_metrics(execution_profile)
        )
        row["environment_class"] = self.collection_environment.environment
        if self.collection_environment.environment == "wsl2":
            from acprof.metric_registry import METRICS
            from acprof.platform import native_only_metric
            for name, metric in METRICS.items():
                if native_only_metric(metric):
                    row[name] = "nan" if metric.kind == "number" else "unavailable"
        return row

    def _execute_window(self, scale_entry, repeat_request_limit, warmup_flag, repeat_idx, requests_f, cpu_idle_values_so_far, gpu_idle_values_so_far, slow_latency_threshold_s, compute_profile_plan, execution_profile_plan):
        """Sample one window, then assemble its result; publication belongs to main."""
        scale_val = float(scale_entry["input_scale"])
        payload_override = scale_entry.get("payload")
        input_num_samples = _input_num_samples(scale_entry.get("input_metadata"))
        scale_label = str(scale_entry["scale_label"])
        phase = "w" if warmup_flag else "r"
        sniff_group_id = f"{self.config.case_name}_{scale_label}_{phase}{repeat_idx}"

        latency_app_s = float("nan")
        resolved_input_scale = scale_val
        input_units_per_request = _input_units_per_request(
            resolved_input_scale,
            self.config.batch_size,
        )
        latency_app_s_per_input_unit = float("nan")
        request_payload_bytes = float("nan")
        output_length_avg = float("nan")
        output_token_count_avg = float("nan")
        executed_task_param: Optional[str] = None
        status = "ok"
        err_msg = ""
        gpu_metrics = _nan_metrics(GPU_METRIC_FIELDS)
        cpu_metrics = _nan_metrics(CPU_METRIC_FIELDS)
        efficiency_metrics = _nan_metrics(EFFICIENCY_METRIC_FIELDS)
        resource_usage_metrics = _nan_metrics(RESOURCE_USAGE_METRIC_FIELDS)
        gpu_runtime_metrics: Dict[str, Any] = _nan_metrics(
            GPU_RUNTIME_STATE_FIELDS
        )
        gpu_runtime_metrics["gpu_pstate"] = "nan"
        mips_metrics = _nan_metrics(MIPS_METRIC_FIELDS)
        latency_packet_distribution_metrics = _nan_metrics(LATENCY_PACKET_DISTRIBUTION_FIELDS)
        latency_app_distribution_metrics = _nan_metrics(LATENCY_APP_DISTRIBUTION_FIELDS)
        effective_input_scale: Optional[float] = None
        cpu_idle_measured_at = "nan"
        gpu_idle_measured_at = "nan"
        idle_debug_snapshot: Optional[Dict[str, Any]] = None
        gpu_idle_debug_snapshot: Optional[Dict[str, Any]] = None
        idle_trace: Dict[str, Any] = {}
        gpu_idle_trace: Dict[str, Any] = {}

        monitors = MonitorGroup()
        gpu_monitor = None
        cpu_monitor = None
        resource_usage_monitor = None
        mips_monitor = None
        gpu_result = None
        _gpu_samples = []
        cpu_result = None
        resource_usage_result = None
        actual_repeat_in_window = 0
        lat_sum = 0.0
        latency_app_values: List[float] = []
        request_payload_bytes_values: List[float] = []
        output_length_values: List[float] = []
        output_token_count_values: List[float] = []
        workload_contracts: List[Dict[str, Any]] = []
        window_error = ""
        primary_error = None
        pending_request_id = ""
        _resource_usage_err = ""
        try:
            try:
                if measurement_requested(self.config.profiling_mode, "gpu_power", gpu=self.use_energy) and self.energy_mod is not None:
                    gpu_monitor = self.energy_mod.GPUEnergyMonitor(
                        sample_hz=self.config.sample_hz,
                        idle_seconds=self.config.idle_seconds,
                        device_index=self.config.device_index,
                        device_uuid=self.config.gpu_device_uuid,
                    )
                    monitors.add("gpu", gpu_monitor)

                if measurement_requested(self.config.profiling_mode, "cpu_energy") and self.cpu_energy_mod is not None:
                    try:
                        cpu_monitor = self.cpu_energy_mod.CPUEnergyMonitor(
                            sample_hz=self.config.sample_hz, idle_seconds=self.config.idle_seconds,
                            container_name=self.config.container_name, dram_energy=self.config.dram_energy)
                        monitors.add("cpu", cpu_monitor)
                    except RuntimeError as exc:
                        if self.config.dram_energy == "required":
                            raise EnergyAbort(str(exc)) from exc
                        raise

                if self.resource_usage_mod is not None:
                    resource_usage_monitor = self.resource_usage_mod.ResourceUsageMonitor(
                        sample_hz=self.config.sample_hz,
                        container_name=self.config.container_name,
                        cpu_cores=_to_float_or_nan(self.config.cpu_cores),
                        mem_cap_gb=_to_float_or_nan(self.config.mem_cap_gb),
                        use_gpu=self.use_energy,
                        device_index=self.config.device_index,
                        device_uuid=self.config.gpu_device_uuid,
                    )
                    monitors.add("resource", resource_usage_monitor)

                if measurement_requested(self.config.profiling_mode, "cpu_instructions") and self.config.use_mips:
                    mips_monitor = self.perf_mips_mod.PerfMIPSMonitor(self.config.container_name)
                    monitors.add("mips", mips_monitor)

                if gpu_monitor is not None or cpu_monitor is not None:
                    self._sleep_before_idle_baseline()
                    run_matched_control_window(monitors, idle_seconds=self.config.idle_seconds,
                                               trace=self.config.idle_debug)
                    measured_at = _now_iso()
                    if gpu_monitor is not None:
                        gpu_idle_measured_at = measured_at
                        gpu_idle_trace = dict(
                            getattr(gpu_monitor, "idle_trace", {}) or {}
                        )
                    if cpu_monitor is not None:
                        cpu_idle_measured_at = measured_at
                        idle_trace = dict(
                            getattr(cpu_monitor, "idle_trace", {}) or {}
                        )

                    if self.config.idle_debug:
                        if gpu_monitor is not None:
                            gpu_idle_debug_snapshot = self._collect_gpu_idle_debug_snapshot()
                        if cpu_monitor is not None:
                            idle_debug_snapshot = client_diagnostics._collect_idle_debug_snapshot()

                monitors.start()

                while self._should_send_window_request(
                    actual_repeat_in_window,
                    lat_sum,
                    repeat_request_limit,
                ):
                    req_id = f"{sniff_group_id}:{actual_repeat_in_window}"
                    pending_request_id = req_id
                    out = self._one_request(scale_val, req_id=req_id, payload_override=payload_override)
                    pending_request_id = ""
                    request_latency_app_s = float(out["latency_app_s"])
                    lat_sum += request_latency_app_s
                    latency_app_values.append(request_latency_app_s)
                    request_payload_bytes_values.append(
                        _to_float_or_nan(out.get("request_payload_bytes"))
                    )
                    output_length_values.append(
                        _to_float_or_nan(out.get("output_length"))
                    )
                    output_token_count_values.append(
                        _to_float_or_nan(out.get("output_token_count"))
                    )
                    workload_contracts.append(out.get("workload_contract"))
                    request_task_param = out.get("task_param")
                    if (
                        request_task_param is not None
                        and executed_task_param is None
                    ):
                        executed_task_param = str(request_task_param)
                    actual_repeat_in_window += 1
                    effective_input_scale = _merge_effective_input_scale(
                        effective_input_scale,
                        out.get("effective_input_scale"),
                        scale_val,
                    )
            except BaseException as exc:
                primary_error = exc
                window_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                monitors.finish(actual_repeat_in_window,
                                lat_sum / actual_repeat_in_window if actual_repeat_in_window else float("nan"))
                if monitors.results.get("resource") is not None:
                    resource_usage_result, _resource_usage_err, _resource_usage_samples = monitors.results["resource"]
                if monitors.results.get("gpu") is not None:
                    gpu_result, _gpu_name_ret, _gpu_err, _gpu_samples = monitors.results["gpu"]
                if monitors.results.get("cpu") is not None:
                    cpu_result, _cpu_err, _cpu_samples = monitors.results["cpu"]
                mips_result = monitors.results.get("mips")
                try:
                    client_publication._append_request_window(
                        requests_f, sniff_group_id=sniff_group_id,
                        input_scale=effective_input_scale if effective_input_scale is not None else scale_val,
                        warmup=warmup_flag, repeat_idx=repeat_idx,
                        latencies=latency_app_values,
                        error="; ".join(part for part in (window_error, monitors.error) if part),
                        failed_request_id=pending_request_id,
                    )
                finally:
                    if primary_error is not None and (
                        not isinstance(primary_error, Exception)
                        or isinstance(primary_error, (RequestTimeoutAbort, EnergyAbort, MIPSAbort))
                    ):
                        raise primary_error
                    monitors.raise_if_failed()

            latency_app_s = _mean(latency_app_values)
            request_payload_bytes = _mean_finite(
                request_payload_bytes_values
            )
            output_length_avg = _mean_finite(output_length_values)
            output_token_count_avg = _mean_finite(
                output_token_count_values
            )
            latency_app_distribution_metrics = _latency_distribution_metrics(
                "latency_app",
                latency_app_values,
                slow_latency_threshold_s=slow_latency_threshold_s,
            )

            if gpu_result is not None:
                gpu_metrics = _gpu_metrics_from_result(
                    gpu_result,
                    actual_repeat_in_window,
                )
            if cpu_result is not None:
                cpu_metrics = _cpu_metrics_from_result(
                    cpu_result,
                    actual_repeat_in_window,
                )
            if self.config.dram_energy == "required" and not all(math.isfinite(cpu_metrics[field]) for field in (
                "dram_window_energy_j", "dram_energy_per_request_j", "dram_window_effective_energy_j"
            )):
                raise EnergyAbort("required DRAM RAPL measurement unavailable or incomplete")
            if resource_usage_result is not None:
                if self.config.profiling_mode == "basic":
                    if _resource_usage_err:
                        raise RuntimeError(f"required CPU/memory measurement failed: {_resource_usage_err}")
                    if not all(math.isfinite(value) for value in (
                        resource_usage_result.container_cpu_util_avg_pct,
                        resource_usage_result.container_mem_usage_avg_bytes,
                    )):
                        raise RuntimeError("required CPU/memory measurement unavailable")
                resource_usage_metrics = _resource_usage_metrics_from_result(
                    resource_usage_result,
                    actual_repeat_in_window,
                )
            resolved_input_scale = (
                effective_input_scale
                if effective_input_scale is not None
                else scale_val
            )
            input_units_per_request = _input_units_per_request(
                resolved_input_scale,
                self.config.batch_size,
            )
            efficiency_metrics = _derived_efficiency_metrics(
                gpu_mode=self.config.gpu_mode,
                batch_size=self.config.batch_size,
                latency_app_s=latency_app_s,
                output_token_count_avg=output_token_count_avg,
                gpu_energy_eff_j=gpu_metrics["gpu_energy_eff_j"],
                vcpu_energy_eff_j=cpu_metrics["vcpu_energy_eff_j"],
                input_units_per_request=input_units_per_request,
            )
            gpu_runtime_metrics = _gpu_runtime_metrics_from_result(
                resource_usage_result
            )
            if mips_result is not None:
                mips_metrics = _mips_metrics_from_result(mips_result)

            warnings = []
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
            if warnings:
                status = "warn"
                err_msg = "; ".join(warnings)

            if latency_app_s == latency_app_s and latency_app_s > 0:
                throughput = float(self.config.batch_size) / float(latency_app_s)
            else:
                throughput = float("nan")
            latency_app_s_per_input_unit = _per_positive_denominator(
                latency_app_s,
                input_units_per_request,
            )
            throughput_per_cpu_core = _per_positive_denominator(
                throughput,
                self.config.cpu_cores,
            )

        except RequestTimeoutAbort:
            raise
        except EnergyAbort:
            raise
        except MIPSAbort:
            raise
        except MonitorCleanupError as exc:
            if self._is_mips_error(exc):
                raise MIPSAbort(str(exc)) from exc
            raise
        except Exception as e:
            if self._is_mips_error(e):
                raise MIPSAbort(str(e)) from None
            status = "error"
            err_msg = repr(e)
            throughput = float("nan")
            throughput_per_cpu_core = float("nan")

        gpu_idle_stats = _idle_power_debug_stats(gpu_idle_values_so_far, "gpu")
        if self.config.idle_debug and _finite_positive(gpu_metrics["gpu_idle_power_w"]):
            gpu_idle_values_so_far.append(_to_float_or_nan(gpu_metrics["gpu_idle_power_w"]))
            gpu_idle_stats = _idle_power_debug_stats(gpu_idle_values_so_far, "gpu")

        idle_stats = _idle_debug_stats(cpu_idle_values_so_far)
        if self.config.idle_debug and _finite_positive(cpu_metrics["cpu_idle_power_w"]):
            cpu_idle_values_so_far.append(_to_float_or_nan(cpu_metrics["cpu_idle_power_w"]))
            idle_stats = _idle_debug_stats(cpu_idle_values_so_far)

        cpu_cycles_est_app = _estimate_cpu_cycles(
            latency_app_s,
            resource_usage_metrics["cpu_freq_avg_hz"],
            _to_float_or_nan(self.config.cpu_cores),
            resource_usage_metrics["container_cpu_util_avg_pct"],
        )

        row = self._build_result_row({
            "environment_class": self.collection_environment.environment,
            "gpu_device_uuid": self.config.gpu_device_uuid if self.use_energy else "nan",
            "gpu_energy_source": getattr(gpu_result, "energy_source", "unavailable") if measurement_requested(self.config.profiling_mode, "gpu_power", gpu=self.use_energy) else "not_requested",
            "gpu_energy_fallback_reason": getattr(gpu_result, "energy_fallback_reason", ""),
            "gpu_idle_energy_source": getattr(gpu_monitor, "idle_energy_source", "unavailable") if measurement_requested(self.config.profiling_mode, "gpu_power", gpu=self.use_energy) else "not_requested",
            "input_scale": str(resolved_input_scale),
            "input_units_per_request": _fmt_float(
                input_units_per_request
            ),
            "input_num_samples": _fmt_float(input_num_samples),
            "request_payload_bytes": _fmt_float(request_payload_bytes),
            "task_param": (
                executed_task_param
                if executed_task_param is not None
                else _canonical_task_param(payload_override)
            ),
            "workload_contract": json.dumps(summarize_workload_contracts(workload_contracts), ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            "output_length_avg": _fmt_float(output_length_avg),
            "output_token_count_avg": _fmt_float(
                output_token_count_avg
            ),
            "repeat_idx": str(repeat_idx),
            "warmup": str(warmup_flag),
            "repeat_in_window": str(actual_repeat_in_window),
            "latency_s": "nan",  # Placeholder: filled by merge_packet_latency
            "latency_s_per_input_unit": "nan",
            "latency_app_s": _fmt_float(latency_app_s),
            "latency_app_s_per_input_unit": _fmt_float(
                latency_app_s_per_input_unit
            ),
            "throughput_samples_per_s": _fmt_float(throughput),
            "throughput_samples_per_s_per_cpu_core": _fmt_float(
                throughput_per_cpu_core
            ),
            "gpu_idle_measured_at": gpu_idle_measured_at if self.config.idle_debug else "nan",
            "gpu_idle_rel_range_so_far": (
                _fmt_float(gpu_idle_stats["gpu_idle_rel_range_so_far"])
                if self.config.idle_debug
                else "nan"
            ),
            "dram_energy_status": (
                "not_requested" if self.config.profiling_mode == "basic" or self.config.dram_energy == "off" else
                getattr(getattr(cpu_result, "dram", None), "status", "unavailable")
            ),
            "dram_energy_error": getattr(getattr(cpu_result, "dram", None), "error", ""),
            "cpu_idle_measured_at": cpu_idle_measured_at if self.config.idle_debug else "nan",
            "cpu_idle_rel_range_so_far": (
                _fmt_float(idle_stats["cpu_idle_rel_range_so_far"])
                if self.config.idle_debug
                else "nan"
            ),
            "cpu_cycles_est_app": _fmt_float(cpu_cycles_est_app),
            "cpu_cycles_est_packet": "nan",
            "status": status,
            "error": err_msg,
        }, metric_groups=(
            gpu_metrics, cpu_metrics, efficiency_metrics, resource_usage_metrics,
            gpu_runtime_metrics, mips_metrics, latency_packet_distribution_metrics,
            latency_app_distribution_metrics,
        ), scale_entry=scale_entry, resolved_input_scale=resolved_input_scale,
            latency_app_s=latency_app_s, compute_profile_plan=compute_profile_plan,
            execution_profile_plan=execution_profile_plan)
        idle_diag_record = None
        if self.config.idle_debug:
            if idle_debug_snapshot is None:
                idle_debug_snapshot = client_diagnostics._collect_idle_debug_snapshot()
            idle_diag_record = {
                "case_name": self.config.case_name,
                "gpu_mode": self.config.gpu_mode,
                "cpu_cores": self.config.cpu_cores,
                "mem_cap_gb": self.config.mem_cap_gb,
                "input_scale": row["input_scale"],
                "warmup": row["warmup"],
                "repeat_idx": row["repeat_idx"],
                "repeat_in_window": row["repeat_in_window"],
                "sniff_group_id": sniff_group_id,
                "gpu_idle_measured_at": row["gpu_idle_measured_at"],
                "gpu_device_uuid": row["gpu_device_uuid"],
                "gpu_energy_source": row["gpu_energy_source"],
                "gpu_energy_counter_start_mj": getattr(gpu_result, "counter_start_mj", None),
                "gpu_energy_counter_end_mj": getattr(gpu_result, "counter_end_mj", None),
                "gpu_energy_duration_s": getattr(gpu_result, "measurement_duration_s", None),
                "gpu_power_samples": [[t - _gpu_samples[0][0], p] for t, p in _gpu_samples],
                "gpu_idle_power_w": _to_float_or_nan(row["gpu_idle_power_w"]),
                **gpu_idle_stats,
                **gpu_idle_trace,
                **(gpu_idle_debug_snapshot or {}),
                "cpu_idle_measured_at": row["cpu_idle_measured_at"],
                "cpu_idle_power_w": _to_float_or_nan(row["cpu_idle_power_w"]),
                **idle_stats,
                **idle_trace,
                **idle_debug_snapshot,
            }
        return row, idle_diag_record, sniff_group_id

    def main(self) -> None:
        from acprof.host.preflight import require_collection_host
        require_collection_host(profiling_mode=self.config.profiling_mode,
                                dram_energy=self.config.dram_energy)
        from acprof.artifacts import read_static_metadata
        from acprof.latency_slo import latency_slo_threshold
        slow_latency_threshold_s = latency_slo_threshold(
            read_static_metadata(ArtifactLayout.from_csv(self.config.out_csv).root)
        )
        if not self.input_scale_entries:
            self.input_scale_entries = self._load_input_scale_entries()
        self.first_predict_app_s = float("nan")

        if self.config.profiling_mode == "basic" and self.resource_usage_mod is None:
            raise RuntimeError("basic profiling requires the container CPU and memory collector")
        if self.config.gpu_mode == "on" and not self.config.gpu_device_uuid:
            raise RuntimeError("GPU_DEVICE_UUID is required: launch the client through the AC-Prof orchestrator")

        if measurement_requested(self.config.profiling_mode, "gpu_power", gpu=self.use_energy) and self.energy_mod is None:
            raise EnergyAbort(
                "GPU energy monitoring is required for gpu_mode=on but NVML/pynvml is unavailable. "
                "Install nvidia-ml-py, verify NVIDIA driver access, or rerun with --gpus off."
            )
        if measurement_requested(self.config.profiling_mode, "cpu_instructions") and self.config.use_mips and self.perf_mips_mod is None:
            raise MIPSAbort(
                "MIPS profiling is enabled but perf_mips.py could not be imported."
            )

        compute_profile_plan = _load_compute_profile_plan(self.config.compute_profile_plan_file)
        execution_profile_plan = _load_execution_profile_plan(
            self.config.execution_profile_plan_file
        )
        need_header = _is_file_empty(self.config.out_csv)
        fieldnames = CSV_FIELDS
        if not need_header:
            with open(self.config.out_csv, "r", newline="", encoding="utf-8-sig") as existing:
                fieldnames = next(csv.reader(existing))
            if len(fieldnames) != len(CSV_FIELDS) or set(fieldnames) != set(CSV_FIELDS):
                raise RuntimeError(f"existing CSV columns do not match current fields: {self.config.out_csv}; use a new output file")
        sidecar_mode = "w" if need_header else "a"
        if self.config.idle_debug:
            diag_path = self._idle_diag_path(self.config.out_csv)
            os.makedirs(os.path.dirname(diag_path) or ".", exist_ok=True)
            diag_context = open(diag_path, sidecar_mode, encoding="utf-8")
        else:
            diag_context = nullcontext(None)
        with open(self.config.out_csv, "a", newline="", encoding="utf-8") as f, open(
            self._sniff_groups_path(self.config.out_csv),
            sidecar_mode,
            encoding="utf-8",
        ) as sidecar_f, open(
            case_sidecar(self.config.out_csv, "requests"), sidecar_mode, encoding="utf-8"
        ) as requests_f, diag_context as diag_f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_MINIMAL,
            )
            if need_header:
                writer.writeheader()
                f.flush()
                os.fsync(f.fileno())

            # /ready check
            try:
                rr = requests.get(self.config.base_url + "/ready", timeout=60, headers={"Connection": "close"})
                if rr.status_code >= 400:
                    raise RuntimeError(f"/ready HTTP {rr.status_code}: {rr.text[:200]}")
            except Exception as e:
                row = {k: "nan" for k in CSV_FIELDS}
                row.update({
                    "cpu_cores": self.config.cpu_cores,
                    "mem_cap_gb": self.config.mem_cap_gb,
                    "gpu_mode": self.config.gpu_mode,
                    **self._cold_start_row_metrics(),
                    "status": "error",
                    "error": f"ready_failed: {e!r}",
                })
                client_publication._append_row(writer, row, f, sidecar_f, "")
                return

            cpu_idle_values_so_far: List[float] = []
            gpu_idle_values_so_far: List[float] = []
            for scale_entry in self.input_scale_entries:
                scale_val = float(scale_entry["input_scale"])
                payload_override = scale_entry.get("payload")
                repeat_request_limit = self._prepare_repeat_window(
                    scale_val,
                    str(scale_entry["scale_label"]),
                    payload_override,
                )
                for idx in range(self.config.warmup + self.config.repeat):
                    warmup_flag = 1 if idx < self.config.warmup else 0
                    repeat_idx = idx if warmup_flag else (idx - self.config.warmup)

                    row, idle_diag_record, sniff_group_id = self._execute_window(
                        scale_entry, repeat_request_limit, warmup_flag, repeat_idx, requests_f,
                        cpu_idle_values_so_far, gpu_idle_values_so_far, slow_latency_threshold_s,
                        compute_profile_plan, execution_profile_plan,
                    )
                    client_publication._append_row(
                        writer,
                        row,
                        f,
                        sidecar_f,
                        sniff_group_id,
                        diag_f=diag_f,
                        idle_diag_record=idle_diag_record,
                    )

    def run_cli(self) -> None:
        try:
            self.main()
        except RequestTimeoutAbort as exc:
            try:
                self._write_client_error_sidecar(exc)
            except OSError as sidecar_exc:
                print(
                    "[case][WARN] failed to persist structured timeout context: "
                    f"{sidecar_exc}",
                    file=sys.stderr,
                )
            print(f"[case][ERROR] {exc}", file=sys.stderr)
            raise SystemExit(CLIENT_REQUEST_TIMEOUT_EXIT_CODE) from None
        except MonitorCleanupError as exc:
            print(f"[monitor][ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        except EnergyAbort as exc:
            print(f"[energy][ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        except MIPSAbort as exc:
            message = str(exc)
            if message.startswith("[mips][ERROR]"):
                print(message, file=sys.stderr)
            else:
                print(f"[mips][ERROR] {message}", file=sys.stderr)
            exit_code = getattr(self.perf_mips_mod, "MIPS_EXIT_CODE", 8) if self.perf_mips_mod else 8
            raise SystemExit(exit_code) from None


def main(config: ClientConfig | None = None) -> None:
    config = ClientConfig.from_env() if config is None else config
    config.validate()
    _ensure_local_proxy_bypass()
    logging.getLogger(__name__).debug("client configuration: pipeline_tag=%s", config.pipeline_tag)
    ClientRunner(config).run_cli()


if __name__ == "__main__":
    main()
