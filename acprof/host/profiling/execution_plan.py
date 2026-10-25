"""Read and resolve Massif/Nsight Systems execution profile plans.

Execution profiling is optional.  A plan contains resource-specific cases,
and each case may contain a Massif profile, an Nsight Systems profile, or
both.  CPU-only rows consume only Massif data while GPU rows consume only
Nsight Systems data.
"""
from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from acprof.artifacts import read_json_object

INPUT_SCALE_ABS_TOLERANCE = 1e-6
MASSIF_TOOL_KEY = "massif"
NSYS_TOOL_KEY = "nsys"

CPU_HEAP_PEAK_BYTES_MASSIF = "cpu_heap_peak_bytes_massif"
CPU_HEAP_EXTRA_PEAK_BYTES_MASSIF = "cpu_heap_extra_peak_bytes_massif"
CPU_STACK_PEAK_BYTES_MASSIF = "cpu_stack_peak_bytes_massif"
CPU_HEAP_PEAK_TOTAL_BYTES_MASSIF = "cpu_heap_peak_total_bytes_massif"
CPU_HEAP_PEAK_AT_MS_MASSIF = "cpu_heap_peak_at_ms_massif"
COMPUTE_PROFILE_ERROR_MASSIF = "compute_profile_error_massif"

HOST_INFERENCE_WALL_TIME_MS_PER_REQUEST_NSYS = (
    "host_inference_wall_time_ms_per_request_nsys"
)
CUDA_API_TIME_SUM_MS_PER_REQUEST_NSYS = (
    "cuda_api_time_sum_ms_per_request_nsys"
)
CUDA_API_CALL_COUNT_PER_REQUEST_NSYS = (
    "cuda_api_call_count_per_request_nsys"
)
GPU_KERNEL_TIME_SUM_MS_PER_REQUEST_NSYS = (
    "gpu_kernel_time_sum_ms_per_request_nsys"
)
GPU_KERNEL_LAUNCH_COUNT_PER_REQUEST_NSYS = (
    "gpu_kernel_launch_count_per_request_nsys"
)
GPU_MEMCPY_TIME_SUM_MS_PER_REQUEST_NSYS = (
    "gpu_memcpy_time_sum_ms_per_request_nsys"
)
GPU_MEMCPY_COUNT_PER_REQUEST_NSYS = "gpu_memcpy_count_per_request_nsys"
GPU_MEMCPY_BYTES_PER_REQUEST_NSYS = "gpu_memcpy_bytes_per_request_nsys"
COMPUTE_PROFILE_ERROR_NSYS = "compute_profile_error_nsys"

# Short aliases match the naming convention used by compute_profile_plan.py.
MASSIF_HEAP_PEAK_FIELD = CPU_HEAP_PEAK_BYTES_MASSIF
MASSIF_HEAP_EXTRA_PEAK_FIELD = CPU_HEAP_EXTRA_PEAK_BYTES_MASSIF
MASSIF_STACK_PEAK_FIELD = CPU_STACK_PEAK_BYTES_MASSIF
MASSIF_HEAP_PEAK_TOTAL_FIELD = CPU_HEAP_PEAK_TOTAL_BYTES_MASSIF
MASSIF_PEAK_AT_MS_FIELD = CPU_HEAP_PEAK_AT_MS_MASSIF
MASSIF_ERROR_FIELD = COMPUTE_PROFILE_ERROR_MASSIF

NSYS_HOST_WALL_TIME_FIELD = HOST_INFERENCE_WALL_TIME_MS_PER_REQUEST_NSYS
NSYS_CUDA_API_TIME_FIELD = CUDA_API_TIME_SUM_MS_PER_REQUEST_NSYS
NSYS_CUDA_API_CALL_COUNT_FIELD = CUDA_API_CALL_COUNT_PER_REQUEST_NSYS
NSYS_GPU_KERNEL_TIME_FIELD = GPU_KERNEL_TIME_SUM_MS_PER_REQUEST_NSYS
NSYS_GPU_KERNEL_LAUNCH_COUNT_FIELD = (
    GPU_KERNEL_LAUNCH_COUNT_PER_REQUEST_NSYS
)
NSYS_GPU_MEMCPY_TIME_FIELD = GPU_MEMCPY_TIME_SUM_MS_PER_REQUEST_NSYS
NSYS_GPU_MEMCPY_COUNT_FIELD = GPU_MEMCPY_COUNT_PER_REQUEST_NSYS
NSYS_GPU_MEMCPY_BYTES_FIELD = GPU_MEMCPY_BYTES_PER_REQUEST_NSYS
NSYS_ERROR_FIELD = COMPUTE_PROFILE_ERROR_NSYS

MASSIF_METRIC_FIELDS = (
    MASSIF_HEAP_PEAK_FIELD,
    MASSIF_HEAP_EXTRA_PEAK_FIELD,
    MASSIF_STACK_PEAK_FIELD,
    MASSIF_HEAP_PEAK_TOTAL_FIELD,
    MASSIF_PEAK_AT_MS_FIELD,
)
NSYS_METRIC_FIELDS = (
    NSYS_HOST_WALL_TIME_FIELD,
    NSYS_CUDA_API_TIME_FIELD,
    NSYS_CUDA_API_CALL_COUNT_FIELD,
    NSYS_GPU_KERNEL_TIME_FIELD,
    NSYS_GPU_KERNEL_LAUNCH_COUNT_FIELD,
    NSYS_GPU_MEMCPY_TIME_FIELD,
    NSYS_GPU_MEMCPY_COUNT_FIELD,
    NSYS_GPU_MEMCPY_BYTES_FIELD,
)
MASSIF_FIELDS = MASSIF_METRIC_FIELDS + (MASSIF_ERROR_FIELD,)
NSYS_FIELDS = NSYS_METRIC_FIELDS + (NSYS_ERROR_FIELD,)
EXECUTION_PROFILE_FIELDS = MASSIF_FIELDS + NSYS_FIELDS


def _empty_result() -> dict[str, Any]:
    result: dict[str, Any] = {
        field: float("nan")
        for field in MASSIF_METRIC_FIELDS + NSYS_METRIC_FIELDS
    }
    result[MASSIF_ERROR_FIELD] = ""
    result[NSYS_ERROR_FIELD] = ""
    return result


def _finite_float_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def load_execution_profile_plan(path: str) -> dict[str, Any]:
    """Load an execution profile plan without raising client-side errors.

    An empty path means that execution profiling was deliberately disabled.
    Other read or schema failures are retained as ``_load_error`` so the
    applicable tool's error column can expose the problem.
    """
    if not path:
        return {"profiles": [], "_disabled": True}
    if not os.path.exists(path):
        return {
            "profiles": [],
            "_load_error": f"execution_profile_plan_not_found:{path}",
        }
    try:
        plan = read_json_object(path, label="execution profile plan")
    except (OSError, ValueError) as exc:
        return {
            "profiles": [],
            "_load_error": f"execution_profile_plan_invalid:{exc!r}",
        }
    if not isinstance(plan.get("profiles"), list):
        return {
            "profiles": [],
            "_load_error": "execution_profile_plan_invalid:missing_profiles",
        }
    return plan


def _normalize_gpu_mode(value: Any) -> str:
    return str(value).strip().lower()


def _integer_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _resource_case_matches(
    profile: Mapping[str, Any],
    *,
    cpu_cores: int | str,
    mem_cap_gb: int | str,
    gpu_mode: str,
) -> bool:
    profile_cpu = _integer_value(profile.get("cpu_cores"))
    target_cpu = _integer_value(cpu_cores)
    profile_mem = _integer_value(profile.get("mem_cap_gb"))
    target_mem = _integer_value(mem_cap_gb)
    return (
        profile_cpu is not None
        and target_cpu is not None
        and profile_cpu == target_cpu
        and profile_mem is not None
        and target_mem is not None
        and profile_mem == target_mem
        and _normalize_gpu_mode(profile.get("gpu_mode")) == gpu_mode
    )


def _tool_profile_enabled(profile: Any) -> bool:
    if not isinstance(profile, Mapping):
        return False
    enabled = profile.get("enabled")
    if enabled is not None:
        if isinstance(enabled, str):
            return enabled.strip().lower() not in {
                "",
                "0",
                "false",
                "no",
                "off",
                "disabled",
            }
        return bool(enabled)
    status = str(profile.get("status", "")).strip().lower()
    return status not in {"disabled", "not_enabled", "off"}


def _tool_names(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {
            str(key).strip().lower()
            for key, enabled in value.items()
            if (
                isinstance(enabled, Mapping)
                and _tool_profile_enabled(enabled)
            )
            or (not isinstance(enabled, Mapping) and bool(enabled))
        }
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().lower() for item in value}
    if isinstance(value, str):
        return {
            item.strip().lower()
            for item in value.replace(";", ",").split(",")
            if item.strip()
        }
    return set()


def _enabled_tools(plan: Mapping[str, Any]) -> set[str]:
    enabled: set[str] = set()
    for key in ("enabled_tools", "execution_profile_tools"):
        enabled.update(_tool_names(plan.get(key)))

    profiles = plan.get("profiles")
    if not isinstance(profiles, list):
        return enabled
    for profile in profiles:
        if not isinstance(profile, Mapping):
            continue
        tools = profile.get("tools")
        if not isinstance(tools, Mapping):
            continue
        for tool_name, tool_profile in tools.items():
            if _tool_profile_enabled(tool_profile):
                enabled.add(str(tool_name).strip().lower())
    return enabled


def _find_resource_case(
    plan: Mapping[str, Any],
    *,
    cpu_cores: int | str,
    mem_cap_gb: int | str,
    gpu_mode: str,
    tool_key: str,
) -> Mapping[str, Any] | None:
    profiles = plan.get("profiles")
    if not isinstance(profiles, list):
        return None

    first_match: Mapping[str, Any] | None = None
    for profile in profiles:
        if not isinstance(profile, Mapping):
            continue
        if not _resource_case_matches(
            profile,
            cpu_cores=cpu_cores,
            mem_cap_gb=mem_cap_gb,
            gpu_mode=gpu_mode,
        ):
            continue
        if first_match is None:
            first_match = profile
        tools = profile.get("tools")
        if isinstance(tools, Mapping) and tool_key in tools:
            return profile
    return first_match


def _profile_error(profile: Mapping[str, Any], error_field: str) -> str:
    if error_field in profile:
        return str(profile.get(error_field) or "")
    return str(profile.get("error") or "")


def _scale_label(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _resolve_tool_entry(
    tool_profile: Mapping[str, Any],
    *,
    tool_key: str,
    error_field: str,
    input_scale: float,
) -> tuple[Mapping[str, Any], str]:
    profile_error = _profile_error(tool_profile, error_field)
    entries = tool_profile.get("entries")
    if not isinstance(entries, list):
        return (
            {},
            profile_error or f"execution_profile_missing_entries:{tool_key}",
        )

    target_scale = _finite_float_or_nan(input_scale)
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_scale = _finite_float_or_nan(entry.get("input_scale"))
        if not (
            math.isfinite(target_scale)
            and math.isfinite(entry_scale)
            and math.isclose(
                entry_scale,
                target_scale,
                rel_tol=0.0,
                abs_tol=INPUT_SCALE_ABS_TOLERANCE,
            )
        ):
            continue
        if error_field in entry:
            entry_error = str(entry.get(error_field) or "")
        elif "error" in entry:
            entry_error = str(entry.get("error") or "")
        else:
            entry_error = profile_error
        return entry, entry_error

    return (
        {},
        f"execution_profile_missing_scale:{tool_key}:{_scale_label(input_scale)}",
    )


def _missing_case_error(
    *,
    tool_key: str,
    cpu_cores: int | str,
    mem_cap_gb: int | str,
    gpu_mode: str,
) -> str:
    return (
        f"execution_profile_missing_case:{tool_key}:"
        f"cpu_cores={cpu_cores},mem_cap_gb={mem_cap_gb},gpu_mode={gpu_mode}"
    )


def _apply_entry(
    result: dict[str, Any],
    *,
    entry: Mapping[str, Any],
    metric_fields: Iterable[str],
    error_field: str,
    error: str,
) -> None:
    for field in metric_fields:
        result[field] = _finite_float_or_nan(entry.get(field))
    result[error_field] = error


@dataclass(frozen=True)
class _ResolvedProfileEntry:
    scale: float
    metrics: tuple[float, ...]
    error: str


@dataclass(frozen=True)
class _ResolvedToolProfile:
    enabled: bool
    entries: tuple[_ResolvedProfileEntry, ...] | None
    error: str = ""


def _snapshot_tool_profile(value: Any, tool_key: str) -> _ResolvedToolProfile | None:
    if not isinstance(value, Mapping):
        return None
    if not _tool_profile_enabled(value):
        return _ResolvedToolProfile(False, None)
    error_field = NSYS_ERROR_FIELD if tool_key == NSYS_TOOL_KEY else MASSIF_ERROR_FIELD
    metric_fields = NSYS_METRIC_FIELDS if tool_key == NSYS_TOOL_KEY else MASSIF_METRIC_FIELDS
    profile_error = _profile_error(value, error_field)
    entries = value.get("entries")
    if not isinstance(entries, list):
        return _ResolvedToolProfile(
            True, None, profile_error or f"execution_profile_missing_entries:{tool_key}"
        )
    resolved = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        scale = _finite_float_or_nan(entry.get("input_scale"))
        if not math.isfinite(scale):
            continue
        if error_field in entry:
            error = str(entry.get(error_field) or "")
        elif "error" in entry:
            error = str(entry.get("error") or "")
        else:
            error = profile_error
        resolved.append(_ResolvedProfileEntry(
            scale,
            tuple(_finite_float_or_nan(entry.get(field)) for field in metric_fields),
            error,
        ))
    return _ResolvedToolProfile(True, tuple(resolved))


class ExecutionProfileResolver:
    """Resolve many result rows from one compact, immutable plan snapshot.

    Only the selected tools' scalar result fields are retained; large reports,
    provenance payloads and unrelated metadata are not copied. Rebuild after
    resume/backfill merges a new plan. Each query returns its own result dict.
    """

    def __init__(self, plan: Any) -> None:
        self._disabled = False
        self._load_error = ""
        self._enabled: frozenset[str] = frozenset()
        self._cases: dict[tuple[int, int, str], _ResolvedToolProfile | None] = {}
        if not isinstance(plan, Mapping):
            self._load_error = "execution_profile_plan_invalid:not_dict"
            return
        self._disabled = bool(plan.get("_disabled"))
        if self._disabled:
            return
        self._load_error = str(plan.get("_load_error") or "")
        if self._load_error:
            return
        self._enabled = frozenset(_enabled_tools(plan))
        profiles = plan.get("profiles")
        if not isinstance(profiles, list):
            return
        for profile in profiles:
            if not isinstance(profile, Mapping):
                continue
            cpu = _integer_value(profile.get("cpu_cores"))
            memory = _integer_value(profile.get("mem_cap_gb"))
            if cpu is None or memory is None:
                continue
            mode = _normalize_gpu_mode(profile.get("gpu_mode"))
            tool_key = NSYS_TOOL_KEY if mode == "on" else MASSIF_TOOL_KEY
            tools = profile.get("tools")
            if not isinstance(tools, Mapping) or tool_key not in tools:
                continue
            key = (cpu, memory, mode)
            # Presence wins even when the first matching tool is invalid or disabled.
            if key not in self._cases:
                self._cases[key] = _snapshot_tool_profile(tools[tool_key], tool_key)

    def resolve(
        self,
        cpu_cores: int | str,
        mem_cap_gb: int | str,
        gpu_mode: str,
        input_scale: float,
    ) -> dict[str, Any]:
        result = _empty_result()
        mode = _normalize_gpu_mode(gpu_mode)
        tool_key = NSYS_TOOL_KEY if mode == "on" else MASSIF_TOOL_KEY
        error_field = NSYS_ERROR_FIELD if tool_key == NSYS_TOOL_KEY else MASSIF_ERROR_FIELD
        metric_fields = NSYS_METRIC_FIELDS if tool_key == NSYS_TOOL_KEY else MASSIF_METRIC_FIELDS
        if self._disabled:
            return result
        if self._load_error:
            result[error_field] = self._load_error
            return result
        if tool_key not in self._enabled:
            return result
        cpu = _integer_value(cpu_cores)
        memory = _integer_value(mem_cap_gb)
        profile = self._cases.get((cpu, memory, mode)) if cpu is not None and memory is not None else None
        if profile is None:
            result[error_field] = _missing_case_error(
                tool_key=tool_key, cpu_cores=cpu_cores, mem_cap_gb=mem_cap_gb, gpu_mode=mode
            )
            return result
        if not profile.enabled:
            return result
        if profile.entries is None:
            result[error_field] = profile.error
            return result
        target_scale = _finite_float_or_nan(input_scale)
        if math.isfinite(target_scale):
            for entry in profile.entries:
                if math.isclose(entry.scale, target_scale, rel_tol=0.0, abs_tol=INPUT_SCALE_ABS_TOLERANCE):
                    result.update(zip(metric_fields, entry.metrics, strict=True))
                    result[error_field] = entry.error
                    return result
        result[error_field] = f"execution_profile_missing_scale:{tool_key}:{_scale_label(input_scale)}"
        return result


def find_execution_profile_entry(
    plan: dict[str, Any] | ExecutionProfileResolver,
    cpu_cores: int | str,
    mem_cap_gb: int | str,
    gpu_mode: str,
    input_scale: float,
) -> dict[str, Any]:
    """Return the flat execution-profile fields for one result row.

    A CPU-only row (``gpu_mode="off"``) applies only Massif.  A GPU row
    (``gpu_mode="on"``) applies only Nsight Systems.  A tool that is absent
    from the whole plan is treated as intentionally disabled. Reuse an
    ExecutionProfileResolver when resolving multiple rows from the same plan.
    """
    if isinstance(plan, ExecutionProfileResolver):
        return plan.resolve(cpu_cores, mem_cap_gb, gpu_mode, input_scale)
    result = _empty_result()
    normalized_gpu_mode = _normalize_gpu_mode(gpu_mode)
    tool_key = (
        NSYS_TOOL_KEY if normalized_gpu_mode == "on" else MASSIF_TOOL_KEY
    )
    error_field = (
        NSYS_ERROR_FIELD if tool_key == NSYS_TOOL_KEY else MASSIF_ERROR_FIELD
    )
    metric_fields = (
        NSYS_METRIC_FIELDS
        if tool_key == NSYS_TOOL_KEY
        else MASSIF_METRIC_FIELDS
    )

    if not isinstance(plan, Mapping):
        result[error_field] = "execution_profile_plan_invalid:not_dict"
        return result
    if plan.get("_disabled"):
        return result
    load_error = str(plan.get("_load_error") or "")
    if load_error:
        result[error_field] = load_error
        return result
    if tool_key not in _enabled_tools(plan):
        return result

    resource_case = _find_resource_case(
        plan,
        cpu_cores=cpu_cores,
        mem_cap_gb=mem_cap_gb,
        gpu_mode=normalized_gpu_mode,
        tool_key=tool_key,
    )
    if resource_case is None:
        result[error_field] = _missing_case_error(
            tool_key=tool_key,
            cpu_cores=cpu_cores,
            mem_cap_gb=mem_cap_gb,
            gpu_mode=normalized_gpu_mode,
        )
        return result

    tools = resource_case.get("tools")
    tool_profile = (
        tools.get(tool_key) if isinstance(tools, Mapping) else None
    )
    if isinstance(tool_profile, Mapping) and not _tool_profile_enabled(
        tool_profile
    ):
        return result
    if not isinstance(tool_profile, Mapping):
        result[error_field] = _missing_case_error(
            tool_key=tool_key,
            cpu_cores=cpu_cores,
            mem_cap_gb=mem_cap_gb,
            gpu_mode=normalized_gpu_mode,
        )
        return result

    entry, error = _resolve_tool_entry(
        tool_profile,
        tool_key=tool_key,
        error_field=error_field,
        input_scale=input_scale,
    )
    _apply_entry(
        result,
        entry=entry,
        metric_fields=metric_fields,
        error_field=error_field,
        error=error,
    )
    return result
