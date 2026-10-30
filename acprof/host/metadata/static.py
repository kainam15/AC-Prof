"""主机、模型与 profiler 计划的静态元数据。"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any

from acprof.artifacts import atomic_write_json, read_json_object
from acprof.config import STATIC_META_FIELDS, STATIC_META_SCHEMA_VERSION
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import (
    PlannedInputScales,
)
from acprof.host.metadata import machine as machine_metadata
from acprof.host.metadata.gpu import probe_gpu_metadata
from acprof.host.metadata.platform import operating_system_label
from acprof.host.model_schema import (
    _inference_precision_by_device,
    _model_io_formats,
)
from acprof.monitors.rapl_topology import discover_rapl_topology
from acprof.platform import detect_environment


@dataclass
class StaticMeta:
    model_name: str
    model_revision: str
    task_family: str
    pipeline_tag: str
    runtime_backend: str
    image_tag: str
    batch_size: int
    input_scale_type: str
    run_command: str
    model_download_url: str
    gpu: str
    gpu_mem_total_bytes: int | None
    model_cache_bytes: int | None
    docker_image_bytes: int | None
    environment: str
    cpu_power_source: str
    vcpu_power_method: str
    cpu_governor: str
    cpu_boost: str
    image_id: str = ""
    image_name: str = ""
    model_storage_mode: str = "baked"
    model_artifact_bytes: int | None = None
    runtime_image_bytes: int | None = None
    total_deployment_bytes: int | None = None
    runtime_environment: dict[str, Any] = field(default_factory=dict)
    runtime_kind: str = "docker"
    runtime_id: str = ""
    runtime_selection: dict[str, Any] = field(default_factory=dict)
    resource_allocation: dict[str, Any] = field(default_factory=dict)
    collection_plan: dict[str, Any] = field(default_factory=dict)
    runtime_validation: dict[str, Any] = field(default_factory=dict)
    latency_slo: dict[str, Any] = field(
        default_factory=lambda: {"threshold_s": None, "source": "unconfigured"}
    )
    gpu_device: dict[str, Any] = field(default_factory=dict)
    gpu_metadata: dict[str, Any] = field(default_factory=dict)
    profiling_mode: str = "full"
    capability_report: dict[str, Any] = field(default_factory=dict)
    platform: dict[str, Any] = field(default_factory=dict)
    collection_tier: str = "unknown"
    comparability_class: str = "unknown"
    environment_class: str = "unknown"
    platform_runtime: dict[str, Any] = field(default_factory=dict)
    rapl_topology: dict[str, Any] = field(default_factory=dict)
    cgroup_version: str = "unknown"
    cgroup_collection_mode: str = "unknown"
    host_mem_total_bytes: int | None = None
    host_swap_total_bytes: int | None = None
    host_swap_used_bytes_at_start: int | None = None
    host_swap_type: str = "unknown"
    host_vm_swappiness: int | None = None
    docker_storage_total_bytes: int | None = None
    docker_storage_available_bytes_at_start: int | None = None
    docker_storage_filesystem: str = "unknown"
    docker_storage_device: str = "unknown"
    docker_storage_type: str = "unknown"
    workload: dict[str, Any] = field(default_factory=dict)
    input_scale_plan_sha256: str = ""
    schema_version: int = STATIC_META_SCHEMA_VERSION
    parameter_count: int | None = None
    parameter_bytes: int | None = None
    precision_dtype: str | None = None
    parameter_dtype_counts: dict[str, int] = field(default_factory=dict)
    inference_precision_by_device: dict[str, str] = field(default_factory=dict)
    static_flops: dict[str, Any] | None = None
    static_macs: dict[str, Any] | None = None
    input_format: dict[str, Any] = field(default_factory=dict)
    output_format: dict[str, Any] = field(default_factory=dict)
    quantized: bool | None = None
    quantization_method: str | None = None
    quantization_config: dict[str, Any] = field(default_factory=dict)
    model_license: str | None = None
    model_metadata_source: str | None = None
    model_source: str = "huggingface"
    model_resolution: dict[str, Any] = field(default_factory=dict)
    compute_profile_tools: list[str] = field(default_factory=list)
    torch_profiler_eager_flop_semantics: str = ""
    torch_profiler_eager_attention_implementation: str = ""
    torch_profiler_eager_repeat_cpu: int | None = None
    torch_profiler_eager_repeat_gpu: int | None = None
    ncu_flop_semantics: str = ""
    ncu_flop_counting_version: int | None = None
    ncu_repeat: int | None = None
    ncu_fma_flop_weight: float | None = None
    ncu_metrics: list[str] = field(default_factory=list)
    torch_version: str = ""
    transformers_version: str = ""
    ncu_version: str = ""
    gpu_compute_capability: str = "unknown"
    gpu_sm_count: Any = None
    gpu_compute_metadata: list[dict[str, Any]] = field(default_factory=list)
    compute_profiles_retained: bool = False
    compute_profile_provenance: str = ""
    execution_profile_schema_version: int | None = None
    execution_profile_tools: list[str] = field(default_factory=list)
    massif_peak_semantics: str = ""
    massif_repeat: int | None = None
    massif_version: str = ""
    massif_sampling_strategy: str = ""
    massif_reference_cpu_cores: int | None = None
    massif_reference_mem_cap_gb: int | None = None
    massif_reused_across_resource_cases: bool = False
    nsys_timeline_semantics: str = ""
    nsys_repeat: int | None = None
    nsys_version: str = ""
    nsys_sampling_strategy: str = ""
    nsys_reference_cpu_cores: int | None = None
    nsys_reference_mem_cap_gb: int | None = None
    nsys_reused_across_resource_cases: bool = False
    execution_profiles_retained: bool = False
    execution_profile_provenance: str = ""


def _build_model_download_url(model_id: str) -> str:
    """Return the canonical Hugging Face model URL."""
    return f"https://huggingface.co/{model_id}"


def collect_static_meta(
    task_info: TaskInfo,
    image_info: ImageInfo,
    batch_size: int,
    input_scale_type: str,
    run_command: str = "",
    device_index: int = 0,
    cgroup_version: str = "unknown",
    cgroup_collection_mode: str = "unknown",
    compute_profile_enabled: bool = True,
    execution_profile_enabled: bool = False,
    profiling_mode: str = "full",
    gpu_device: dict[str, Any] | None = None,
) -> StaticMeta:
    """Collect static metadata for the current model/image pair."""
    from acprof.capabilities import measurement_requested
    from acprof.host.metadata.platform import collect_platform_metadata
    from acprof.host.runtimes.contracts import runtime_identity, runtime_kind
    from acprof.installation import resource_root
    process = runtime_kind(image_info) == 'process'
    environment = detect_environment()
    cpu_power_source, vcpu_power_method = (
        machine_metadata.cpu_power_metadata() if measurement_requested(profiling_mode, "cpu_energy")
        else ("not_requested", "not_requested")
    )
    cpu_governor, cpu_boost = machine_metadata.cpu_frequency_policy_metadata()
    host_swap = machine_metadata.host_swap_metadata()
    docker_storage: dict[str, Any] = dict.fromkeys(('docker_storage_total_bytes', 'docker_storage_available_bytes_at_start'), None) if process else machine_metadata.docker_storage_metadata()
    if process:
        docker_storage.update(docker_storage_filesystem='not_applicable', docker_storage_device='not_applicable', docker_storage_type='not_applicable')
    input_format, output_format = _model_io_formats(task_info)
    manifest = getattr(image_info, "runtime_environment", {})
    store = manifest.get("model_store")
    model_bytes = store["model_artifact_bytes"] if store else None if process else machine_metadata.docker_model_cache_bytes(image_info.tag)
    image_bytes = None if process else machine_metadata.docker_image_size_bytes(image_info.tag)
    gpu_probe = probe_gpu_metadata(gpu_device=gpu_device, device_index=device_index)
    gpu_memory = gpu_probe["fields"]["memory_total_bytes"]["value"]
    static_meta = StaticMeta(
        **environment.metadata(),
        platform_runtime=collect_platform_metadata(environment, resource_root(), runtime_kind=runtime_kind(image_info)),
        runtime_kind=runtime_kind(image_info), runtime_id=runtime_identity(image_info),
        model_name=task_info.model_id,
        profiling_mode=profiling_mode,
        model_revision=task_info.model_revision,
        parameter_count=task_info.parameter_count,
        parameter_bytes=task_info.parameter_bytes,
        precision_dtype=task_info.precision_dtype,
        parameter_dtype_counts=dict(task_info.parameter_dtype_counts),
        inference_precision_by_device=_inference_precision_by_device(task_info),
        input_format=input_format,
        output_format=output_format,
        quantized=task_info.quantized,
        quantization_method=task_info.quantization_method,
        quantization_config=dict(task_info.quantization_config),
        model_license=task_info.model_license,
        model_metadata_source=task_info.model_metadata_source,
        model_source=getattr(task_info, "model_source", "huggingface"),
        model_resolution=dict(task_info.model_resolution),
        task_family=task_info.task_family,
        pipeline_tag=task_info.pipeline_tag,
        runtime_backend=task_info.runtime_backend,
        image_tag='' if process else image_info.tag,
        image_id='' if process else image_info.tag if image_info.tag.startswith("sha256:") else "",
        image_name=getattr(image_info, "name", ""),
        runtime_environment=dict(getattr(image_info, "runtime_environment", {})),
        batch_size=batch_size,
        input_scale_type=input_scale_type,
        run_command=run_command,
        model_download_url=(f"https://modelscope.cn/models/{task_info.model_id}"
                            if getattr(task_info, "model_source", "huggingface") == "modelscope"
                            else _build_model_download_url(task_info.model_id)),
        gpu_device=dict(gpu_device or {}),
        gpu_metadata=dict(gpu_probe),
        gpu=str(gpu_probe["fields"]["name"]["value"] or "unknown"),
        gpu_mem_total_bytes=gpu_memory if isinstance(gpu_memory, int) else None,
        host_mem_total_bytes=machine_metadata.host_mem_total_bytes(),
        host_swap_total_bytes=host_swap["host_swap_total_bytes"],
        host_swap_used_bytes_at_start=host_swap[
            "host_swap_used_bytes_at_start"
        ],
        host_swap_type=host_swap["host_swap_type"],
        host_vm_swappiness=host_swap["host_vm_swappiness"],
        model_cache_bytes=model_bytes,
        docker_image_bytes=image_bytes,
        model_storage_mode="mounted" if store else "baked",
        model_artifact_bytes=model_bytes if store else None,
        runtime_image_bytes=image_bytes if store else None,
        total_deployment_bytes=(model_bytes + image_bytes if store and model_bytes is not None and image_bytes is not None
                                else image_bytes),
        docker_storage_total_bytes=docker_storage[
            "docker_storage_total_bytes"
        ],
        docker_storage_available_bytes_at_start=docker_storage[
            "docker_storage_available_bytes_at_start"
        ],
        docker_storage_filesystem=docker_storage[
            "docker_storage_filesystem"
        ],
        docker_storage_device=docker_storage["docker_storage_device"],
        docker_storage_type=docker_storage["docker_storage_type"],
        environment=operating_system_label(),
        cgroup_version=cgroup_version,
        cgroup_collection_mode=cgroup_collection_mode,
        cpu_power_source=cpu_power_source,
        vcpu_power_method=vcpu_power_method,
        rapl_topology=discover_rapl_topology() if environment.native else {},
        cpu_governor=cpu_governor,
        cpu_boost=cpu_boost,
    )
    disabled_metadata: dict[str, Any] = {}
    if not compute_profile_enabled:
        disabled_metadata.update({
            "compute_profile_tools": [],
            "torch_profiler_eager_flop_semantics": (
                "logical_operator_shape_flops"
            ),
            "torch_profiler_eager_attention_implementation": "eager",
            "ncu_flop_semantics": (
                "gpu_executed_floating_point_operations"
            ),
            "ncu_fma_flop_weight": 2,
            "ncu_metrics": [],
            "torch_version": "unknown",
            "transformers_version": "unknown",
            "ncu_version": "unknown",
            "gpu_compute_capability": "unknown",
            "gpu_sm_count": "unknown",
            "compute_profiles_retained": False,
            "compute_profile_provenance": "disabled",
        })
    if not execution_profile_enabled:
        disabled_metadata.update({
            "execution_profile_schema_version": 1,
            "execution_profile_tools": [],
            "massif_peak_semantics": (
                "process_lifetime_heap_peak_including_model_load_and_warmup"
            ),
            "massif_version": "unknown",
            "nsys_timeline_semantics": "nvtx_acprof_compute_range",
            "nsys_version": "unknown",
            "execution_profiles_retained": False,
            "execution_profile_provenance": "disabled",
        })
    if getattr(image_info, "runtime_environment", {}):
        packages = image_info.runtime_environment.get("packages", {})
        disabled_metadata.update({
            "torch_version": packages.get("torch", "unknown"),
            "transformers_version": packages.get("transformers", "unknown"),
        })
    return (
        enrich_static_meta(static_meta, disabled_metadata)
        if disabled_metadata
        else static_meta
    )


def enrich_static_meta(
    static_meta: StaticMeta,
    metadata: dict[str, Any],
) -> StaticMeta:
    """Return static metadata enriched with recognized profiling fields."""
    updates: dict[str, Any] = {}
    for field_name in STATIC_META_FIELDS:
        if field_name not in metadata:
            continue
        value = metadata[field_name]
        if isinstance(value, tuple):
            value = list(value)
        updates[field_name] = value
    return replace(static_meta, **updates) if updates else static_meta


def enrich_static_meta_from_input_plan(
    static_meta: StaticMeta,
    planned: PlannedInputScales,
) -> StaticMeta:
    """Attach the exact workload provenance used to build the payload plan."""
    if not isinstance(static_meta, StaticMeta):
        raise TypeError("static_meta must be a StaticMeta instance")
    return replace(
        static_meta,
        workload=dict(planned.workload),
        input_scale_type=str(planned.workload.get("input_scale_type") or static_meta.input_scale_type),
        input_scale_plan_sha256=str(planned.plan_sha256 or ""),
    )


def _static_flops_from_compute_plan(
    plan: dict[str, Any],
    static_meta: StaticMeta,
) -> dict[str, Any] | None:
    profiles = plan.get("profiles", {})
    if not isinstance(profiles, dict):
        return None

    for profile_name in ("gpu", "cpu"):
        profile_group = profiles.get(profile_name, {})
        if not isinstance(profile_group, dict):
            continue
        torch_profile = profile_group.get("torch_profiler_eager", {})
        if not isinstance(torch_profile, dict):
            continue
        entries = torch_profile.get("entries", [])
        if not isinstance(entries, list):
            continue

        values: list[dict[str, Any]] = []
        seen_scales = set()
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("error"):
                continue
            try:
                input_scale = float(entry["input_scale"])
                mflop_per_request = float(
                    entry[
                        "model_logical_mflop_per_request_torch_profiler_eager"
                    ]
                )
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not math.isfinite(input_scale)
                or not math.isfinite(mflop_per_request)
                or mflop_per_request < 0
            ):
                continue
            normalized_scale: Any = (
                int(input_scale) if input_scale.is_integer() else input_scale
            )
            if normalized_scale in seen_scales:
                continue
            seen_scales.add(normalized_scale)
            values.append({
                "input_scale": normalized_scale,
                "flops_per_request": int(round(mflop_per_request * 1_000_000)),
            })

        if values:
            values.sort(key=lambda item: float(item["input_scale"]))
            static_metadata = plan.get("static_metadata", {})
            semantics = torch_profile.get("flop_semantics")
            if not semantics and isinstance(static_metadata, dict):
                semantics = static_metadata.get(
                    "torch_profiler_eager_flop_semantics"
                )
            return {
                "source": "torch_profiler_eager",
                "profile": profile_name,
                "semantics": semantics or "logical_operator_shape_flops",
                "unit": "FLOP/request",
                "input_scale_type": static_meta.input_scale_type,
                "batch_size": static_meta.batch_size,
                "values": values,
            }
    return None


def enrich_static_meta_from_compute_plan(
    static_meta: StaticMeta,
    plan_path: str,
) -> StaticMeta:
    """Read compute-profile metadata without making a failed probe fatal."""
    if not plan_path or not os.path.exists(plan_path):
        return static_meta
    try:
        plan = read_json_object(plan_path, label="compute profile plan")
    except (OSError, ValueError, TypeError) as exc:
        print(f"[meta][WARN] Cannot read compute profile metadata: {exc}")
        return static_meta
    metadata = plan.get("static_metadata", {})
    if not isinstance(metadata, dict):
        print("[meta][WARN] compute_profile_plan static_metadata is not an object")
        return static_meta
    enriched = enrich_static_meta(static_meta, metadata)
    static_flops = _static_flops_from_compute_plan(plan, enriched)
    if static_flops is not None:
        enriched = replace(enriched, static_flops=static_flops)
    return enriched


def enrich_static_meta_from_execution_plan(
    static_meta: StaticMeta,
    plan_path: str,
) -> StaticMeta:
    """Read execution-profile metadata without making a failed probe fatal."""
    if not plan_path or not os.path.exists(plan_path):
        return static_meta
    try:
        plan = read_json_object(plan_path, label="execution profile plan")
    except (OSError, ValueError, TypeError) as exc:
        print(f"[meta][WARN] Cannot read execution profile metadata: {exc}")
        return static_meta
    metadata = plan.get("static_metadata", {})
    if not isinstance(metadata, dict):
        print("[meta][WARN] execution_profile_plan static_metadata is not an object")
        return static_meta
    return enrich_static_meta(static_meta, metadata)


def write_static_meta_json(static_meta: StaticMeta, output_path: str) -> None:
    """Atomically write static metadata as one JSON object."""
    payload = {
        field: getattr(static_meta, field)
        for field in STATIC_META_FIELDS
    }

    atomic_write_json(output_path, payload)
    print(f"[meta] Static meta JSON: {output_path}")
