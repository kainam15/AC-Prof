"""Select input scale policies while preserving reproducible workload identity."""

import math

from acprof.config import SCALING_DIMENSIONS
from acprof.extensions import CATALOG
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan.families import (
    default_family_max_scale,
    plan_audio_scales,
    plan_manual_nlp_scales,
    plan_nlp_auto_scales,
    plan_timeseries_scales,
)
from acprof.host.input_plan.materialize import get_task_generator, materialize_scale_plan
from acprof.host.input_plan.scales import (
    AUTO_INPUT_SCALE_COUNT,
    float_auto_scales,
    integer_auto_scales,
    resolve_input_scales,
    serialize_input_scales,
)
from acprof.host.input_plan.types import PlannedInputScales


def plan_input_scales(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: list[int],
    mem_list: list[int],
    gpu_list: list[str],
    batch_size: int,
    output_dir: str,
    input_scales: str | None = None,
    workload_spec_path: str | None = None,
    input_scale_policy: str = "auto",
) -> PlannedInputScales:
    """Select a policy while retaining model-specific validation and payload bytes."""
    if input_scale_policy not in {"auto", "minimal"}:
        raise ValueError("input_scale_policy must be auto or minimal")
    minimal = input_scale_policy == "minimal" and not input_scales
    if minimal:
        generator = get_task_generator(
            task_info, batch_size, workload_spec_path=workload_spec_path,
        )
        candidates = generator.default_input_scales()
        if not candidates:
            capabilities = CATALOG.describe(task_info).declaration.input_plan
            family = "nlp" if capabilities.text_payload else task_info.task_family
            candidates = resolve_input_scales(family)
        minimum = min(float(value) for value in candidates)
        if not math.isfinite(minimum) or minimum <= 0:
            raise ValueError("workload has no positive finite minimal scale")
        input_scales = serialize_input_scales([minimum])
    plan = _plan_input_scales(
        task_info,
        image_info,
        cpu_list,
        mem_list,
        gpu_list,
        batch_size,
        output_dir,
        input_scales,
        workload_spec_path,
    )
    if minimal:
        plan.source = "minimal"
    return plan


def input_plan_summary(task_info: TaskInfo, plan: PlannedInputScales) -> dict:
    """Small presentation event derived from the completed authoritative plan."""
    axis = plan.workload.get("input_scale_type") or plan.workload.get("input_scale", {}).get("type")
    if not axis:
        capabilities = CATALOG.describe(task_info).declaration.input_plan
        axis = "seq_length" if capabilities.text_payload else SCALING_DIMENSIONS[task_info.task_family].param_name
    return {"scales": list(plan.scales), "scale_type": axis}


def _plan_input_scales(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: list[int],
    mem_list: list[int],
    gpu_list: list[str],
    batch_size: int,
    output_dir: str,
    input_scales: str | None = None,
    workload_spec_path: str | None = None,
) -> PlannedInputScales:
    capabilities = CATALOG.describe(task_info).declaration.input_plan
    if workload_spec_path and not capabilities.accepts_workload_spec:
        raise ValueError(f"--workload-spec is not declared for {task_info.pipeline_tag}")
    if capabilities.requires_scale_meta:
        return plan_timeseries_scales(
            task_info,
            image_info,
            cpu_list,
            mem_list,
            gpu_list,
            batch_size,
            output_dir,
            input_scales,
        )

    if input_scales:
        manual_scales = resolve_input_scales(task_info.task_family, input_scales=input_scales)
        if capabilities.audio_payload:
            return plan_audio_scales(
                task_info=task_info,
                image_info=image_info,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                scales=manual_scales,
                batch_size=batch_size,
                output_dir=output_dir,
                source="manual",
                workload_spec_path=workload_spec_path,
            )
        if capabilities.text_payload:
            return plan_manual_nlp_scales(
                task_info=task_info,
                image_info=image_info,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                scales=manual_scales,
                batch_size=batch_size,
                output_dir=output_dir,
                **(
                    {"workload_spec_path": workload_spec_path}
                    if workload_spec_path or capabilities.text_requires_workload_spec
                    else {}
                ),
            )
        print(f"[scale] Using manual input scales: {serialize_input_scales(manual_scales)}")
        return materialize_scale_plan(
            task_info=task_info,
            scales=manual_scales,
            batch_size=batch_size,
            output_dir=output_dir,
            source="manual",
            workload_spec_path=workload_spec_path,
        )

    if capabilities.text_payload:
        return plan_nlp_auto_scales(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            batch_size=batch_size,
            output_dir=output_dir,
            **(
                {"workload_spec_path": workload_spec_path}
                if workload_spec_path or capabilities.text_requires_workload_spec
                else {}
            ),
        )

    if capabilities.audio_payload:
        workload_gen = get_task_generator(
            task_info, batch_size, workload_spec_path=workload_spec_path,
        )
        default_scales = workload_gen.default_input_scales()
        if not default_scales:
            raise RuntimeError("audio workload manifest did not provide default input scales")
        return plan_audio_scales(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            scales=[float(scale) for scale in default_scales],
            batch_size=batch_size,
            output_dir=output_dir,
            source="workload_spec",
            workload_spec_path=workload_spec_path,
        )

    if capabilities.workload_scales or (capabilities.fractional_scales and workload_spec_path):
        workload_gen = get_task_generator(
            task_info, batch_size, workload_spec_path=workload_spec_path,
        )
        default_scales = workload_gen.default_input_scales()
        if not default_scales and capabilities.fractional_scales:
            default_scales = float_auto_scales(
                default_family_max_scale(task_info, batch_size),
                count=AUTO_INPUT_SCALE_COUNT,
            )
        if not default_scales:
            raise RuntimeError(f"{task_info.task_family} workload did not provide default scales")
        scales = [float(scale) for scale in default_scales]
        print(
            f"[scale] Using {task_info.task_family} workload scales: "
            f"{serialize_input_scales(scales)}"
        )
        return materialize_scale_plan(
            task_info=task_info,
            scales=scales,
            batch_size=batch_size,
            output_dir=output_dir,
            source="workload_default",
            workload_spec_path=workload_spec_path,
        )

    max_scale = default_family_max_scale(task_info, batch_size)
    if capabilities.fractional_scales:
        scales = float_auto_scales(max_scale, count=AUTO_INPUT_SCALE_COUNT)
    else:
        scales = integer_auto_scales(int(max_scale), count=AUTO_INPUT_SCALE_COUNT)

    print(f"[scale] Auto-planned {task_info.task_family} scales: {serialize_input_scales(scales)}")
    return materialize_scale_plan(
        task_info=task_info,
        scales=scales,
        batch_size=batch_size,
        output_dir=output_dir,
        source="auto",
        workload_spec_path=workload_spec_path,
    )
