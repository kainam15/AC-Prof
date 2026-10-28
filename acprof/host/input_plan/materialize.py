"""Materialize deterministic workload payloads and input plan artifacts."""

import hashlib
import json
import os
from itertools import chain
from typing import Any, TextIO

from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES, atomic_write
from acprof.extensions import CATALOG
from acprof.host.detect import TaskInfo
from acprof.host.input_plan.scales import format_scale_value
from acprof.host.input_plan.types import PlannedInputScales


def scale_plan_file_path(output_dir: str) -> str:
    from acprof.artifact_layout import ArtifactLayout
    return str(ArtifactLayout.discover(output_dir).path("input_scale_plan.json"))


def write_scale_plan_file(
    path: str,
    task_info: TaskInfo,
    entries: list[dict[str, Any]],
    *,
    workload: dict[str, Any] | None = None,
    model_constraints: dict[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": 2,
        "model_id": task_info.model_id,
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        **({"task_workload": task_info.selected_tasks[0]} if task_info.selected_tasks else {}),
        "workload": dict(workload or {}),
        "model_constraints": dict(model_constraints or {}),
        "scenario": {"type": "serial"},
        "entries": entries,
    }
    digest = hashlib.sha256()

    def write(stream: TextIO) -> None:
        size = 0
        encoder = json.JSONEncoder(ensure_ascii=True, indent=2, allow_nan=False)
        for chunk in chain(encoder.iterencode(payload), ("\n",)):
            chunk = chunk.replace("\n", os.linesep)
            data = chunk.encode("utf-8")
            size += len(data)
            if size > MAX_JSON_ARTIFACT_BYTES:
                raise ValueError(f"{path}: input scale plan exceeds the 4 MiB read limit")
            stream.write(chunk)
            digest.update(data)

    atomic_write(path, write)
    return digest.hexdigest()


def materialize_scale_plan(
    *,
    task_info: TaskInfo,
    scales: list[float],
    batch_size: int,
    output_dir: str,
    source: str,
    workload_spec_path: str | None = None,
    model_constraints: dict[str, Any] | None = None,
) -> PlannedInputScales:
    """Generate one reusable payload plan for non-NLP task families."""
    workload_gen = get_task_generator(
        task_info, batch_size, workload_spec_path=workload_spec_path,
    )
    entries: list[dict[str, Any]] = []
    effective_scales: list[float] = []

    for scale in scales:
        requested_scale = float(scale)
        payload = workload_gen.generate(requested_scale)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"workload generator returned a non-object payload for "
                f"task_family={task_info.task_family}, "
                f"input_scale={format_scale_value(requested_scale)}"
            )

        effective_scale = workload_gen.effective_input_scale(requested_scale, payload)
        if effective_scale is None:
            raise RuntimeError(
                f"cannot determine effective input scale for "
                f"task_family={task_info.task_family}, "
                f"input_scale={format_scale_value(requested_scale)}"
            )

        actual_scale = float(effective_scale)
        if effective_scales and actual_scale <= effective_scales[-1]:
            raise RuntimeError(
                f"input scale plan is not strictly increasing for "
                f"task_family={task_info.task_family}: "
                f"{format_scale_value(effective_scales[-1])}, "
                f"{format_scale_value(actual_scale)}"
            )

        effective_scales.append(actual_scale)
        input_metadata: dict[str, Any] = {}
        metadata_fn = getattr(workload_gen, "input_metadata", None)
        if callable(metadata_fn):
            candidate_metadata = metadata_fn(actual_scale, payload)
            if candidate_metadata is not None:
                if not isinstance(candidate_metadata, dict):
                    raise RuntimeError(
                        "workload generator returned non-object input metadata "
                        f"for input_scale={format_scale_value(actual_scale)}"
                    )
                input_metadata = candidate_metadata

        entries.append({
            "input_scale": actual_scale,
            "scale_label": workload_gen.scale_label(actual_scale),
            "input_metadata": input_metadata,
            "payload": payload,
        })

    workload_metadata: dict[str, Any] = {}
    metadata_fn = getattr(workload_gen, "plan_metadata", None)
    if callable(metadata_fn):
        candidate_metadata = metadata_fn()
        if candidate_metadata is not None:
            if not isinstance(candidate_metadata, dict):
                raise RuntimeError("workload generator returned non-object plan metadata")
            workload_metadata = candidate_metadata

    plan_file = scale_plan_file_path(output_dir)
    constraints = dict(model_constraints or {})
    plan_sha256 = write_scale_plan_file(
        plan_file,
        task_info,
        entries,
        workload=workload_metadata,
        model_constraints=constraints,
    )
    static_workload = dict(workload_metadata)
    if constraints:
        static_workload["model_constraints"] = constraints
    return PlannedInputScales(
        scales=effective_scales,
        source=source,
        plan_file=plan_file,
        workload=static_workload,
        plan_sha256=plan_sha256,
    )


def get_task_generator(task_info: TaskInfo, batch_size: int, **kwargs: Any):
    """Carry the selected adapter into workload defaults, including model aliases."""
    from acprof.model_spec import task_model_spec
    from acprof.workloads import get_generator

    if task_info.model_adapter != "family-default":
        kwargs["model_adapter"] = task_info.model_adapter
    spec = task_model_spec(task_info)
    if CATALOG.describe(task_info).declaration.input_plan.model_feature_dim and spec.get("feature_dim") is not None:
        kwargs["model_feature_dim"] = spec["feature_dim"]
    return get_generator(task_info.task_family, task_info.model_id, task_info.pipeline_tag, batch_size, **kwargs)
