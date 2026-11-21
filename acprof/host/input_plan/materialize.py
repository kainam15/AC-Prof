"""Materialize deterministic workload payloads and input plan artifacts."""

import hashlib
import json
import os
from itertools import chain
from pathlib import Path
from typing import Any, TextIO

from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES, atomic_write, read_json_object_snapshot
from acprof.extensions import CATALOG
from acprof.host.configuration_transport import (
    bound_inference_configuration,
    verify_embedding_plan_parameters,
)
from acprof.host.detect import TaskInfo
from acprof.host.input_plan.scales import format_scale_value
from acprof.host.input_plan.types import PlannedInputScales
from acprof.models.evidence import content_digest
from acprof.quality import runtime_quality_constraints


def scale_plan_file_path(output_dir: str) -> str:
    from acprof.results.layout import ArtifactLayout
    return str(ArtifactLayout.discover(output_dir).path("input_scale_plan.json"))


def write_scale_plan_file(
    path: str,
    task_info: TaskInfo,
    entries: list[dict[str, Any]],
    *,
    workload: dict[str, Any] | None = None,
    model_constraints: dict[str, Any] | None = None,
) -> str:
    frozen = bound_inference_configuration(task_info)
    payload = {
        "schema_version": 2,
        "model_id": task_info.model_id,
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        **({"task_workload": task_info.selected_tasks[0]} if task_info.selected_tasks else {}),
        "workload": dict(workload or {}),
        "model_constraints": dict(model_constraints or {}),
        "quality_constraints": runtime_quality_constraints(),
        "scenario": {"type": "serial"},
        "entries": entries,
    }
    if frozen:
        verify_embedding_plan_parameters(task_info, payload, frozen)
        payload["inference_configuration_sha256"] = frozen["identity_sha256"]
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


def _plan_metadata(generator: Any) -> dict[str, Any]:
    read_metadata = getattr(generator, "plan_metadata", None)
    metadata = read_metadata() if callable(read_metadata) else None
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise RuntimeError("workload generator returned non-object plan metadata")
    return metadata


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

    workload_metadata = _plan_metadata(workload_gen)
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


def _expected_spec_bytes_sha256(task_info: TaskInfo, path: str | None) -> str | None:
    """Bind the bytes consumed by the generator to the reviewed JSON content."""
    workload = getattr(task_info, "inference_configuration", {}).get("workload", {})
    expected_bytes = workload.get("specification_bytes_sha256")
    # NLP task runs consume a derived manifest. Historical configurations may
    # have no source identity at all.
    if task_info.task_family == "nlp" or "specification_sha256" not in workload:
        return expected_bytes
    expected_content = workload["specification_sha256"]
    if path and expected_content:
        if expected_bytes:
            return expected_bytes
        snapshot = read_json_object_snapshot(Path(path).expanduser(), label="workload specification")
        if content_digest(snapshot.payload) == expected_content:
            return hashlib.sha256(snapshot.content).hexdigest()
    elif not path and expected_content is None:
        return None
    raise ValueError("workload specification changed after inference confirmation")


def get_task_generator(task_info: TaskInfo, batch_size: int, **kwargs: Any):
    """Carry the selected adapter into workload defaults, including model aliases."""
    from acprof.models.spec import task_model_spec
    from acprof.workloads import get_generator

    if task_info.model_adapter != "family-default":
        kwargs["model_adapter"] = task_info.model_adapter
    spec = task_model_spec(task_info)
    if task_info.selected_tasks:
        parameters = task_info.selected_tasks[0]["config"].get("inference", {})
        if parameters:
            bound_inference_configuration(task_info)
            from acprof.extensions.parameters import validate_parameter_values
            from acprof.models.configuration import parameter_declaration
            declaration = parameter_declaration(task_info, task_info.selected_tasks[0])
            validate_parameter_values(declaration.parameters, parameters)
            kwargs["inference_parameters"] = parameters
    if CATALOG.describe(task_info).declaration.input_plan.model_feature_dim and spec.get("feature_dim") is not None:
        kwargs["model_feature_dim"] = spec["feature_dim"]
    # Inspection and default scale lookups omit the workload argument.
    expected = (_expected_spec_bytes_sha256(task_info, kwargs["workload_spec_path"])
                if "workload_spec_path" in kwargs else None)
    generator = get_generator(task_info.task_family, task_info.model_id, task_info.pipeline_tag, batch_size, **kwargs)
    if expected:
        metadata = _plan_metadata(generator)
        observed = metadata.get("workload_spec_sha256", metadata.get("manifest_sha256"))
        if not observed:
            raise ValueError("workload generator did not report the SHA-256 of its workload specification")
        if observed != expected:
            raise ValueError("workload specification changed after inference confirmation")
    return generator
