"""Validate a selected candidate interface, never load inference weights."""
from __future__ import annotations

from typing import Any

from acprof.model_spec import (
    FORMAT_BACKENDS,
    custom_code_files,
    task_model_spec,
    validate_model_spec,
)
from acprof.model_support import supports_pipeline_task, supports_transformers_task


def require_resolved_candidate(task_info: Any) -> None:
    resolution = getattr(task_info, "model_resolution", {}) or {}
    if resolution.get("failure", {}).get("reason_code") == "model_contract_required":
        from acprof.failures import Failure, RuntimeFailure
        raise RuntimeFailure(Failure(**resolution["failure"]))
    if (resolution.get("status") in {"ambiguous", "needs_configuration", "selection_required"}
            or resolution.get("semantics", {}).get("status") in {"conflict", "unresolved"}):
        choices = ", ".join(f"{item['task']}/{item['backend']}" for item in resolution.get("candidates", []))
        details = "; ".join([*resolution.get("conflicts", []), *resolution.get("missing", [])])
        raise ValueError(f"model resolution {resolution['status']}: {details}; candidates: {choices or 'none'}; "
                         "use --task / --backend / --model-spec to complete the interface")


def resolve_model_interface(task_info: Any) -> dict:
    """Reject known incompatible layouts; a successful resolution is only a candidate."""
    from acprof.extensions import select_extension
    from acprof.failures import Failure, RuntimeFailure
    from acprof.runtime_profiles import locked_transformers_version, select_runtime_profile

    require_resolved_candidate(task_info)
    extension = select_extension(task_info)
    backend, library = task_info.runtime_backend, task_info.library_name
    config = getattr(task_info, "model_config", {}) or {}
    files: set[str] = set(getattr(task_info, "repository_files", ()) or ())
    metadata = getattr(task_info, "repository_metadata", {}) or {}
    errors = getattr(task_info, "metadata_errors", ())
    if errors:
        raise ValueError("Model metadata could not be resolved: " + "; ".join(errors))
    spec = task_model_spec(task_info)
    if extension.handler_options.get("requires_model_spec") and not spec:
        failure = Failure("preflight", "model_contract_required",
                          f"{backend}/{task_info.pipeline_tag} requires acprof_model.json; input semantics cannot be inferred",
                          runtime_profile=getattr(task_info, "runtime_profile_id", ""),
                          retryability="after_configuration", evidence={"required_artifact": "acprof_model.json", "backend": backend})
        task_info.model_resolution.update(status="needs_configuration", failure=failure.to_dict())
        raise RuntimeFailure(failure)
    if spec:
        validate_model_spec(spec)
        if spec["task"] != task_info.pipeline_tag or FORMAT_BACKENDS[spec["format"]] != backend:
            raise ValueError("model spec task/format conflicts with selected interface")
        if files and spec.get("model_file") and spec["model_file"] not in files:
            raise ValueError(f"model spec model_file is absent from snapshot: {spec['model_file']}")
    code_files = set()
    for item in [config, *(value for value in metadata.values() if isinstance(value, dict))]:
        code_files.update(custom_code_files(item))
    if files and code_files - files:
        raise ValueError(f"custom code is absent from snapshot: {sorted(code_files - files)}")
    pipeline_name = spec.get("pipeline_task", task_info.pipeline_tag)
    if spec.get("format") == "transformers-pipeline":
        from acprof.model_spec import declared_multimodal_pipeline
        if (task_info.task_family not in {"nlp", "cv", "audio"} and not declared_multimodal_pipeline(task_info)) or task_info.pipeline_tag in {
            "video-classification", "keypoint-detection", "sentence-similarity", "text-ranking",
        }:
            raise ValueError("custom pipeline needs an adapter for this task protocol")
        if pipeline_name not in (config.get("custom_pipelines") or {}):
            raise ValueError(f"pipeline_task {pipeline_name!r} is not declared in config.custom_pipelines")
    if any(item["repo_id"] == task_info.model_id for item in spec.get("dependencies", [])):
        raise ValueError("model dependencies must not override the primary snapshot")
    format_name, loader, operation = "unknown", backend, "predict"
    profile = select_runtime_profile(task_info)
    task_info.runtime_profile_id, task_info.model_adapter = profile.profile_id, profile.adapter
    version = locked_transformers_version(profile.environment)
    if (extension.handler_options.get("uses_pipeline") and backend not in {"sentence_transformers", "cross_encoder"}
            and pipeline_name not in (config.get("custom_pipelines") or {})
            and version and not supports_pipeline_task(version, pipeline_name)):
        raise RuntimeFailure(Failure("preflight", "runtime_task_unsupported",
            f"Pipeline task {pipeline_name!r} is not registered in transformers=={version}",
            runtime_profile=profile.profile_id,
            evidence={"transformers_version": version, "pipeline_task": pipeline_name}))
    if backend in {"transformers_model", "transformers_pipeline", "sentence_transformers", "cross_encoder"}:
        # Unknown ecosystem tags may still wrap a registered native checkpoint;
        # accept actual native metadata, never an unrelated task tag alone.
        native = version and supports_transformers_task(version, task_info.pipeline_tag, str(config.get("model_type", "")), config)
        if library not in {"", "unknown", "transformers", "sentence-transformers", "timm"} and not native and extension.adapter == "family-default":
            raise ValueError(f"library {library!r} has no shared {backend} interface; select a registered backend")
        if files and "config.json" not in files and "modules.json" not in files:
            suffix = "GGUF" if any(name.lower().endswith(".gguf") for name in files) else "missing config.json/modules.json"
            raise RuntimeFailure(Failure("preflight", "artifact_layout_unsupported", f"{backend} cannot load this artifact layout ({suffix})", runtime_profile=profile.profile_id, evidence={"files": sorted(files)}))
        if "adapter_config.json" in files:
            base = (metadata.get("adapter_config.json") or {}).get("base_model_name_or_path")
            raise ValueError(f"adapter requires an offline base model dependency ({base or 'unknown'}); a standalone snapshot is incomplete")
        format_name = "sentence_transformers" if backend == "sentence_transformers" else "transformers"
        if backend == "sentence_transformers":
            loader, operation = "SentenceTransformer", "encode"
        elif backend == "cross_encoder":
            loader = "CrossEncoder"
        else:
            loader = "Transformers Auto/pipeline"
    elif backend == "chronos":
        format_name, loader = "chronos", "BaseChronosPipeline"
    elif backend == "diffusers":
        if files and "model_index.json" not in files:
            format_name = "GGUF" if any(name.lower().endswith(".gguf") for name in files) else "single-file/components"
            raise RuntimeFailure(Failure("preflight", "artifact_layout_unsupported", f"DiffusionPipeline requires model_index.json; {format_name} needs a compatible artifact loader", runtime_profile=profile.profile_id, evidence={"files": sorted(files)}))
        format_name, loader, operation = "diffusers_pipeline", "DiffusionPipeline", "__call__"
    elif backend == "onnxruntime":
        format_name, loader, operation = "onnx", "InferenceSession", "run"
        artifacts = [name for name in files if name.lower().endswith(".onnx")]
        if files and not spec and len(artifacts) != 1:
            raise ValueError("ONNX requires one artifact or --model-spec selecting model_file")
        if files and task_info.task_family in {"cv", "nlp"} and not spec:
            raise RuntimeFailure(Failure("preflight", "model_contract_required", "ONNX image/text preprocessing requires --model-spec (acprof_model.json)", runtime_profile=profile.profile_id))
    elif backend in {"torchscript", "skops"}:
        format_name = backend
    return {**getattr(task_info, "model_resolution", {}),
            "schema_version": 1, "status": "candidate", "task": task_info.pipeline_tag,
            "backend": backend, "library": library, "artifact_format": format_name,
            "loader": loader, "operation": operation, "model_revision": task_info.model_revision,
            "metadata_files": sorted(metadata), "model_type": config.get("model_type"),
            "interface_kind": "custom_pipeline" if pipeline_name in (config.get("custom_pipelines") or {}) else
                              "custom_auto" if config.get("auto_map") else "standard",
            "pipeline_task": pipeline_name if backend == "transformers_pipeline" else None,
            "code_revision": task_info.model_revision if code_files else None,
            "code_files": sorted(code_files), "model_spec": spec, "runtime_profile": profile.profile_id,
            "transformers_version": version}
