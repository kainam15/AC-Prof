"""Validate a selected candidate interface, never load inference weights."""
from __future__ import annotations

from typing import Any

from acprof.models.spec import (
    FORMAT_BACKENDS,
    custom_code_files,
    pipeline_route,
    task_model_spec,
    validate_model_spec,
)
from acprof.models.support import (
    keypoint_model_loader,
    supports_pipeline_task,
    supports_transformers_task,
)


def require_resolved_candidate(task_info: Any) -> None:
    resolution = getattr(task_info, "model_resolution", {}) or {}
    from acprof.failures import REASON_CODES, Failure, RuntimeFailure

    if resolution.get("failure", {}).get("reason_code") in REASON_CODES:
        raise RuntimeFailure(Failure(**resolution["failure"]))
    if (resolution.get("status") in {"ambiguous", "needs_configuration", "selection_required"}
            or resolution.get("semantics", {}).get("status") in {"conflict", "unresolved"}):
        choices = ", ".join(f"{item['task']}/{item['backend']}" for item in resolution.get("candidates", []))
        details = "; ".join([*resolution.get("conflicts", []), *resolution.get("missing", [])])
        raise ValueError(f"model resolution {resolution['status']}: {details}; candidates: {choices or 'none'}; "
                         "use --task / --backend / --model-spec to complete the interface")


def describe_static_compatibility(task_info: Any, profile: Any, extension: Any) -> dict[str, Any]:
    """Describe registered interface and artifact evidence without runtime claims."""
    from acprof.runtime.profiles import locked_transformers_version

    version = locked_transformers_version(profile.environment) or ""
    config = task_info.model_config or {}
    native_backend = task_info.runtime_backend in {"transformers_model", "transformers_pipeline"}
    custom = bool(config.get("auto_map") or config.get("custom_pipelines"))
    support = (supports_transformers_task(version, task_info.pipeline_tag, config.get("model_type", ""), config)
               if native_backend and version and not custom else None)
    declared = not native_backend or extension.adapter != "family-default"
    interface_status = "supported" if declared or support is True else "unsupported" if support is False else "unknown"
    layout = task_info.model_resolution.get("artifact_layout", {})
    layout_status = layout.get("status", "unknown")
    overall_status = "unknown" if interface_status == "supported" and layout_status == "unknown" else interface_status
    compatibility = {
        "status": overall_status, "interface_status": interface_status, "artifact_status": layout_status,
        "evidence_level": "static", "source": "registered_extension" if declared else "locked_transformers_mapping",
        "runtime_profile": profile.profile_id, "transformers_version": version or None,
        "checkpoint_verified": False,
    }
    if not declared and support is True and task_info.pipeline_tag == "keypoint-detection":
        compatibility["source"] = "registered_keypoint_loader"
    if overall_status == "unknown":
        compatibility.update(
            detail=layout.get("detail") if layout_status == "unknown" else
            "Architecture support is not established by the registered runtime metadata.",
            next_action="Review the pinned artifacts and registered backend capability; "
            "static evidence does not replace a separately authorized interface probe.",
        )
    return compatibility


def resolve_model_interface(task_info: Any) -> dict:
    """Reject known incompatible layouts; a successful resolution is only a candidate."""
    from acprof.extensions import select_extension
    from acprof.failures import Failure, RuntimeFailure
    from acprof.models.audiocpp import artifact_failure, gguf_checkpoint
    from acprof.runtime.profiles import locked_transformers_version, select_runtime_profile

    if failure := artifact_failure(task_info, interface=task_info.pipeline_tag, backend=task_info.runtime_backend):
        raise RuntimeFailure(failure)
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
    if backend in {"onnxruntime", "skops", "torchscript"}:
        from acprof.models.artifact_layout import export_only_failure
        failure = export_only_failure(task_info, files)
        if failure and not (backend == "onnxruntime" and failure.evidence["artifact_format"] == "ONNX"):
            raise RuntimeFailure(failure)
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
    variant = getattr(task_info, "model_variant", "")
    if variant and backend != "audio_cpp":
        from acprof.models.files import checkpoint_variants, has_default_checkpoint
        if not files:
            raise RuntimeFailure(Failure("preflight", "artifact_layout_unknown",
                                         "The selected checkpoint variant requires a pinned repository file inventory."))
        has_variant = (has_default_checkpoint(files) if variant.endswith("/") else variant in checkpoint_variants(files))
        if (backend not in {"transformers_model", "transformers_pipeline"} or extension.adapter != "family-default"
                or task_info.task_family not in {"audio", "nlp", "cv", "multimodal"}
                or config.get("auto_map") or config.get("custom_pipelines") or config.get("quantization_config")
                or "adapter_config.json" in files or not has_variant):
            raise RuntimeFailure(Failure("preflight", "artifact_layout_unsupported",
                f"The selected variant {variant!r} requires matching native Transformers weights and a registered loader.",
                evidence={"backend": backend, "variant": variant, "available_variants": checkpoint_variants(files)}))
    code_files = set()
    for item in [config, *(value for value in metadata.values() if isinstance(value, dict))]:
        code_files.update(custom_code_files(item))
    if files and code_files - files:
        raise ValueError(f"custom code is absent from snapshot: {sorted(code_files - files)}")
    pipeline_name, native_chat = pipeline_route(task_info.pipeline_tag, backend=backend, adapter=extension.adapter,
                                               spec=spec, config=config)
    if spec.get("format") == "transformers-pipeline":
        from acprof.models.spec import declared_multimodal_pipeline
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
    layout: dict[str, Any] = {"status": "not_applicable", "weights": []}
    if backend in {"transformers_model", "transformers_pipeline", "sentence_transformers", "cross_encoder", "chronos", "diffusers", "kokoro"}:
        from acprof.models.artifact_layout import inspect_artifact_layout

        layout = inspect_artifact_layout(task_info, adapter=extension.adapter)
    chat_evidence = {}
    if native_chat:
        from acprof.models.chat import CHAT_INPUT_PROTOCOL, native_chat_template
        chat_evidence = {"chat_template_source": native_chat_template(task_info, version or ""),
                         "input_protocol": CHAT_INPUT_PROTOCOL}
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
            raise RuntimeFailure(Failure(
                "preflight", "backend_unsupported",
                f"library {library!r} has no shared {backend} interface; select a registered backend",
                runtime_profile=profile.profile_id, evidence={"library": library, "backend": backend},
            ))
        format_name = "sentence_transformers" if backend == "sentence_transformers" else "transformers"
        if backend == "sentence_transformers":
            loader, operation = "SentenceTransformer", "encode"
        elif backend == "cross_encoder":
            loader = "CrossEncoder"
        else:
            loader = "Transformers Auto/pipeline"
            if (task_info.pipeline_tag == "keypoint-detection" and extension.adapter == "family-default"
                    and version and not config.get("auto_map") and not config.get("custom_pipelines")):
                loader = keypoint_model_loader(version, config.get("model_type", "")) or loader
    elif backend == "audio_cpp":
        format_name, loader, operation = "gguf", "audiocpp_server", "transcribe"
        layout = {"status": "identified", "source": "audiocpp_package", "weights": [gguf_checkpoint(variant)]}
    elif backend == "chronos":
        format_name, loader = "chronos", "BaseChronosPipeline"
    elif backend == "kokoro":
        format_name, loader, operation = "kokoro", "kokoro.KModel + kokoro.KPipeline", "generate_from_tokens"
    elif backend == "diffusers":
        format_name, loader, operation = "diffusers_pipeline", "DiffusionPipeline", "__call__"
    elif backend == "onnxruntime":
        format_name, loader, operation = "onnx", "InferenceSession", "run"
        artifacts = [name for name in files if name.lower().endswith(".onnx")]
        if files and not spec and len(artifacts) != 1:
            raise ValueError("ONNX requires one artifact or --model-spec selecting model_file")
        if files:
            artifact = spec["model_file"] if spec else artifacts[0]
            layout = {"status": "identified", "source": "onnx_model_file", "weights": [
                {"component": ".", "format": "onnx", "variant": None, "files": [artifact]}]}
        if files and task_info.task_family in {"cv", "nlp"} and not spec:
            raise RuntimeFailure(Failure("preflight", "model_contract_required", "ONNX image/text preprocessing requires --model-spec (acprof_model.json)", runtime_profile=profile.profile_id))
    elif backend in {"torchscript", "skops"}:
        format_name = backend
    if layout["status"] == "unknown":
        format_name = "unknown"
    return {**getattr(task_info, "model_resolution", {}),
            "schema_version": 1, "status": "candidate", "task": task_info.pipeline_tag,
            "backend": backend, "library": library, "artifact_format": format_name,
            "artifact_layout": layout,
            "loader": loader, "operation": operation, "model_revision": task_info.model_revision,
            "metadata_files": sorted(metadata), "model_type": config.get("model_type"),
            "interface_kind": "custom_pipeline" if pipeline_name in (config.get("custom_pipelines") or {}) else
                              "custom_auto" if config.get("auto_map") else "standard",
            "pipeline_task": pipeline_name if backend == "transformers_pipeline" else None,
            **chat_evidence,
            "code_revision": task_info.model_revision if code_files else None,
            "code_files": sorted(code_files), "model_spec": spec, "runtime_profile": profile.profile_id,
            "transformers_version": version}
