"""Static task candidates and evidence ranking, independent of model execution."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from acprof.model_spec import FORMAT_BACKENDS, task_model_spec, validate_model_spec
from acprof.model_support import TASK_MAPPINGS, transformers_support

_AUTO_TASKS = {
    "AutoModelForCausalLM": "text-generation", "AutoModelForSeq2SeqLM": "text2text-generation",
    "AutoModelForMaskedLM": "fill-mask", "AutoModelForSequenceClassification": "text-classification",
    "AutoModelForTokenClassification": "token-classification", "AutoModelForQuestionAnswering": "question-answering",
    "AutoModelForImageClassification": "image-classification", "AutoModelForAudioClassification": "audio-classification",
    "AutoModelForCTC": "automatic-speech-recognition", "AutoModelForSpeechSeq2Seq": "automatic-speech-recognition",
    "AutoModelForImageTextToText": "image-text-to-text", "AutoModelForVision2Seq": "image-to-text",
    "AutoModelForMultimodalLM": "image-text-to-text",
}


@lru_cache(maxsize=None)
def _architecture_tasks(architecture: str) -> tuple[str, ...]:
    """Reverse locked Auto registries, retaining ambiguity between task protocols."""
    tasks = set()
    for version in ("4.57.6", "5.6.0"):
        mappings = transformers_support(version)
        for task, operations in TASK_MAPPINGS.items():
            if task in {"conversational", "sentence-similarity", "replaced-token-detection"}:
                # PRETRAINING includes heads such as GPT2LMHeadModel and
                # BertForPreTraining. Only the reviewed ELECTRA discriminator
                # extension declares an executable replaced-token contract.
                continue
            for operation in operations:
                name = f"MODEL_FOR_{operation}_MAPPING_NAMES" if operation else "MODEL_MAPPING_NAMES"
                for classes in mappings.get(name, {}).values():
                    if architecture in ([classes] if isinstance(classes, str) else classes):
                        tasks.add(task)
    return tuple(sorted(tasks))


def _generic_pipeline_loader_hint(config: dict, transformers_info: dict) -> bool:
    """A custom Pipeline's generic AutoModel says nothing about its workload.

    Hub can label that loader as feature extraction while the authored Pipeline
    implements a different protocol. Keep this observation as a loader hint;
    task declarations and the static contract still have to establish semantics.
    """
    if (transformers_info.get("auto_model") != "AutoModel"
            or transformers_info.get("pipeline_tag") != "feature-extraction"):
        return False
    pipelines = config.get("custom_pipelines")
    if not isinstance(pipelines, dict):
        return False
    for pipeline in pipelines.values():
        if not isinstance(pipeline, dict) or not isinstance(pipeline.get("impl"), str):
            continue
        models = pipeline.get("pt", [])
        if models == "AutoModel" or isinstance(models, list) and "AutoModel" in models:
            return True
    return False


def _sentence_transformer_encoder(task_info: Any, task: str | None, backend: str) -> bool:
    """The declared Transformer module loads AutoModel, not the checkpoint head."""
    modules = (task_info.repository_metadata or {}).get("modules.json")
    return (backend == "sentence_transformers" and task in {"feature-extraction", "sentence-similarity"}
            and isinstance(modules, list) and any(
                isinstance(module, dict) and module.get("path") == ""
                and module.get("type") == "sentence_transformers.models.Transformer"
                for module in modules))


def discover_model_candidates(task_info: Any, *, override_tag: str | None = None,
                              override_backend: str | None = None) -> dict:
    """Collect static evidence before selecting a route. Never import model code."""
    from acprof.extensions import CATALOG

    config = task_info.model_config or {}
    metadata = task_info.repository_metadata or {}
    files: set[str] = set(task_info.repository_files or ())
    spec = task_model_spec(task_info)
    errors: list[str] = []
    conflicts: list[str] = []
    candidates: list[dict[str, Any]] = []
    if spec:
        try:
            validate_model_spec(spec)
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
            spec = {}
    hub_metadata = getattr(task_info, "hub_metadata", {}) or {}
    hub_task = hub_metadata.get("pipeline_tag", task_info.pipeline_tag)
    raw_hub_task = hub_task
    # Retain the raw Hub label as evidence, but do not let an unsupported generic
    # tag override a registered, architecture-specific executable protocol.
    if (hub_task not in CATALOG.task_families
            and hub_metadata.get("task_hint_source") == "registered_architecture"):
        hub_task = hub_metadata.get("task_hint")
    if hub_task == "unknown" or task_info.detection_method == "config_infer":
        hub_task = None
    if spec.get("pipeline_task") and hub_task == spec["pipeline_task"]:
        hub_task = spec["task"]
    declared_task = spec.get("task")
    transformers_info = hub_metadata.get("transformers_info", {})
    if declared_task and hub_task and hub_task != declared_task and not override_tag:
        conflicts.append(f"task conflict: Hub={hub_task}, model spec={declared_task}; select --task explicitly")
    if override_tag and declared_task and override_tag != declared_task:
        conflicts.append("--task conflicts with model spec task; update --model-spec")
    declared_format = spec.get("format")
    declared_backend = FORMAT_BACKENDS.get(declared_format) if isinstance(declared_format, str) else None
    if override_backend and declared_backend and override_backend != declared_backend:
        conflicts.append("--backend conflicts with model spec format; update --model-spec")

    route_errors: dict[str, str] = {}

    def backend_for(task: str, *, required: bool = False) -> str:
        from acprof.extensions import UnsupportedExtensionError
        try:
            return CATALOG.resolve(
                task=task, library=task_info.library_name, config=config,
                backend=override_backend or declared_backend or "", model_id=task_info.model_id,
                files=files, metadata=metadata,
            ).backend
        except UnsupportedExtensionError as exc:
            route_errors[task] = str(exc)
            if required and str(exc) not in errors:
                errors.append(str(exc))
            return "unknown"

    primary_task = declared_task or hub_task
    encoder = _sentence_transformer_encoder(
        task_info, primary_task, backend_for(primary_task) if primary_task else "unknown")
    loader_hint = None
    if _generic_pipeline_loader_hint(config, transformers_info):
        loader_hint = "AutoModel is a generic loader for config.custom_pipelines, not a task declaration"
    else:
        auto_model = transformers_info.get("auto_model")
        generic_task = "feature-extraction" if auto_model == "AutoModel" else _AUTO_TASKS.get(auto_model)
        # Only the canonical library task for a matching Auto class is a hint.
        # Two authored tasks sharing a loader (e.g. translation/summarization)
        # still disagree. A library tag without its Auto class is not enough.
        if (primary_task and generic_task and primary_task != generic_task
                and transformers_info.get("pipeline_tag") == generic_task):
            shared_operation = set(TASK_MAPPINGS.get(primary_task, ())) & set(TASK_MAPPINGS.get(generic_task, ()))
            if encoder:
                loader_hint = "modules.json selects SentenceTransformer AutoModel encoding, not the checkpoint head"
            elif shared_operation and auto_model != "AutoModel":
                loader_hint = f"{auto_model} is a shared loader for {primary_task}; its library tag does not declare another workload"
    transformers_task = None if loader_hint else transformers_info.get("pipeline_tag")
    if (hub_metadata.get("task_hint_source") == "registered_architecture"
            and transformers_task == hub_metadata.get("pipeline_tag")
            and transformers_task not in CATALOG.task_families):
        transformers_task = None
    if spec.get("pipeline_task") and transformers_task == spec["pipeline_task"]:
        transformers_task = declared_task
    observed_tasks = {value for value in (declared_task, hub_task, transformers_task) if value}
    overridden_conflicts: list[str] = []
    if len(observed_tasks) > 1:
        disagreement = "task conflict between declarations/Hub: " + ", ".join(sorted(observed_tasks))
        (overridden_conflicts if override_tag else conflicts).append(disagreement)

    def add(task: str | None, source: str) -> None:
        if not task:
            return
        backend = backend_for(task)
        previous = next((item for item in candidates if (item["task"], item["backend"]) == (task, backend)), None)
        if previous is not None:
            if source not in previous["evidence"]:
                previous["evidence"].append(source)
            return
        family = CATALOG.task_families.get(task, "unknown")
        candidate = {"task": task, "family": family, "backend": backend, "evidence": [source]}
        if task in route_errors:
            candidate["unsupported_reason"] = route_errors[task]
        candidates.append(candidate)

    add(override_tag, "explicit_task")
    spec_source = ("local_model_spec" if getattr(task_info, "model_spec", {}) else
                   "repository_model_spec" if metadata.get("acprof_model.json") else "generated_contract")
    add(declared_task, spec_source)
    add(hub_task, ("registered_architecture" if raw_hub_task != hub_task
                   else "hub_metadata" if task_info.detection_method == "hub_api" else task_info.detection_method))
    add(hub_metadata.get("task_hint"), hub_metadata.get("task_hint_source", "extension_catalog"))
    add(transformers_task, "hub.transformers_info.pipeline_tag")
    if not transformers_task:
        add(_AUTO_TASKS.get(transformers_info.get("auto_model")), "hub.transformers_info.auto_model")
    elif transformers_info.get("auto_model"):
        # A loader hint is correlated metadata, not another semantic vote.
        add(_AUTO_TASKS.get(transformers_info["auto_model"]), "hub.transformers_info.auto_model")
    for architecture in config.get("architectures") or []:
        if isinstance(architecture, str):
            inferred = CATALOG.infer_task([architecture], str(config.get("model_type") or ""))
            for task in (inferred,) if inferred else _architecture_tasks(architecture):
                add(task, "config.architectures")
    if config.get("model_type"):
        add(CATALOG.infer_task([], str(config["model_type"])), "config.model_type")
    auto_map = config.get("auto_map") or {}
    if isinstance(auto_map, dict):
        for auto_class in auto_map:
            add(_AUTO_TASKS.get(auto_class), "config.auto_map")
    custom_tasks = config.get("custom_pipelines") or {}
    if isinstance(custom_tasks, dict):
        for custom_task in custom_tasks:
            if custom_task in CATALOG.task_families:
                add(custom_task, "config.custom_pipelines")
    if not candidates and "modules.json" in metadata:
        add("feature-extraction", "modules.json")

    selected = override_tag or declared_task or hub_task or transformers_task
    selected_operations = set(TASK_MAPPINGS.get(selected or "", ()))
    hints: list[tuple[str, tuple[str | None, ...]]] = [] if encoder else [
        ("transformersInfo.auto_model", (_AUTO_TASKS.get(transformers_info.get("auto_model")),))
    ]
    if not encoder and not config.get("auto_map") and not config.get("custom_pipelines"):
        hints.extend((f"config.architectures:{architecture}", _architecture_tasks(architecture))
                     for architecture in config.get("architectures") or [] if isinstance(architecture, str))
    for source, hint_tasks in hints:
        hint_operations = {operation for task in hint_tasks for operation in TASK_MAPPINGS.get(task or "", ())}
        if selected_operations and hint_operations and selected_operations.isdisjoint(hint_operations):
            disagreement = f"task conflict: {selected} is incompatible with {source}"
            (overridden_conflicts if override_tag else conflicts).append(disagreement)
    status = "candidate"
    if not selected:
        if len(candidates) == 1:
            selected = candidates[0]["task"]
        elif len(candidates) > 1:
            status = "ambiguous"
        else:
            status = "needs_configuration"
            if any(name.lower().endswith(".onnx") for name in files):
                for extension in CATALOG.extensions.values():
                    if "onnxruntime" in extension.backends:
                        for candidate_task in extension.tasks:
                            add(candidate_task, "artifact.onnx; task semantics required")
            errors.append("task semantics are unknown; provide --task or --model-spec")
    if selected:
        task_info.pipeline_tag = selected
        task_info.task_family = CATALOG.task_families.get(selected, "unknown")
        task_info.runtime_backend = backend_for(selected, required=True)
    if errors:
        status = "needs_configuration"
    if conflicts:
        status = "ambiguous"
    from acprof.model_evidence import resolution_provenance
    provenance = resolution_provenance(task_info, candidates, hub_task=(
        raw_hub_task if raw_hub_task != hub_task and hub_metadata.get("task_hint_source") == "registered_architecture"
        else hub_task), selected=selected,
                                        status=status, explicit_task=override_tag, explicit_backend=override_backend,
                                        overridden_conflicts=overridden_conflicts, loader_hint=loader_hint)
    semantic_status = ("conflict" if conflicts else "unresolved" if errors or not selected else
                       "explicit" if override_tag else "declared" if (declared_task and spec_source != "generated_contract"
                        or hub_task and hub_metadata.get("task_hint_source") != "registered_architecture"
                        or transformers_task) else "inferred")
    return {"schema_version": 1, "status": status, "model_revision": task_info.model_revision,
            "provenance": provenance, "semantics": {"status": semantic_status, "task": selected,
                                                     "scope": "selected_workload; not_author_intent_or_model_quality"},
            "benchmark_kind": "task_pipeline", "adapter_origin": "repository" if spec.get("format") == "transformers-pipeline" else "builtin",
            "execution": {"interface": task_info.runtime_backend if selected else None,
                          "entrypoint": "handler.predict", "stages": ["preprocess", "predict", "completion", "postprocess"],
                          "timing_protocol": "acprof_task_pipeline"},
            "runtime_validation": {"status": "not_run"},
            "candidates": candidates, "conflicts": conflicts, "missing": errors,
            "selection": {"task": selected, "backend": task_info.runtime_backend if selected else None,
                          "source": "explicit" if override_tag or override_backend or
                                    getattr(task_info, "model_spec", {}) else "metadata"}}
