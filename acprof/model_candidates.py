"""Static task candidates and evidence ranking, independent of model execution."""
from __future__ import annotations

import copy
from functools import cache
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


@cache
def _architecture_tasks(architecture: str) -> tuple[str, ...]:
    """Reverse locked Auto registries, retaining ambiguity between task protocols."""
    tasks = set()
    for version in ("4.57.6", "5.10.4"):
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


def apply_model_candidate(task_info: Any, resolution: dict) -> dict:
    """Explicitly apply a discovery decision at an orchestration boundary."""
    from acprof.extensions import CATALOG

    task = resolution.get("selection", {}).get("task")
    if task:
        task_info.pipeline_tag = task
        task_info.task_family = CATALOG.task_families.get(task, "unknown")
        task_info.runtime_backend = resolution["selection"]["backend"]
    return resolution


def discover_model_candidates(task_info: Any, *, override_tag: str | None = None,
                              override_backend: str | None = None) -> dict:
    """Collect static evidence before selecting a route. Never import model code."""
    from acprof.extensions import CATALOG

    # Evidence collection is a query, not an update to the caller's task.
    # The selected route is applied explicitly by the owning workflow.
    task_info = copy.copy(task_info)
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
    unsupported_transformers_hint = None
    if transformers_task and transformers_task not in CATALOG.task_families:
        # E.g. `pretraining` is a Hub metadata label, not a runnable task.
        # Do not let it outrank an executable head registered by an extension.
        unsupported_transformers_hint = (
            f"{transformers_task!r} is not a registered executable task; "
            "task semantics still require a declaration or a recognized checkpoint head"
        )
        transformers_task = None
    if spec.get("pipeline_task") and transformers_task == spec["pipeline_task"]:
        transformers_task = declared_task
    observed_tasks = {value for value in (declared_task, hub_task, transformers_task) if value}
    overridden_conflicts: list[str] = []
    # Without a primary Hub task, a *supported* TransformersInfo task must not
    # silently contradict a uniquely registered checkpoint head.
    head_tasks = {inferred for architecture in config.get("architectures") or []
                  if isinstance(architecture, str)
                  if (inferred := CATALOG.infer_task([architecture], ""))}
    if (not hub_task and not declared_task and len(head_tasks) == 1 and transformers_task
            and transformers_task not in head_tasks):
        disagreement = (f"task conflict: transformers_info={transformers_task} "
                        f"disagrees with registered checkpoint head={next(iter(head_tasks))}")
        (overridden_conflicts if override_tag else conflicts).append(disagreement)
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
                                        overridden_conflicts=overridden_conflicts, loader_hint=loader_hint,
                                         unsupported_task_hint=unsupported_transformers_hint)
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


def resolve_task_backends(task_info: Any, tasks: list[dict], *, override_backend: str = "") -> dict[str, str]:
    """Resolve the actual loader of each interface, not the Hub's generic task backend."""
    backends: dict[str, str] = {}
    for interface in sorted({item["interface"] for item in tasks}):
        if interface == task_info.pipeline_tag and not override_backend:
            backend = task_info.runtime_backend
        else:
            decision = discover_model_candidates(task_info, override_tag=interface,
                                                 override_backend=override_backend or None)
            backend = decision["selection"].get("backend")
        if not backend or backend == "unknown":
            raise ValueError(f"no runnable backend for declared inference interface: {interface}")
        backends[interface] = backend
    return backends


def discover_declared_tasks(task_info: Any) -> dict:
    """Collect explicit workload declarations without loading weights or model code."""
    from acprof.extensions import CATALOG
    from acprof.task_workloads import canonical_task_id, parse_task

    config = task_info.model_config or {}
    hub = task_info.hub_metadata or {}
    metadata = task_info.repository_metadata or {}
    encoder = config.get("is_encoder_decoder") is True or any(
        "SEQ_TO_SEQ_CAUSAL_LM" in TASK_MAPPINGS.get(task, ())
        for architecture in config.get("architectures", []) for task in _architecture_tasks(architecture))
    found: dict[str, dict] = {}
    blocked: set[str] = set()
    unresolved, conflicts = [], []
    declarations = config.get("task_specific_params") or {}
    defaults = metadata.get("generation_config.json") or {}
    if not isinstance(declarations, dict):
        unresolved.append({"source": "config.task_specific_params", "reason": "expected an object"})
        declarations = {}

    def add(name, params, source):
        identifier = canonical_task_id(name)
        try:
            candidate = parse_task(name, params, encoder_decoder=encoder, source=source,
                                   default_generation=defaults)
            identifier = candidate["task_id"]
            if identifier in blocked:
                return
            if identifier in found:
                if found[identifier]["fingerprint"] != candidate["fingerprint"]:
                    conflicts.append(f"{source}: conflicting configurations for {identifier}")
                    blocked.add(identifier)
                    del found[identifier]
                    return
                found[identifier]["evidence"].append(source)
                return
            # Incompatible metadata is evidence of a conflict, not an extra executable task.
            hints = {operation for architecture in config.get("architectures", [])
                     for task in _architecture_tasks(architecture) for operation in TASK_MAPPINGS.get(task, ())}
            operations = set(TASK_MAPPINGS.get(candidate["interface"], ()))
            if hints and operations and hints.isdisjoint(operations):
                conflicts.append(f"{source}: {name} conflicts with checkpoint loader")
                return
            found[identifier] = candidate
        except (ValueError, TypeError) as exc:
            blocked.add(identifier)
            unresolved.append({"task": name, "source": source, "reason": str(exc)})

    for name, params in sorted(declarations.items()):
        add(name, params, f"config.task_specific_params.{name}")
    labels = [(hub.get("pipeline_tag"), "hub.pipeline_tag")]
    labels.append(((hub.get("transformers_info") or {}).get("pipeline_tag"),
                   "hub.transformers_info.pipeline_tag"))
    labels.extend((tag, "hub.tags") for tag in hub.get("tags", []) if isinstance(tag, str)
                  and (tag in CATALOG.task_families or tag.startswith("translation_") or tag == "sentiment-analysis"))
    if not hub and task_info.detection_method == "hub_api":
        labels.append((task_info.pipeline_tag, "hub.pipeline_tag"))
    spec = task_model_spec(task_info)
    if spec and (task_info.model_spec or metadata.get("acprof_model.json")):
        labels.append((spec.get("task"), "model_spec"))
    for name, source in labels:
        if not name or name == "unknown":
            continue
        if canonical_task_id(name) in blocked:
            continue
        matching = [item for item in found.values() if item["task_id"] == name or item["kind"] == name
                    or item["interface"] == name]
        if matching:
            for item in matching:
                if source not in item["evidence"]:
                    item["evidence"].append(source)
        elif not (declarations and name in {"text2text-generation", "text-generation", "translation"}):
            add(name, {}, source)
    for identifier, generic in list(found.items()):
        if identifier != generic["interface"] or not all(source.startswith("hub.") for source in generic["evidence"]):
            continue
        specific = [item for item in found.values() if item["interface"] == identifier and item["task_id"] != identifier]
        if specific:
            for item in specific:
                item["evidence"] = list(dict.fromkeys([*item["evidence"], *generic["evidence"]]))
            del found[identifier]
    tasks = sorted(found.values(), key=lambda item: item["task_id"])
    from acprof.inference_groups import group_inference_tasks
    profiles = group_inference_tasks(tasks, backend=resolve_task_backends(task_info, tasks))
    return {"schema_version": 1, "status": "selection_required" if len(profiles) > 1 or conflicts else
            "ready" if profiles else "needs_configuration", "tasks": tasks,
            "profiles": [dict(profile_id=item["task_id"], **item["execution_profile"]) for item in profiles],
            "unresolved": unresolved, "conflicts": conflicts,
            "inferred": list(config.get("architectures") or []),
            "model_revision": task_info.model_revision}
