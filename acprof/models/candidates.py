"""Static task candidates and evidence ranking, independent of model execution."""
from __future__ import annotations

import copy
from functools import cache
from typing import Any

from acprof.models.spec import FORMAT_BACKENDS, task_model_spec, validate_model_spec
from acprof.models.support import TASK_MAPPINGS, transformers_support

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


def apply_model_candidate(task_info: Any, resolution: dict) -> dict:
    """Explicitly apply a discovery decision at an orchestration boundary."""
    from acprof.extensions import CATALOG

    task = resolution.get("selection", {}).get("task")
    if task:
        task_info.pipeline_tag = task
        task_info.task_family = CATALOG.task_families.get(task, "unknown")
        task_info.runtime_backend = resolution["selection"]["backend"]
    task_info.task_discovery = copy.deepcopy(resolution["task_discovery"])
    return resolution


class _CandidateDiscovery:
    def __init__(self, task_info: Any, override_tag: str | None, override_backend: str | None):
        from acprof.extensions import CATALOG

        self.catalog = CATALOG
        self.override_tag = override_tag
        self.override_backend = override_backend
        # Discovery is a query; only the owning workflow applies the selected route.
        self.task_info = copy.copy(task_info)
        self.config = self.task_info.model_config or {}
        self.metadata = self.task_info.repository_metadata or {}
        self.files: set[str] = set(self.task_info.repository_files or ())
        self.spec = task_model_spec(self.task_info)
        self.hub_metadata = getattr(self.task_info, "hub_metadata", {}) or {}
        self.errors: list[str] = []
        self.conflicts: list[str] = []
        self.overridden_conflicts: list[str] = []
        self.candidates: list[dict[str, Any]] = []
        self.route_errors: dict[str, str] = {}
        self.hub_task: str | None = None
        self.raw_hub_task: str | None = None
        self.declared_task: str | None = None
        self.declared_backend: str | None = None
        self.transformers_info: dict[str, Any] = {}
        self.structure_task: str | None = None
        self.primary_task: str | None = None
        self.encoder = False
        self.loader_hint: str | None = None
        self.transformers_task: str | None = None
        self.unsupported_transformers_hint: str | None = None
        self.spec_source = ""

    def _read_declarations(self) -> None:
        if self.spec:
            try:
                validate_model_spec(self.spec)
            except (ValueError, TypeError) as exc:
                self.errors.append(str(exc))
                self.spec = {}
        hub_task = self.hub_metadata.get("pipeline_tag", self.task_info.pipeline_tag)
        self.raw_hub_task = hub_task
        # Generic Hub labels must not override a registered executable head.
        if (hub_task not in self.catalog.task_families
                and self.hub_metadata.get("task_hint_source") == "registered_architecture"):
            hub_task = self.hub_metadata.get("task_hint")
        if hub_task == "unknown" or self.task_info.detection_method == "config_infer":
            hub_task = None
        if self.spec.get("pipeline_task") and hub_task == self.spec["pipeline_task"]:
            hub_task = self.spec["task"]
        self.hub_task = hub_task
        self.declared_task = self.spec.get("task")
        self.transformers_info = self.hub_metadata.get("transformers_info", {})
        if self.declared_task and hub_task and hub_task != self.declared_task and not self.override_tag:
            self.conflicts.append(
                f"task conflict: Hub={hub_task}, model spec={self.declared_task}; select --task explicitly"
            )
        if self.override_tag and self.declared_task and self.override_tag != self.declared_task:
            self.conflicts.append("--task conflicts with model spec task; update --model-spec")
        declared_format = self.spec.get("format")
        self.declared_backend = FORMAT_BACKENDS.get(declared_format) if isinstance(declared_format, str) else None
        if self.override_backend and self.declared_backend and self.override_backend != self.declared_backend:
            self.conflicts.append("--backend conflicts with model spec format; update --model-spec")

    def _backend_for(self, task: str, *, required: bool = False) -> str:
        from acprof.extensions import UnsupportedExtensionError
        try:
            return self.catalog.resolve(
                task=task, library=self.task_info.library_name, config=self.config,
                backend=self.override_backend or self.declared_backend or "", model_id=self.task_info.model_id,
                files=self.files, metadata=self.metadata,
            ).backend
        except UnsupportedExtensionError as exc:
            self.route_errors[task] = str(exc)
            if required and str(exc) not in self.errors:
                self.errors.append(str(exc))
            return "unknown"

    def _resolve_structure(self) -> None:
        from acprof.tasks.interfaces import component_scoped, structured_interface
        try:
            self.structure_task = structured_interface(self.task_info)
        except ValueError as exc:
            self.errors.append(str(exc))
        self.primary_task = self.declared_task or self.hub_task or self.structure_task
        self.encoder = component_scoped(
            self.task_info, self.primary_task,
            self._backend_for(self.primary_task) if self.primary_task else "unknown",
        )

    def _resolve_transformers_task(self) -> None:
        if _generic_pipeline_loader_hint(self.config, self.transformers_info):
            self.loader_hint = "AutoModel is a generic loader for config.custom_pipelines, not a task declaration"
        else:
            auto_model = self.transformers_info.get("auto_model", "")
            generic_task = "feature-extraction" if auto_model == "AutoModel" else _AUTO_TASKS.get(auto_model)
            # A shared loader is a hint, not another authored workload declaration.
            if (self.primary_task and generic_task and self.primary_task != generic_task
                    and self.transformers_info.get("pipeline_tag") == generic_task):
                shared_operation = (set(TASK_MAPPINGS.get(self.primary_task, ()))
                                    & set(TASK_MAPPINGS.get(generic_task, ())))
                if self.encoder:
                    self.loader_hint = "modules.json selects SentenceTransformer AutoModel encoding, not the checkpoint head"
                elif shared_operation and auto_model != "AutoModel":
                    self.loader_hint = (
                        f"{auto_model} is a shared loader for {self.primary_task}; "
                        "its library tag does not declare another workload"
                    )
        transformers_task = None if self.loader_hint else self.transformers_info.get("pipeline_tag")
        if transformers_task and transformers_task not in self.catalog.task_families:
            self.unsupported_transformers_hint = (
                f"{transformers_task!r} is not a registered executable task; "
                "task semantics still require a declaration or a recognized checkpoint head"
            )
            transformers_task = None
        if self.spec.get("pipeline_task") and transformers_task == self.spec["pipeline_task"]:
            transformers_task = self.declared_task
        self.transformers_task = transformers_task

    def _record_declaration_conflicts(self) -> None:
        observed_tasks = {value for value in (self.declared_task, self.hub_task, self.transformers_task) if value}
        head_tasks = {inferred for architecture in self.config.get("architectures") or []
                      if isinstance(architecture, str)
                      if (inferred := self.catalog.infer_task([architecture], ""))}
        if (not self.hub_task and not self.declared_task and len(head_tasks) == 1
                and self.transformers_task and self.transformers_task not in head_tasks):
            disagreement = (f"task conflict: transformers_info={self.transformers_task} "
                            f"disagrees with registered checkpoint head={next(iter(head_tasks))}")
            (self.overridden_conflicts if self.override_tag else self.conflicts).append(disagreement)
        if len(observed_tasks) > 1:
            disagreement = "task conflict between declarations/Hub: " + ", ".join(sorted(observed_tasks))
            (self.overridden_conflicts if self.override_tag else self.conflicts).append(disagreement)

    def _add(self, task: str | None, source: str) -> None:
        if not task:
            return
        backend = self._backend_for(task)
        previous = next((item for item in self.candidates
                         if (item["task"], item["backend"]) == (task, backend)), None)
        if previous is not None:
            if source not in previous["evidence"]:
                previous["evidence"].append(source)
            return
        candidate = {
            "task": task,
            "family": self.catalog.task_families.get(task, "unknown"),
            "backend": backend,
            "evidence": [source],
        }
        if task in self.route_errors:
            candidate["unsupported_reason"] = self.route_errors[task]
        self.candidates.append(candidate)

    def _collect_primary_candidates(self) -> None:
        self._add(self.override_tag, "explicit_task")
        self.spec_source = ("local_model_spec" if getattr(self.task_info, "model_spec", {}) else
                            "repository_model_spec" if self.metadata.get("acprof_model.json") else "generated_contract")
        self._add(self.declared_task, self.spec_source)
        if not self.declared_task and not self.hub_task:
            self._add(self.structure_task, "modules.json")
        self._add(self.hub_task, ("registered_architecture" if self.raw_hub_task != self.hub_task
                                  else "hub_metadata" if self.task_info.detection_method == "hub_api"
                                  else self.task_info.detection_method))
        self._add(self.hub_metadata.get("task_hint"),
                  self.hub_metadata.get("task_hint_source", "extension_catalog"))
        self._add(self.transformers_task, "hub.transformers_info.pipeline_tag")
        if not self.transformers_task:
            self._add(_AUTO_TASKS.get(self.transformers_info.get("auto_model", "")),
                      "hub.transformers_info.auto_model")
        elif self.transformers_info.get("auto_model", ""):
            self._add(_AUTO_TASKS.get(self.transformers_info["auto_model"]),
                      "hub.transformers_info.auto_model")

    def _collect_config_candidates(self) -> None:
        for architecture in self.config.get("architectures") or []:
            if isinstance(architecture, str):
                inferred = self.catalog.infer_task([architecture], str(self.config.get("model_type") or ""))
                for task in (inferred,) if inferred else _architecture_tasks(architecture):
                    self._add(task, "config.architectures")
        if self.config.get("model_type"):
            self._add(self.catalog.infer_task([], str(self.config["model_type"])), "config.model_type")
        auto_map = self.config.get("auto_map") or {}
        if isinstance(auto_map, dict):
            for auto_class in auto_map:
                self._add(_AUTO_TASKS.get(auto_class), "config.auto_map")
        custom_tasks = self.config.get("custom_pipelines") or {}
        if isinstance(custom_tasks, dict):
            for custom_task in custom_tasks:
                if custom_task in self.catalog.task_families:
                    self._add(custom_task, "config.custom_pipelines")
        if not self.candidates and "modules.json" in self.metadata:
            self._add("feature-extraction", "modules.json")

    def _validate_loader_compatibility(self, selected: str | None) -> None:
        selected_operations = set(TASK_MAPPINGS.get(selected or "", ()))
        hints: list[tuple[str, tuple[str | None, ...]]] = [] if self.encoder else [
            ("transformersInfo.auto_model", (_AUTO_TASKS.get(self.transformers_info.get("auto_model", "")),))
        ]
        if not self.encoder and not self.config.get("auto_map") and not self.config.get("custom_pipelines"):
            architectures = [item for item in self.config.get("architectures") or [] if isinstance(item, str)]
            hints.append(("config.architectures:" + ",".join(architectures), tuple({task
                          for architecture in architectures for task in _architecture_tasks(architecture)})))
        for source, hint_tasks in hints:
            hint_operations = {operation for task in hint_tasks for operation in TASK_MAPPINGS.get(task or "", ())}
            if selected_operations and hint_operations and selected_operations.isdisjoint(hint_operations):
                self.conflicts.append(f"task conflict: {selected} is incompatible with {source}")

    def _add_onnx_candidates(self) -> None:
        if not any(name.lower().endswith(".onnx") for name in self.files):
            return
        for extension in self.catalog.extensions.values():
            if "onnxruntime" in extension.backends:
                for candidate_task in extension.tasks:
                    self._add(candidate_task, "artifact.onnx; task semantics required")

    def _select(self) -> tuple[str | None, str]:
        selected = (self.override_tag or self.declared_task or self.hub_task
                    or self.structure_task or self.transformers_task)
        self._validate_loader_compatibility(selected)
        status = "candidate"
        if not selected:
            if len(self.candidates) == 1:
                selected = self.candidates[0]["task"]
            elif len(self.candidates) > 1:
                status = "ambiguous"
            else:
                status = "needs_configuration"
                self._add_onnx_candidates()
                self.errors.append("task semantics are unknown; provide --task or --model-spec")
        if selected:
            self.task_info.pipeline_tag = selected
            self.task_info.task_family = self.catalog.task_families.get(selected, "unknown")
            self.task_info.runtime_backend = self._backend_for(selected, required=True)
        if self.errors:
            status = "needs_configuration"
        if self.conflicts:
            status = "ambiguous"
        return selected, status

    def _base_resolution(self, selected: str | None, status: str) -> dict:
        from acprof.models.evidence import resolution_provenance
        provenance = resolution_provenance(
            self.task_info, self.candidates,
            hub_task=(self.raw_hub_task if self.raw_hub_task != self.hub_task
                      and self.hub_metadata.get("task_hint_source") == "registered_architecture" else self.hub_task),
            selected=selected, status=status, explicit_task=self.override_tag,
            explicit_backend=self.override_backend, overridden_conflicts=self.overridden_conflicts,
            loader_hint=self.loader_hint, unsupported_task_hint=self.unsupported_transformers_hint,
        )
        semantic_status = (
            "conflict" if self.conflicts else "unresolved" if self.errors or not selected else
            "explicit" if self.override_tag else "declared" if (
                self.declared_task and self.spec_source != "generated_contract"
                or self.hub_task and self.hub_metadata.get("task_hint_source") != "registered_architecture"
                or self.transformers_task
            ) else "inferred"
        )
        return {"schema_version": 1, "status": status, "model_revision": self.task_info.model_revision,
                "provenance": provenance, "semantics": {"status": semantic_status, "task": selected,
                                                         "scope": "selected_workload; not_author_intent_or_model_quality"},
                "benchmark_kind": "task_pipeline",
                "adapter_origin": "repository" if self.spec.get("format") == "transformers-pipeline" else "builtin",
                "execution": {"interface": self.task_info.runtime_backend if selected else None,
                              "entrypoint": "handler.predict",
                              "stages": ["preprocess", "predict", "completion", "postprocess"],
                              "timing_protocol": "acprof_task_pipeline"},
                "runtime_validation": {"status": "not_run"},
                "candidates": self.candidates, "conflicts": self.conflicts, "missing": self.errors,
                "selection": {"task": selected,
                              "backend": self.task_info.runtime_backend if selected else None,
                              "source": "explicit" if self.override_tag or self.override_backend or
                                        getattr(self.task_info, "model_spec", {}) else "metadata"}}

    def _attach_execution_paths(self, selected: str | None) -> None:
        from acprof.tasks.interfaces import execution_path
        for item in self.candidates:
            if not item.get("unsupported_reason"):
                try:
                    item["execution_path"] = execution_path(self.task_info, item["task"], backend=item["backend"])
                except ValueError as exc:
                    item["unsupported_reason"] = str(exc)
                    if item["task"] == selected:
                        self.errors.append(str(exc))

    def _apply_task_discovery(self, resolution: dict, selected: str | None) -> dict:
        from acprof.models.tasks import build_task_discovery
        from acprof.tasks.interfaces import inference_evidence
        discovery = build_task_discovery(
            self.task_info, resolution, component_scope=self.encoder,
            override_backend=self.override_backend or "",
            explicit_task=bool(self.override_tag or self.declared_task),
        )
        resolution["task_discovery"] = discovery
        resolution["evidence"] = inference_evidence(self.task_info, self.candidates, component_scope=self.encoder)
        executable_tasks = {item["interface"] for item in discovery["tasks"]}
        for item in resolution["evidence"]:
            item["disposition"] = ("component_hint" if item["scope"] == "component" else
                                   "usage_only" if item["kind"] == "usage" else
                                   "executable_candidate" if item.get("task") in executable_tasks else "not_selected")
        if discovery["status"] == "ready":
            executable = discovery["tasks"][0]
            resolution["execution"] = copy.deepcopy(executable["execution_path"])
            resolution["status"] = "candidate"
            if not selected:
                resolution["selection"].update(task=executable["interface"],
                                                backend=executable["execution_path"]["backend"])
                resolution["semantics"].update(status="inferred", task=executable["interface"])
        elif discovery["status"] == "selection_required":
            resolution["status"] = "selection_required"
        else:
            resolution["status"] = "ambiguous" if self.conflicts else "needs_configuration"
            self.errors.extend(item["reason"] for item in discovery["unresolved"] if item["reason"] not in self.errors)
            self.conflicts.extend(reason for reason in discovery["conflicts"] if reason not in self.conflicts)
        if self.conflicts or self.errors:
            resolution["semantics"]["status"] = "conflict" if self.conflicts else "unresolved"
        return discovery

    def _finalize_provenance(self, resolution: dict, discovery: dict) -> None:
        resolution["provenance"]["decision"] = (
            "abstain" if resolution["status"] != "candidate" else
            "explicit" if self.override_tag else "consistent_evidence")
        resolution["provenance"].update(status=resolution["status"],
                                        selected_task=resolution["selection"]["task"])
        resolution["provenance"]["raw_evidence"] = copy.deepcopy({
            "hub": self.hub_metadata, "config": self.config, "metadata": self.metadata})
        from acprof.models.evidence import content_digest
        resolution["provenance"]["resolver_version"] = "model-interfaces-v1"
        resolution["provenance"]["identity_sha256"] = content_digest({
            "sources": resolution["provenance"]["sources"], "raw": resolution["provenance"]["raw_evidence"],
            "evidence": resolution["evidence"], "discovery": discovery,
            "selection": resolution["selection"], "resolver_version": "model-interfaces-v1"})

    def run(self) -> dict:
        self._read_declarations()
        self._resolve_structure()
        self._resolve_transformers_task()
        self._record_declaration_conflicts()
        self._collect_primary_candidates()
        self._collect_config_candidates()
        selected, status = self._select()
        resolution = self._base_resolution(selected, status)
        self._attach_execution_paths(selected)
        discovery = self._apply_task_discovery(resolution, selected)
        self._finalize_provenance(resolution, discovery)
        return resolution


def discover_model_candidates(task_info: Any, *, override_tag: str | None = None,
                              override_backend: str | None = None) -> dict:
    """Collect static evidence before selecting a route. Never import model code."""
    return _CandidateDiscovery(task_info, override_tag, override_backend).run()


def resolve_task_backends(task_info: Any, tasks: list[dict], *, override_backend: str = "") -> dict[str, str]:
    """Validate each explicit interface through the same evidence decision."""
    backends: dict[str, str] = {}
    for interface in sorted({item["interface"] for item in tasks}):
        decision = discover_model_candidates(task_info, override_tag=interface,
                                             override_backend=override_backend or None)
        if decision["conflicts"] or decision["missing"]:
            raise ValueError("; ".join([*decision["conflicts"], *decision["missing"]]))
        backend = decision["selection"].get("backend")
        if not backend or backend == "unknown":
            raise ValueError(f"no runnable backend for inference interface: {interface}")
        backends[interface] = backend
    return backends
