"""Describe registered execution paths using JSON; never import model code."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from acprof.extensions import CATALOG
from acprof.models.evidence import content_digest
from acprof.models.spec import task_model_spec


def structured_interface(task_info: Any) -> str | None:
    """Discover a registered wrapper even when Hub has no task label."""
    matches = set()
    for extension in CATALOG.extensions.values():
        for protocol in extension.backend_inference.values():
            if protocol.get("structure") == "modules" and "modules.json" in task_info.repository_metadata:
                module_components(task_info.repository_metadata, protocol)
                if protocol.get("default_task"):
                    matches.add(protocol["default_task"])
    if len(matches) > 1:
        raise ValueError("conflicting registered module interfaces")
    return next(iter(matches), None)


def module_components(metadata: dict, protocol: dict) -> list[dict]:
    modules = metadata.get("modules.json")
    if not isinstance(modules, list) or not modules or len(modules) > 64:
        raise ValueError("modules.json must declare a bounded, nonempty module chain")
    components = []
    available: set[str] = set()
    allowed = protocol.get("module_protocols", {})
    for index, module in enumerate(modules):
        if not isinstance(module, dict) or type(module.get("idx")) is not int or module["idx"] != index:
            raise ValueError("modules.json must declare ordered module indices")
        kind, path = module.get("type"), module.get("path")
        if not isinstance(kind, str) or kind not in allowed:
            raise ValueError(f"unregistered module protocol: {kind}; register an Adapter before execution")
        contract = allowed[kind]
        if index == 0 and not contract.get("accepts_input"):
            raise ValueError("module chain has no registered input processor")
        if set(contract.get("requires", [])) - available:
            raise ValueError(f"module chain cannot supply required inputs for {kind}")
        available.update(contract["outputs"])
        if (not isinstance(path, str) or "\\" in path or PurePosixPath(path).is_absolute()
                or ".." in path.split("/")):
            raise ValueError("module paths must stay inside the model snapshot")
        configs = {name: value for name, value in metadata.items() if name in {
            str(PurePosixPath(path) / "config.json"),
            str(PurePosixPath(path) / "sentence_bert_config.json")}}
        components.append({"index": index, "type": kind, "path": path,
                           "configuration_sha256": content_digest(configs)})
    if set(protocol.get("output_features", [])) - available:
        raise ValueError("module chain does not produce the registered output features")
    return components


def execution_path(task_info: Any, task: str, *, backend: str = "") -> dict:
    """Bind the Adapter, IO, components and reviewed parameters of one route."""
    resolved = CATALOG.resolve(task=task, library=task_info.library_name, config=task_info.model_config,
                               backend=backend, model_id=task_info.model_id,
                               files=task_info.repository_files, metadata=task_info.repository_metadata)
    extension = resolved.declaration
    protocol = extension.inference
    metadata = task_info.repository_metadata or {}
    components: list[dict] = []
    structure = protocol.get("structure", "single")
    required_manifest = {"modules": "modules.json", "components": "model_index.json"}.get(structure)
    if required_manifest and task_info.repository_files and required_manifest not in metadata:
        raise ValueError(f"{required_manifest} is required to establish the execution chain")
    if structure == "modules" and "modules.json" in metadata:
        components = module_components(metadata, protocol)
    elif structure == "components":
        index = metadata.get("model_index.json", {})
        if not isinstance(index, dict):
            raise ValueError("model_index.json must declare component protocols")
        components = [{"name": name, "library": value[0], "class": value[1]}
                      for name, value in sorted(index.items()) if not name.startswith("_")
                      and isinstance(value, list) and len(value) == 2 and value != [None, None]]
    spec = task_model_spec(task_info)
    pipeline = (task_info.model_config.get("custom_pipelines") or {}).get(spec.get("pipeline_task", task))
    loader = protocol.get("loader", extension.handler_entrypoint + ".load")
    operation = protocol.get("operation", "predict")
    if pipeline:
        if not isinstance(pipeline, dict):
            raise ValueError("custom pipeline must declare its implementation and loader")
        loader = pipeline.get("impl", loader)
    path = {
        "backend": resolved.backend, "adapter": extension.adapter,
        "handler": extension.handler_entrypoint, "loader": loader, "operation": operation,
        "structure": structure, "components": components,
        "stages": {stage: extension.handler_entrypoint + "." + stage
                   for stage in ("load", "preprocess", "predict", "postprocess")},
        "io": resolved.io_format, "input_modalities": list(extension.input_modalities),
        "configuration_sha256": content_digest({"config": task_info.model_config, "spec": spec,
            "pipeline": pipeline, "metadata": metadata,
            "handler_options": extension.handler_options}),
    }
    return path


def component_scoped(task_info: Any, task: str | None, backend: str) -> bool:
    """A registered module loader owns the chain, including its backbone."""
    if not task or backend == "unknown" or "modules.json" not in task_info.repository_metadata:
        return False
    try:
        path = execution_path(task_info, task, backend=backend)
    except ValueError:
        return False
    return path["structure"] == "modules" and bool(path["components"])


def inference_evidence(task_info: Any, candidates: list[dict], *, component_scope: bool) -> list[dict]:
    """Retain raw facts and their scope separately from executable choices."""
    observations = []
    for candidate in candidates:
        for source in candidate["evidence"]:
            structural = source.startswith(("config.architectures", "config.model_type", "config.auto_map",
                                            "hub.transformers_info.auto_model"))
            component = component_scope and (structural or source.startswith("hub.transformers_info"))
            observations.append({"source": source, "task": candidate["task"],
                                 "kind": "capability" if structural or component else "interface",
                                 "scope": "component" if component else "model",
                                 "sufficient": not component and not candidate.get("unsupported_reason"),
                                 "reason": "owned by the registered module chain" if component else
                                           candidate.get("unsupported_reason", "registered protocol; runtime validation pending")})
    hub = task_info.hub_metadata or {}
    for tag in hub.get("tags", []):
        observations.append({"source": "hub.tags", "value": tag, "kind": "usage",
                             "scope": "model", "sufficient": False,
                             "reason": "usage labels do not establish another executable interface"})
    declarations = task_info.model_config.get("task_specific_params") or {}
    for name, params in declarations.items() if isinstance(declarations, dict) else ():
        observations.append({"source": f"config.task_specific_params.{name}", "task": name, "value": params,
                             "kind": "usage", "scope": "model", "sufficient": False,
                             "reason": "input and generation configuration for a registered interface"})
    if "modules.json" in task_info.repository_metadata:
        observations.append({"source": "modules.json", "kind": "interface", "scope": "model",
                             "sufficient": component_scope, "reason": "registered module chain" if component_scope else
                             "module chain requires configuration or is not used by the selected backend"})
    return observations


def validate_execution_identity(task_info: Any) -> None:
    """Check new frozen paths before restoring a runtime; legacy records stay readable."""
    from acprof.tasks.groups import inference_signature
    from acprof.tasks.workloads import task_fingerprint
    for selected in task_info.selected_tasks:
        profile = selected.get("execution_profile", {})
        signature = profile.get("signature", {})
        if "execution" not in signature:
            continue  # Historical experiment identities are not rewritten.
        if (task_fingerprint(selected) != selected["fingerprint"] or
                inference_signature(selected, backend=signature["backend"]) != signature or
                execution_path(task_info, selected["interface"], backend=signature["backend"]) != signature["execution"]):
            raise ValueError("inference identity changed; start a new experiment")
