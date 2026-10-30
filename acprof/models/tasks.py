"""Materialize workloads from a single interface decision, not from loader hints."""
from __future__ import annotations

from typing import Any

from acprof.models.support import TASK_MAPPINGS
from acprof.tasks.groups import group_inference_tasks
from acprof.tasks.interfaces import execution_path
from acprof.tasks.workloads import canonical_task_id, parse_task, task_fingerprint


def build_task_discovery(task_info: Any, resolution: dict, *, component_scope: bool,
                         override_backend: str = "", explicit_task: bool = False) -> dict:
    config = task_info.model_config or {}
    selected = resolution["selection"]["task"]
    candidates = resolution["candidates"]
    selected_operations = set(TASK_MAPPINGS.get(selected or "", ()))

    def independent(item: dict) -> bool:
        if component_scope or explicit_task:
            return False
        if "config.custom_pipelines" in item["evidence"]:
            return True
        return ("config.architectures" in item["evidence"] and
                selected_operations.isdisjoint(TASK_MAPPINGS.get(item["task"], ())))

    eligible = [item for item in candidates if not item.get("unsupported_reason") and item["backend"] != "unknown"
                and (not selected or item["task"] == selected or independent(item))]
    interfaces = {item["task"]: item for item in eligible}
    encoder = config.get("is_encoder_decoder") is True or any(
        "SEQ_TO_SEQ_CAUSAL_LM" in TASK_MAPPINGS.get(task, ()) for task in interfaces)
    generative = encoder or "text-generation" in interfaces
    declarations = config.get("task_specific_params") or {}
    defaults = task_info.repository_metadata.get("generation_config.json") or {}
    unresolved: list[dict] = []
    conflicts = list(resolution["conflicts"])
    found: dict[str, dict] = {}
    blocked: set[str] = set()
    if not isinstance(declarations, dict):
        unresolved.append({"source": "config.task_specific_params", "reason": "expected an object"})
        declarations = {}

    def observe(name: str, source: str) -> bool:
        matches = [item for item in found.values() if name in {item["task_id"], item["kind"], item["interface"]}]
        for item in matches:
            if source not in item["evidence"]:
                item["evidence"].append(source)
        return bool(matches)

    def add(name: str, params: dict, source: str, *, usage: bool = False) -> None:
        identifier = canonical_task_id(name)
        try:
            workload = parse_task(name, params, encoder_decoder=encoder, generative=generative,
                                  source=source, default_generation=defaults)
            identifier = workload["task_id"]
            if identifier in blocked:
                return
            interface = workload["interface"]
            # A semantic use can select an existing generation route, but it
            # cannot introduce another head or modality without interface evidence.
            operations = set(TASK_MAPPINGS.get(interface, ()))
            compatible = interface in interfaces or any(
                operations & set(TASK_MAPPINGS.get(task, ())) for task in interfaces)
            if not compatible:
                if not usage:
                    conflicts.append(f"{source}: {name} conflicts with the declared execution interface")
                return
            backend = override_backend or next((item["backend"] for item in eligible
                                               if item["task"] == interface), "")
            workload["execution_path"] = execution_path(task_info, interface, backend=backend)
            workload["fingerprint"] = task_fingerprint(workload)
            if identifier in found:
                if found[identifier]["fingerprint"] != workload["fingerprint"]:
                    conflicts.append(f"{source}: conflicting configurations for {identifier}")
                    blocked.add(identifier)
                    del found[identifier]
                elif source not in found[identifier]["evidence"]:
                    found[identifier]["evidence"].append(source)
                return
            found[identifier] = workload
        except (ValueError, TypeError) as exc:
            if not usage:
                blocked.add(identifier)
                unresolved.append({"task": name, "source": source, "reason": str(exc)})

    for name, params in sorted(declarations.items()):
        add(name, params, f"config.task_specific_params.{name}")
    for item in eligible:
        name = item["task"]
        source = "hub.pipeline_tag" if "hub_metadata" in item["evidence"] else item["evidence"][0]
        if observe(name, source):
            continue
        # Generic translation labels supply no language direction. A declared
        # language pair already specifies the shared generate interface.
        if name == "translation" and any(workload["kind"] == "translation" for workload in found.values()):
            continue
        if name == "translation" and not declarations:
            name = "text2text-generation" if encoder else "text-generation"
        if not unresolved and not conflicts:
            add(name, {}, source)
    for name in (task_info.hub_metadata or {}).get("tags", []):
        if isinstance(name, str) and not component_scope and not observe(name, "hub.tags"):
            add(name, {}, "hub.tags", usage=True)
    # Prefer a declared semantic input exemplar to a generic label for the same
    # interface. Both raw facts remain in the evidence snapshot.
    for identifier, item in list(found.items()):
        if identifier == item["interface"] and any(other["interface"] == identifier and
                                                   other["task_id"] != identifier for other in found.values()):
            for other in found.values():
                if other["interface"] == identifier and other is not item:
                    other["evidence"] = list(dict.fromkeys([*other["evidence"], *item["evidence"]]))
            del found[identifier]
    tasks = sorted(found.values(), key=lambda item: item["task_id"])
    backends = {item["interface"]: item["execution_path"]["backend"] for item in tasks}
    blocked_decision = bool(conflicts or unresolved or resolution["missing"])
    profiles = [] if blocked_decision else group_inference_tasks(tasks, backend=backends)
    return {"schema_version": 2, "status": "needs_configuration" if blocked_decision or not profiles else
            "selection_required" if len(profiles) > 1 else "ready", "tasks": tasks,
            "profiles": [dict(profile_id=item["task_id"], **item["execution_profile"]) for item in profiles],
            "unresolved": unresolved, "conflicts": conflicts,
            "model_revision": task_info.model_revision}
