"""Resolve declared workload choices before environment or weight preparation."""
from __future__ import annotations

import copy

from acprof.task_workloads import parse_task, task_fingerprint, task_label


def activate_task(task, selected: dict, *, backend: str = ""):
    from acprof.model_candidates import apply_model_candidate, discover_model_candidates

    result = copy.deepcopy(task)
    result.selected_tasks = [copy.deepcopy(selected)]
    result.runtime_profile_id = ""
    if result.pipeline_tag != selected["interface"] or backend and backend != result.runtime_backend:
        result.model_resolution = apply_model_candidate(result, discover_model_candidates(
            result, override_tag=selected["interface"], override_backend=backend or None))
    profile = selected.get("execution_profile")
    if profile and result.runtime_backend != profile["signature"]["backend"]:
        raise ValueError("inference profile backend differs from selected runtime backend")
    result.model_resolution["selected_task"] = copy.deepcopy(selected)
    result.model_resolution["task_discovery"] = copy.deepcopy(task.task_discovery)
    return result


def select_tasks(task, args, workflow):
    if task.selected_tasks:
        return task
    discovery = task.task_discovery
    if not discovery:
        return task  # Historical frozen experiments predate declared task discovery.
    candidates = discovery["tasks"]
    manual = False
    overrides: dict = {}
    if getattr(args, "workload_spec", ""):
        from pathlib import Path

        from acprof.artifacts import read_json_object
        supplied = read_json_object(Path(args.workload_spec).expanduser(), label="workload specification")
        # A user workload can change Beam/Sampling, so grouping must use the
        # effective compute options, not only checkpoint task declarations.
        from acprof.task_workloads import GENERATION_KEYS
        params = supplied.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("workload params must be an object")
        overrides = {key: value for key, value in params.items() if key in GENERATION_KEYS}
        declaration = supplied.get("task_declaration")
        if declaration is not None:
            if (supplied.get("schema_version") != 1 or not isinstance(declaration, dict)
                    or set(declaration) - {"task_id", "params"} or not declaration.get("task_id")):
                raise ValueError("task_declaration requires task_id and optional params")
            candidates = [parse_task(declaration["task_id"], declaration.get("params", {}),
                                     encoder_decoder=task.model_config.get("is_encoder_decoder") is True,
                                     default_generation=task.repository_metadata.get("generation_config.json"),
                                     source="workload_spec.task_declaration")]
            manual = True
        elif supplied.get("task_workload"):
            frozen = supplied["task_workload"]
            if (supplied.get("schema_version") != 1 or frozen.get("fingerprint") != task_fingerprint(frozen)
                    or supplied.get("task") != frozen["interface"]):
                raise ValueError("invalid frozen task workload")
            candidates = [copy.deepcopy(frozen)]
            manual = True
            if "execution_profile" in frozen and overrides:
                from acprof.inference_groups import inference_signature
                from acprof.task_workloads import (
                    GENERATIVE_INTERFACES,
                    merge_generation_parameters,
                    validate_generation,
                )
                effective = copy.deepcopy(frozen)
                if effective["interface"] in GENERATIVE_INTERFACES:
                    effective["config"]["generation"] = merge_generation_parameters(
                        effective["config"].get("generation", {}), overrides)
                    validate_generation(effective["config"]["generation"])
                original = frozen["execution_profile"]["signature"]
                if inference_signature(effective, backend=original["backend"]) != original:
                    raise ValueError("frozen inference profile cannot change compute settings; "
                                     "select a new task declaration instead")
    if not candidates and args.task:
        candidates = [parse_task(args.task, {}, encoder_decoder=task.model_config.get("is_encoder_decoder") is True,
                                 source="explicit_task")]
    if not candidates:
        task.model_resolution.update(status="needs_configuration",
                                     missing=["No explicit task declaration; provide --task and workload configuration"])
        return task
    from acprof.inference_groups import group_inference_tasks
    from acprof.model_candidates import resolve_task_backends

    backends = resolve_task_backends(task, candidates, override_backend=args.backend)
    # Users select independent inference paths, not translations or prompts.
    # Explicit semantic IDs remain useful for choosing the input exemplar.
    profiles = ([copy.deepcopy(candidates[0])] if len(candidates) == 1 and
                "execution_profile" in candidates[0] else
                group_inference_tasks(candidates, backend=backends,
                                      generation_override=overrides))
    available = {item["task_id"]: item for item in profiles}
    semantics = {item["task_id"]: item for profile in profiles
                 for item in profile["semantic_tasks"]}
    selected = getattr(args, "tasks", "")
    if selected:
        choices = list(available) if selected == "all" else [part.strip() for part in selected.split(",") if part.strip()]
        if len(set(choices)) != len(choices) or not choices or set(choices) - (available.keys() | semantics.keys()):
            raise ValueError("invalid or duplicate task selection; available tasks: " + ",".join(semantics))
        if all(identifier in available for identifier in choices):
            chosen = [available[identifier] for identifier in choices]
        else:
            # Expand selected profiles and task IDs to a single set, then group
            # once. Mixed/exact duplicates never schedule duplicate matrices.
            task_ids = {identifier for choice in choices for identifier in (
                available[choice]["execution_profile"]["semantic_task_ids"]
                if choice in available else [choice])}
            chosen = group_inference_tasks([semantics[key] for key in sorted(task_ids)],
                                           backend=backends,
                                           generation_override=overrides)
    elif len(profiles) == 1 and (not discovery["conflicts"] or manual):
        chosen = profiles
    else:
        if not workflow.interactive:
            raise ValueError("Inference profile selection required; pass --tasks " +
                             ",".join(available) + " (or --tasks all)")
        reply = workflow.ask("resolution", "review", resolved=False,
                             fields={"Model": task.model_id, "Revision": task.model_revision},
                             questions=[{"path": "tasks", "kind": "multi", "value": [],
                                         "options": list(available),
                                         "labels": {key: task_label(value) for key, value in available.items()},
                                         "reason": "请选择要采集的推理方式，同一方式下的不同输入任务会自动合并。"}],
                             summary="\n".join(f"{key}: {', '.join(value['execution_profile']['semantic_task_ids'])}"
                                               for key, value in available.items()))
        choices = reply.get("answers", {}).get("tasks")
        if (not isinstance(choices, list) or not choices or
                any(not isinstance(item, str) for item in choices) or
                len(set(choices)) != len(choices) or set(choices) - available.keys()):
            raise ValueError("select at least one valid inference profile")
        chosen = [available[key] for key in choices]
    if args.task and any(args.task not in {item["interface"], item["task_id"],
                                          *item["execution_profile"]["semantic_task_ids"]} for item in chosen):
        raise ValueError("--task conflicts with the selected inference interface; use --tasks for workload selection")
    result = activate_task(task, chosen[0], backend=args.backend)
    result.selected_tasks = copy.deepcopy(chosen)
    if args.task:
        args.task = result.pipeline_tag
    return result
