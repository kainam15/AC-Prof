"""Resolve declared workload choices before environment or weight preparation."""
from __future__ import annotations

import copy

from acprof.task_workloads import parse_task, task_fingerprint


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
    from acprof.inference_interfaces import validate_execution_identity
    validate_execution_identity(result)
    if result.model_resolution.get("status") == "selection_required":
        result.model_resolution["status"] = "candidate"
    if selected.get("execution_path"):
        result.model_resolution["execution"] = copy.deepcopy(selected["execution_path"])
    result.model_resolution["selected_task"] = copy.deepcopy(selected)
    result.model_resolution["task_discovery"] = copy.deepcopy(task.task_discovery)
    return result


def select_tasks(task, args, workflow):
    if task.selected_tasks:
        from acprof.inference_interfaces import validate_execution_identity
        validate_execution_identity(task)
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
                from acprof.inference_groups import matches_recorded_signature
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
                if not matches_recorded_signature(effective, original):
                    raise ValueError("frozen inference profile cannot change compute settings; "
                                     "select a new task declaration instead")
    if not candidates and args.task:
        candidates = [parse_task(args.task, {}, encoder_decoder=task.model_config.get("is_encoder_decoder") is True,
                                 source="explicit_task")]
    if not candidates or (discovery.get("status") == "needs_configuration" and not manual and not args.task):
        reasons = [item["reason"] for item in discovery.get("unresolved", [])]
        task.model_resolution.update(status="needs_configuration",
                                     missing=[*task.model_resolution.get("missing", []), *reasons,
                                              "无法确认完整推理接口，请补充模型声明或输入配置。"])
        return task
    from acprof.inference_groups import group_inference_tasks
    from acprof.model_candidates import resolve_task_backends

    backends = resolve_task_backends(task, candidates, override_backend=args.backend)
    from acprof.inference_interfaces import execution_path
    candidates = copy.deepcopy(candidates)
    for candidate in candidates:
        # Frozen historical groups keep their original identity. New workload
        # declarations bind the same registered path as automatic discovery.
        if "execution_profile" not in candidate and "execution_path" not in candidate:
            candidate["execution_path"] = execution_path(task, candidate["interface"],
                                                         backend=backends[candidate["interface"]])
            candidate["fingerprint"] = task_fingerprint(candidate)
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
                                         "profiles": copy.deepcopy(profiles)}])
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
    result.model_resolution["selected_task"] = copy.deepcopy(result.selected_tasks[0])
    if args.task:
        args.task = result.pipeline_tag
    return result


def review_selected_task_parameters(task, args, workflow):
    """Confirm workload controls once, *after* the full interface review.

    select_tasks may run repeatedly as a user resolves missing model metadata.
    Its pure selection step must never trigger this modal. Resume restores a
    frozen task snapshot instead of asking for new values.
    """
    if (not task.selected_tasks or not workflow.interactive
            or bool(getattr(args, "workload_spec", None)) or getattr(args, "resume", False)):
        return task
    from acprof.task_parameters import apply_parameter_review, parameter_review

    result = copy.deepcopy(task)
    batch_size = getattr(args, "batch_size", 1)
    input_scales = getattr(args, "input_scales", "") or ""
    review = parameter_review(result, result.selected_tasks,
                              batch_size=batch_size, input_scales=input_scales)
    if not review["questions"]:
        return result
    error = ""
    while True:
        reply = workflow.ask(
            "resolution", "review", resolved=False, purpose="workload_parameters",
            fields={"Model": result.model_id, "Revision": result.model_revision},
            questions=review["questions"], parameter_advanced=review["parameter_advanced"],
            detail=error,
        )
        try:
            selected = apply_parameter_review(
                result, result.selected_tasks, reply.get("answers"),
                batch_size=batch_size, input_scales=input_scales,
            )
        except (TypeError, ValueError) as exc:
            error = str(exc)
            continue
        result.selected_tasks = selected
        result.model_resolution["selected_task"] = copy.deepcopy(selected[0])
        return result
