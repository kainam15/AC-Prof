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
    if getattr(args, "workload_spec", ""):
        from pathlib import Path

        from acprof.artifacts import read_json_object
        supplied = read_json_object(Path(args.workload_spec).expanduser(), label="workload specification")
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
    if not candidates and args.task:
        candidates = [parse_task(args.task, {}, encoder_decoder=task.model_config.get("is_encoder_decoder") is True,
                                 source="explicit_task")]
    if not candidates:
        task.model_resolution.update(status="needs_configuration",
                                     missing=["No explicit task declaration; provide --task and workload configuration"])
        return task
    available = {item["task_id"]: item for item in candidates}
    selected = getattr(args, "tasks", "")
    choices = list(available) if selected == "all" else [part.strip() for part in selected.split(",") if part.strip()]
    if not choices and len(candidates) == 1 and (not discovery["conflicts"] or manual):
        choices = list(available)
    while not choices:
        if not workflow.interactive:
            raise ValueError("Task selection required; pass --tasks " + ",".join(available) + " (or --tasks all)")
        reply = workflow.ask("resolution", "review", resolved=False,
                             fields={"Model": task.model_id, "Revision": task.model_revision},
                             questions=[{"path": "tasks", "kind": "multi", "value": [],
                                         "options": list(available), "labels": {key: task_label(value) for key, value in available.items()},
                                         "reason": "请选择要采集的任务，可多选或全选。"}],
                             summary="\n".join(f"{key}: {', '.join(value['evidence'])}" for key, value in available.items()))
        choices = reply.get("answers", {}).get("tasks")
        if not isinstance(choices, list) or not choices or any(not isinstance(item, str) for item in choices):
            raise ValueError("select at least one task")
    if len(set(choices)) != len(choices) or set(choices) - available.keys():
        raise ValueError("invalid or duplicate task selection; available tasks: " + ",".join(available))
    chosen = [available[key] for key in choices]
    if args.task and any(args.task not in {item["interface"], item["task_id"]} for item in chosen):
        raise ValueError("--task conflicts with the selected task interface; use --tasks for workload selection")
    result = activate_task(task, chosen[0], backend=args.backend)
    result.selected_tasks = copy.deepcopy(chosen)
    # Parameter choices are a preparation-time decision, never a measurement-
    # time callback. Explicit workload specifications have higher priority and
    # are already editable through their own declaration.
    if workflow.interactive and not getattr(args, "workload_spec", ""):
        from acprof.task_parameters import apply_parameter_review, parameter_review

        batch_size = getattr(args, "batch_size", 1)
        input_scales = getattr(args, "input_scales", "") or ""
        review = parameter_review(result, result.selected_tasks,
                                  batch_size=batch_size, input_scales=input_scales)
        if review["questions"]:
            error = ""
            while True:
                reply = workflow.ask(
                    "resolution", "review", resolved=False, purpose="workload_parameters",
                    fields={"Model": result.model_id, "Revision": result.model_revision},
                    questions=review["questions"],
                    parameter_advanced=review["parameter_advanced"],
                    detail=error,
                )
                try:
                    result.selected_tasks = apply_parameter_review(
                        result, result.selected_tasks, reply.get("answers"),
                        batch_size=batch_size, input_scales=input_scales,
                    )
                except (TypeError, ValueError) as exc:
                    error = str(exc)
                    continue
                break
    result.model_resolution["selected_task"] = copy.deepcopy(result.selected_tasks[0])
    if args.task:
        args.task = result.pipeline_tag
    return result
