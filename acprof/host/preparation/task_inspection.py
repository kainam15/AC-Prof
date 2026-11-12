"""Inspect selected workloads with the same selection and preflight as collection."""
from __future__ import annotations

import copy
from dataclasses import replace

from acprof.failures import Failure, RuntimeFailure
from acprof.host.preparation.task_selection import (
    TaskSelectionRequired,
    activate_task,
    select_tasks,
)
from acprof.models.inspection.decision import record_resolution_failure


def _record_failure(task, failure: Failure) -> RuntimeFailure:
    failure = record_resolution_failure(task, failure)
    return RuntimeFailure(failure)


def inspect_selected_tasks(task, args):
    """Resolve choices without prompts, then inspect every selected profile.

    All profile checks start from one unchanged selection snapshot. Their reports
    remain separate so a successful first profile cannot hide a blocked sibling.
    """
    failure = task.model_resolution.get("failure")
    retry_selection = (failure and failure.get("reason_code") == "task_selection_required"
                       and getattr(args, "tasks", ""))
    if failure and not retry_selection:
        return task, _record_failure(task, Failure(**failure))
    if task.model_resolution.get("status") in {"ambiguous", "needs_configuration", "blocked"}:
        return task, None
    try:
        task = select_tasks(task, args)
    except TaskSelectionRequired as exc:
        return task, _record_failure(task, Failure(
            "preflight", "task_selection_required", str(exc), retryability="after_configuration",
            evidence={"profile_ids": list(exc.profile_ids), "semantic_task_ids": list(exc.semantic_task_ids)},
        ))
    except RuntimeFailure as exc:
        return task, _record_failure(task, exc.failure)
    except ValueError as exc:
        return task, _record_failure(task, Failure(
            "preflight", "evidence_conflict", str(exc), retryability="after_configuration",
        ))

    selected = copy.deepcopy(task.selected_tasks)
    task.model_resolution["selected_tasks"] = copy.deepcopy(selected)
    if getattr(args, "probe_interface", False) and len(selected) > 1:
        commands = ", ".join("--tasks " + item["task_id"] for item in selected)
        return task, _record_failure(task, Failure(
            "preflight", "task_selection_required",
            "Interface probing requires one inference profile; inspect profiles separately with " + commands,
            retryability="after_configuration", evidence={"profile_ids": [item["task_id"] for item in selected]},
        ))

    from acprof.host.preparation.task_support import TaskSupportError, require_task_support

    reports = {}
    inspected = []
    failures = []
    for choice in selected or [None]:
        current = activate_task(task, choice, backend=args.backend) if len(selected) > 1 else task
        current.model_resolution["selected_tasks"] = copy.deepcopy(current.selected_tasks)
        try:
            require_task_support(current, devices=("cpu",))
        except TaskSupportError as exc:
            # Also normalize failures from external or mocked preflight providers.
            error = _record_failure(current, exc.failure)
            failures.append((choice, error))
        inspected.append(current)
        if choice is not None:
            reports[choice["task_id"]] = copy.deepcopy(current.model_resolution)

    result = inspected[0]
    if len(selected) > 1:
        result.selected_tasks = selected
        result.model_resolution["selected_tasks"] = copy.deepcopy(selected)
        result.model_resolution["task_resolutions"] = reports
        states = {item.get("compatibility", {}).get("status", "unknown") for item in reports.values()}
        result.model_resolution["compatibility"] = {
            "status": "unsupported" if "unsupported" in states else "supported" if states == {"supported"} else "unknown",
            "source": "selected_task_profiles", "evidence_level": "static", "checkpoint_verified": False,
        }
    if failures:
        choice, error = failures[0]
        failure = error.failure
        if len(selected) > 1:
            failure = replace(
                failure, detail=f"{choice['task_id']}: {failure.detail}",
                evidence={**failure.evidence, "task_id": choice["task_id"]},
            )
        return result, _record_failure(result, failure)
    return result, None
