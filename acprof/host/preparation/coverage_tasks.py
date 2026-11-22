"""Bind one frozen coverage sample entry to a discovered workload."""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING

from acprof.failures import Failure, RuntimeFailure
from acprof.host.preparation.task_selection import TaskSelectionRequired, select_tasks
from acprof.models.inspection.decision import record_resolution_failure

if TYPE_CHECKING:
    from acprof.host.detect import TaskInfo


def _one_profile(task: TaskInfo, requested: str | None) -> TaskInfo:
    if not task.task_discovery and not task.selected_tasks:
        raise TaskSelectionRequired(())
    selected = select_tasks(task, SimpleNamespace(tasks=requested or "", task="", backend=""), single_profile=True)
    if not requested or selected.model_resolution.get("status") == "needs_configuration":
        return selected
    if not selected.selected_tasks:
        raise TaskSelectionRequired(())
    choice, = selected.selected_tasks
    profile_id = choice["task_id"]
    representative = (choice.get("execution_profile") or {}).get("representative_task_id")
    # inspect preserves preselected tasks; a different exemplar cannot satisfy a frozen request.
    if requested not in {profile_id, representative}:
        raise TaskSelectionRequired([profile_id], [representative] if representative else [])
    return selected


def _check_original_blocker(task: TaskInfo, devices: tuple[str, ...]) -> None:
    """Keep a failed checkpoint/adapter decision without attaching a sibling's successful preflight."""
    from acprof.host.preparation.task_support import require_task_support

    candidate = copy.deepcopy(task)
    try:
        require_task_support(candidate, devices=devices)
    except RuntimeFailure:
        task.model_resolution = candidate.model_resolution
        raise


def select_coverage_task(task: TaskInfo, requested: str | None, *, devices: tuple[str, ...]) -> TaskInfo:
    """Use inspect's selector only for an explicit choice or an existing selection blocker."""
    decision = task.model_resolution
    code = (decision.get("failure") or {}).get("reason_code")
    if (task.metadata_errors or code and code != "task_selection_required"
            or decision.get("status") in {"blocked", "ambiguous", "needs_configuration"}):
        return task
    if requested is None and (code or decision.get("status") != "selection_required"
                              or task.task_discovery.get("status") != "selection_required"):
        return task
    try:
        return _one_profile(task, requested)
    except TaskSelectionRequired as exc:
        if decision.get("status") in {"candidate", "resolved"}:
            _check_original_blocker(task, devices)
        action = ("Set task_id on this model in a new coverage sample. Available profile IDs: "
                  + ", ".join(exc.profile_ids) + "." if exc.profile_ids else
                  "Inspect the pinned model to establish a supported workload before setting task_id.")
        detail = (f"Cannot select frozen coverage task_id={requested!r}." if requested else
                  "The frozen coverage model requires one inference profile.")
        failure = Failure("resolution", "task_selection_required", detail, retryability="after_configuration",
                          evidence={"profile_ids": list(exc.profile_ids), "semantic_task_ids": list(exc.semantic_task_ids),
                                    "next_action": action})
        raise RuntimeFailure(record_resolution_failure(task, failure)) from exc
    except RuntimeFailure as exc:
        raise RuntimeFailure(record_resolution_failure(task, exc.failure)) from exc
    except ValueError as exc:
        raise RuntimeFailure(record_resolution_failure(task, Failure(
            "resolution", "evidence_conflict", str(exc), retryability="after_configuration",
        ))) from exc
