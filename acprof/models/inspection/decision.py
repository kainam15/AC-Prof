"""Shared static failure decisions; runtime evidence remains independently scoped."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from acprof.failures import Failure

_FAILURE_STATUSES = {
    "task_selection_required": "selection_required",
    "evidence_conflict": "ambiguous",
    "model_contract_required": "needs_configuration",
}
_NEXT_ACTIONS = {
    "task_selection_required": "Choose the listed task/profile IDs with --tasks; --task selects the interface.",
    "evidence_conflict": "Review the conflicting evidence and select a compatible --task, --backend or --model-spec.",
    "model_contract_required": "Provide the missing configuration with --model-spec and inspect the model again.",
    "runtime_unavailable": "Select a compatible registered backend/runtime profile or add and validate its adapter.",
    "backend_unsupported": "Select an implemented backend for these artifacts or add a reviewed adapter.",
    "artifact_layout_unsupported": "Select artifacts supported by the registered loader; changing a Hub tag is insufficient.",
    "artifact_incomplete": "Provide the missing checkpoint files or select a complete pinned artifact snapshot.",
    "artifact_layout_unknown": "Obtain the pinned repository file list before claiming artifact compatibility.",
    "metadata_invalid": "Review the source file and field diagnostics; correct the metadata or select a valid revision.",
    "runtime_dependency_missing": "Prepare the registered runtime dependencies and inspect the same model again.",
    "runtime_dependency_incompatible": "Select a registered runtime with compatible pinned dependency versions.",
    "runtime_dependency_unknown": "Review and declare the unresolved runtime dependencies before collection.",
    "remote_code_disallowed": "Use a reviewed registered implementation whose remote-code policy permits this model.",
    "access_denied": "Check repository access and credentials, then retry the same pinned model.",
    "precision_mismatch": "Choose a supported precision or a compatible registered runtime.",
    "runtime_task_unsupported": "Review the task/backend mapping and use a supported collection interface.",
}


def clear_preflight_failure(task_info: Any) -> None:
    """Re-evaluate a support attempt while preserving upstream detection failures."""
    resolution = task_info.model_resolution
    previous = resolution.get("failure", {})
    evidence = previous.get("evidence", {})
    if "preflight_input_status" not in evidence:
        return
    if resolution.get("status") == _FAILURE_STATUSES.get(previous["reason_code"], "blocked"):
        if evidence["preflight_input_status"] is None:
            resolution.pop("status", None)
        else:
            resolution["status"] = evidence["preflight_input_status"]
    resolution.pop("failure")
    resolution.pop("compatibility", None)
    resolution.pop("precision", None)


def record_resolution_failure(task_info: Any, failure: Failure) -> Failure:
    """Keep the original cause and interface provenance while blocking collection."""
    action = failure.evidence.get("next_action") or _NEXT_ACTIONS.get(
        failure.reason_code, "Review the recorded failure and resolve it before collection.",
    )
    failure = replace(failure, evidence={**failure.evidence, "next_action": action})
    task_info.model_resolution.update(
        status=_FAILURE_STATUSES.get(failure.reason_code, "blocked"),
        failure=failure.to_dict(),
    )
    return failure
