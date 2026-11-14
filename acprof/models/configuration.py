"""Static configuration discovery shared by inspection and task parameter review.

This module describes declarations and observed defaults. It does not resolve
Auto, freeze an execution configuration, or authorize a runtime implementation.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from acprof.extensions import CATALOG, select_extension
from acprof.extensions.parameters import parameter_spec_to_dict


def parameter_declaration(task_info: Any, selected_task: dict | None = None):
    """Resolve the existing extension for one selected task without mutating it."""
    selected = selected_task or {}
    view = copy.copy(task_info)
    view.pipeline_tag = selected.get("interface", task_info.pipeline_tag)
    view.task_family = CATALOG.task_families.get(view.pipeline_tag, task_info.task_family)
    view.runtime_backend = selected.get("execution_profile", {}).get(
        "signature", {},
    ).get("backend", task_info.runtime_backend)
    declaration = select_extension(view)
    config = view.model_config
    pipeline_class = config.get("_class_name") or view.repository_metadata.get("model_index.json", {}).get("_class_name")
    return replace(declaration, parameters=tuple(
        spec for spec in declaration.parameters
        if (not spec.model_types or config.get("model_type") in spec.model_types)
        and (not spec.pipeline_classes or pipeline_class in spec.pipeline_classes)
    ))


def describe_configuration_options(
    task_info: Any, *, runtime_profile_constraint: str = "",
) -> dict:
    """Describe registered options, with an optional explicit profile constraint.

    Discovery is independent of a previously auto-selected runtime profile.
    Callers that have an explicit profile request must pass it as a constraint.
    The input's current selection and all validation evidence remain untouched.
    """
    from acprof.failures import RuntimeFailure
    from acprof.models.backend_candidates import describe_backend_candidates
    from acprof.models.configuration_sources import discover_configuration_sources

    if not isinstance(runtime_profile_constraint, str):
        raise ValueError("runtime_profile_constraint must be a string")
    selected_tasks = getattr(task_info, "selected_tasks", ()) or (None,)
    interfaces = sorted({
        selected["interface"] if selected else task_info.pipeline_tag
        for selected in selected_tasks
    })
    # Only the optional profile constraint changes here. Backend discovery owns
    # its isolated working state and validates metadata before copying it.
    candidate_view = copy.copy(task_info)
    candidate_view.runtime_profile_id = runtime_profile_constraint
    candidates = []
    for interface in interfaces:
        candidates.extend(describe_backend_candidates(candidate_view, task=interface))

    task_options, diagnostics = [], []
    for selected in selected_tasks:
        interface = selected["interface"] if selected else task_info.pipeline_tag
        backend = (selected or {}).get("execution_profile", {}).get(
            "signature", {},
        ).get("backend", task_info.runtime_backend)
        task_id = selected["task_id"] if selected else None
        try:
            declaration = parameter_declaration(task_info, selected)
        except (ValueError, RuntimeFailure) as exc:
            specs, extension_id = (), None
            diagnostics.append({
                "task_id": task_id, "task": interface, "backend": backend,
                "reason_code": "parameter_declaration_unavailable", "detail": str(exc),
            })
        else:
            specs, extension_id = declaration.parameters, declaration.extension_id
        task_options.append({
            "task_id": task_id, "task": interface, "backend": backend,
            "extension_id": extension_id,
            "parameters": [parameter_spec_to_dict(spec) for spec in specs],
            "source_discovery": discover_configuration_sources(task_info, specs, selected),
        })
    return {
        "schema_version": 1, "phase": "discovery", "evidence_level": "static",
        "runtime_validation": "not_run", "frozen": False,
        "runtime_profile_constraint": runtime_profile_constraint or None,
        "backend_candidates": candidates, "task_options": task_options,
        "diagnostics": diagnostics,
    }
