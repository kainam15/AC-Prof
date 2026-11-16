"""Read-only backend offers from registered routes and available static evidence."""
from __future__ import annotations

import copy
from typing import Any

from acprof.extensions import CATALOG, UnsupportedExtensionError, select_extension
from acprof.failures import Failure, RuntimeFailure
from acprof.models.candidates import apply_model_candidate, discover_model_candidates
from acprof.models.inspection.decision import (
    clear_preflight_failure,
    record_resolution_failure,
)
from acprof.models.metadata_validation import finite_metadata_diagnostics
from acprof.models.resolution import describe_static_compatibility, resolve_model_interface
from acprof.models.spec import task_model_spec, validate_model_spec
from acprof.models.support import supports_transformers_task
from acprof.precision import remote_code_allowed
from acprof.runtime.dependencies import dependency_preflight
from acprof.runtime.profiles import locked_transformers_version, select_runtime_profile
from acprof.tasks.interfaces import validate_execution_identity

_UNKNOWN_FAILURES = {
    "metadata_invalid", "model_contract_required", "evidence_conflict",
    "task_selection_required", "artifact_layout_unknown", "runtime_dependency_unknown",
}


def _upstream_failure(task_info: Any, *, explicit_task: bool) -> Failure | None:
    resolution = task_info.model_resolution
    saved = resolution.get("failure")
    if saved and not (explicit_task and saved.get("reason_code") == "task_selection_required"):
        return Failure(**saved)
    diagnostics = list(getattr(task_info, "metadata_diagnostics", ()))
    sources = getattr(task_info, "metadata_sources", {})
    documents = dict(task_info.repository_metadata)
    if task_info.model_config is not documents.get("config.json"):
        documents["model_config"] = task_info.model_config
    documents["hub_metadata"] = task_info.hub_metadata
    documents["model_spec"] = task_info.model_spec
    for filename, value in documents.items():
        source = sources.get(filename, {})
        raw = source.get("sha256") if isinstance(source, dict) and source.get("encoding") == "raw" else None
        diagnostics.extend(finite_metadata_diagnostics(value, filename, raw_sha256=raw))
    errors = getattr(task_info, "metadata_errors", ())
    if diagnostics or errors:
        return Failure(
            "resolution", "metadata_invalid", "; ".join(errors) or "Model metadata has invalid values.",
            evidence={"diagnostics": list(diagnostics), "metadata_errors": list(errors)},
        )
    if (resolution.get("status") == "ambiguous" or resolution.get("conflicts")
            or resolution.get("semantics", {}).get("status") == "conflict"):
        return Failure(
            "resolution", "evidence_conflict", "; ".join(resolution.get("conflicts", ())) or "Model evidence conflicts.",
            evidence={"conflicts": copy.deepcopy(resolution.get("conflicts", []))},
        )
    spec = task_model_spec(task_info)
    if spec:
        try:
            validate_model_spec(spec)
        except (ValueError, TypeError) as exc:
            return Failure("resolution", "model_contract_required", str(exc), evidence={"source": "model_spec"})
    contract = resolution.get("contract", {})
    if resolution.get("status") in {"needs_configuration", "blocked"} or (contract and contract.get("status") != "resolved"):
        return Failure(
            "resolution", "model_contract_required",
            "; ".join(resolution.get("missing", ())) or "Model interface configuration is incomplete.",
            evidence={"missing": copy.deepcopy(resolution.get("missing", [])),
                      "contract_status": contract.get("status")},
        )
    if not explicit_task and resolution.get("status") == "selection_required":
        return Failure("resolution", "task_selection_required", "Choose an inference task before selecting its backend.")
    return None


def _record_failure(candidate: dict[str, Any], task_info: Any, failure: Failure) -> None:
    failure = record_resolution_failure(task_info, failure)
    candidate.update(status=task_info.model_resolution["status"], selectable=False, failure=failure.to_dict())
    candidate["compatibility"].update(
        status="unknown" if failure.reason_code in _UNKNOWN_FAILURES else "unsupported",
        reason_code=failure.reason_code, detail=failure.detail,
        next_action=failure.evidence["next_action"],
    )
    if failure.runtime_profile and not candidate["runtime_profile"]:
        candidate["runtime_profile"] = failure.runtime_profile


def _available_source(task_info: Any, name: str) -> str:
    source = getattr(task_info, "repository_sources", {}).get(name)
    if not isinstance(source, str):
        raise OSError(f"Source metadata {name!r} is not available in this inspection snapshot.")
    return source


def _value_failure(task_info: Any, exc: ValueError) -> Failure:
    resolution = task_info.model_resolution
    if resolution.get("conflicts") or resolution.get("semantics", {}).get("status") == "conflict":
        code = "evidence_conflict"
    elif resolution.get("status") == "selection_required":
        code = "task_selection_required"
    elif resolution.get("missing") or resolution.get("status") == "needs_configuration":
        code = "model_contract_required"
    else:
        code = "runtime_task_unsupported"
    return Failure(
        "resolution", code, str(exc), runtime_profile=task_info.runtime_profile_id,
        evidence={"task": task_info.pipeline_tag, "backend": task_info.runtime_backend},
    )


def _restore_workload_selection(task_info: Any) -> None:
    """Retain a validated active group without rewriting its backend identity."""
    selected = [item for item in task_info.selected_tasks if item["interface"] == task_info.pipeline_tag]
    if not selected or task_info.model_resolution.get("status") != "selection_required":
        return
    backends = {item.get("execution_profile", {}).get("signature", {}).get("backend")
                or item.get("execution_path", {}).get("backend") for item in selected}
    if backends != {task_info.runtime_backend}:
        raise RuntimeFailure(Failure(
            "resolution", "task_selection_required",
            "Select the workload for this backend; its recorded execution profile belongs to another backend.",
            evidence={"backend": task_info.runtime_backend, "recorded_backends": sorted(str(item) for item in backends)},
        ))
    try:
        validate_execution_identity(task_info)
    except ValueError as exc:
        raise RuntimeFailure(Failure(
            "resolution", "evidence_conflict", str(exc), evidence={"source": "selected_tasks"},
        )) from exc
    task_info.model_resolution.update(status="candidate", selected_task=copy.deepcopy(selected[0]))
    if selected[0].get("execution_path"):
        task_info.model_resolution["execution"] = copy.deepcopy(selected[0]["execution_path"])


def describe_backend_candidates(task_info: Any, *, task: str | None = None) -> list[dict[str, Any]]:
    """Describe preparable registered routes, without choosing one or loading code.

    A runtime_profile_id supplied on TaskInfo remains a hard constraint. Callers
    evaluating a fresh Auto choice must provide a copy without a profile constraint.
    Static eligibility never establishes successful loading, inference or measurement.
    """
    selected_task = task if task is not None else task_info.pipeline_tag
    family = CATALOG.task_families.get(selected_task)
    if family is None:
        return []
    registered = {
        backend for (route_family, backend, _adapter), declaration in CATALOG.routes.items()
        if route_family == family and selected_task in declaration.backend_tasks.get(backend, declaration.tasks)
    }
    backends = set(registered)
    if task_info.pipeline_tag == selected_task and task_info.runtime_backend not in {"", "unknown"}:
        backends.add(task_info.runtime_backend)
    original = copy.copy(task_info)
    original.model_resolution = dict(task_info.model_resolution)
    clear_preflight_failure(original)
    upstream = _upstream_failure(original, explicit_task=task is not None)
    if upstream is None and original.model_resolution.get("failure", {}).get("reason_code") == "task_selection_required":
        original.model_resolution.pop("failure")
    candidates = []
    for backend in sorted(backends):
        context = copy.copy(original) if upstream is not None else copy.deepcopy(original)
        context.model_resolution = dict(context.model_resolution)
        for key in ("compatibility", "precision", "dependency_preflight", "runtime_validation", "task_resolutions"):
            context.model_resolution.pop(key, None)
        context.pipeline_tag, context.task_family, context.runtime_backend = selected_task, family, backend
        candidate = {
            "schema_version": 1, "id": f"{selected_task}/{backend}", "task": selected_task, "family": family,
            "backend": backend, "registered": backend in registered, "status": "candidate", "selectable": False,
            "extension_id": None, "engine": None, "handler": None, "adapter": None,
            "runtime_profile": context.runtime_profile_id or None, "environment": None,
            "loader": None, "operation": None,
            "compatibility": {"status": "unknown", "evidence_level": "static", "checkpoint_verified": False},
            "runtime_validation": {"status": "not_run"},
            "author_recommendation": {"status": "unknown", "evidence": []},
            "observed_library": context.library_name,
        }
        candidates.append(candidate)
        try:
            if upstream is not None:
                raise RuntimeFailure(upstream)
            if not candidate["registered"]:
                raise RuntimeFailure(Failure(
                    "resolution", "backend_unsupported", f"No registered {selected_task}/{backend} route.",
                    evidence={"task": selected_task, "backend": backend},
                ))
            extension = select_extension(context)
            candidate.update(
                extension_id=extension.extension_id, engine=extension.runtime,
                handler=extension.handler_entrypoint, adapter=extension.adapter,
            )
            decision = discover_model_candidates(context, override_tag=selected_task, override_backend=backend)
            context.model_resolution.update(apply_model_candidate(context, decision))
            _restore_workload_selection(context)
            context.model_resolution = resolve_model_interface(context)
            profile = select_runtime_profile(context)
            if profile.environment is None:
                raise ValueError(f"Runtime profile {profile.profile_id!r} has no dependency environment")
            candidate.update(
                runtime_profile=profile.profile_id, environment=profile.environment.environment_key,
                loader=context.model_resolution["loader"], operation=context.model_resolution["operation"],
                artifact_layout=copy.deepcopy(context.model_resolution.get("artifact_layout", {})),
                compatibility=describe_static_compatibility(context, profile, extension),
            )
            config = context.model_config or {}
            version = locked_transformers_version(profile.environment) or ""
            remote_required = config.get("custom_pipelines") or (config.get("auto_map") and version and
                supports_transformers_task(version, selected_task, config.get("model_type", ""), config) is not True)
            if remote_required and not remote_code_allowed(profile, extension):
                raise RuntimeFailure(Failure(
                    "preflight", "remote_code_disallowed",
                    "Model requires remote code but its registered runtime/extension policy forbids it.",
                    runtime_profile=profile.profile_id,
                    evidence={"trust_remote_code": False, "code_files": context.model_resolution.get("code_files", [])},
                ))
            dependency_preflight(context, profile, read_source=lambda name: _available_source(context, name))
            if not context.repository_files and candidate["compatibility"]["status"] == "supported":
                raise RuntimeFailure(Failure(
                    "resolution", "artifact_layout_unknown",
                    "Repository file evidence is unavailable; artifact compatibility has not been verified.",
                    runtime_profile=profile.profile_id,
                ))
            candidate["selectable"] = candidate["compatibility"]["status"] == "supported"
        except RuntimeFailure as exc:
            _record_failure(candidate, context, exc.failure)
        except UnsupportedExtensionError as exc:
            _record_failure(candidate, context, upstream or Failure(
                "resolution", "backend_unsupported", str(exc),
                evidence={"task": selected_task, "backend": backend},
            ))
        except ValueError as exc:
            _record_failure(candidate, context, upstream or _value_failure(context, exc))
        except OSError as exc:
            _record_failure(candidate, context, upstream or Failure(
                "dependency_preflight", "runtime_dependency_unknown", str(exc),
                runtime_profile=context.runtime_profile_id,
            ))
        if context.model_resolution.get("dependency_preflight"):
            candidate["dependency_preflight"] = copy.deepcopy(context.model_resolution["dependency_preflight"])
    return candidates
