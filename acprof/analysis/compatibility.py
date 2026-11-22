"""Render compatibility evidence without reinterpreting exception messages."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import NamedTuple, TextIO

from acprof.artifacts import (
    atomic_write,
    loads_finite_json,
    recorded_artifact_present,
    reject_isolated_surrogates,
)
from acprof.failures import Failure, compatibility_status, ordered_validation_failures
from acprof.hub.repository import MODEL_SOURCES
from acprof.models.evidence import pinned_revision
from acprof.quality import combine_quality, read_quality, summarize_quality
from acprof.results.validation_cleanup import (
    cleanup_summary as _cleanup_summary,
    public_cleanup_error as _public_cleanup_error,
)
from acprof.system.redaction import redact_text

MAX_COMPATIBILITY_ARTIFACT_BYTES = 4 * 1024 * 1024


def _read_recorded_object(path: Path) -> dict:
    with path.open("rb") as stream:
        content = stream.read(MAX_COMPATIBILITY_ARTIFACT_BYTES + 1)
    if len(content) > MAX_COMPATIBILITY_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds the 4 MiB read limit")
    payload = loads_finite_json(content)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    reject_isolated_surrogates(payload)
    return payload


def _recorded_evidence_failure(path: Path, error: Exception) -> dict:
    return Failure(
        "compatibility_report", "recorded_evidence_invalid", f"{path.name}: {error}",
        evidence={"artifact": str(path.absolute()), "artifact_name": path.name},
        exception_type=type(error).__name__,
    ).to_dict()


class RecordedSourceError(ValueError):
    def __init__(self, artifact: str, detail: str, *, additional: tuple[tuple[str, str], ...] = ()):
        super().__init__(detail)
        self.artifact = artifact
        self.problems = ((artifact, detail), *additional)


class RecordedModelIdentity(NamedTuple):
    source: str = "unknown"
    model_id: str | None = None
    commit: str | None = None

    def revision(self, *historical: object) -> str:
        if self.source == "unknown":
            return "unknown"
        return self.commit or next((ref for ref in historical if isinstance(ref, str) and ref), "unknown")


def _primary_model_coordinates(records: tuple[tuple[str, str, object, object], ...]) -> tuple[str | None, str | None]:
    known: dict[str, tuple[str, str]] = {}
    conflicts: dict[tuple[str, str], str] = {}
    for artifact, location, model, revision in records:
        for field, value in (("model_id", model), ("revision", revision)):
            if value in (None, ""):
                continue
            if not isinstance(value, str):
                conflicts.setdefault((artifact, field), f"{location} {field} must be a string")
                continue
            if field == "revision" and not pinned_revision(value):
                # A saved branch or tag cannot establish a different resolved snapshot.
                continue
            previous = known.get(field)
            if previous and previous[0] != value:
                conflicts.setdefault((artifact, field), f"{location} {field} conflicts with {previous[1]}")
            known.setdefault(field, (value, f"{artifact} {location}"))
    if conflicts:
        first, *others = [(name, detail) for (name, _field), detail in conflicts.items()]
        raise RecordedSourceError(*first, additional=tuple(others))
    model = known.get("model_id")
    revision = known.get("revision")
    return model[0] if model else None, revision[0] if revision else None


def recorded_model_identity(metadata: dict, resolution: dict, state: dict, *,
                          resolution_name: str = "model_resolution.json", interface: dict | None = None,
                          interface_name: str = "interface_validation.json") -> RecordedModelIdentity:
    """Resolve the source and immutable coordinates of the recorded primary model."""
    provenance = resolution.get("provenance", {})
    if not isinstance(provenance, dict):
        raise RecordedSourceError(resolution_name, "model provenance must be an object")
    sources = provenance.get("sources", {})
    if not isinstance(sources, dict):
        raise RecordedSourceError(resolution_name, "model provenance sources must be an object")
    snapshot = sources.get("repository_snapshot", {})
    if not isinstance(snapshot, dict):
        raise RecordedSourceError(resolution_name, "repository snapshot provenance must be an object")
    task = _dict_or_empty(_dict_or_empty(state.get("runtime")).get("task"))
    options = _dict_or_empty(state.get("options"))
    iface = interface or {}
    model, commit = _primary_model_coordinates((
        ("static_meta.json", "static metadata", metadata.get("model_id"), metadata.get("model_revision")),
        ("static_meta.json", "model_name", metadata.get("model_name"), None),
        (resolution_name, "model resolution", resolution.get("model_id"), resolution.get("model_revision")),
        (resolution_name, "provenance", provenance.get("model_id"), provenance.get("revision")),
        (resolution_name, "repository_snapshot", snapshot.get("model_id"), snapshot.get("revision")),
        (interface_name, "interface validation", iface.get("model_id"), iface.get("revision")),
        ("run_state.json", "runtime.task", task.get("model_id"), task.get("model_revision")),
    ))
    source = None
    for artifact, record, key in (
        ("static_meta.json", metadata, "model_source"),
        (resolution_name, snapshot, "source"),
        ("run_state.json", task, "model_source"),
        ("run_state.json", options, "model_source"),
        ("run_state.json", _dict_or_empty(options.get("extra_options")), "model_source"),
    ):
        if key not in record:
            continue
        value = record[key]
        if value not in MODEL_SOURCES:
            raise RecordedSourceError(artifact, "recorded model source must be huggingface/modelscope")
        if source is not None and value != source:
            raise RecordedSourceError(artifact, f"conflicting recorded model sources: {source} and {value}")
        source = value
    return RecordedModelIdentity(source if source is not None else "huggingface", model, commit)


def recorded_model_source(metadata: dict, resolution: dict, state: dict, *,
                          resolution_name: str = "model_resolution.json", interface: dict | None = None,
                          interface_name: str = "interface_validation.json") -> str:
    """Resolve the repository source, including from preparation-only runs."""
    return recorded_model_identity(
        metadata, resolution, state, resolution_name=resolution_name, interface=interface, interface_name=interface_name,
    ).source


def _validate_run_outcome(status: object, outcome: object) -> None:
    # Historical reports can also record error/cancelled. Import does not require a resumable state.
    if status not in ("preparing", "running", "complete", "interrupted", "failed", "error", "cancelled"):
        raise ValueError("missing or invalid run status")
    if outcome not in (None, "ok", "partial"):
        raise ValueError("invalid run outcome")


def result_status(row):
    """Combine recorded stages without treating a completed probe as a completed collection."""
    if row.get("failure"):
        return compatibility_status(row["failure"])
    run_status = row.get("run_status")
    if "run_status" in row or "run_outcome" in row:
        _validate_run_outcome(run_status, row.get("run_outcome"))
        if run_status != "complete":
            return run_status
        if row.get("run_outcome") == "partial":
            return "partial"
    runtime_status = row.get("runtime_status", "unverified")
    if runtime_status not in ("ok", "unverified", "not_run", "not_requested"):
        return runtime_status
    if row.get("cleanup_status") == "incomplete":
        return "error"
    if row.get("full_profile_complete") is True:
        severities = {check["severity"] for check in (row.get("quality_checks") or [])}
        if row.get("quality_status") == "blocked" or "error" in severities:
            return "full_success_quality_blocked"
        if row.get("quality_status") == "unknown":
            return "full_success_quality_unknown"
        return "full_success_with_warnings" if "warning" in severities else "full_success"
    return runtime_status


def write_compatibility_report(root: Path, rows: list[dict], *, recorded_results: bool = False):
    fields: tuple[str, ...] = (
        "model_id", "revision", "status", "stage", "reason_code", "detail", "device",
        "runtime_profile", "retryability", "evidence", "full_profile_complete", "quality_status", "quality_checks",
        "quality_reasons", "auto_selection_eligible", "attempt_id", "attempt_path", "configuration_sha256", "source",
    )
    if recorded_results:
        fields += ("run_status", "run_outcome", "runtime_status")
    headings = ["Source", "Model", "Revision", "Status", "reason_code", "Detail"]
    include_task_selection = any(row.get("requested_task_id") for row in rows)
    if include_task_selection:
        fields += ("requested_task_id", "selected_task_id", "task_selection_source")
        headings.extend(("Requested task", "Selected profile"))
    include_cleanup = any(row.get("cleanup_status") == "incomplete" or row.get("cleanup_errors") for row in rows)
    if include_cleanup:
        fields += ("cleanup_status", "cleanup_errors")
        headings.append("Cleanup")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    lines = ["# Compatibility report", "", "状态仅适用于所记录的 source、revision、设备、环境和验证范围。", "",
             "| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
    for row in rows:
        failure = row.get("failure") or {}
        quality = summarize_quality(row.get("quality_checks"))
        if row.get("quality_status") == "unknown":
            quality = combine_quality([quality, summarize_quality(None)])
        if "quality_reasons" in row:
            quality.update({key: row[key] for key in quality if key in row})
        row = {"source": "huggingface", **row, **quality}
        checks = row["quality_checks"]
        values = {key: failure.get(key, row.get(key, "")) for key in fields}
        values.update(status=result_status(row), evidence=json.dumps(failure.get("evidence", row.get("evidence", {})), ensure_ascii=False),
                      quality_checks=json.dumps(checks, ensure_ascii=False),
                      quality_reasons=json.dumps(row["quality_reasons"], ensure_ascii=False))
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")
        detail = failure.get("detail", "") or row.get("detail", "") or ", ".join(check["code"] for check in checks)
        cells = [row["source"], row["model_id"], values["revision"], values["status"], values["reason_code"], detail]
        if include_task_selection:
            cells.extend((row.get("requested_task_id") or "", row.get("selected_task_id") or ""))
        if include_cleanup:
            cleanup_status = row.get("cleanup_status") or ""
            cleanup_errors = [_public_cleanup_error(error) for error in row.get("cleanup_errors", [])]
            values["cleanup_status"] = cleanup_status
            values["cleanup_errors"] = json.dumps(cleanup_errors, ensure_ascii=False) if cleanup_status or cleanup_errors else ""
            cells.append(" — ".join(filter(None, (cleanup_status, _cleanup_summary(cleanup_errors)))))
        writer.writerow(values)
        lines.append("| " + " | ".join(cell(value) for value in cells) + " |")
    csv_text = stream.getvalue()
    report_text = "\n".join(lines) + "\n"
    def write_text(stream: TextIO, content: str) -> None:
        stream.write(content)

    atomic_write(root / "models.csv", lambda output: write_text(output, csv_text))
    atomic_write(root / "REPORT.md", lambda output: write_text(output, report_text))


_DIRECT_EVIDENCE_FILES = (
    "static_meta.json", "model_resolution.json", "capability_report.json",
    "runtime_validation.json", "runtime_failures.json", "interface_validation.json",
    "run_state.json",
)


def _expand_collection(collection_root: Path) -> list[tuple[Path, list[dict]]]:
    task_runs_dir = collection_root / "metadata" / "task_runs"
    if task_runs_dir.is_symlink() or not task_runs_dir.is_dir():
        return [(collection_root, [])]
    manifests = sorted(path for path in task_runs_dir.glob("*.json") if not path.is_symlink())
    if not manifests:
        return [(collection_root, [])]
    expanded: list[tuple[Path, list[dict]]] = []
    resolved_root = collection_root.resolve()
    for manifest_path in manifests:
        try:
            manifest = _read_recorded_object(manifest_path)
            run_id = manifest.get("run_id")
            tasks = manifest.get("tasks")
            if (manifest.get("schema_version") != 1 or not isinstance(run_id, str) or not run_id
                    or not isinstance(tasks, list) or not tasks):
                raise ValueError("task_runs manifest requires schema_version=1, run_id, and non-empty tasks")
        except (OSError, UnicodeError, ValueError, RecursionError) as error:
            expanded.append((collection_root, [_recorded_evidence_failure(manifest_path, error)]))
            continue
        for index, item in enumerate(tasks):
            task_id = item.get("task_id") if isinstance(item, dict) else None
            if not isinstance(task_id, str) or not task_id:
                expanded.append((collection_root, [
                    _recorded_evidence_failure(manifest_path, ValueError(f"tasks[{index}].task_id must be a string")),
                ]))
                continue
            task_dir = collection_root / "tasks" / task_id / run_id
            try:
                valid_dir = (not task_dir.is_symlink() and task_dir.is_dir()
                             and task_dir.resolve().is_relative_to(resolved_root))
            except OSError:
                valid_dir = False
            if valid_dir:
                expanded.append((task_dir, []))
            else:
                expanded.append((collection_root, [
                    _recorded_evidence_failure(manifest_path, ValueError(f"missing task run directory for {task_id}")),
                ]))
    return expanded


def _expand_result_sources(sources: list[Path]) -> list[tuple[Path, list[dict]]]:
    from acprof.results.layout import ArtifactLayout
    expanded: list[tuple[Path, list[dict]]] = []
    for source in sources:
        if (source / "metadata" / "task_runs").is_dir() and not (source / "metadata" / "task_runs").is_symlink():
            expanded.extend(_expand_collection(source))
            continue
        if source.is_dir():
            try:
                layout = ArtifactLayout.discover(source)
                has_direct = any(layout.path(name).is_file() for name in _DIRECT_EVIDENCE_FILES)
            except (OSError, ValueError):
                has_direct = False
            if not has_direct:
                children = sorted(
                    child for child in source.iterdir()
                    if not child.name.startswith(".") and not child.is_symlink() and child.is_dir()
                    and (child / "metadata" / "task_runs").is_dir()
                    and not (child / "metadata" / "task_runs").is_symlink()
                )
                if children:
                    for child in children:
                        expanded.extend(_expand_collection(child))
                    continue
        expanded.append((source, []))
    return expanded


def _read_compatibility_artifact(layout, source: Path, name: str, invalid: list[dict]) -> dict | None:
    try:
        path = layout.path(name) if name in _DIRECT_EVIDENCE_FILES else layout.contained(name)
    except (OSError, ValueError) as error:
        invalid.append(_recorded_evidence_failure(source / name, error))
        return {}
    try:
        if not recorded_artifact_present(path):
            return None
        record = _read_recorded_object(path)
        if name == "run_state.json":
            _validate_run_outcome(record.get("status"), record.get("outcome"))
        return record
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        invalid.append(_recorded_evidence_failure(path, error))
        return {}


def _invalid_compatibility_shape(layout, name: str, detail: str, invalid: list[dict]) -> None:
    invalid.append(_recorded_evidence_failure(layout.path(name), ValueError(detail)))


def _validated_compatibility_failure(
    layout, name: str, value: object, location: str, invalid: list[dict],
) -> dict | None:
    try:
        if not isinstance(value, dict):
            raise ValueError("must be an object")
        failure = Failure(**value)
        if (not isinstance(failure.stage, str) or not failure.stage
                or not isinstance(failure.detail, str)
                or not isinstance(failure.evidence, dict)):
            raise ValueError("has invalid typed failure fields")
        return failure.to_dict()
    except (TypeError, ValueError) as error:
        _invalid_compatibility_shape(layout, name, f"{location} must be a valid typed failure: {error}", invalid)
        return None


def _recorded_cleanup(layout, validation: dict, validation_name: str, interface: dict,
                      interface_name: str, invalid: list[dict]) -> dict:
    records = [(validation_name, "runtime validation", validation)]
    records.extend((validation_name, f"device {device!r}", value) for device, value in
                   _dict_or_empty(validation.get("devices")).items() if isinstance(value, dict))
    records.append((interface_name, "interface validation", interface))
    incomplete = False
    errors = []
    for name, location, record in records:
        status = record.get("cleanup_status")
        error = record.get("cleanup_error")
        if status not in (None, "complete", "incomplete"):
            _invalid_compatibility_shape(layout, name, f"{location}: invalid cleanup_status", invalid)
        if status == "incomplete":
            incomplete = True
        if error is None:
            continue
        try:
            if not isinstance(error, dict):
                raise ValueError("cleanup_error must be an object")
            if status not in (None, "incomplete") or error.get("status") not in (None, "incomplete"):
                raise ValueError("cleanup_error conflicts with the recorded cleanup status")
            errors.append(_public_cleanup_error(error))
            incomplete = True
        except ValueError as exc:
            _invalid_compatibility_shape(layout, name, f"{location}: {exc}", invalid)
    return {"cleanup_status": "incomplete", "cleanup_errors": errors} if incomplete else {}


def _load_compatibility_evidence(
    layout, source: Path, invalid: list[dict],
) -> tuple[dict[str, dict], list[dict], dict[str, str]]:
    stage_errors: list[dict] = []

    def read(name: str) -> dict:
        return _read_compatibility_artifact(layout, source, name, invalid) or {}

    def read_with_fallback(name: str, parent: dict, parent_name: str, errors: list[dict]) -> tuple[dict, str]:
        direct = _read_compatibility_artifact(layout, source, name, invalid)
        key = name.removesuffix(".json")
        if direct is not None or key not in parent:
            return direct or {}, name
        embedded = parent[key]
        if not isinstance(embedded, dict):
            _invalid_compatibility_shape(layout, parent_name, f"{key} must be an object", errors)
            return {}, parent_name
        return embedded, parent_name

    metadata = read("static_meta.json")
    resolution, resolution_name = read_with_fallback("model_resolution.json", metadata, "static_meta.json", invalid)
    capability, _ = read_with_fallback("capability_report.json", metadata, "static_meta.json", stage_errors)
    validation, validation_name = read_with_fallback("runtime_validation.json", metadata, "static_meta.json", stage_errors)
    interface, interface_name = read_with_fallback("interface_validation.json", resolution, resolution_name, stage_errors)
    state = read("run_state.json")
    preparation = {}
    stages = [(validation, validation_name), (interface, interface_name)]
    attempts = state.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        prep_rel = attempts[-1].get("preparation_evidence")
        if isinstance(prep_rel, str) and prep_rel:
            preparation = read(prep_rel)
            stages.append((preparation, prep_rel))
    for record, name in stages:
        if "status" in record and (not isinstance(record["status"], str) or not record["status"].strip()):
            path = layout.path(name) if name in _DIRECT_EVIDENCE_FILES else layout.contained(name)
            stage_errors.append(_recorded_evidence_failure(path, ValueError("stage status must be a nonempty string")))
    cleanup = _recorded_cleanup(layout, validation, validation_name, interface, interface_name, stage_errors)
    return {
        "metadata": metadata, "resolution": resolution, "capability": capability, "cleanup": cleanup,
        "validation": validation, "failure_report": read("runtime_failures.json"),
        "interface": interface, "state": state, "preparation": preparation,
    }, stage_errors, {"resolution": resolution_name, "validation": validation_name, "interface": interface_name}


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _check_identity_consistency(
    layout, metadata: dict, resolution: dict, state: dict, invalid: list[dict], *,
    resolution_name: str = "model_resolution.json", interface: dict | None = None,
    interface_name: str = "interface_validation.json",
) -> tuple[object, object, object, object, RecordedModelIdentity]:
    provenance = _dict_or_empty(resolution.get("provenance"))
    sources_map = _dict_or_empty(provenance.get("sources"))
    snapshot = _dict_or_empty(sources_map.get("repository_snapshot"))
    meta_model = metadata.get("model_id") or metadata.get("model_name")
    res_model = resolution.get("model_id") or provenance.get("model_id") or snapshot.get("model_id")
    meta_rev = metadata.get("model_revision")
    res_rev = resolution.get("model_revision") or provenance.get("revision") or snapshot.get("revision")
    invalid_identity = any(
        problem["evidence"]["artifact_name"] in {"static_meta.json", "model_resolution.json", "run_state.json"}
        for problem in invalid
    )
    try:
        # Corrupt identity evidence is not a historical document with missing fields.
        identity = RecordedModelIdentity() if invalid_identity else recorded_model_identity(
            metadata, resolution, state, resolution_name=resolution_name,
            interface=interface, interface_name=interface_name,
        )
    except RecordedSourceError as error:
        for name, detail in error.problems:
            _invalid_compatibility_shape(layout, name, detail, invalid)
        identity = RecordedModelIdentity()
    return meta_model, res_model, meta_rev, res_rev, identity


def _collect_compatibility_failures(
    layout, failure_report: dict, validation: dict, resolution: dict, invalid: list[dict], origins: dict[str, str],
) -> list[dict]:
    recorded_failures = failure_report.get("failures", [])
    if not isinstance(recorded_failures, list):
        _invalid_compatibility_shape(layout, "runtime_failures.json", "failures must be a list", invalid)
        recorded_failures = []
    else:
        recorded_failures = [
            failure for index, value in enumerate(recorded_failures)
            if (failure := _validated_compatibility_failure(
                layout, "runtime_failures.json", value, f"failures[{index}]", invalid,
            )) is not None
        ]
    devices = validation.get("devices", {})
    if not isinstance(devices, dict):
        _invalid_compatibility_shape(layout, origins["validation"], "devices must be an object", invalid)
        devices = {}
    device_failures = []
    for device, value in devices.items():
        if not isinstance(value, dict):
            _invalid_compatibility_shape(
                layout, origins["validation"], f"device {device!r} must be an object", invalid,
            )
            continue
        raw_failure = value.get("failure")
        if raw_failure is None:
            continue
        validated = _validated_compatibility_failure(
            layout, origins["validation"], raw_failure, f"device {device!r} failure", invalid,
        )
        if validated is not None:
            device_failures.append((value, validated))
    resolution_failure = resolution.get("failure")
    if resolution_failure is not None:
        resolution_failure = _validated_compatibility_failure(
            layout, origins["resolution"], resolution_failure, "failure", invalid,
        )
    failures = [*recorded_failures, *ordered_validation_failures(validation, device_failures)]
    if resolution_failure:
        failures.append(resolution_failure)
    failures.extend(invalid)
    return failures


def _resolve_compatibility_runtime_status(
    validation: dict, interface: dict, preparation: dict, state: dict, failures: list[dict],
) -> tuple[str, dict[str, object]]:
    validation_status = validation.get("status")
    if isinstance(validation_status, str) and validation_status and validation_status != "not_run":
        runtime_status = validation_status
    else:
        observed_statuses = (interface.get("status"), preparation.get("status"), state.get("status"))
        if "error" in observed_statuses or "failed" in observed_statuses:
            runtime_status = "failed"
        elif "cancelled" in observed_statuses:
            runtime_status = "cancelled"
        else:
            runtime_status = "unverified"
    unsuccessful = ("failed", "error", "interrupted", "cancelled")
    if failures or (runtime_status not in unsuccessful and state.get("status") not in unsuccessful):
        return runtime_status, {}
    prep = preparation if preparation.get("status") in unsuccessful else {}
    iface = interface if interface.get("status") in ("cancelled", "error") else {}
    if not prep and not iface:
        return runtime_status, {}
    interface_reason = {"error": "interface_validation_failed", "cancelled": "interface_validation_cancelled"}
    return runtime_status, {
        "stage": prep.get("last_stage") or iface.get("failed_stage", ""),
        "reason_code": prep.get("reason_code") or interface_reason.get(iface.get("status", ""), ""),
        "detail": _dict_or_empty(prep.get("exception")).get("message") or iface.get("error", ""),
    }


def _report_single_source(source: Path, pre_invalid: list[dict]) -> dict:
    from acprof.results.layout import ArtifactLayout
    layout = ArtifactLayout.discover(source)
    invalid = list(pre_invalid)
    evidence, stage_errors, origins = _load_compatibility_evidence(layout, source, invalid)
    metadata = evidence["metadata"]
    resolution = evidence["resolution"]
    capability = evidence["capability"]
    validation = evidence["validation"]
    failure_report = evidence["failure_report"]
    interface = evidence["interface"]
    state = evidence["state"]
    preparation = evidence["preparation"]
    if not any((*evidence.values(), invalid)):
        raise ValueError(f"No recorded compatibility evidence: {source}")

    meta_model, res_model, meta_rev, res_rev, identity = _check_identity_consistency(
        layout, metadata, resolution, state, invalid, resolution_name=origins["resolution"],
        interface=interface, interface_name=origins["interface"],
    )
    failures = _collect_compatibility_failures(
        layout, failure_report, validation, resolution, [*invalid, *stage_errors], origins,
    )

    state_runtime = _dict_or_empty(state.get("runtime"))
    state_task = _dict_or_empty(state_runtime.get("task"))
    state_options = _dict_or_empty(state.get("options"))
    runtime_env = _dict_or_empty(metadata.get("runtime_environment"))
    model_id = (
        identity.model_id or meta_model or res_model or interface.get("model_id")
        or state_task.get("model_id") or state_options.get("model") or source.name
    )
    revision = identity.revision(
        meta_rev, res_rev, interface.get("revision"), state_task.get("model_revision"), state_options.get("revision"),
    )
    runtime_profile = (
        metadata.get("runtime_profile_id") or runtime_env.get("profile_id")
        or resolution.get("runtime_profile") or interface.get("runtime_profile")
        or state_task.get("runtime_profile_id") or ""
    )
    runtime_status, extra_row = _resolve_compatibility_runtime_status(
        validation, interface, preparation, state, failures,
    )
    cleanup = evidence["cleanup"]
    if cleanup and isinstance(extra_row.get("detail"), str):
        # Interface probes may repeat the same cleanup exception in their primary message.
        extra_row["detail"] = redact_text(str(extra_row["detail"]))
    if cleanup and not failures and not extra_row:
        extra_row = {"stage": "cleanup", "reason_code": "cleanup_incomplete",
                     "detail": _cleanup_summary(cleanup["cleanup_errors"])}
    quality = read_quality(source)
    row = {
        "source": identity.source,
        "model_id": model_id,
        "revision": revision,
        "runtime_profile": runtime_profile,
        "full_profile_complete": capability.get("full_profile_complete") is True,
        "runtime_status": runtime_status,
        **extra_row,
        **cleanup,
        **quality,
        "failure": failures[0] if failures else None,
        "failures": failures,
        "evidence": {"result_directory": str(source.resolve())},
    }
    if state:
        row["run_status"] = state["status"]
        if "outcome" in state:
            row["run_outcome"] = state["outcome"]
    row["status"] = result_status(row)
    return row


def report_results(sources: list[Path], output: Path) -> dict:
    """Read recorded results, including failed preflight directories; never rerun."""
    from acprof.artifacts import atomic_write_json
    from acprof.results.outputs import validate_analysis_destinations
    expanded = _expand_result_sources(sources)
    validate_analysis_destinations(sources, [output], expanded_sources=(source for source, _ in expanded))
    rows = [_report_single_source(source, pre_invalid) for source, pre_invalid in expanded]
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "scope": "recorded_results; no_reexecution", "rows": rows}
    atomic_write_json(output / "coverage.json", report)
    write_compatibility_report(output, rows, recorded_results=True)
    return report
