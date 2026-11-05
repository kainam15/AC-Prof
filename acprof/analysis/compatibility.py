"""Render compatibility evidence without reinterpreting exception messages."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import TextIO

from acprof.artifacts import atomic_write
from acprof.failures import Failure, compatibility_status
from acprof.hub.repository import MODEL_SOURCES
from acprof.quality import combine_quality, read_quality, summarize_quality

MAX_COMPATIBILITY_ARTIFACT_BYTES = 4 * 1024 * 1024


def _finite_json_number(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def _read_recorded_object(path: Path) -> dict:
    with path.open("rb") as stream:
        content = stream.read(MAX_COMPATIBILITY_ARTIFACT_BYTES + 1)
    if len(content) > MAX_COMPATIBILITY_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds the 4 MiB read limit")
    payload = json.loads(content, parse_float=_finite_json_number,
                         parse_constant=_finite_json_number)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    return payload


def _recorded_evidence_failure(path: Path, error: Exception) -> dict:
    return Failure(
        "compatibility_report", "recorded_evidence_invalid", f"{path.name}: {error}",
        evidence={"artifact": str(path.resolve()), "artifact_name": path.name},
        exception_type=type(error).__name__,
    ).to_dict()


class _RecordedSourceError(ValueError):
    def __init__(self, artifact: str, detail: str):
        super().__init__(detail)
        self.artifact = artifact


def _recorded_model_source(metadata: dict, resolution: dict) -> str:
    provenance = resolution.get("provenance", {})
    if not isinstance(provenance, dict):
        raise _RecordedSourceError("model_resolution.json", "model provenance must be an object")
    sources = provenance.get("sources", {})
    if not isinstance(sources, dict):
        raise _RecordedSourceError("model_resolution.json", "model provenance sources must be an object")
    snapshot = sources.get("repository_snapshot", {})
    if not isinstance(snapshot, dict):
        raise _RecordedSourceError("model_resolution.json", "repository snapshot provenance must be an object")
    source = metadata.get("model_source", snapshot.get("source", "huggingface"))
    if not isinstance(source, str) or source not in MODEL_SOURCES:
        raise _RecordedSourceError("static_meta.json" if "model_source" in metadata else "model_resolution.json",
                                   "recorded model source must be huggingface/modelscope")
    if "source" in snapshot and snapshot["source"] != source:
        raise _RecordedSourceError("model_resolution.json", "recorded source conflicts with static_meta.json")
    return source


def result_status(row):
    if row.get("failure"):
        return compatibility_status(row["failure"])
    if row.get("full_profile_complete"):
        if row.get("quality_status") == "unknown":
            return "full_success_quality_unknown"
        return "full_success_with_warnings" if any(check["severity"] == "warning" for check in row.get("quality_checks", [])) else "full_success"
    return row.get("runtime_status", "unverified")


def write_compatibility_report(root: Path, rows: list[dict]):
    fields = ("model_id", "revision", "status", "stage", "reason_code", "detail", "device",
              "runtime_profile", "retryability", "evidence", "full_profile_complete", "quality_status", "quality_checks",
              "quality_reasons", "auto_selection_eligible", "attempt_id", "attempt_path", "configuration_sha256", "source")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    lines = ["# Compatibility report", "", "状态仅适用于所记录的 source、revision、设备、环境和验证范围。", "",
             "| Source | Model | Revision | Status | reason_code | Detail |", "| --- | --- | --- | --- | --- | --- |"]
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
        writer.writerow(values)
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")
        detail = failure.get("detail", "") or row.get("detail", "") or ", ".join(check["code"] for check in checks)
        lines.append("| " + " | ".join(cell(value) for value in (
            row["source"], row["model_id"], values["revision"], values["status"], values["reason_code"], detail)) + " |")
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


def _read_compatibility_artifact(layout, source: Path, name: str, invalid: list[dict]) -> dict:
    try:
        path = layout.path(name) if name in _DIRECT_EVIDENCE_FILES else layout.contained(name)
    except (OSError, ValueError) as error:
        invalid.append(_recorded_evidence_failure(source / name, error))
        return {}
    if not path.is_file():
        return {}
    try:
        return _read_recorded_object(path)
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


def _load_compatibility_evidence(layout, source: Path, invalid: list[dict]) -> dict[str, dict]:
    metadata = _read_compatibility_artifact(layout, source, "static_meta.json", invalid)
    resolution = _read_compatibility_artifact(layout, source, "model_resolution.json", invalid)
    capability = _read_compatibility_artifact(layout, source, "capability_report.json", invalid)
    validation = _read_compatibility_artifact(layout, source, "runtime_validation.json", invalid)
    failure_report = _read_compatibility_artifact(layout, source, "runtime_failures.json", invalid)
    interface = _read_compatibility_artifact(layout, source, "interface_validation.json", invalid)
    state = _read_compatibility_artifact(layout, source, "run_state.json", invalid)
    invalid_names = {problem["evidence"]["artifact_name"] for problem in invalid}
    fallbacks = (
        ("model_resolution.json", resolution, metadata.get("model_resolution")),
        ("capability_report.json", capability, metadata.get("capability_report")),
        ("runtime_validation.json", validation, metadata.get("runtime_validation")),
        ("interface_validation.json", interface, resolution.get("interface_validation")),
    )
    resolved = {
        "model_resolution.json": resolution,
        "capability_report.json": capability,
        "runtime_validation.json": validation,
        "interface_validation.json": interface,
    }
    for artifact_name, current, fallback in fallbacks:
        if not current and artifact_name not in invalid_names and isinstance(fallback, dict):
            resolved[artifact_name] = fallback
    preparation = {}
    attempts = state.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        prep_rel = attempts[-1].get("preparation_evidence")
        if isinstance(prep_rel, str) and prep_rel:
            preparation = _read_compatibility_artifact(layout, source, prep_rel, invalid)
    return {
        "metadata": metadata,
        "resolution": resolved["model_resolution.json"],
        "capability": resolved["capability_report.json"],
        "validation": resolved["runtime_validation.json"],
        "failure_report": failure_report,
        "interface": resolved["interface_validation.json"],
        "state": state,
        "preparation": preparation,
    }


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _check_identity_consistency(
    layout, metadata: dict, resolution: dict, invalid: list[dict],
) -> tuple[object, object, object, object, str]:
    provenance = _dict_or_empty(resolution.get("provenance"))
    sources_map = _dict_or_empty(provenance.get("sources"))
    snapshot = _dict_or_empty(sources_map.get("repository_snapshot"))
    meta_model = metadata.get("model_id") or metadata.get("model_name")
    res_model = resolution.get("model_id") or provenance.get("model_id") or snapshot.get("model_id")
    if isinstance(meta_model, str) and meta_model and isinstance(res_model, str) and res_model and meta_model != res_model:
        _invalid_compatibility_shape(
            layout, "model_resolution.json", "recorded model_id conflicts with static_meta.json", invalid,
        )
    meta_rev = metadata.get("model_revision")
    res_rev = resolution.get("model_revision") or provenance.get("revision") or snapshot.get("revision")
    if isinstance(meta_rev, str) and meta_rev and isinstance(res_rev, str) and res_rev and meta_rev != res_rev:
        _invalid_compatibility_shape(
            layout, "model_resolution.json", "recorded model_revision conflicts with static_meta.json", invalid,
        )
    invalid_identity = any(
        problem["evidence"]["artifact_name"] in {"static_meta.json", "model_resolution.json"}
        for problem in invalid
    )
    try:
        # Corrupt identity evidence is not a historical document with missing fields.
        model_source = "unknown" if invalid_identity else _recorded_model_source(metadata, resolution)
    except _RecordedSourceError as error:
        _invalid_compatibility_shape(layout, error.artifact, str(error), invalid)
        model_source = "unknown"
    return meta_model, res_model, meta_rev, res_rev, model_source


def _collect_compatibility_failures(
    layout, failure_report: dict, validation: dict, resolution: dict, invalid: list[dict],
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
        _invalid_compatibility_shape(layout, "runtime_validation.json", "devices must be an object", invalid)
        devices = {}
    device_failures = []
    for device, value in devices.items():
        if not isinstance(value, dict):
            _invalid_compatibility_shape(
                layout, "runtime_validation.json", f"device {device!r} must be an object", invalid,
            )
            continue
        raw_failure = value.get("failure")
        if raw_failure is None:
            continue
        validated = _validated_compatibility_failure(
            layout, "runtime_validation.json", raw_failure, f"device {device!r} failure", invalid,
        )
        if validated is not None:
            device_failures.append(validated)
    resolution_failure = resolution.get("failure")
    if resolution_failure is not None:
        resolution_failure = _validated_compatibility_failure(
            layout, "model_resolution.json", resolution_failure, "failure", invalid,
        )
    failures = [*recorded_failures, *device_failures]
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
        observed_statuses = {interface.get("status"), preparation.get("status"), state.get("status")}
        if observed_statuses & {"error", "failed"}:
            runtime_status = "failed"
        elif "cancelled" in observed_statuses:
            runtime_status = "cancelled"
        else:
            runtime_status = "unverified"
    extra_row: dict[str, object] = {}
    if not failures and runtime_status in {"failed", "cancelled"}:
        prep_exception = preparation.get("exception")
        prep_exc = prep_exception if isinstance(prep_exception, dict) else {}
        extra_row = {
            "stage": preparation.get("last_stage") or interface.get("failed_stage") or "",
            "reason_code": (
                preparation.get("reason_code")
                or ("interface_validation_failed" if interface.get("status") == "error" else "")
            ),
            "detail": prep_exc.get("message") or interface.get("error") or "",
        }
    return runtime_status, extra_row


def _report_single_source(source: Path, pre_invalid: list[dict]) -> dict:
    from acprof.results.layout import ArtifactLayout
    layout = ArtifactLayout.discover(source)
    invalid = list(pre_invalid)
    evidence = _load_compatibility_evidence(layout, source, invalid)
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

    meta_model, res_model, meta_rev, res_rev, model_source = _check_identity_consistency(
        layout, metadata, resolution, invalid,
    )
    failures = _collect_compatibility_failures(layout, failure_report, validation, resolution, invalid)

    state_runtime = _dict_or_empty(state.get("runtime"))
    state_task = _dict_or_empty(state_runtime.get("task"))
    state_options = _dict_or_empty(state.get("options"))
    runtime_env = _dict_or_empty(metadata.get("runtime_environment"))
    model_id = (
        meta_model or res_model or interface.get("model_id")
        or state_task.get("model_id") or state_options.get("model") or source.name
    )
    revision = (
        meta_rev or res_rev or interface.get("revision")
        or state_task.get("model_revision") or state_options.get("revision") or "unknown"
    )
    runtime_profile = (
        metadata.get("runtime_profile_id") or runtime_env.get("profile_id")
        or resolution.get("runtime_profile") or interface.get("runtime_profile")
        or state_task.get("runtime_profile_id") or ""
    )
    runtime_status, extra_row = _resolve_compatibility_runtime_status(
        validation, interface, preparation, state, failures,
    )
    quality = read_quality(source)
    return {
        "source": model_source,
        "model_id": model_id,
        "revision": revision,
        "runtime_profile": runtime_profile,
        "full_profile_complete": capability.get("full_profile_complete") is True,
        "runtime_status": runtime_status,
        **extra_row,
        **quality,
        "failure": failures[0] if failures else None,
        "failures": failures,
        "evidence": {"result_directory": str(source.resolve())},
    }


def report_results(sources: list[Path], output: Path) -> dict:
    """Read recorded results, including failed preflight directories; never rerun."""
    from acprof.artifacts import atomic_write_json
    rows = [
        _report_single_source(source, pre_invalid)
        for source, pre_invalid in _expand_result_sources(sources)
    ]
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "scope": "recorded_results; no_reexecution", "rows": rows}
    atomic_write_json(output / "coverage.json", report)
    write_compatibility_report(output, rows)
    return report
