"""Result quality evidence, independent of execution/measurement capabilities."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acprof.artifacts import read_json_object, reject_isolated_surrogates
from acprof.cancellation import cancellable, check_cancelled

if TYPE_CHECKING:
    from acprof.results.metadata import ResultMetadataSnapshot


def runtime_quality_constraints() -> dict[str, Any]:
    """Declare collection's existing sanity gate, without claiming model accuracy."""
    return {
        "schema_version": 1,
        "scope": "sanity_only_not_accuracy_benchmark",
        "required_status": {"protocol": "verified", "task": "verified"},
    }


@dataclass(frozen=True)
class QualityCheck:
    code: str
    severity: str
    observed: Any
    threshold: Any
    detail: str
    evidence: dict

    def to_dict(self):
        return asdict(self)


def loading_quality(info: dict, *, source: str) -> list[dict]:
    checks = []
    for key, code in (("missing_keys", "weights_reinitialized"), ("mismatched_keys", "weights_reinitialized"),
                      ("unexpected_keys", "unused_checkpoint_weights")):
        values = info.get(key) or []
        if values:
            checks.append(QualityCheck(code, "warning", sorted(str(item) for item in values), 0,
                f"Checkpoint loading reported {key}", {"source": source, "loading_info_key": key}).to_dict())
    return checks


def cli_exit_quality(exit_code, *, source: str) -> list[dict]:
    if exit_code is not None:
        return []
    return [QualityCheck("missing_cli_exit_code", "warning", None, "recorded process exit code",
                         "Process termination evidence is missing", {"source": source}).to_dict()]


def write_quality(path: Path, checks: list[dict], *, append=False):
    from acprof.artifacts import atomic_write_json
    previous = read_json_object(path, label="quality evidence").get("checks", []) if append and path.exists() else []
    unique = {json.dumps(value, sort_keys=True): value for value in [*previous, *checks]}
    atomic_write_json(path, {"schema_version": 1, "checks": list(unique.values())})


def collect_quality(root: Path, case_csvs=()) -> list[dict]:
    from acprof.results.layout import ArtifactLayout, case_sidecar
    layout = ArtifactLayout.discover(root)
    checks = []
    validation_path = layout.path("runtime_validation.json")
    if validation_path.exists():
        for device, result in read_json_object(validation_path, label="runtime validation").get("devices", {}).items():
            for check in result.get("quality_checks", []):
                checks.append({**check, "evidence": {**check["evidence"], "device": device}})
    for csv in case_csvs:
        path = case_sidecar(csv, "quality_checks")
        if path.exists():
            checks.extend(read_json_object(path, label="quality evidence")["checks"])
    write_quality(layout.path("quality_checks.json"), checks)
    return checks


QUALITY_FIELDS = ("quality_status", "quality_checks", "quality_reasons", "auto_selection_eligible")


def summarize_quality(checks: list[dict] | None) -> dict:
    """Describe recorded checks, never turn missing evidence into a passed check."""
    recorded = checks is not None
    checks = list(checks or [])
    blocked, unverified, warnings = [], [], []
    for check in checks:
        code, severity = check["code"], check["severity"]
        if code == "weights_reinitialized" or severity == "error":
            blocked.append(code)
        elif severity == "warning":
            warnings.append(code)
            # Unused checkpoint tensors did not enter the executed model. Keep
            # the loader's exact evidence; this is not an accuracy certificate.
            if code != "unused_checkpoint_weights":
                unverified.append(code)
    reasons = sorted(set(blocked + unverified + warnings + ([] if recorded else ["quality_evidence_missing"])))
    status = ("blocked" if blocked else "unknown" if not recorded or unverified else
              "warning" if warnings else "passed")
    return {"quality_status": status, "quality_checks": checks, "quality_reasons": reasons,
            "auto_selection_eligible": status in {"passed", "warning"}}


def combine_quality(summaries: list[dict]) -> dict:
    """Merge reports while preserving an unknown member of a partially known group."""
    unique = {json.dumps(check, sort_keys=True): check for item in summaries for check in item["quality_checks"]}
    result = summarize_quality(list(unique.values()) if summaries else None)
    statuses = {item["quality_status"] for item in summaries}
    if "blocked" in statuses:
        result["quality_status"] = "blocked"
    elif "unknown" in statuses:
        result["quality_status"] = "unknown"
    result["auto_selection_eligible"] = result["quality_status"] in {"passed", "warning"}
    result["quality_reasons"] = sorted(set(result["quality_reasons"]).union(
        *(item["quality_reasons"] for item in summaries)))
    return result


def _invalid_quality_check(path: Path, error: Exception, *, device: str | None = None) -> dict:
    evidence = {"source": str(path), "artifact": str(path)}
    if device is not None:
        evidence["device"] = device
    return QualityCheck("quality_evidence_invalid", "warning", str(error),
        "schema v1 checks", "Quality evidence could not be verified", evidence).to_dict()


def _quality_from_checks(checks: object, path: Path, *, device: str | None = None,
                         recorded_device: str | None = None, cancelled: Callable[[], bool] | None) -> dict:
    if not isinstance(checks, list):
        raise ValueError("quality checks must be an explicit list")
    result = []
    for index, check in enumerate(cancellable(checks, cancelled)):
        if (not isinstance(check, dict) or not isinstance(check.get("code"), str)
                or not check["code"] or check.get("severity") not in ("info", "warning", "error")
                or not isinstance(check.get("detail"), str) or not isinstance(check.get("evidence"), dict)):
            result.append(_invalid_quality_check(path, ValueError(f"invalid quality check at index {index}"),
                                                 device=recorded_device))
            continue
        evidence = {"source": str(path), **check["evidence"]}
        if recorded_device is not None:
            evidence["device"] = recorded_device
        evidence["artifact"] = str(path)
        target = evidence.get("device")
        if target is not None and (not isinstance(target, str) or not target.strip()):
            result.append(_invalid_quality_check(path, ValueError(f"invalid device in quality check at index {index}")))
        try:
            if device is not None and target not in (None, device):
                reject_isolated_surrogates(target)
                continue
            item = {**check, "evidence": evidence}
            reject_isolated_surrogates(item)
            result.append(item)
        except ValueError as error:
            result.append(_invalid_quality_check(path, ValueError(f"quality check at index {index}: {error}"),
                                                 device=recorded_device))
    return summarize_quality(result)


def _validation_quality(validation: object, path: Path, device: str | None,
                        cancelled: Callable[[], bool] | None) -> dict:
    if not isinstance(validation, dict) or not isinstance(validation.get("devices", {}), dict):
        raise ValueError("invalid runtime-validation device records")
    summaries = []
    for mode, record in cancellable(validation.get("devices", {}).items(), cancelled):
        try:
            reject_isolated_surrogates(mode)
        except ValueError as error:
            summaries.append(summarize_quality([_invalid_quality_check(path, error)]))
            continue
        if device is not None and mode != device:
            continue
        try:
            if not isinstance(record, dict):
                raise ValueError("runtime-validation device must be an object")
            if "quality_checks" in record:
                summary = _quality_from_checks(record["quality_checks"], path, recorded_device=mode, cancelled=cancelled)
            else:
                summary = summarize_quality(None)
            summaries.append(summary)
        except ValueError as error:
            summaries.append(summarize_quality([_invalid_quality_check(path, error, device=mode)]))
    return combine_quality(summaries)


def read_quality(source: str | Path, *, device: str | None = None,
                 metadata_snapshot: ResultMetadataSnapshot | None = None,
                 cancelled: Callable[[], bool] | None = None) -> dict:
    """Use explicit quality checks, then standalone or embedded probe evidence in that order."""
    from acprof.results.metadata import ResultMetadataSnapshot
    check_cancelled(cancelled)
    metadata_snapshot = metadata_snapshot or ResultMetadataSnapshot(source, cancelled=cancelled)
    metadata_snapshot.validate_source(source, cancelled=cancelled)
    for name in ("quality_checks.json", "runtime_validation.json", "static_meta.json"):
        try:
            if not metadata_snapshot.exists(name):
                continue
            path = metadata_snapshot.layout.path(name)
            payload = metadata_snapshot.read(name, label="quality evidence", cancelled=cancelled).payload
            if name == "quality_checks.json":
                if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
                    raise ValueError("unsupported quality evidence schema")
                return _quality_from_checks(payload.get("checks"), path, device=device, cancelled=cancelled)
            validation = payload.get("runtime_validation", {}) if name == "static_meta.json" else payload
            return _validation_quality(validation, path, device, cancelled)
        except (OSError, ValueError, TypeError) as error:
            try:
                path = metadata_snapshot.layout.path(name)
            except (ValueError, OSError):
                # The layout rejected this path; only its containing experiment is known safe.
                path = metadata_snapshot.layout.root
            return summarize_quality([_invalid_quality_check(path, error)])
    return summarize_quality(None)
