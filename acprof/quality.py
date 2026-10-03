"""Result quality evidence, independent of execution/measurement capabilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
    previous = json.loads(path.read_text()).get("checks", []) if append and path.exists() else []
    unique = {json.dumps(value, sort_keys=True): value for value in [*previous, *checks]}
    atomic_write_json(path, {"schema_version": 1, "checks": list(unique.values())})


def collect_quality(root: Path, case_csvs=()) -> list[dict]:
    from acprof.artifact_layout import ArtifactLayout, case_sidecar
    layout = ArtifactLayout.discover(root)
    checks = []
    validation_path = layout.path("runtime_validation.json")
    if validation_path.exists():
        for device, result in json.loads(validation_path.read_text()).get("devices", {}).items():
            for check in result.get("quality_checks", []):
                checks.append({**check, "evidence": {**check["evidence"], "device": device}})
    for csv in case_csvs:
        path = case_sidecar(csv, "quality_checks")
        if path.exists():
            checks.extend(json.loads(path.read_text())["checks"])
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


def read_quality(source: str | Path, *, device: str | None = None) -> dict:
    """Read canonical checks, or explicit runtime-validation checks in older runs."""
    from acprof.artifact_layout import ArtifactLayout
    source = Path(source)
    layout = ArtifactLayout.discover(source) if source.is_dir() else ArtifactLayout.from_csv(source)
    quality_path, validation_path = (layout.path(name) for name in ("quality_checks.json", "runtime_validation.json"))
    path = quality_path if quality_path.exists() else validation_path
    if not path.exists():
        return summarize_quality(None)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("quality evidence must be an object")
        if path == quality_path:
            if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
                raise ValueError("unsupported quality evidence schema")
            checks = payload.get("checks")
        else:
            devices = payload.get("devices", {})
            if not isinstance(devices, dict):
                raise ValueError("invalid runtime-validation devices")
            relevant = [(name, item) for name, item in devices.items() if device is None or name == device]
            if not relevant or any(not isinstance(item, dict) or "quality_checks" not in item for _, item in relevant):
                return summarize_quality(None)
            checks = []
            for name, item in relevant:
                if not isinstance(item["quality_checks"], list):
                    raise ValueError("invalid runtime-validation quality checks")
                for check in item["quality_checks"]:
                    if not isinstance(check, dict) or not isinstance(check.get("evidence"), dict):
                        raise ValueError("invalid runtime-validation quality check")
                    checks.append({**check, "evidence": {**check["evidence"], "device": name}})
        if not isinstance(checks, list):
            raise ValueError("quality checks must be an explicit list")
        result = []
        for check in checks:
            if (not isinstance(check, dict) or not isinstance(check.get("code"), str)
                    or not check["code"] or check.get("severity") not in {"info", "warning", "error"}
                    or not isinstance(check.get("detail"), str) or not isinstance(check.get("evidence"), dict)):
                raise ValueError("invalid quality check")
            evidence = check["evidence"]
            if device is not None and evidence.get("device") not in {None, device}:
                continue
            result.append({**check, "evidence": {"source": str(path), **evidence, "artifact": str(path)}})
        return summarize_quality(result)
    except (OSError, ValueError, TypeError) as error:
        return summarize_quality([QualityCheck("quality_evidence_invalid", "warning", str(error),
            "schema v1 checks", "Quality evidence could not be verified", {"source": str(path)}).to_dict()])
