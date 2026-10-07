"""Render compatibility evidence without reinterpreting exception messages."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from acprof.artifacts import atomic_write
from acprof.failures import Failure, compatibility_status
from acprof.model_repository import MODEL_SOURCES
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
    if source not in MODEL_SOURCES:
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
        detail = failure.get("detail", "") or ", ".join(check["code"] for check in checks)
        lines.append("| " + " | ".join(cell(value) for value in (
            row["source"], row["model_id"], values["revision"], values["status"], values["reason_code"], detail)) + " |")
    csv_text = stream.getvalue()
    report_text = "\n".join(lines) + "\n"
    atomic_write(root / "models.csv", lambda output: output.write(csv_text))
    atomic_write(root / "REPORT.md", lambda output: output.write(report_text))


def report_results(sources: list[Path], output: Path) -> dict:
    """Read recorded results, including failed preflight directories; never rerun."""
    from acprof.artifact_layout import ArtifactLayout
    from acprof.artifacts import atomic_write_json
    rows = []
    for source in sources:
        layout = ArtifactLayout.discover(source)
        invalid = []

        def read(name):
            path = layout.path(name)
            if not path.is_file():
                return {}
            try:
                return _read_recorded_object(path)
            except (OSError, UnicodeError, ValueError, RecursionError) as error:
                invalid.append(_recorded_evidence_failure(path, error))
                return {}

        def invalid_shape(name, detail):
            invalid.append(_recorded_evidence_failure(layout.path(name), ValueError(detail)))

        def validated_failure(name, value, location):
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
                invalid_shape(name, f"{location} must be a valid typed failure: {error}")
                return None

        metadata, resolution = read("static_meta.json"), read("model_resolution.json")
        capability = read("capability_report.json")
        validation = read("runtime_validation.json")
        failure_report = read("runtime_failures.json")
        if not any((metadata, resolution, capability, validation, failure_report, invalid)):
            raise ValueError(f"No recorded compatibility evidence: {source}")

        invalid_identity = any(problem["evidence"]["artifact_name"] in {
            "static_meta.json", "model_resolution.json"} for problem in invalid)
        try:
            # Corrupt identity evidence is not a historical document with missing fields.
            model_source = "unknown" if invalid_identity else _recorded_model_source(metadata, resolution)
        except _RecordedSourceError as error:
            invalid_shape(error.artifact, str(error))
            model_source = "unknown"

        recorded_failures = failure_report.get("failures", [])
        if not isinstance(recorded_failures, list):
            invalid_shape("runtime_failures.json", "failures must be a list")
            recorded_failures = []
        else:
            recorded_failures = [
                failure for index, value in enumerate(recorded_failures)
                if (failure := validated_failure(
                    "runtime_failures.json", value, f"failures[{index}]",
                )) is not None
            ]

        devices = validation.get("devices", {})
        if not isinstance(devices, dict):
            invalid_shape("runtime_validation.json", "devices must be an object")
            devices = {}
        device_failures = []
        for device, value in devices.items():
            if not isinstance(value, dict):
                invalid_shape("runtime_validation.json", f"device {device!r} must be an object")
                continue
            failure = value.get("failure")
            if failure is None:
                continue
            failure = validated_failure(
                "runtime_validation.json", failure, f"device {device!r} failure",
            )
            if failure is not None:
                device_failures.append(failure)

        resolution_failure = resolution.get("failure")
        if resolution_failure is not None:
            resolution_failure = validated_failure(
                "model_resolution.json", resolution_failure, "failure",
            )
        failures = [*recorded_failures, *device_failures]
        if resolution_failure:
            failures.append(resolution_failure)
        failures.extend(invalid)

        quality = read_quality(source)
        rows.append({"source": model_source, "model_id": metadata.get("model_id", resolution.get("model_id", source.name)),
                     "revision": metadata.get("model_revision", resolution.get("model_revision", "unknown")),
                     "runtime_profile": metadata.get("runtime_profile_id", resolution.get("runtime_profile", "")),
                     "full_profile_complete": capability.get("full_profile_complete") is True,
                     "runtime_status": validation.get("status", "unverified"),
                     **quality, "failure": failures[0] if failures else None,
                     "failures": failures, "evidence": {"result_directory": str(source.resolve())}})
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "scope": "recorded_results; no_reexecution", "rows": rows}
    atomic_write_json(output / "coverage.json", report)
    write_compatibility_report(output, rows)
    return report
