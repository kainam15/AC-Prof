"""Render compatibility evidence without reinterpreting exception messages."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from acprof.failures import compatibility_status
from acprof.quality import combine_quality, read_quality, summarize_quality


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
              "quality_reasons", "auto_selection_eligible", "attempt_id", "attempt_path", "configuration_sha256")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    lines = ["# Compatibility report", "", "状态仅适用于所记录的 revision、设备、环境和验证范围。", "",
             "| Model | Status | reason_code | Detail |", "| --- | --- | --- | --- |"]
    for row in rows:
        failure = row.get("failure") or {}
        quality = summarize_quality(row.get("quality_checks"))
        if row.get("quality_status") == "unknown":
            quality = combine_quality([quality, summarize_quality(None)])
        if "quality_reasons" in row:
            quality.update({key: row[key] for key in quality if key in row})
        row = {**row, **quality}
        checks = row["quality_checks"]
        values = {key: failure.get(key, row.get(key, "")) for key in fields}
        values.update(status=result_status(row), evidence=json.dumps(failure.get("evidence", row.get("evidence", {})), ensure_ascii=False),
                      quality_checks=json.dumps(checks, ensure_ascii=False),
                      quality_reasons=json.dumps(row["quality_reasons"], ensure_ascii=False))
        writer.writerow(values)
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")
        detail = failure.get("detail", "") or ", ".join(check["code"] for check in checks)
        lines.append("| " + " | ".join(cell(value) for value in (row["model_id"], values["status"], values["reason_code"], detail)) + " |")
    (root / "models.csv").write_text(stream.getvalue(), encoding="utf-8")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_results(sources: list[Path], output: Path) -> dict:
    """Read recorded results, including failed preflight directories; never rerun."""
    from acprof.artifact_layout import ArtifactLayout
    from acprof.artifacts import atomic_write_json
    rows = []
    for source in sources:
        layout = ArtifactLayout.discover(source)
        def read(name):
            path = layout.path(name)
            return json.loads(path.read_text()) if path.is_file() else {}
        metadata, resolution = read("static_meta.json"), read("model_resolution.json")
        capability = read("capability_report.json")
        validation = read("runtime_validation.json")
        if not any((metadata, resolution, capability, validation)):
            raise ValueError(f"No recorded compatibility evidence: {source}")
        failures = read("runtime_failures.json").get("failures", [])
        failures.extend(value["failure"] for value in validation.get("devices", {}).values() if value.get("failure"))
        if resolution.get("failure"):
            failures.append(resolution["failure"])
        quality = read_quality(source)
        rows.append({"model_id": metadata.get("model_id", resolution.get("model_id", source.name)),
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
