"""Replay immutable measurements and explicit derivations without recomputing metrics."""
from __future__ import annotations

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import file_sha256
from acprof.result_csv import ResultValidationError, measurement_key
from acprof.result_store.protocol import IDENTITY_FIELDS, read_reference
from acprof.window_records import read_window_record, verify_window_record_evidence


def window_row(layout: ArtifactLayout, source: dict) -> tuple[dict, tuple[str, ...]]:
    ref = source["reference"]
    path = layout.contained(ref["path"])
    if not path.is_relative_to(layout.path("raw/windows")):
        raise ResultValidationError("window source is outside raw/windows")
    if not isinstance(ref["sha256"], str):
        raise ResultValidationError("invalid window receipt SHA256")
    record = read_window_record(path, verify_evidence=False, expected_sha256=ref["sha256"])
    identity = record["identity"]
    if path != layout.window_evidence_path("windows", identity):
        raise ResultValidationError("window receipt path disagrees with its identity")
    telemetry = layout.window_evidence_path("telemetry", identity)
    if (path.parent / record["telemetry"]["path"]).absolute().resolve() != telemetry.resolve():
        raise ResultValidationError("window telemetry path disagrees with its identity")
    verify_window_record_evidence(path, record)
    if any(name in record["row"] and record["row"][name] != identity[name] for name in IDENTITY_FIELDS):
        raise ResultValidationError("window row disagrees with its receipt identity")
    return {**record["row"], **identity}, tuple(identity[name] for name in IDENTITY_FIELDS)


def apply_source(layout, manifest, source, rows, windows):
    """Mutate ordered rows only after validating the complete source operation."""
    fields = manifest["wide_fields"]
    if source.get("kind") == "window":
        row, identity = window_row(layout, source)
        if identity[0] != manifest["run_id"]:
            raise ResultValidationError("window belongs to another run")
        receipt_digest = source["reference"]["sha256"]
        if identity in windows:
            if windows[identity] != receipt_digest:
                raise ResultValidationError("conflicting committed window identity")
            return
        key = measurement_key(row)
        if set(row) - set(fields):
            raise ResultValidationError("unknown window fields")
        if row.get("status") == "error" and not row.get("error", "").strip():
            raise ResultValidationError("window error requires a diagnostic")
        if key in rows:
            raise ResultValidationError("duplicate planned measurement across attempts")
        rows[key] = {name: row.get(name, "nan") for name in fields}
        windows[identity] = receipt_digest
        return
    if source.get("kind") != "derivation":
        raise ResultValidationError("unknown result source")
    ref = source["reference"]
    if not layout.contained(ref["path"]).is_relative_to(layout.path("raw/result_updates")):
        raise ResultValidationError("derivation source is outside raw/result_updates")
    update = read_reference(layout, ref)
    if update.get("schema_version") != 1 or update.get("run_id") != manifest["run_id"]:
        raise ResultValidationError("invalid result derivation identity")
    if not isinstance(update.get("reason"), str) or not update["reason"]:
        raise ResultValidationError("result derivation requires provenance")
    for evidence in update.get("evidence", []):
        path = layout.contained(evidence["path"])
        if file_sha256(path) != evidence["sha256"]:
            raise ResultValidationError("result derivation evidence changed")
    for row in update.get("insert", []):
        if not isinstance(row, dict) or set(row) - set(fields) or not all(isinstance(value, str) for value in row.values()):
            raise ResultValidationError("invalid derived outcome fields")
        key = measurement_key(row)
        if key in rows:
            raise ResultValidationError("duplicate derived outcome")
        rows[key] = {name: row.get(name, "nan") for name in fields}
    for change in update.get("changes", []):
        key = tuple(change["key"])
        values = change["values"]
        if key not in rows or not isinstance(values, dict) or not set(values) <= set(fields):
            raise ResultValidationError("invalid derived measurement key or fields")
        if any(name in IDENTITY_FIELDS or name in ("cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx")
               for name in values):
            raise ResultValidationError("derivation cannot change measurement identity")
        rows[key] = {**rows[key], **values}


def replay(layout, manifest):
    """Verify the commit chain and all authority once, outside acquisition."""
    commits = []
    ref = manifest["commit"]
    expected_sequence = manifest["sequence"]
    while ref is not None:
        path = layout.contained(ref["path"])
        if not path.is_relative_to(layout.result_store_path("commits")):
            raise ResultValidationError("result commit path is outside its journal")
        commit = read_reference(layout, ref)
        after = commit.get("after", {})
        if (commit.get("schema_version") != 1 or after.get("sequence") != expected_sequence
                or after.get("run_id") != manifest["run_id"]
                or after.get("schema_sha256") != manifest["schema_sha256"]):
            raise ResultValidationError("broken result commit chain")
        # Keep replay inputs without retaining every historical manifest.
        commits.append((after["row_count"], commit["sources"]))
        expected_sequence -= 1
        ref = commit.get("previous")
    if expected_sequence != -1:
        raise ResultValidationError("incomplete result commit chain")
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    windows: dict[tuple[str, ...], str] = {}
    for expected_rows, sources in reversed(commits):
        for source in sources:
            apply_source(layout, manifest, source, rows, windows)
        if len(rows) != expected_rows:
            raise ResultValidationError("result transaction row count disagrees with its sources")
    return rows, windows
