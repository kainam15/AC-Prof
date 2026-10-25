"""Immutable, verified inputs for retrying postprocessing without inference.

The caller owns the result-directory lock and has stopped every sampler and
packet capture before publishing. Schema 2 can include resumed windows from
separate attempts; it retains their identities and independent PCAP segments.
"""
from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from acprof.artifact_layout import CaseArtifacts
from acprof.artifacts import (
    atomic_write,
    atomic_write_json,
    file_sha256,
    loads_finite_json,
    read_json_object,
    sync_directory,
)
from acprof.capabilities import require_profiling_mode
from acprof.result_csv import measurement_key, read_result_csv

CHECKPOINT_SCHEMA = 1
_FILE_NAMES = {"csv": "measurement.csv", "groups": "sniff_groups.jsonl",
               "requests": "requests.jsonl", "pcap": "sniff.pcap"}


def _evidence(case: CaseArtifacts, path: Path) -> dict[str, Any]:
    path = case.layout.contained(path.relative_to(case.layout.root))
    if not path.is_file():
        raise ValueError(f"Measurement evidence is not a regular file: {path}")
    return {"path": path.relative_to(case.layout.root).as_posix(),
            "sha256": file_sha256(path), "size": path.stat().st_size}


def _verify_file(case: CaseArtifacts, record: dict) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
        raise ValueError("Invalid measurement evidence reference")
    path = case.layout.contained(record["path"])
    if (not path.is_file() or path.stat().st_size != record["size"]
            or file_sha256(path) != record["sha256"]):
        raise ValueError(f"Measurement evidence is missing or changed: {path}")
    return path


def _validate_processing(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "capture_available", "case_incomplete", "completed_rows_before_failure",
        "incomplete_case_reason", "require_packet_latency", "profiling_mode", "gpu", "server_port",
    }:
        raise ValueError("Invalid measurement processing context")
    for name in ("capture_available", "case_incomplete", "require_packet_latency"):
        if type(value[name]) is not bool:
            raise ValueError(f"Invalid processing flag: {name}")
    count = value["completed_rows_before_failure"]
    if type(count) is not int or count < 0:
        raise ValueError("Invalid completed measurement count")
    if not isinstance(value["incomplete_case_reason"], str) or value["gpu"] not in {"off", "on"}:
        raise ValueError("Invalid case processing context")
    require_profiling_mode(value["profiling_mode"])
    if type(value["server_port"]) is not int or not 0 < value["server_port"] < 65536:
        raise ValueError("Invalid packet parser port")


def _receipt_paths(case: CaseArtifacts, attempt_id: str, csv_path: Path) -> set[Path]:
    from acprof.window_records import window_filename
    paths = set(case.window_records(attempt_id).glob("*.json"))
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("window_id", "").lower() not in {"", "nan", "none"}:
                path = case.window_records(row["case_attempt_id"]) / window_filename(row["window_id"])
                if not path.is_file():
                    raise ValueError("Measured CSV contains a row without a committed window receipt")
                paths.add(path)
    return paths


def _verify_window_coverage(case: CaseArtifacts, attempt_id: str, run_id: str,
                            csv_path: Path) -> list[dict[str, Any]]:
    from acprof.window_records import read_window_record

    records = [read_window_record(path) for path in sorted(_receipt_paths(case, attempt_id, csv_path))]
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = {measurement_key(row): row for row in csv.DictReader(stream)}
    seen = set()
    for record in records:
        if record["identity"]["case_id"] != case.case_id or record["identity"]["run_id"] != run_id:
            raise ValueError("Window receipt belongs to another case")
        from acprof.window_records import window_filename
        record_attempt = record["identity"]["case_attempt_id"]
        expected_telemetry = case.telemetry(record_attempt) / window_filename(record["identity"]["window_id"])
        actual_telemetry = case.window_records(record_attempt) / record["telemetry"]["path"]
        if actual_telemetry.resolve() != expected_telemetry.resolve():
            raise ValueError("Window telemetry belongs to another case attempt")
        case.layout.contained(expected_telemetry.relative_to(case.layout.root))
        key = measurement_key(record["row"])
        if key in seen or key not in rows:
            raise ValueError("Window receipts have duplicate or unplanned measurements")
        seen.add(key)
        # The orchestrator may append an OOM diagnostic to a committed error row.
        changed = {name for name, value in record["row"].items() if rows[key].get(name) != value}
        if changed and not (changed == {"error"} and record["row"].get("status") == "error"
                            and record["row"]["error"] in rows[key].get("error", "")):
            raise ValueError("Window receipt disagrees with the measured CSV row")
    if any(key not in seen and row.get("status") != "error" for key, row in rows.items()):
        raise ValueError("Measured CSV contains a successful row without a committed window receipt")
    return records


def _require_inputs(processing: dict, files: dict, *, has_receipts: bool) -> None:
    required = {"csv"}
    if processing["capture_available"]:
        required.add("pcap")
    if has_receipts:
        required.update({"groups", "requests"})
    if required - set(files):
        raise ValueError(f"Missing required measurement inputs: {sorted(required - set(files))}")


def _verify_projections(csv_path: Path, files: dict[str, Path], records: list[dict], *,
                        case_incomplete: bool) -> None:
    """Every committed receipt must agree with its published journal projections."""
    if not records:
        return
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with files["groups"].open(encoding="utf-8") as stream:
        groups = [loads_finite_json(line) for line in stream]
    if len(groups) != len(rows):
        raise ValueError("Sniff groups do not cover the measured CSV")
    with files["requests"].open(encoding="utf-8") as stream:
        requests = [loads_finite_json(line) for line in stream]
    by_id = {record["identity"]["window_id"]: record for record in records}
    committed_keys = {measurement_key(record["row"]) for record in records}
    failed_keys = {measurement_key(row) for row in rows if row.get("status") == "error"} - committed_keys
    uncommitted_keys = set()
    request_ids = set()
    for request in requests:
        group = request.get("sniff_group_id") if isinstance(request, dict) else None
        if not isinstance(group, str) or not group or group in request_ids:
            raise ValueError("Request journal disagrees with committed window receipts")
        if group in by_id:
            if request != by_id[group]["request"]:
                raise ValueError("Request journal disagrees with committed window receipts")
        else:
            # A timed-out request window is durable diagnostic evidence, but
            # has no commit receipt. It may only accompany its planned error
            # placeholder; it can never supply a successful measurement.
            key = measurement_key({**rows[0], **{name: request.get(name, "")
                                  for name in ("input_scale", "warmup", "repeat_idx")}})
            if (not case_incomplete or request.get("status") != "error"
                    or not isinstance(request.get("error"), str) or not request["error"].strip()
                    or key not in failed_keys or key in uncommitted_keys):
                raise ValueError("Uncommitted request does not match a planned failed window")
            uncommitted_keys.add(key)
        request_ids.add(group)
    if not set(by_id) <= request_ids:
        raise ValueError("Request journal is missing committed windows")
    group_ids = set()
    for row, group in zip(rows, groups, strict=True):
        identifier = group.get("sniff_group_id") if isinstance(group, dict) else None
        if identifier == "" and row.get("status") == "error":
            continue  # An unattempted OOM/timeout placeholder has no window receipt.
        if (identifier not in by_id or identifier in group_ids
                or measurement_key(row) != measurement_key(by_id[identifier]["row"])):
            raise ValueError("Sniff group disagrees with committed window receipt")
        group_ids.add(identifier)
    if group_ids != set(by_id):
        raise ValueError("Sniff groups are missing committed windows")


def publish_measurement_checkpoint(case: CaseArtifacts, *, run_id: str, attempt_id: str,
                                   processing: dict, expected, capture_segments=()) -> dict:
    """Copy durable raw inputs, then publish one immutable checkpoint directory."""
    _validate_processing(processing)
    read_result_csv(case.csv, expected=expected)
    window_records = _verify_window_coverage(case, attempt_id, run_id, case.csv)
    destination = case.attempt_directory("measurements", attempt_id)
    if destination.exists():
        raise ValueError(f"Measurement attempt already has a checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination.parent))
    try:
        sources = {"csv": case.csv, "groups": case.sidecar("sniff_groups"),
                   "requests": case.requests, "pcap": case.pcap}
        files = {}
        for name, source in sources.items():
            case.layout.contained(source.relative_to(case.layout.root))
            if not source.exists():
                if name == "csv":
                    raise ValueError("Measurement CSV disappeared before checkpoint")
                continue
            target = staging / _FILE_NAMES[name]
            shutil.copyfile(source, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            files[name] = {"path": (destination / target.name).relative_to(case.layout.root).as_posix(),
                           "sha256": file_sha256(target), "size": target.stat().st_size}
        _require_inputs(processing, files, has_receipts=bool(window_records))
        read_result_csv(staging / "measurement.csv", expected=expected)
        _verify_window_coverage(case, attempt_id, run_id, staging / "measurement.csv")
        _verify_projections(staging / "measurement.csv", {key: staging / _FILE_NAMES[key] for key in files},
                            window_records, case_incomplete=processing["case_incomplete"])
        receipts = [_evidence(case, path) for path in sorted(_receipt_paths(case, attempt_id, case.csv))]
        payload = {"schema_version": CHECKPOINT_SCHEMA, "run_id": run_id, "case_id": case.case_id,
                   "case_attempt_id": attempt_id, "processing": processing, "files": files,
                   "window_records": receipts}
        if capture_segments or any(record["schema_version"] == 2 for record in window_records):
            segments = []
            for index, source in enumerate(capture_segments):
                source = case.layout.contained(Path(source).relative_to(case.layout.root))
                target = staging / f"capture-{index}.pcap"
                shutil.copyfile(source, target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                segments.append({"path": (destination / target.name).relative_to(case.layout.root).as_posix(),
                                 "size": target.stat().st_size, "sha256": file_sha256(target)})
            payload.update(schema_version=2, capture_segments=segments,
                           continuity="separate_runtime_attempts", window_resume_protocol="window-v1")
        atomic_write_json(staging / "checkpoint.json", payload)
        sync_directory(staging)
        os.rename(staging, destination)
        parent = destination.parent
        while parent != case.layout.root.parent:
            sync_directory(parent)
            if parent == case.layout.root:
                break
            parent = parent.parent
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _evidence(case, destination / "checkpoint.json")


def read_measurement_checkpoint(case: CaseArtifacts, reference: dict, *, run_id: str,
                                attempt_id: str, expected) -> dict:
    """Reject changed evidence before any derived projection is overwritten."""
    path = _verify_file(case, reference)
    directory = case.attempt_directory("measurements", attempt_id)
    if path != directory / "checkpoint.json":
        raise ValueError("Measurement checkpoint belongs to another case attempt")
    payload = read_json_object(path, label="measurement checkpoint")
    if (type(payload.get("schema_version")) is not int or payload["schema_version"] not in {CHECKPOINT_SCHEMA, 2}
            or payload.get("run_id") != run_id or payload.get("case_id") != case.case_id
            or payload.get("case_attempt_id") != attempt_id):
        raise ValueError("Measurement checkpoint identity mismatch")
    _validate_processing(payload.get("processing"))
    files = payload.get("files")
    if not isinstance(files, dict) or "csv" not in files or set(files) - {"csv", "pcap", "groups", "requests"}:
        raise ValueError("Invalid measurement checkpoint files")
    for role, entry in files.items():
        if _verify_file(case, entry) != directory / _FILE_NAMES[role]:
            raise ValueError("Measurement file belongs to another case attempt")
    records = payload.get("window_records")
    if not isinstance(records, list):
        raise ValueError("Invalid measurement window references")
    _require_inputs(payload["processing"], files, has_receipts=bool(records))
    verified_paths = {_verify_file(case, entry) for entry in records}
    csv_path = case.layout.contained(files["csv"]["path"])
    observed_paths = _receipt_paths(case, attempt_id, csv_path)
    if len(verified_paths) != len(records) or verified_paths != observed_paths:
        raise ValueError("Measurement window receipt set changed")
    if payload["schema_version"] == 2:
        for index, segment in enumerate(payload["capture_segments"]):
            if _verify_file(case, segment) != directory / f"capture-{index}.pcap":
                raise ValueError("Capture segment belongs to another checkpoint")
    read_result_csv(csv_path, expected=expected)
    window_records = _verify_window_coverage(case, attempt_id, run_id, csv_path)
    _verify_projections(csv_path, {key: case.layout.contained(value["path"]) for key, value in files.items()},
                        window_records, case_incomplete=payload["processing"]["case_incomplete"])
    return payload


def restore_measurement_projection(case: CaseArtifacts, checkpoint: dict) -> None:
    """Restore only mutable projections; raw evidence remains byte-for-byte intact."""
    for name, destination in (("csv", case.csv), ("groups", case.sidecar("sniff_groups")),
                              ("requests", case.requests)):
        case.layout.contained(destination.relative_to(case.layout.root))
        reference = checkpoint["files"].get(name)
        if reference is None:
            if destination.exists():
                raise ValueError(f"Unexpected derived input absent from checkpoint: {destination}")
            continue
        source = _verify_file(case, reference)
        def copy(stream, source=source):
            with source.open(encoding="utf-8", newline="") as incoming:
                shutil.copyfileobj(incoming, stream)
        atomic_write(destination, copy)


def discover_measurement_checkpoint(case: CaseArtifacts, *, run_id: str, attempt_id: str,
                                    expected) -> dict | None:
    """Reconcile a committed directory whose RunState pointer was never saved."""
    directory = case.attempt_directory("measurements", attempt_id)
    if not directory.exists():
        return None
    reference = _evidence(case, directory / "checkpoint.json")
    read_measurement_checkpoint(case, reference, run_id=run_id, attempt_id=attempt_id, expected=expected)
    return reference
