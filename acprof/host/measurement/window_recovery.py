"""Window-level resume from receipts; runtime segments remain explicitly distinct."""
from __future__ import annotations

import csv
import json
import time

from acprof.artifacts import atomic_write, atomic_write_json, file_sha256, read_json_object
from acprof.config import CSV_FIELDS
from acprof.result_csv import measurement_key
from acprof.result_store.writer import ResultWriter
from acprof.window_records import read_window_record


def is_incremental(layout) -> bool:
    return layout.result_csv.is_file() and read_json_object(layout.result_csv).get("schema_version") == 3


def repair_result_projections(layout, data):
    """Restore disposable files only after the runner validated its frozen identity."""
    if not is_incremental(layout):
        return
    from acprof.result_store.protocol import write_bytes
    with ResultWriter(layout.root) as writer:
        if writer.manifest["run_id"] != data["run_id"]:
            raise ValueError("result store belongs to another run")
    if data.get("status") == "complete":
        return  # Completed runs have already cleaned their case work directories.
    for name, record in data.get("cases", {}).items():
        ref = record.get("case_result")
        if record.get("status") != "complete" or not ref:
            continue
        case = layout.case_from_csv(layout.contained(name))
        authority = case.completed_result(record["sha256"])
        if (ref != {"path": authority.relative_to(layout.root).as_posix(), "sha256": record["sha256"]}
                or file_sha256(authority) != record["sha256"]):
            raise ValueError("immutable case result is damaged")
        if not case.csv.is_file() or file_sha256(case.csv) != record["sha256"]:
            write_bytes(case.csv, authority.read_bytes())


def case_receipts(case, run_id):
    """Only committed receipts, sorted by explicit acquisition sequence."""
    records = []
    for path in case.layout.window_records_root(case.case_id).glob("*/*.json"):
        record = read_window_record(path)
        identity = record["identity"]
        if identity["run_id"] != run_id or identity["case_id"] != case.case_id:
            raise ValueError("window receipt belongs to another run or case")
        if path != case.layout.window_evidence_path("windows", identity):
            raise ValueError("window receipt path disagrees with its identity")
        records.append((path, record))
    records.sort(key=lambda item: (item[1].get("sequence", 0), item[1]["identity"]["case_attempt_id"],
                                   item[1]["identity"]["window_id"]))
    keys = [measurement_key(record["row"]) for _, record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate planned measurement across case attempts")
    return records


def mark_packet_processing_failed(layout, case_id):
    if not is_incremental(layout):
        return
    with ResultWriter(layout.root) as writer:
        changed = [{**row, "packet_status": "error"} for row in writer.rows.values()
                   if row.get("case_id") == case_id and row.get("packet_status") == "pending"]
        if changed:
            writer.update_rows(changed, reason="case postprocessing failed; original checkpoint retained")


def restore_case_windows(case, *, run_id, expected):
    """Rebuild disposable case inputs and replay only missing root projections."""
    records = case_receipts(case, run_id)
    if not records:
        return 0
    if not {measurement_key(record["row"]) for _, record in records} <= set(expected):
        raise ValueError("window receipts do not belong to the frozen measurement plan")
    with ResultWriter(case.layout.root) as writer:
        for path, _ in records:
            writer.append_window(path)
    def csv_projection(stream):
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(record["row"] for _, record in records)
    atomic_write(case.csv, csv_projection)
    atomic_write(case.sidecar("sniff_groups"), lambda stream: stream.writelines(
        json.dumps({"sniff_group_id": record["identity"]["window_id"]}) + "\n" for _, record in records))
    atomic_write(case.requests, lambda stream: stream.writelines(
        json.dumps(record["request"], allow_nan=False) + "\n" for _, record in records))
    return len(records)


def archived_captures(case, record):
    """Capture segments are never concatenated at the binary PCAP level."""
    captures = []
    for previous in record.get("previous_attempts", []):
        backup = previous.get("backup")
        if backup:
            candidate = case.layout.contained(backup) / case.pcap.relative_to(case.layout.root)
            case.layout.contained(candidate.relative_to(case.layout.root))
            if candidate.is_file():
                captures.append(candidate)
    return captures


def prepare_resumed_scale(config, scale_entry, completed, writer, one_request):
    """Rewarm a new runtime without recording the old planned windows twice."""
    count = sum(int(float(writer.rows[key].get("repeat_in_window", "0")))
                for key in completed if writer.rows[key].get("warmup") == "1")
    started = time.monotonic()
    for index in range(count):
        one_request(float(scale_entry["input_scale"]), payload_override=scale_entry.get("payload"),
                    req_id=f"{config.request_namespace}_{scale_entry['scale_label']}_resume_warmup{index}")
    layout = writer.layout
    path = layout.recovery_warmup_path(config.case_id, config.case_attempt_id, str(scale_entry["scale_label"]))
    atomic_write_json(path, {"schema_version": 1, "run_id": config.run_id, "case_id": config.case_id,
                            "case_attempt_id": config.case_attempt_id, "input_scale": scale_entry["input_scale"],
                            "request_count": count, "elapsed_s": time.monotonic() - started,
                            "scope": "unmeasured_recovery_preparation", "reused_windows": len(completed)})
