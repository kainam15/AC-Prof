"""Immutable authority for completed measurement windows.

Schema 2 receipts precede their disposable CSV projections. Schema 1 retains
the historical case publication order. Neither a telemetry file alone nor a CSV
tail constitutes a receipt. Postprocessing never mutates the original row.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from acprof.artifacts import atomic_write, loads_finite_json
from acprof.telemetry import canonical_bytes, read_window_evidence, write_window_evidence

WINDOW_RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WindowPublicationContext:
    run_id: str
    case_id: str
    case_attempt_id: str
    records_dir: str
    telemetry_dir: str

    def identity(self, window_id: str) -> dict[str, str]:
        values = {"run_id": self.run_id, "case_id": self.case_id,
                  "case_attempt_id": self.case_attempt_id, "window_id": window_id}
        if not all(isinstance(value, str) and value for value in values.values()):
            raise ValueError("window publication requires complete run/case/attempt/window identity")
        return values

    def record_path(self, window_id: str) -> Path:
        return Path(self.records_dir) / window_filename(window_id)


def window_filename(window_id: str) -> str:
    return hashlib.sha256(window_id.encode("utf-8")).hexdigest()[:24] + ".json"


def canonical_csv_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in row.items()}


def content_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def prepare_window_telemetry(context: WindowPublicationContext, window_id: str,
                             evidence: dict[str, Any]) -> dict[str, Any]:
    identity = context.identity(window_id)
    if evidence.get("identity") != identity:
        raise ValueError("window telemetry identity mismatch")
    reference = write_window_evidence(Path(context.telemetry_dir) / window_filename(window_id), evidence)
    reference["path"] = os.path.relpath(reference["path"], context.record_path(window_id).parent)
    return reference


def commit_window_record(context: WindowPublicationContext, *, window_id: str,
                         row: Mapping[str, Any], request_record: dict[str, Any],
                         telemetry_reference: dict[str, Any],
                         projections: dict[str, Any], sequence: int | None = None) -> dict[str, Any]:
    """Persist verified measurement authority; schema 1 requires prior projections."""
    identity = context.identity(window_id)
    request = request_record.get("record")
    if not isinstance(request, dict) or request.get("sniff_group_id") != window_id:
        raise ValueError("request record does not belong to the committed window")
    destination = context.record_path(window_id)
    # Verify existing evidence before making the final commit marker visible.
    evidence = read_window_evidence((destination.parent / telemetry_reference["path"]).resolve(),
                                    expected_sha256=telemetry_reference["sha256"])
    if evidence.get("identity") != identity:
        raise ValueError("referenced window telemetry identity mismatch")
    normalized_row = canonical_csv_row(row)
    journal = request_record.get("journal")
    if isinstance(journal, dict) and journal.get("path"):
        journal = {**journal, "path": os.path.relpath(journal["path"], destination.parent)}
    record = {
        "schema_version": WINDOW_RECORD_SCHEMA_VERSION,
        "identity": identity, "row": normalized_row, "row_sha256": content_hash(normalized_row),
        "request": request, "request_sha256": content_hash(request),
        "sniff_group": {"sniff_group_id": window_id},
        "telemetry": telemetry_reference,
        "projections": {**projections, "request_journal": journal},
        "projection_semantics": "publication_time_only; later postprocessing may rebuild CSV",
    }
    if sequence is not None:
        if type(sequence) is not int or sequence < 1:
            raise ValueError("invalid window publication sequence")
        record.update(schema_version=2, sequence=sequence,
                      projection_semantics="immutable_authority_before_result_projection")
    record["content_sha256"] = content_hash(record)
    payload = canonical_bytes(record) + b"\n"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f"refusing to replace a committed window record: {destination}")
    else:
        def write(stream: TextIO) -> None:
            stream.write(payload.decode("utf-8"))
        atomic_write(destination, write)
    return record


def read_window_record(path: str | Path, *, verify_evidence: bool = True,
                       expected_sha256: str | None = None) -> dict[str, Any]:
    """Verify and parse one receipt snapshot, optionally deferring evidence I/O."""
    source = Path(path)
    try:
        payload = source.read_bytes()
        if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("window receipt SHA256 mismatch")
        record = loads_finite_json(payload.decode("utf-8"))
        if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
                or record.get("schema_version") not in {WINDOW_RECORD_SCHEMA_VERSION, 2}):
            raise ValueError("unsupported window record schema")
        if record["schema_version"] == 2 and (type(record.get("sequence")) is not int or record["sequence"] < 1):
            raise ValueError("invalid window publication sequence")
        body = {key: value for key, value in record.items() if key != "content_sha256"}
        if content_hash(body) != record.get("content_sha256"):
            raise ValueError("window record content hash mismatch")
        identity = record.get("identity")
        if not isinstance(identity, dict) or set(identity) != {"run_id", "case_id", "case_attempt_id", "window_id"}:
            raise ValueError("invalid window identity")
        if not all(isinstance(value, str) and value for value in identity.values()):
            raise ValueError("incomplete window identity")
        if source.name != window_filename(identity["window_id"]):
            raise ValueError("window record filename does not match its identity")
        if not isinstance(record.get("row"), dict) or not all(
                isinstance(value, str) for value in record["row"].values()):
            raise ValueError("invalid canonical window row")
        if not isinstance(record.get("request"), dict):
            raise ValueError("invalid window request record")
        if content_hash(record.get("row")) != record.get("row_sha256"):
            raise ValueError("window row hash mismatch")
        if content_hash(record.get("request")) != record.get("request_sha256"):
            raise ValueError("window request hash mismatch")
        if record.get("request", {}).get("sniff_group_id") != identity["window_id"]:
            raise ValueError("window request identity mismatch")
        if record.get("sniff_group") != {"sniff_group_id": identity["window_id"]}:
            raise ValueError("window sidecar identity mismatch")
        if verify_evidence:
            verify_window_record_evidence(source, record)
        return record
    except (OSError, KeyError, TypeError, AttributeError, UnicodeError) as exc:
        raise ValueError(f"invalid window record {source}: {exc}") from exc


def verify_window_record_evidence(path: str | Path, record: dict[str, Any]) -> None:
    """Verify the loaded receipt's evidence without reopening the receipt.

    Callers may check layout-specific paths after read_window_record has checked
    the receipt itself, and before this function reads any referenced evidence.
    """
    source = Path(path)
    try:
        reference = record["telemetry"]
        evidence = read_window_evidence(source.parent / reference["path"],
                                        expected_sha256=reference["sha256"])
        if evidence.get("identity") != record["identity"]:
            raise ValueError("window telemetry identity mismatch")
    except (OSError, KeyError, TypeError, AttributeError, UnicodeError) as exc:
        raise ValueError(f"invalid window record {source}: {exc}") from exc


def verify_case_window_records(directory: str | Path, *, run_id: str | None = None,
                               case_id: str | None = None,
                               case_attempt_id: str | None = None) -> list[dict[str, Any]]:
    records = []
    identities: set[tuple[str, str, str, str]] = set()
    for path in sorted(Path(directory).glob("*.json")):
        record = read_window_record(path)
        identity = record["identity"]
        if run_id is not None and identity["run_id"] != run_id:
            raise ValueError("window record belongs to a different run")
        if case_id is not None and identity["case_id"] != case_id:
            raise ValueError("window record belongs to a different case")
        if case_attempt_id is not None and identity["case_attempt_id"] != case_attempt_id:
            raise ValueError("window record belongs to a different case attempt")
        key = tuple(identity[field] for field in ("run_id", "case_id", "case_attempt_id", "window_id"))
        if key in identities:
            raise ValueError("duplicate committed window identity")
        identities.add(key)
        records.append(record)
    return records
