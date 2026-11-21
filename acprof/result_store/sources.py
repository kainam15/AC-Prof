"""Replay immutable measurements and explicit derivations without recomputing metrics."""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, DecimalException, localcontext
from typing import TYPE_CHECKING

from acprof.artifacts import file_sha256
from acprof.result_store.protocol import IDENTITY_FIELDS, read_reference, reference_path
from acprof.results.csv import (
    KEY_FIELDS,
    ResultValidationError,
    measurement_key,
    require_matching_run_id,
    run_id_column,
)
from acprof.results.layout import ArtifactLayout
from acprof.results.windows import read_window_record, verify_window_record_evidence

if TYPE_CHECKING:
    from acprof.result_store.writer import ResultWriter

# New derivations identify rows with context-independent decimal keys. Older
# readers can still consume the existing key field for ordinary measurements.
EXACT_KEY_ENCODING = "exact-decimal-v1"


def window_row(layout: ArtifactLayout, source: dict) -> tuple[dict, tuple[str, ...]]:
    return _window_row(layout, source, None)


def _window_row(layout: ArtifactLayout, source: dict, verified_receipts: Mapping[tuple[str, ...], str] | None
                ) -> tuple[dict, tuple[str, ...]]:
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
    referenced_telemetry = path.parent / record["telemetry"]["path"]
    if referenced_telemetry.absolute().resolve() != telemetry.resolve():
        raise ResultValidationError("window telemetry path disagrees with its identity")
    key = tuple(identity[name] for name in IDENTITY_FIELDS)
    if verified_receipts is not None and verified_receipts.get(key) == ref["sha256"]:
        # This exact receipt bound the telemetry digest to a decoded identity in this lease.
        try:
            if file_sha256(referenced_telemetry) != record["telemetry"]["sha256"]:
                raise ValueError("window telemetry hash mismatch")
        except OSError as exc:
            raise ValueError(f"invalid window record {path}: {exc}") from exc
    else:
        verify_window_record_evidence(path, record)
    if any(name in record["row"] and record["row"][name] != identity[name] for name in IDENTITY_FIELDS):
        raise ResultValidationError("window row disagrees with its receipt identity")
    return {**record["row"], **identity}, key


class _LegacyChangeKeys:
    """Index original spellings once per row while replaying pre-encoding derivations."""

    def __init__(self):
        self.aliases: dict[tuple[str, ...], tuple[str, ...] | None] = {}
        self.last_key: tuple[str, ...] | None = None

    def update(self, rows):
        # Replay only appends rows or patches their non-identity fields in place.
        with localcontext():
            for key in reversed(rows):
                if key == self.last_key:
                    break
                try:
                    legacy = tuple(part if part in {"off", "on", "inherited"} else
                                   str(Decimal(rows[key][field]).normalize())
                                   for field, part in zip(KEY_FIELDS, key, strict=True))
                except DecimalException as error:
                    raise ResultValidationError("cannot reproduce historical measurement key") from error
                self.aliases[legacy] = None if legacy in self.aliases else key
        self.last_key = next(reversed(rows), None)

    def resolve(self, key):
        if key not in self.aliases:
            raise ResultValidationError("invalid derived measurement key or fields")
        if self.aliases[key] is None:
            raise ResultValidationError("ambiguous historical measurement key")
        return self.aliases[key]


def _read_derivation(layout, manifest, source):
    ref = source["reference"]
    if not layout.contained(ref["path"]).is_relative_to(layout.path("raw/result_updates")):
        raise ResultValidationError("derivation source is outside raw/result_updates")
    update = read_reference(layout, ref)
    if update.get("schema_version") != 1 or update.get("run_id") != manifest["run_id"]:
        raise ResultValidationError("invalid result derivation identity")
    encoding = update.get("measurement_key_encoding")
    if "measurement_key_encoding" in update and encoding != EXACT_KEY_ENCODING:
        raise ResultValidationError("unknown measurement key encoding in result derivation")
    if not isinstance(update.get("reason"), str) or not update["reason"]:
        raise ResultValidationError("result derivation requires provenance")
    for evidence in update.get("evidence", []):
        path = layout.contained(evidence["path"])
        if file_sha256(path) != evidence["sha256"]:
            raise ResultValidationError("result derivation evidence changed")
    return update


def apply_source(layout, manifest, source, rows, windows, *, legacy_keys=None):
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
    update = _read_derivation(layout, manifest, source)
    encoding = update.get("measurement_key_encoding")
    identity_column = run_id_column(fields) or "run_id"
    source_path = source["reference"]["path"]
    for row in update.get("insert", []):
        if not isinstance(row, dict) or set(row) - set(fields) or not all(isinstance(value, str) for value in row.values()):
            raise ResultValidationError("invalid derived outcome fields")
        require_matching_run_id(manifest["run_id"], row.get(identity_column), source=source_path)
        key = measurement_key(row)
        if key in rows:
            raise ResultValidationError("duplicate derived outcome")
        rows[key] = {name: row.get(name, "nan") for name in fields}
    changes = update.get("changes", [])
    if changes and encoding is None:
        if legacy_keys is None:
            legacy_keys = _LegacyChangeKeys()
        legacy_keys.update(rows)
    for change in changes:
        key = tuple(change["key"])
        if encoding is None:
            key = legacy_keys.resolve(key)
        values = change["values"]
        if key not in rows or not isinstance(values, dict) or not set(values) <= set(fields):
            raise ResultValidationError("invalid derived measurement key or fields")
        if any(name in IDENTITY_FIELDS or name in ("cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx")
               for name in values):
            raise ResultValidationError("derivation cannot change measurement identity")
        if identity_column in values:
            require_matching_run_id(manifest["run_id"], values[identity_column], source=source_path)
        rows[key] = {**rows[key], **values}


def _verified_commits(layout, manifest):
    """Validate the complete chain before accessing the referenced sources."""
    commits = []
    ref = manifest["commit"]
    expected_sequence = manifest["sequence"]
    while ref is not None:
        path = reference_path(layout, ref)
        if not path.is_relative_to(layout.result_store_path("commits")):
            raise ResultValidationError("result commit path is outside its journal")
        commit = read_reference(layout, ref)
        after = commit.get("after")
        if (type(commit.get("schema_version")) is not int or commit["schema_version"] != 1
                or not isinstance(after, dict) or type(after.get("sequence")) is not int
                or after["sequence"] != expected_sequence
                or after.get("run_id") != manifest["run_id"]
                or after.get("schema_sha256") != manifest["schema_sha256"]):
            raise ResultValidationError("broken result commit chain")
        if type(after.get("row_count")) is not int or after["row_count"] < 0:
            raise ResultValidationError("invalid result commit row_count")
        if not isinstance(commit.get("sources"), list):
            raise ResultValidationError("invalid result commit sources")
        if "previous" not in commit:
            raise ResultValidationError("missing result commit previous reference")
        for source in commit["sources"]:
            if not isinstance(source, dict) or source.get("kind") not in ("window", "derivation"):
                raise ResultValidationError("invalid result commit source")
            reference_path(layout, source.get("reference"))
        # Keep replay inputs without retaining every historical manifest.
        commits.append((after["row_count"], commit["sources"]))
        expected_sequence -= 1
        ref = commit.get("previous")
    if expected_sequence != -1:
        raise ResultValidationError("incomplete result commit chain")
    commits.reverse()
    return commits


def verify_source_files(layout, manifest, *, writer: ResultWriter | None = None) -> None:
    """Recheck immutable files under the same writer lease before final freeze."""
    receipts = writer.verified_receipts(layout, manifest) if writer is not None else None
    for _, sources in _verified_commits(layout, manifest):
        for source in sources:
            if source["kind"] == "window":
                _window_row(layout, source, receipts)
            else:
                _read_derivation(layout, manifest, source)


def replay(layout, manifest):
    """Rebuild ordered rows only after checking the complete commit chain."""
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    windows: dict[tuple[str, ...], str] = {}
    legacy_keys = _LegacyChangeKeys()
    for expected_rows, sources in _verified_commits(layout, manifest):
        for source in sources:
            apply_source(layout, manifest, source, rows, windows, legacy_keys=legacy_keys)
        if len(rows) != expected_rows:
            raise ResultValidationError("result transaction row count disagrees with its sources")
    return rows, windows
