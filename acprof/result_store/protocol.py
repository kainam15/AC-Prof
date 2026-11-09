"""Versioned local-file protocol shared by result readers and the single writer."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO, TypeGuard

from acprof.artifacts import (
    atomic_write,
    atomic_write_json,
    create_durable_directory,
    file_sha256,
    read_json_object,
    sync_directory,
)
from acprof.cancellation import cancellable, check_cancelled
from acprof.result_store.types import (
    CommitRecord,
    PendingTransaction,
    ResultManifest,
    ResultManifestBody,
    ResultReference,
    ResultSource,
)
from acprof.results.csv import ResultValidationError, ValidatedResultRows, parse_result_stream
from acprof.results.layers import BASE_LAYERS, LAYER_FILES, layer_fields
from acprof.results.layout import ArtifactLayout
from acprof.system.cleanup import cleanup_on_exit

SCHEMA_VERSION = 3
IDENTITY_FIELDS = ("run_id", "case_id", "case_attempt_id", "window_id")
STATES = frozenset(("initialized", "collecting", "processing", "complete", "interrupted"))
_HASH_BUFFER_BYTES = 2**18


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def csv_bytes(fields, rows=(), *, header=False) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", restval="nan")
    if header:
        writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_bytes(path: Path, raw: bytes) -> None:
    # atomic_write uses newline="", so byte offsets also cover UTF-8 and quoted newlines.
    def write(stream: TextIO) -> None:
        stream.write(raw.decode("utf-8"))
    atomic_write(path, write)


@contextmanager
def publication_lock(layout: ArtifactLayout, *, exclusive: bool, blocking: bool = True,
                     cancelled: Callable[[], bool] | None = None):
    """Separate from the parent's lifetime result lock; readers never create files."""
    check_cancelled(cancelled)
    path = layout.result_store_path("lock")
    if exclusive:
        create_durable_directory(path.parent)
    stream = path.open("a+b" if exclusive else "rb")
    with cleanup_on_exit(stream.close, description=f"publication lock stream {path}"):
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if blocking and cancelled is not None:
            while True:
                check_cancelled(cancelled)
                try:
                    fcntl.flock(stream.fileno(), operation | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    check_cancelled(cancelled)
                    time.sleep(0.05)
        else:
            fcntl.flock(stream.fileno(), operation | (0 if blocking else fcntl.LOCK_NB))
        with cleanup_on_exit(lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN),
                             description=f"publication lock release {path}"):
            check_cancelled(cancelled)
            yield


def reference(layout: ArtifactLayout, path: Path) -> ResultReference:
    return {"path": path.relative_to(layout.root).as_posix(), "sha256": file_sha256(path)}


def validate_run_identity(manifest: dict, expected_run_id: str) -> None:
    """Compare the caller's frozen identity while its publication lock is held."""
    if not isinstance(expected_run_id, str) or not expected_run_id:
        raise ResultValidationError("invalid expected result run identity")
    if manifest.get("run_id") != expected_run_id:
        raise ResultValidationError("result store belongs to another run")


def check_pending_replacement(layout: ArtifactLayout, manifest: dict, *,
                              cancelled: Callable[[], bool] | None = None,
                              byte_budget: list[int] | None = None) -> None:
    """Under the publication lock, reject replacements whose commit is not visible."""
    transaction = read_pending(layout, manifest, cancelled=cancelled, byte_budget=byte_budget)
    if transaction is not None and transaction["replacements"] and manifest == transaction["before"]:
        raise ResultValidationError("result module replacement requires recovery")


def reference_path(layout: ArtifactLayout, ref: ResultReference | None) -> Path:
    """Check reference structure and containment without reading its target."""
    if (not isinstance(ref, dict) or any(not isinstance(name, str) for name in ref)
            or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str) or not ref["path"]
            or not isinstance(ref["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", ref["sha256"]) is None):
        raise ResultValidationError("invalid result transaction reference")
    try:
        return layout.contained(ref["path"])
    except ValueError as error:
        raise ResultValidationError(f"invalid result transaction reference: {error}") from error


def read_reference(layout: ArtifactLayout, ref: ResultReference | None, *,
                   cancelled: Callable[[], bool] | None = None, byte_budget: list[int] | None = None) -> dict:
    path = reference_path(layout, ref)
    assert ref is not None  # reference_path rejects absent references.
    return read_json_object(path, expected_sha256=ref["sha256"], bounded=False,
                            cancelled=cancelled, byte_budget=byte_budget)


def _validate_layer_structure(layer: str, info: object) -> None:
    """Reject malformed layer fields before comparisons or hash checks."""
    prefix = f"invalid incremental {layer} layer: layers.{layer}"
    if not isinstance(info, dict):
        raise ResultValidationError(f"{prefix} must be an object")
    if any(not isinstance(name, str) for name in info):
        raise ResultValidationError(f"{prefix} field names must be strings")
    for name in ("path", "sha256"):
        if not isinstance(info.get(name), str):
            raise ResultValidationError(f"{prefix}.{name} must be a string")
    fields = info.get("fields")
    if not isinstance(fields, list) or any(not isinstance(name, str) for name in fields):
        raise ResultValidationError(f"{prefix}.fields must be a list of strings")
    for name in ("row_count", "committed_bytes", "revision"):
        if type(info.get(name)) is not int:
            raise ResultValidationError(f"{prefix}.{name} must be an integer")


def validate_manifest_fields(layout: ArtifactLayout, manifest: object, *, committed: bool) -> dict[str, Path]:
    """Validate types, nested structure and values without reading projections."""
    prefix = "invalid incremental result manifest"
    if not isinstance(manifest, dict):
        raise ResultValidationError(f"{prefix}: expected an object")
    if any(not isinstance(name, str) for name in manifest):
        raise ResultValidationError(f"{prefix}: field names must be strings")
    for name in ("schema_version", "sequence", "row_count"):
        if type(manifest.get(name)) is not int:
            raise ResultValidationError(f"{prefix}: {name} must be an integer")
    for name in ("state", "run_id", "schema_sha256"):
        if not isinstance(manifest.get(name), str):
            raise ResultValidationError(f"{prefix}: {name} must be a string")
    wide_fields = manifest.get("wide_fields")
    if not isinstance(wide_fields, list) or any(not isinstance(name, str) for name in wide_fields):
        raise ResultValidationError(f"{prefix}: wide_fields must be a list of strings")
    layers = manifest.get("layers")
    if not isinstance(layers, dict):
        raise ResultValidationError("missing or unknown incremental result layer: layers must be an object")
    if set(layers) != set(LAYER_FILES):
        raise ResultValidationError("missing or unknown incremental result layer: layers must contain every result layer")
    for layer, info in layers.items():
        _validate_layer_structure(layer, info)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ResultValidationError(f"{prefix}: schema_version must be {SCHEMA_VERSION}")
    if manifest["state"] not in STATES:
        raise ResultValidationError(f"{prefix}: state is not a supported result state")
    if not manifest["run_id"]:
        raise ResultValidationError(f"{prefix}: run_id must be non-empty")
    for name in ("sequence", "row_count"):
        if manifest[name] < 0:
            raise ResultValidationError(f"{prefix}: {name} must be non-negative")
    try:
        assigned = layer_fields(wide_fields)
    except ResultValidationError as error:
        raise ResultValidationError(f"{prefix}: wide_fields: {error}") from error
    if manifest["schema_sha256"] != digest(json_bytes(assigned)):
        raise ResultValidationError("incremental result field ownership changed: schema_sha256 does not match wide_fields")
    paths = {}
    for layer, info in layers.items():
        layer_prefix = f"invalid incremental {layer} layer: layers.{layer}"
        if info["path"] != LAYER_FILES[layer]:
            raise ResultValidationError(f"{layer_prefix}.path must be {LAYER_FILES[layer]!r}")
        if info["fields"] != assigned[layer]:
            raise ResultValidationError(f"{layer_prefix}.fields disagree with wide_fields")
        for name, minimum in (("row_count", 0), ("committed_bytes", 1), ("revision", 0)):
            if info[name] < minimum:
                raise ResultValidationError(f"{layer_prefix}.{name} must be at least {minimum}")
        if re.fullmatch(r"[a-f0-9]{64}", info["sha256"]) is None:
            raise ResultValidationError(f"{layer_prefix}.sha256 must be a lowercase SHA256")
        if layer in BASE_LAYERS and info["row_count"] != manifest["row_count"]:
            raise ResultValidationError(f"incremental {layer} row count mismatch: layers.{layer}.row_count")
        paths[layer] = layout.layer_csv(layer)
    if committed:
        if not reference_path(layout, manifest.get("commit")).is_relative_to(layout.result_store_path("commits")):
            raise ResultValidationError("result commit path is outside its journal")
    elif "commit" in manifest:
        raise ResultValidationError("uncommitted result manifest must not contain commit")
    if "snapshot" in manifest:
        if not reference_path(layout, manifest["snapshot"]).is_relative_to(layout.result_store_path("snapshots")):
            raise ResultValidationError("result snapshot path is outside its journal")
    return paths


def validate_manifest_structure(manifest: object, layout: ArtifactLayout) -> TypeGuard[ResultManifest]:
    """Narrow the original dictionary after the shared field contract succeeds."""
    validate_manifest_fields(layout, manifest, committed=True)
    return True


def validate_manifest(layout: ArtifactLayout, manifest: dict, *,
                      cancelled: Callable[[], bool] | None = None, byte_budget: list[int] | None = None) -> dict[str, Path]:
    check_cancelled(cancelled)
    paths = validate_manifest_fields(layout, manifest, committed=True)
    commit = read_reference(layout, manifest["commit"], cancelled=cancelled, byte_budget=byte_budget)
    if commit.get("after") != {k: v for k, v in manifest.items() if k != "commit"}:
        raise ResultValidationError("result manifest disagrees with its committed transaction")
    return paths


def validate_pending_transaction(transaction: object, layout: ArtifactLayout,
                                 manifest: dict) -> TypeGuard[PendingTransaction]:
    """One runtime contract for both readable prefixes and recoverable transactions."""
    if not isinstance(transaction, dict) or not {"before", "after", "replacements"} <= transaction.keys():
        raise ResultValidationError("pending transaction requires before, after and replacements")
    if any(not isinstance(name, str) for name in transaction):
        raise ResultValidationError("pending transaction field names must be strings")
    before, after, replacements = transaction["before"], transaction["after"], transaction["replacements"]
    validate_manifest_fields(layout, before, committed=True)
    validate_manifest_fields(layout, after, committed=False)
    if (not isinstance(replacements, list)
            or any(not isinstance(layer, str) or layer not in LAYER_FILES for layer in replacements)
            or len(replacements) != len(set(replacements))):
        raise ResultValidationError("invalid pending replacement layers")
    if (any(before[name] != after[name] for name in ("schema_version", "run_id", "schema_sha256", "wide_fields"))
            or after["sequence"] != before["sequence"] + 1 or after["row_count"] < before["row_count"]):
        raise ResultValidationError("pending result transaction changes identity or sequence")
    for layer in LAYER_FILES:
        old, new = before["layers"][layer], after["layers"][layer]
        replaced = layer in replacements
        if (new["revision"] != old["revision"] + int(replaced)
                or (not replaced and (new["committed_bytes"] < old["committed_bytes"]
                                      or new["row_count"] < old["row_count"]))):
            raise ResultValidationError(f"pending {layer} transition disagrees with replacements")
    if manifest != before and {k: v for k, v in manifest.items() if k != "commit"} != after:
        raise ResultValidationError("pending result transaction does not match current commit")
    return True


def read_pending(layout: ArtifactLayout, manifest: dict, *,
                 cancelled: Callable[[], bool] | None = None,
                 byte_budget: list[int] | None = None) -> PendingTransaction | None:
    """Read and validate pending metadata without repairing or consuming evidence."""
    check_cancelled(cancelled)
    path = layout.result_store_path("pending.json")
    if not path.exists():
        return None
    try:
        transaction = read_json_object(path, bounded=False, cancelled=cancelled, byte_budget=byte_budget)
        if not validate_pending_transaction(transaction, layout, manifest):
            raise ResultValidationError("invalid pending result transaction")
        if manifest != transaction["before"]:
            commit = read_reference(layout, manifest.get("commit"), cancelled=cancelled, byte_budget=byte_budget)
            if commit.get("previous") != transaction["before"]["commit"]:
                raise ResultValidationError("pending transaction disagrees with committed previous reference")
    except ValueError as error:
        raise ResultValidationError(f"invalid pending result transaction: {error}") from error
    check_cancelled(cancelled)
    return transaction

class _CommittedPrefixReader(io.RawIOBase):
    """Hash one raw prefix without prefetching bytes from an uncommitted tail."""

    def __init__(self, stream, path: Path, info: dict, cancelled: Callable[[], bool] | None = None):
        super().__init__()
        self._stream = stream
        self._path = path
        self._remaining = info["committed_bytes"]
        self._expected_digest = info["sha256"]
        self._digest = hashlib.sha256()
        self._cancelled = cancelled

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        check_cancelled(self._cancelled)
        if not self._remaining or not buffer:
            return 0
        view = memoryview(buffer)[:min(self._remaining, _HASH_BUFFER_BYTES)]
        size = self._stream.readinto(view)
        if size is None:
            raise BlockingIOError("result layer read would block")
        if not size:
            raise ResultValidationError(f"missing or modified {self._path} committed prefix")
        self._digest.update(view[:size])
        self._remaining -= size
        return size

    def verify(self, *, exact=False) -> None:
        check_cancelled(self._cancelled)
        if self._remaining:
            buffer = bytearray(min(self._remaining, _HASH_BUFFER_BYTES))
            while self._remaining:
                self.readinto(buffer)
        check_cancelled(self._cancelled)
        if exact:
            tail = self._stream.read(1)
            if tail is None:
                raise BlockingIOError("result layer tail check would block")
            if tail:
                raise ResultValidationError(f"uncommitted tail in {self._path}")
        check_cancelled(self._cancelled)
        if self._digest.hexdigest() != self._expected_digest:
            raise ResultValidationError(f"missing or modified {self._path} committed prefix")


def validate_layer_files(layout: ArtifactLayout, manifest: dict, *, exact=False,
                         cancelled: Callable[[], bool] | None = None) -> None:
    """Check every layer with bounded buffers, without retaining its payload."""
    check_cancelled(cancelled)
    for layer, path in cancellable(validate_manifest(layout, manifest, cancelled=cancelled).items(), cancelled):
        info = manifest["layers"][layer]
        with path.open("rb", buffering=0) as raw, _CommittedPrefixReader(raw, path, info, cancelled) as prefix:
            prefix.verify(exact=exact)


def read_layer_rows(path: Path, info: dict, *, cancelled: Callable[[], bool] | None = None) -> ValidatedResultRows:
    """Hash and parse the same bounded prefix before returning any rows."""
    check_cancelled(cancelled)
    with path.open("rb", buffering=0) as raw:
        prefix = _CommittedPrefixReader(raw, path, info, cancelled)
        with io.TextIOWrapper(io.BufferedReader(prefix, buffer_size=_HASH_BUFFER_BYTES),
                              encoding="utf-8", newline="") as stream:
            try:
                columns, validated = parse_result_stream(stream, path, None, allow_empty=True, cancelled=cancelled)
            except (ValueError, csv.Error):
                # Preserve integrity-error precedence even if decoding/CSV failed
                # before reaching the end of the committed bytes.
                prefix.verify()
                raise
            prefix.verify()
    if columns != info["fields"]:
        raise ResultValidationError(f"result layer header changed: {path}")
    check_cancelled(cancelled)
    return validated


def save_commit(layout: ArtifactLayout, before: ResultManifest | None, after: ResultManifestBody,
                sources: list[ResultSource], *, transaction: str) -> ResultManifest:
    path = layout.result_store_path(f"commits/{after['sequence']:020d}-{transaction}.json")
    payload: CommitRecord = {"schema_version": 1, "previous": before["commit"] if before else None,
                             "after": after, "sources": sources}
    if path.exists():
        if read_json_object(path, bounded=False) != payload:
            raise ResultValidationError("refusing to overwrite a result transaction")
    else:
        atomic_write_json(path, payload)
    sync_directory(path.parent.parent)
    return {**after, "commit": reference(layout, path)}
