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
from typing import Any, TextIO

from acprof.artifacts import (
    atomic_write,
    atomic_write_json,
    create_durable_directory,
    file_sha256,
    read_json_object,
    sync_directory,
)
from acprof.cancellation import cancellable, check_cancelled
from acprof.results.csv import ResultValidationError, ValidatedResultRows, parse_result_stream
from acprof.results.layers import BASE_LAYERS, LAYER_FILES, layer_fields
from acprof.results.layout import ArtifactLayout

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
    with path.open("a+b" if exclusive else "rb") as stream:
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
        try:
            check_cancelled(cancelled)
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def reference(layout: ArtifactLayout, path: Path) -> dict:
    return {"path": path.relative_to(layout.root).as_posix(), "sha256": file_sha256(path)}


def read_reference(layout: ArtifactLayout, ref: dict | None) -> dict:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str) or not isinstance(ref["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", ref["sha256"]) is None):
        raise ResultValidationError("invalid result transaction reference")
    return read_json_object(layout.contained(ref["path"]), expected_sha256=ref["sha256"], bounded=False)


def validate_manifest(layout: ArtifactLayout, manifest: dict) -> dict[str, Path]:
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or type(manifest.get("schema_version")) is not int
            or manifest.get("state") not in STATES
            or not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]
            or type(manifest.get("sequence")) is not int or manifest["sequence"] < 0
            or type(manifest.get("row_count")) is not int or manifest["row_count"] < 0
            or not isinstance(manifest.get("wide_fields"), list)):
        raise ResultValidationError("invalid incremental result manifest")
    assigned = layer_fields(manifest["wide_fields"])
    if manifest.get("schema_sha256") != digest(json_bytes(assigned)):
        raise ResultValidationError("incremental result field ownership changed")
    layers = manifest.get("layers")
    if not isinstance(layers, dict) or set(layers) != set(LAYER_FILES):
        raise ResultValidationError("missing or unknown incremental result layer")
    paths = {}
    for layer, info in layers.items():
        if (not isinstance(info, dict) or info.get("path") != LAYER_FILES[layer]
                or info.get("fields") != assigned[layer]
                or type(info.get("row_count")) is not int or info["row_count"] < 0
                or type(info.get("committed_bytes")) is not int or info["committed_bytes"] < 1
                or not isinstance(info.get("sha256"), str) or re.fullmatch(r"[a-f0-9]{64}", info["sha256"]) is None
                or type(info.get("revision")) is not int or info["revision"] < 0):
            raise ResultValidationError(f"invalid incremental {layer} layer")
        if layer in BASE_LAYERS and info["row_count"] != manifest["row_count"]:
            raise ResultValidationError(f"incremental {layer} row count mismatch")
        paths[layer] = layout.layer_csv(layer)
    commit = read_reference(layout, manifest.get("commit"))
    if not layout.contained(manifest["commit"]["path"]).is_relative_to(layout.result_store_path("commits")):
        raise ResultValidationError("result commit path is outside its journal")
    if commit.get("after") != {k: v for k, v in manifest.items() if k != "commit"}:
        raise ResultValidationError("result manifest disagrees with its committed transaction")
    return paths


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
    for layer, path in cancellable(validate_manifest(layout, manifest).items(), cancelled):
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


def save_commit(layout: ArtifactLayout, before: dict | None, after: dict, sources: list[dict], *, transaction: str) -> dict:
    path = layout.result_store_path(f"commits/{after['sequence']:020d}-{transaction}.json")
    payload = {"schema_version": 1, "previous": before.get("commit") if before else None,
               "after": after, "sources": sources}
    if path.exists():
        if read_json_object(path, bounded=False) != payload:
            raise ResultValidationError("refusing to overwrite a result transaction")
    else:
        atomic_write_json(path, payload)
    sync_directory(path.parent.parent)
    return {**after, "commit": reference(layout, path)}
