"""Repair mutable projections only; immutable source corruption is always fatal."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from acprof.artifacts import read_json_object, sync_directory
from acprof.result_store.protocol import (
    csv_bytes,
    read_pending,
    validate_manifest,
    validate_run_identity,
    write_bytes,
)
from acprof.result_store.sources import replay
from acprof.results.csv import ResultValidationError
from acprof.results.layers import PROFILER_LAYERS, has_layer_evidence, layer_identity_fields

_COMPARE_CHUNK_BYTES = 64 * 1024


def _matches_prefix(stream, raw: bytes) -> bool:
    with memoryview(raw) as expected:
        offset = 0
        while offset < len(expected):
            block = stream.read(min(_COMPARE_CHUNK_BYTES, len(expected) - offset))
            if not block or block != expected[offset:offset + len(block)]:
                return False
            offset += len(block)
    return True


def _repair_projection(target: Path, raw: bytes) -> None:
    if target.is_file():
        # Raw I/O keeps each scan within the prefix and one tail-probe byte.
        with target.open("rb", buffering=0) as stream:
            matches = _matches_prefix(stream, raw)
            if matches and not stream.read(1):
                return
        if matches:
            # Recheck the same writable descriptor that will be truncated.
            # Complete projections never need to be opened for writing.
            with target.open("r+b", buffering=0) as stream:
                if _matches_prefix(stream, raw):
                    if stream.read(1):
                        stream.truncate(len(raw))
                        stream.flush()
                        os.fsync(stream.fileno())
                    return
    write_bytes(target, raw)


def projected_rows(manifest, layer, rows):
    fields = manifest["layers"][layer]["fields"]
    identity = layer_identity_fields(manifest["wide_fields"])
    return [row for row in rows if layer not in PROFILER_LAYERS
            or any(has_layer_evidence(row.get(name, "")) for name in fields if name not in identity)]


def clear_pending(layout):
    layout.result_store_path("pending.json").unlink(missing_ok=True)
    sync_directory(layout.result_store_path(""))


def recover_pending(layout, manifest=None):
    """Validate the same pending contract as readers before any repair writes."""
    if manifest is None:
        manifest = read_json_object(layout.result_csv)
        validate_manifest(layout, manifest)
    read_pending(layout, manifest)


def _recover_layer(layout, manifest, layer, rows):
    """Keep this layer's verified hash state, releasing its CSV buffer on return."""
    info = manifest["layers"][layer]
    selected = projected_rows(manifest, layer, rows.values())
    raw = csv_bytes(info["fields"], selected, header=True)
    checksum = hashlib.sha256(raw)
    if (checksum.hexdigest() != info["sha256"] or len(raw) != info["committed_bytes"]
            or len(selected) != info["row_count"]):
        raise ResultValidationError(f"{layer} commit disagrees with immutable sources")
    _repair_projection(layout.layer_csv(layer), raw)
    return checksum


def recover_projection(layout, *, expected_run_id: str | None = None):
    """Recover projections and hash states under the publication lock and writer lease."""
    manifest = read_json_object(layout.result_csv)
    validate_manifest(layout, manifest)
    if expected_run_id is not None:
        validate_run_identity(manifest, expected_run_id)
    recover_pending(layout, manifest)
    rows, windows = replay(layout, manifest)
    hashes = {layer: _recover_layer(layout, manifest, layer, rows) for layer in manifest["layers"]}
    from acprof.result_store.metadata import recover_metadata
    recover_metadata(layout, manifest)
    clear_pending(layout)
    return manifest, rows, windows, hashes
