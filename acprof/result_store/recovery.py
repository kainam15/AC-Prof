"""Repair mutable projections only; immutable source corruption is always fatal."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from acprof.artifacts import read_json_object, sync_directory
from acprof.result_csv import ResultValidationError
from acprof.result_layers import PROFILER_LAYERS, has_layer_evidence, layer_identity_fields
from acprof.result_store.protocol import csv_bytes, validate_manifest, write_bytes
from acprof.result_store.sources import replay

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


def recover_pending(layout):
    path = layout.result_store_path("pending.json")
    if not path.exists():
        return
    pending = read_json_object(path, bounded=False)
    current = read_json_object(layout.result_csv)
    if current.get("commit") == pending["before"].get("commit"):
        before = pending["before"]
        validate_manifest(layout, before)
    elif {k: v for k, v in current.items() if k != "commit"} != pending["after"]:
        raise ResultValidationError("pending result transaction does not match current commit")
    else:
        validate_manifest(layout, current)


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


def recover_projection(layout):
    """Recover projections and hash states under the publication lock and writer lease."""
    recover_pending(layout)
    manifest = read_json_object(layout.result_csv)
    validate_manifest(layout, manifest)
    rows, windows = replay(layout, manifest)
    hashes = {layer: _recover_layer(layout, manifest, layer, rows) for layer in manifest["layers"]}
    from acprof.result_store.metadata import recover_metadata
    recover_metadata(layout, manifest)
    clear_pending(layout)
    return manifest, rows, windows, hashes
