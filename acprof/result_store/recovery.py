"""Repair mutable projections only; immutable source corruption is always fatal."""
from __future__ import annotations

import os

from acprof.artifacts import read_json_object, sync_directory
from acprof.result_csv import ResultValidationError
from acprof.result_layers import PROFILER_LAYERS, has_layer_evidence, layer_identity_fields
from acprof.result_store.protocol import csv_bytes, digest, validate_manifest, write_bytes
from acprof.result_store.sources import replay


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


def recover_projection(layout):
    """Caller holds the publication lock and the single-writer lease."""
    recover_pending(layout)
    manifest = read_json_object(layout.result_csv)
    validate_manifest(layout, manifest)
    rows, windows = replay(layout, manifest)
    payloads = {}
    for layer, info in manifest["layers"].items():
        selected = projected_rows(manifest, layer, list(rows.values()))
        raw = csv_bytes(info["fields"], selected, header=True)
        if (digest(raw) != info["sha256"] or len(raw) != info["committed_bytes"]
                or len(selected) != info["row_count"]):
            raise ResultValidationError(f"{layer} commit disagrees with immutable sources")
        target = layout.layer_csv(layer)
        current = target.read_bytes() if target.is_file() else None
        if current is not None and len(current) > len(raw) and current.startswith(raw):
            with target.open("r+b") as stream:
                stream.truncate(len(raw))
                stream.flush()
                os.fsync(stream.fileno())
        elif current != raw:
            write_bytes(target, raw)
        payloads[layer] = raw
    from acprof.result_store.metadata import recover_metadata
    recover_metadata(layout, manifest)
    clear_pending(layout)
    return manifest, rows, windows, payloads
