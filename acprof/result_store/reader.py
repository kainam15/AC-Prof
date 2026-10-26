"""Read one committed prefix from all root CSVs, never their uncommitted tails."""
from __future__ import annotations

import hashlib

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import loads_finite_json, read_json_object
from acprof.result_csv import ResultCsvSnapshot, ResultValidationError, measurement_key
from acprof.result_layers import BASE_LAYERS, LAYER_FILES, layer_identity_fields
from acprof.result_store.protocol import (
    digest,
    parse_layer,
    publication_lock,
    read_layer_bytes,
    validate_manifest,
)


def committed_content_digest(root, *, cancelled, budget):
    """Bounded, nonblocking verification for duplicate entries in experiment lists."""
    if cancelled():
        raise InterruptedError("experiment scan cancelled")
    if budget[0] <= 0:
        raise ValueError("duplicate_evidence_unverified: CSV byte budget exceeded")
    layout = ArtifactLayout.discover(root)
    with publication_lock(layout, exclusive=False, blocking=False):
        raw = layout.result_csv.read_bytes()
        if len(raw) > budget[0]:
            budget[0] -= len(raw)
            raise ValueError("duplicate_evidence_unverified: CSV byte budget exceeded")
        manifest = loads_finite_json(raw.decode("utf-8"))
        paths = validate_manifest(layout, manifest)
        size = len(raw) + sum(info["committed_bytes"] for info in manifest["layers"].values())
        budget[0] -= size
        if budget[0] < 0:
            raise ValueError("duplicate_evidence_unverified: CSV byte budget exceeded")
        checksum = hashlib.sha256(raw)
        for layer, path in sorted(paths.items()):
            if cancelled():
                raise InterruptedError("experiment scan cancelled")
            checksum.update(layer.encode())
            info = manifest["layers"][layer]
            remaining = info["committed_bytes"]
            layer_checksum = hashlib.sha256()
            # Raw reads do not prefetch an uncommitted tail into an I/O buffer.
            with path.open("rb", buffering=0) as stream:
                while remaining:
                    if cancelled():
                        raise InterruptedError("experiment scan cancelled")
                    payload = stream.read(min(65536, remaining))
                    if not payload:
                        raise ResultValidationError(f"missing or modified {layer} committed prefix")
                    remaining -= len(payload)
                    checksum.update(payload)
                    layer_checksum.update(payload)
            if layer_checksum.hexdigest() != info["sha256"]:
                raise ResultValidationError(f"missing or modified {layer} committed prefix")
        if cancelled():
            raise InterruptedError("experiment scan cancelled")
        return checksum.hexdigest()


def read_snapshot(root):
    layout = ArtifactLayout.discover(root)
    with publication_lock(layout, exclusive=False):
        manifest_bytes = layout.result_csv.read_bytes()
        manifest = loads_finite_json(manifest_bytes.decode("utf-8"))
        pending = layout.result_store_path("pending.json")
        if pending.exists():
            transaction = read_json_object(pending, bounded=False)
            if transaction.get("replacements") and manifest["commit"] == transaction["before"]["commit"]:
                raise ResultValidationError("result module replacement requires recovery")
        payloads = read_layer_bytes(layout, manifest)
        identity = layer_identity_fields(manifest["wide_fields"])
        combined, expected = {}, []
        for layer in LAYER_FILES:
            info = manifest["layers"][layer]
            rows = parse_layer(layout.layer_csv(layer), info["fields"], payloads[layer])
            if len(rows) != info["row_count"]:
                raise ResultValidationError(f"invalid {layer} row count")
            keys = [measurement_key(row) for row in rows]
            if layer == "summary":
                expected = keys
                combined = dict(zip(keys, rows, strict=True))
            elif layer in BASE_LAYERS and keys != expected:
                raise ResultValidationError(f"measurement order mismatch in {layer}")
            for key, row in zip(keys, rows, strict=True):
                if key not in combined:
                    raise ResultValidationError(f"orphan profiler row in {layer}")
                if any(row[name] != combined[key][name] for name in identity):
                    raise ResultValidationError(f"measurement identity mismatch in {layer}")
                combined[key].update({name: value for name, value in row.items() if name in info["fields"]})
        fields = manifest["wide_fields"]
        rows = [{field: combined[key].get(field, "nan") for field in fields} for key in expected]
        return ResultCsvSnapshot(layout.result_csv, fields, rows, digest(manifest_bytes))
