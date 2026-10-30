"""Read one committed prefix from all root CSVs, never their uncommitted tails."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import CancelledError

from acprof.artifacts import read_json_object, read_json_object_snapshot
from acprof.cancellation import cancellable, check_cancelled
from acprof.result_store.protocol import (
    digest,
    publication_lock,
    read_layer_rows,
    validate_manifest,
)
from acprof.results.csv import MeasurementKey, ResultCsvSnapshot, ResultValidationError
from acprof.results.layers import (
    BASE_LAYERS,
    LAYER_FILES,
    consume_layer_rows,
    layer_identity_fields,
)
from acprof.results.layout import ArtifactLayout


def committed_content_digest(root, *, cancelled, budget):
    """Bounded, nonblocking verification for duplicate entries in experiment lists."""
    if cancelled():
        raise InterruptedError("experiment scan cancelled")
    if budget[0] <= 0:
        raise ValueError("duplicate_evidence_unverified: CSV byte budget exceeded")
    layout = ArtifactLayout.discover(root)
    with publication_lock(layout, exclusive=False, blocking=False):
        try:
            snapshot = read_json_object_snapshot(layout.result_csv, cancelled=cancelled, byte_budget=budget)
            manifest = snapshot.payload
            paths = validate_manifest(layout, manifest, cancelled=cancelled, byte_budget=budget)
        except CancelledError as exc:
            raise InterruptedError("experiment scan cancelled") from exc
        size = sum(info["committed_bytes"] for info in manifest["layers"].values())
        budget[0] -= size
        if budget[0] < 0:
            raise ValueError("duplicate_evidence_unverified: CSV byte budget exceeded")
        checksum = hashlib.sha256(snapshot.content)
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


def read_snapshot(root, *, cancelled: Callable[[], bool] | None = None):
    check_cancelled(cancelled)
    layout = ArtifactLayout.discover(root)
    with publication_lock(layout, exclusive=False, cancelled=cancelled):
        snapshot = read_json_object_snapshot(layout.result_csv, cancelled=cancelled)
        manifest = snapshot.payload
        pending = layout.result_store_path("pending.json")
        if pending.exists():
            transaction = read_json_object(pending, bounded=False, cancelled=cancelled)
            if transaction.get("replacements") and manifest["commit"] == transaction["before"]["commit"]:
                raise ResultValidationError("result module replacement requires recovery")
        paths = validate_manifest(layout, manifest, cancelled=cancelled)
        identity = layer_identity_fields(manifest["wide_fields"])
        combined = {}
        expected: tuple[MeasurementKey, ...] = ()
        for layer in cancellable(LAYER_FILES, cancelled):
            info = manifest["layers"][layer]
            validated = read_layer_rows(paths[layer], info, cancelled=cancelled)
            rows, keys = validated.rows, validated.keys
            assert keys is not None  # Layer parsing always validates measurements.
            if len(rows) != info["row_count"]:
                raise ResultValidationError(f"invalid {layer} row count")
            if layer == "summary":
                expected = keys
                combined = dict(cancellable(zip(keys, rows, strict=True), cancelled))
            elif layer in BASE_LAYERS and keys != expected:
                raise ResultValidationError(f"measurement order mismatch in {layer}")
            if layer != "summary":
                value_fields = [name for name in info["fields"] if name not in identity]
                for key, row in cancellable(zip(keys, rows, strict=True), cancelled):
                    if key not in combined:
                        raise ResultValidationError(f"orphan profiler row in {layer}")
                    if any(row[name] != combined[key][name] for name in identity):
                        raise ResultValidationError(f"measurement identity mismatch in {layer}")
                    combined[key].update({name: row[name] for name in value_fields})
            del validated, rows, keys
        fields = manifest["wide_fields"]
        rows = consume_layer_rows(fields, expected, combined, cancelled=cancelled)
        check_cancelled(cancelled)
        return ResultCsvSnapshot(layout.result_csv, fields, rows, digest(snapshot.content))
