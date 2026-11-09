"""Idempotent creation of zero-row experiment projections."""
from __future__ import annotations

from typing import TypeGuard
from uuid import uuid4

from acprof.artifacts import (
    atomic_write_json,
    create_durable_directory,
    read_json_object,
    sync_directory,
)
from acprof.config import CSV_FIELDS
from acprof.result_store.protocol import (
    SCHEMA_VERSION,
    csv_bytes,
    digest,
    json_bytes,
    publication_lock,
    save_commit,
    validate_layer_files,
    validate_manifest_structure,
    validate_run_identity,
    write_bytes,
)
from acprof.result_store.types import (
    InitializationReceipt,
    LayerManifest,
    ResultManifest,
    ResultManifestBody,
)
from acprof.results.csv import ResultValidationError
from acprof.results.layers import LAYER_FILES, layer_fields
from acprof.results.layout import ArtifactLayout


def _validate_initialization_receipt(schema: object) -> TypeGuard[InitializationReceipt]:
    """Prove the stored identity and fields without discarding extension keys."""
    if not isinstance(schema, dict):
        raise ResultValidationError("invalid result initialization receipt: expected an object")
    if any(not isinstance(name, str) for name in schema):
        raise ResultValidationError("invalid result initialization receipt: field names must be strings")
    fields = schema.get("fields")
    if (not isinstance(fields, list) or any(not isinstance(name, str) for name in fields)
            or fields != list(CSV_FIELDS)):
        raise ResultValidationError("invalid result initialization receipt: fields changed")
    if not isinstance(schema.get("run_id"), str) or not schema["run_id"]:
        raise ResultValidationError("invalid result initialization receipt: run_id must be a non-empty string")
    return True


def _read_initialization_receipt(layout: ArtifactLayout) -> InitializationReceipt | None:
    """Validate a retained identity before any initialization publication."""
    intent = layout.result_store_path("initialization.json")
    if not intent.exists():
        return None
    schema = read_json_object(intent)
    if not _validate_initialization_receipt(schema):
        raise ResultValidationError("invalid result initialization receipt")
    return schema


def initial_files(layout: ArtifactLayout) -> set[str]:
    """Only a verified empty store can be reused during preparation retries."""
    schema = _read_initialization_receipt(layout)
    if schema is None:
        return set()
    if layout.result_csv.exists():
        manifest = read_json_object(layout.result_csv)
        validate_run_identity(manifest, schema["run_id"])
        if manifest.get("sequence") != 0 or manifest.get("state") != "initialized":
            return set()
        validate_layer_files(layout, manifest, exact=True)
    else:
        for layer, fields in layer_fields(CSV_FIELDS).items():
            path = layout.layer_csv(layer)
            if path.exists() and path.read_bytes() != csv_bytes(fields, header=True):
                raise ResultValidationError("nonempty result encountered during initialization")
    paths = {"result_layers.json", *LAYER_FILES.values()}
    paths.update(path.relative_to(layout.root).as_posix() for path in layout.result_store_path("").rglob("*") if path.is_file())
    return paths


def initialize_results(layout: ArtifactLayout) -> ResultManifest:
    # Fail without even creating a publication lock for an invalid retained intent.
    _read_initialization_receipt(layout)
    with publication_lock(layout, exclusive=True):
        # Re-read under the lock; an earlier preflight is not publication authority.
        schema = _read_initialization_receipt(layout)
        if layout.result_csv.exists():
            existing = read_json_object(layout.result_csv)
            validate_layer_files(layout, existing, exact=True)
            if schema is not None:
                validate_run_identity(existing, schema["run_id"])
            if not validate_manifest_structure(existing, layout):
                raise ResultValidationError("invalid incremental result manifest")
            return existing
        intent = layout.result_store_path("initialization.json")
        if schema is None:
            schema = {"run_id": uuid4().hex, "fields": list(CSV_FIELDS)}
            atomic_write_json(intent, schema)
        assigned = layer_fields(schema["fields"])
        records: dict[str, LayerManifest] = {}
        for directory in ("metadata", "raw", "plots", "logs", "profiling", "debug"):
            create_durable_directory(layout.path(directory))
        for layer, fields in assigned.items():
            raw = csv_bytes(fields, header=True)
            path = layout.layer_csv(layer)
            if path.exists() and path.read_bytes() != raw:
                raise ResultValidationError(f"refusing to replace existing {layer} results")
            write_bytes(path, raw)
            records[layer] = {"path": LAYER_FILES[layer], "fields": fields, "row_count": 0,
                              "committed_bytes": len(raw), "sha256": digest(raw), "revision": 0}
        after: ResultManifestBody = {"schema_version": SCHEMA_VERSION, "run_id": schema["run_id"],
                 "state": "initialized", "sequence": 0, "wide_fields": schema["fields"],
                 "schema_sha256": digest(json_bytes(assigned)), "row_count": 0, "layers": records}
        manifest = save_commit(layout, None, after, [], transaction="initial")
        atomic_write_json(layout.result_csv, manifest)
        sync_directory(layout.root / ".acprof")
        sync_directory(layout.root)
        return manifest
