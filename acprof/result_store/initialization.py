"""Idempotent creation of zero-row experiment projections."""
from __future__ import annotations

from uuid import uuid4

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import (
    atomic_write_json,
    create_durable_directory,
    read_json_object,
    sync_directory,
)
from acprof.config import CSV_FIELDS
from acprof.result_csv import ResultValidationError
from acprof.result_layers import LAYER_FILES, layer_fields
from acprof.result_store.protocol import (
    SCHEMA_VERSION,
    csv_bytes,
    digest,
    json_bytes,
    publication_lock,
    save_commit,
    validate_layer_files,
    write_bytes,
)


def initial_files(layout: ArtifactLayout) -> set[str]:
    """Only a verified empty store can be reused during preparation retries."""
    intent = layout.result_store_path("initialization.json")
    if not intent.exists():
        return set()
    schema = read_json_object(intent)
    if schema.get("fields") != list(CSV_FIELDS) or not schema.get("run_id"):
        raise ResultValidationError("invalid result initialization receipt")
    if layout.result_csv.exists():
        manifest = read_json_object(layout.result_csv)
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


def initialize_results(layout: ArtifactLayout) -> dict:
    with publication_lock(layout, exclusive=True):
        if layout.result_csv.exists():
            manifest = read_json_object(layout.result_csv)
            validate_layer_files(layout, manifest, exact=True)
            return manifest
        intent = layout.result_store_path("initialization.json")
        if intent.exists():
            schema = read_json_object(intent)
            if schema.get("fields") != list(CSV_FIELDS):
                raise ResultValidationError("initial result schema changed")
        else:
            schema = {"run_id": uuid4().hex, "fields": list(CSV_FIELDS)}
            atomic_write_json(intent, schema)
        assigned = layer_fields(schema["fields"])
        records = {}
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
        after = {"schema_version": SCHEMA_VERSION, "run_id": schema["run_id"],
                 "state": "initialized", "sequence": 0, "wide_fields": schema["fields"],
                 "schema_sha256": digest(json_bytes(assigned)), "row_count": 0, "layers": records}
        manifest = save_commit(layout, None, after, [], transaction="initial")
        atomic_write_json(layout.result_csv, manifest)
        sync_directory(layout.root / ".acprof")
        sync_directory(layout.root)
        return manifest
