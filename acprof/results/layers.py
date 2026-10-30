"""Authoritative, joinable experiment CSV layers. Wide CSV is an explicit export."""
from __future__ import annotations

import csv
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from acprof.artifacts import (
    atomic_write,
    atomic_write_json,
    file_sha256,
    read_json_object,
    sync_directory,
)
from acprof.cancellation import cancellable, check_cancelled
from acprof.results.metrics import METRICS, order_csv_fields
from acprof.results.csv import (
    KEY_FIELDS,
    ResultValidationError,
    measurement_key,
    read_result_csv_snapshot,
    result_csv_snapshot_unchanged,
)


def _write_csv_text(stream: TextIO, raw: bytes) -> None:
    stream.write(raw.decode("utf-8"))


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset((1, SCHEMA_VERSION))
MANIFEST_NAME = "result_layers.json"
GENERATIONS_DIR = "result_generations"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
LAYER_FILES = {
    "summary": "summary.csv",
    "performance": "performance.csv",
    "resources": "resources.csv",
    "energy": "energy.csv",
    "network": "network.csv",
    "torch": "profiling/torch_profiler.csv",
    "ncu": "profiling/ncu.csv",
    "nsys": "profiling/nsys.csv",
    "massif": "profiling/massif.csv",
}
PROFILER_LAYERS = frozenset(("torch", "ncu", "nsys", "massif"))
BASE_LAYERS = ("summary", "performance", "resources", "energy", "network")


def field_layer(name: str) -> str:
    """Classify fields by their declared measurement source, not UI labels."""
    if name in KEY_FIELDS:
        return "summary"
    metric = METRICS.get(name)
    if metric is None:
        return "summary"  # Keep unknown historical extension fields without loss.
    if metric.tool:
        return metric.tool
    if name.startswith("packet_"):
        return "network"
    if "energy" in name or "power" in name or metric.source.startswith("rapl"):
        return "energy"
    if metric.source in {"cgroup", "perf_stat", "host_cpufreq", "nvml_selected_device",
                         "psutil", "psutil_observed_process_tree"}:
        return "resources"
    if (name.startswith(("latency", "throughput", "cold_start"))
            or name == "process_startup_s" or metric.source in {"pcap", "docker_and_server"}):
        return "performance"
    return "summary"


def layer_identity_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Inherited resource axes require execution context in every projection.

    Historical files without runtime_kind keep their original layer headers.
    """
    return (*KEY_FIELDS, *(name for name in ("runtime_kind", "run_id", "case_id", "case_attempt_id", "window_id") if name in fields))


def layer_fields(fields: Sequence[str]) -> dict[str, list[str]]:
    """Exactly one owner for every non-key column, including unknown columns."""
    if (any(not isinstance(name, str) or not name for name in fields)
            or len(fields) != len(set(fields)) or not set(KEY_FIELDS) <= set(fields)):
        raise ResultValidationError("invalid layered result header or measurement keys")
    identity = layer_identity_fields(fields)
    mapping = {name: list(identity) for name in LAYER_FILES}
    for name in fields:
        if name not in identity:
            mapping[field_layer(name)].append(name)
    return mapping


def has_layer_evidence(value: object) -> bool:
    return str(value).strip().lower() not in {"", "nan", "none", "null", "compute_profile_disabled"}


def _write_csv_rows(stream: TextIO, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    writer = csv.DictWriter(stream, fieldnames=fields, restval="nan", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    import io
    stream = io.StringIO(newline="")
    _write_csv_rows(stream, fields, rows)
    return stream.getvalue().encode("utf-8")


def _result_root(root: str | Path) -> Path:
    root = Path(root).absolute()
    if root.is_symlink():
        raise ResultValidationError(f"result directory must not be a symlink: {root}")
    return root


def _safe_path(root: Path, relative: str) -> Path:
    """Reject symlinks in every component, including in-root directory aliases."""
    parts = relative.split("/")
    if (not relative or any(part in {"", ".", ".."} for part in parts)
            or "\\" in relative or Path(relative).is_absolute()):
        raise ResultValidationError(f"unsafe result layer path: {relative!r}")
    path = root
    for part in parts:
        path /= part
        if path.is_symlink():
            raise ResultValidationError(f"result path must not be a symlink: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ResultValidationError(f"result layer escapes its directory: {path}")
    return path


def _generation_id(fields: Sequence[str], row_count: int, records: Mapping[str, Mapping]) -> str:
    identity = {
        "wide_fields": list(fields), "row_count": row_count,
        "layers": {
            layer: {key: record[key] for key in ("sha256", "row_count", "fields")}
            for layer, record in records.items()
        },
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def result_layer_paths(root: str | Path, manifest: Mapping) -> dict[str, Path]:
    """Validate a v1/v2 manifest and resolve only its declared, safe layer paths.

    Consumers that stream/hash CSVs themselves use this same path contract.
    The returned paths have not yet been checked against their content digests.
    """
    root = _result_root(root)
    if manifest.get("schema_version") == 3:
        from acprof.results.layout import ArtifactLayout
        from acprof.result_store.protocol import validate_manifest
        return validate_manifest(ArtifactLayout.discover(root), dict(manifest))
    version = manifest.get("schema_version")
    if (type(version) is not int or version not in SUPPORTED_SCHEMA_VERSIONS
            or not isinstance(manifest.get("layers"), dict)
            or not isinstance(manifest.get("wide_fields"), list)
            or type(manifest.get("row_count")) is not int or manifest["row_count"] < 1):
        raise ResultValidationError("unsupported result layers manifest")
    assigned = layer_fields(manifest["wide_fields"])
    layers = manifest["layers"]
    if not set(BASE_LAYERS) <= set(layers) <= set(LAYER_FILES):
        raise ResultValidationError("missing or unknown result layer")
    generation = manifest.get("generation")
    if version == SCHEMA_VERSION and (
        not isinstance(generation, str) or _SHA256.fullmatch(generation) is None
    ):
        raise ResultValidationError("invalid result generation identity")
    paths = {}
    for layer, record in layers.items():
        relative = LAYER_FILES[layer]
        if version == SCHEMA_VERSION:
            relative = f"{GENERATIONS_DIR}/{generation}/{relative}"
        if (not isinstance(record, dict) or record.get("path") != relative
                or record.get("fields") != assigned[layer]
                or type(record.get("row_count")) is not int or record["row_count"] < 1
                or not isinstance(record.get("sha256"), str)
                or _SHA256.fullmatch(record["sha256"]) is None):
            raise ResultValidationError(f"invalid {layer} layer schema")
        paths[layer] = _safe_path(root, relative)
    if version == SCHEMA_VERSION and generation != _generation_id(
        manifest["wide_fields"], manifest["row_count"], layers
    ):
        raise ResultValidationError("result generation identity does not match its layers")
    return paths


def _verify_generation(root: Path, generation: str, payloads: Mapping[str, bytes]) -> None:
    for layer, raw in payloads.items():
        path = _safe_path(root, f"{GENERATIONS_DIR}/{generation}/{LAYER_FILES[layer]}")
        if not path.is_file() or path.read_bytes() != raw:
            raise ResultValidationError(f"missing or modified immutable result layer: {path}")


def _prepare_generation(root: Path, generation: str, payloads: Mapping[str, bytes]) -> None:
    """Install one complete immutable directory; a racing identical writer reuses it."""
    generations = _safe_path(root, GENERATIONS_DIR)
    generations.mkdir(parents=True, exist_ok=True)
    sync_directory(generations.parent)
    destination = _safe_path(root, f"{GENERATIONS_DIR}/{generation}")
    if destination.exists():
        _verify_generation(root, generation, payloads)
        return
    stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=generations))
    try:
        for layer, raw in payloads.items():
            path = stage / LAYER_FILES[layer]
            path.parent.mkdir(parents=True, exist_ok=True)
            def write_layer(stream: TextIO, value: bytes = raw) -> None:
                _write_csv_text(stream, value)
            atomic_write(path, write_layer)
        # atomic_write syncs each file's parent; also persist newly created
        # profiler subdirectories before publishing the whole generation.
        sync_directory(stage)
        try:
            os.rename(stage, destination)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            # Another publisher may have installed the same content identity.
            # Never repair or replace an existing, damaged generation.
            _verify_generation(root, generation, payloads)
        sync_directory(generations)
    finally:
        # A real process crash may leave an unreferenced .pending directory.
        # Only remove this invocation's staging area, never published versions.
        if stage.exists():
            shutil.rmtree(stage)


def publish_result_layers(wide_path: str | Path, *, output_dir: str | Path | None = None) -> dict:
    """Import an explicitly supplied CSV into authoritative result layers.

    Normal collection publishes layers directly from validated rows; it never
    persists a canonical wide result.
    """
    wide = Path(wide_path)
    root = Path(output_dir) if output_dir is not None else wide.parent
    snapshot = read_result_csv_snapshot(wide)
    if not result_csv_snapshot_unchanged(snapshot):
        raise ResultValidationError("source CSV changed while partitioning")
    return publish_result_rows(snapshot.fields, snapshot.rows, root, source=snapshot)


def prepare_result_snapshot(fields: Sequence[str], rows: Sequence[Mapping[str, str]], root: str | Path,
                        *, source=None) -> dict:
    """Create a verified immutable generation without switching the live manifest."""
    root = _result_root(root)
    _safe_path(root, MANIFEST_NAME)
    if not rows:
        raise ResultValidationError("cannot partition empty result")
    assigned = layer_fields(fields)
    identity = layer_identity_fields(fields)
    # The semantic key is unique even when a profiler has no evidence for a row.
    keys = [measurement_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ResultValidationError("duplicate measurement keys in result layers")
    rows_by_layer: dict[str, list[dict[str, str]]] = {}
    payloads: dict[str, bytes] = {}
    for layer, names in assigned.items():
        selected = [
            {name: row.get(name, "nan") for name in names}
            for row in rows
            if layer not in PROFILER_LAYERS
            or any(has_layer_evidence(row.get(name, "")) for name in names if name not in identity)
        ]
        if layer in PROFILER_LAYERS and not selected:
            continue
        rows_by_layer[layer] = selected
        payloads[layer] = _csv_bytes(names, selected)
    records: dict[str, dict] = {}
    for layer, raw in payloads.items():
        records[layer] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "row_count": len(rows_by_layer[layer]), "fields": assigned[layer],
        }
    generation = _generation_id(fields, len(rows), records)
    for layer, record in records.items():
        record["path"] = f"{GENERATIONS_DIR}/{generation}/{LAYER_FILES[layer]}"
    manifest = {
        "schema_version": SCHEMA_VERSION, "generation": generation, "row_count": len(rows),
        "wide_fields": list(fields), "layers": records,
    }
    _prepare_generation(root, generation, payloads)
    if source is not None:
        if not result_csv_snapshot_unchanged(source):
            raise ResultValidationError("source CSV changed before layer manifest publication")
        manifest.update(source_name=source.path.name, source_sha256=source.sha256)
    # Check the manifest path again after staging; existing generations are
    # retained even when this last publication fails or a later writer wins.
    _safe_path(root, MANIFEST_NAME)
    return manifest


def publish_result_rows(fields: Sequence[str], rows: Sequence[Mapping[str, str]], root: str | Path,
                        *, source=None) -> dict:
    """Explicit immutable snapshot import; live acquisition uses ResultWriter."""
    path = _safe_path(_result_root(root), MANIFEST_NAME)
    def publish():
        manifest = prepare_result_snapshot(fields, rows, root, source=source)
        if not path.is_file() or read_json_object(path) != manifest:
            atomic_write_json(path, manifest)
        return manifest

    if path.is_file() and read_json_object(path).get("schema_version") == 3:
        from acprof.result_store.protocol import publication_lock
        from acprof.result_store.writer import ResultWriter
        with ResultWriter(path.parent) as writer:
            if writer.manifest["row_count"]:
                raise ResultValidationError("cannot replace incremental measurements with a snapshot import")
            with publication_lock(writer.layout, exclusive=True):
                return publish()
    return publish()


def consume_layer_rows(fields: Sequence[str], keys: Sequence[tuple[str, ...]],
                       combined: dict[tuple[str, ...], dict[str, str]], *,
                       cancelled: Callable[[], bool] | None = None) -> list[dict[str, str]]:
    """Return canonical rows while releasing this read's owned join mapping."""
    rows = []
    for key in cancellable(keys, cancelled):
        row = combined.pop(key)
        rows.append({field: row.get(field, "nan") for field in fields})
    return rows


def _validate_imported_source(root: Path, manifest: Mapping, *, cancelled: Callable[[], bool] | None = None) -> None:
    source_name = manifest.get("source_name")
    if source_name is not None:
        source_digest = manifest.get("source_sha256")
        if (not isinstance(source_name, str) or source_name in {"", ".", ".."}
                or "\\" in source_name or Path(source_name).name != source_name
                or not isinstance(source_digest, str) or _SHA256.fullmatch(source_digest) is None):
            raise ResultValidationError("invalid layered source identity")
        # In v2 the imported CSV is provenance, not a mutable dependency of an
        # immutable snapshot. Keep v1's existing stale-source check unchanged.
        if manifest["schema_version"] == 1:
            source_path = _safe_path(root, source_name)
            if source_path.exists() and file_sha256(source_path, cancelled=cancelled) != source_digest:
                raise ResultValidationError("source CSV has changed since layers were published")


def verify_result_layer_files(root: str | Path, manifest: Mapping, *,
                              cancelled: Callable[[], bool] | None = None) -> None:
    """Recheck v1/v2 snapshot content without parsing previously validated rows."""
    check_cancelled(cancelled)
    root = _result_root(root)
    version = manifest.get("schema_version")
    if type(version) is not int or version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ResultValidationError("content verification requires a v1/v2 snapshot")
    paths = result_layer_paths(root, manifest)
    _validate_imported_source(root, manifest, cancelled=cancelled)
    for layer, path in cancellable(paths.items(), cancelled):
        if not path.is_file() or file_sha256(path, cancelled=cancelled) != manifest["layers"][layer]["sha256"]:
            raise ResultValidationError(f"missing or modified {layer} layer: {path}")


def read_result_layers(root: str | Path, *, expected_manifest_sha256: str | None = None,
                       cancelled: Callable[[], bool] | None = None,
                       ) -> tuple[list[str], list[dict[str, str]]]:
    """Read hashed partitions, optionally bound to an outer consumer's manifest."""
    check_cancelled(cancelled)
    root = _result_root(root)
    manifest = read_json_object(_safe_path(root, MANIFEST_NAME), label="result layers manifest",
                                expected_sha256=expected_manifest_sha256)
    if manifest.get("schema_version") == 3:
        from acprof.result_store.reader import read_snapshot
        snapshot = read_snapshot(root, cancelled=cancelled)
        if expected_manifest_sha256 is not None and snapshot.sha256 != expected_manifest_sha256:
            raise ResultValidationError("layered results changed during read")
        return snapshot.fields, snapshot.rows
    paths = result_layer_paths(root, manifest)
    wide_fields = manifest["wide_fields"]
    _validate_imported_source(root, manifest, cancelled=cancelled)
    assigned = layer_fields(wide_fields)
    identity = layer_identity_fields(wide_fields)
    layers = manifest["layers"]
    expected_keys: list[tuple[str, ...]] | None = None
    combined: dict[tuple[str, ...], dict[str, str]] = {}
    for layer in cancellable(LAYER_FILES, cancelled):
        if layer not in layers:
            continue
        record = layers[layer]
        path = paths[layer]
        if not path.is_file():
            raise ResultValidationError(f"missing or modified {layer} layer: {path}")
        if layer in PROFILER_LAYERS and not set(assigned[layer]) - set(identity):
            raise ResultValidationError(f"empty profiler schema: {layer}")
        try:
            snapshot = read_result_csv_snapshot(path, cancelled=cancelled)
        except (ValueError, csv.Error) as exc:
            raise ResultValidationError(f"missing or modified {layer} layer: {path}: {exc}") from exc
        if snapshot.sha256 != record["sha256"]:
            raise ResultValidationError(f"missing or modified {layer} layer: {path}")
        columns, rows = snapshot.fields, snapshot.rows
        if columns != assigned[layer] or len(rows) != record.get("row_count"):
            raise ResultValidationError(f"invalid {layer} layer rows or columns")
        keys = [measurement_key(row) for row in cancellable(rows, cancelled)]
        if layer == "summary":
            expected_keys = keys
            if len(keys) != manifest["row_count"]:
                raise ResultValidationError("summary row count mismatch")
            combined = dict(cancellable(zip(keys, rows, strict=True), cancelled))
        elif expected_keys is None:
            raise ResultValidationError("summary layer must be first")
        elif layer not in PROFILER_LAYERS and keys != expected_keys:
            raise ResultValidationError(f"measurement order mismatch in {layer}")
        if layer != "summary":
            value_fields = [name for name in assigned[layer] if name not in identity]
            for key, row in cancellable(zip(keys, rows, strict=True), cancelled):
                if key not in combined:
                    raise ResultValidationError(f"orphan profiler measurement in {layer}: {key}")
                if "runtime_kind" in identity and row["runtime_kind"] != combined[key]["runtime_kind"]:
                    raise ResultValidationError(f"runtime identity mismatch in {layer}")
                combined[key].update({name: row[name] for name in value_fields})
        del snapshot, columns, rows, keys
    if expected_keys is None:
        raise ResultValidationError("missing summary measurements")
    ordered = order_csv_fields(wide_fields)
    return ordered, consume_layer_rows(ordered, expected_keys, combined, cancelled=cancelled)


def export_result_layers(root: str | Path, destination: str | Path) -> int:
    """Explicit on-demand wide export; never write over a layered source file."""
    root, destination = _result_root(root), Path(destination)
    resolved_destination = destination.resolve()
    if (resolved_destination in {(root / name).resolve() for name in LAYER_FILES.values()}
            or resolved_destination.is_relative_to((root / GENERATIONS_DIR).resolve())):
        raise ResultValidationError("cannot overwrite a result layer with a wide export")
    if destination.exists() or destination.is_symlink():
        raise ResultValidationError(f"refusing to overwrite existing CSV: {destination}")
    fields, rows = read_result_layers(root)
    try:
        atomic_write(destination, lambda stream: _write_csv_rows(stream, fields, rows), overwrite=False)
    except FileExistsError as exc:
        raise ResultValidationError(f"refusing to overwrite existing CSV: {destination}") from exc
    return len(rows)
