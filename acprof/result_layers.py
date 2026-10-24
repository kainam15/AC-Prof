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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from acprof.artifacts import atomic_write, atomic_write_json, read_json_object, sync_directory
from acprof.metric_registry import METRICS, order_csv_fields
from acprof.result_csv import (
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
    if metric.source in {"cgroup", "perf_stat", "host_cpufreq", "nvml_selected_device"}:
        return "resources"
    if (name.startswith(("latency", "throughput", "cold_start"))
            or metric.source in {"pcap", "docker_and_server"}):
        return "performance"
    return "summary"


def layer_fields(fields: Sequence[str]) -> dict[str, list[str]]:
    """Exactly one owner for every non-key column, including unknown columns."""
    if (any(not isinstance(name, str) or not name for name in fields)
            or len(fields) != len(set(fields)) or not set(KEY_FIELDS) <= set(fields)):
        raise ResultValidationError("invalid layered result header or measurement keys")
    mapping = {name: list(KEY_FIELDS) for name in LAYER_FILES}
    for name in fields:
        if name not in KEY_FIELDS:
            mapping[field_layer(name)].append(name)
    return mapping


def _has_evidence(value: object) -> bool:
    return str(value).strip().lower() not in {"", "nan", "none", "null", "compute_profile_disabled"}


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, restval="nan", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
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


def publish_result_rows(fields: Sequence[str], rows: Sequence[Mapping[str, str]], root: str | Path,
                        *, source=None) -> dict:
    """Publish an immutable generation, then atomically switch its manifest.

    Concurrent complete publications use last-manifest-replacement-wins semantics;
    no layer belonging to an earlier reader is overwritten. Read-modify-write
    operations such as Posthoc still require the experiment directory lock.
    """
    root = _result_root(root)
    manifest_path = _safe_path(root, MANIFEST_NAME)
    if not rows:
        raise ResultValidationError("cannot partition empty result")
    assigned = layer_fields(fields)
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
            or any(_has_evidence(row.get(name, "")) for name in names if name not in KEY_FIELDS)
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
    if manifest_path.is_file() and read_json_object(manifest_path) == manifest:
        return manifest
    atomic_write_json(manifest_path, manifest)
    return manifest


def read_result_layers(root: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a full logical result from independently hashed partition CSVs."""
    root = _result_root(root)
    manifest = read_json_object(_safe_path(root, MANIFEST_NAME), label="result layers manifest")
    paths = result_layer_paths(root, manifest)
    wide_fields = manifest["wide_fields"]
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
            if (source_path.exists() and
                    hashlib.sha256(source_path.read_bytes()).hexdigest() != source_digest):
                raise ResultValidationError("source CSV has changed since layers were published")
    assigned = layer_fields(wide_fields)
    layers = manifest["layers"]
    expected_keys: list[tuple[str, ...]] | None = None
    combined: dict[tuple[str, ...], dict[str, str]] = {}
    for layer in LAYER_FILES:
        if layer not in layers:
            continue
        record = layers[layer]
        path = paths[layer]
        if not path.is_file():
            raise ResultValidationError(f"missing or modified {layer} layer: {path}")
        if layer in PROFILER_LAYERS and not set(assigned[layer]) - set(KEY_FIELDS):
            raise ResultValidationError(f"empty profiler schema: {layer}")
        try:
            snapshot = read_result_csv_snapshot(path)
        except (ValueError, csv.Error) as exc:
            raise ResultValidationError(f"missing or modified {layer} layer: {path}: {exc}") from exc
        if snapshot.sha256 != record["sha256"]:
            raise ResultValidationError(f"missing or modified {layer} layer: {path}")
        columns, rows = snapshot.fields, snapshot.rows
        if columns != assigned[layer] or len(rows) != record.get("row_count"):
            raise ResultValidationError(f"invalid {layer} layer rows or columns")
        keys = [measurement_key(row) for row in rows]
        if layer == "summary":
            expected_keys = keys
            if len(keys) != manifest["row_count"]:
                raise ResultValidationError("summary row count mismatch")
            combined = {key: dict(row) for key, row in zip(keys, rows, strict=True)}
        elif expected_keys is None:
            raise ResultValidationError("summary layer must be first")
        elif layer not in PROFILER_LAYERS and keys != expected_keys:
            raise ResultValidationError(f"measurement order mismatch in {layer}")
        for key, row in zip(keys, rows, strict=True):
            if key not in combined:
                raise ResultValidationError(f"orphan profiler measurement in {layer}: {key}")
            if layer != "summary":
                combined[key].update({name: row[name] for name in assigned[layer] if name not in KEY_FIELDS})
    if expected_keys is None:
        raise ResultValidationError("missing summary measurements")
    ordered = order_csv_fields(wide_fields)
    return ordered, [
        {field: combined[key].get(field, "nan") for field in ordered}
        for key in expected_keys
    ]


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
    raw = _csv_bytes(fields, rows)
    atomic_write(destination, lambda stream: _write_csv_text(stream, raw))
    return len(rows)
