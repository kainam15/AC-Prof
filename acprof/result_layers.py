"""Authoritative, joinable experiment CSV layers. Wide CSV is an explicit export."""
from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from acprof.artifacts import atomic_write, atomic_write_json, read_json_object
from acprof.metric_registry import METRICS, order_csv_fields
from acprof.result_csv import (
    KEY_FIELDS,
    ResultValidationError,
    measurement_key,
    read_result_csv,
    read_result_csv_snapshot,
    result_csv_snapshot_unchanged,
)

SCHEMA_VERSION = 1
MANIFEST_NAME = "result_layers.json"
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
    if len(fields) != len(set(fields)) or not set(KEY_FIELDS) <= set(fields):
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
    """Publish authoritative layers, committing the integrity manifest last."""
    root = Path(root)
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
        relative = LAYER_FILES[layer]
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise ResultValidationError(f"result layer must not be a symlink: {destination}")
        if not destination.exists() or destination.read_bytes() != raw:
            atomic_write(destination, lambda stream, raw=raw: stream.write(raw.decode("utf-8")))
        records[layer] = {
            "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
            "row_count": len(rows_by_layer[layer]), "fields": assigned[layer],
        }
    # Remove an obsolete profiler output only once the new manifest is live.
    # Orphans are ignored by readers; never delete another run's data here.
    manifest = {
        "schema_version": SCHEMA_VERSION, "row_count": len(rows),
        "wide_fields": list(fields), "layers": records,
    }
    if source is not None:
        if not result_csv_snapshot_unchanged(source):
            raise ResultValidationError("source CSV changed before layer manifest publication")
        manifest.update(source_name=source.path.name, source_sha256=source.sha256)
    atomic_write_json(root / MANIFEST_NAME, manifest)
    return manifest


def read_result_layers(root: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a full logical result from independently hashed partition CSVs."""
    root = Path(root)
    manifest = read_json_object(root / MANIFEST_NAME, label="result layers manifest")
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or not isinstance(manifest.get("layers"), dict)
            or not isinstance(manifest.get("wide_fields"), list)
            or not isinstance(manifest.get("row_count"), int)):
        raise ResultValidationError("unsupported result layers manifest")
    wide_fields = manifest["wide_fields"]
    source_name = manifest.get("source_name")
    if source_name is not None:
        source_digest = manifest.get("source_sha256")
        if (not isinstance(source_name, str) or Path(source_name).name != source_name
                or not isinstance(source_digest, str) or len(source_digest) != 64):
            raise ResultValidationError("invalid layered source identity")
        source_path = root / source_name
        if (source_path.exists() and
                hashlib.sha256(source_path.read_bytes()).hexdigest() != source_digest):
            raise ResultValidationError("source CSV has changed since layers were published")
    assigned = layer_fields(wide_fields)
    expected_layers = set(BASE_LAYERS)
    layers = manifest["layers"]
    if not expected_layers <= set(layers) or set(layers) - set(LAYER_FILES):
        raise ResultValidationError("missing or unknown result layer")
    expected_keys: list[tuple[str, ...]] | None = None
    combined: dict[tuple[str, ...], dict[str, str]] = {}
    for layer in LAYER_FILES:
        if layer not in layers:
            continue
        record = layers[layer]
        if (not isinstance(record, dict) or record.get("path") != LAYER_FILES[layer]
                or record.get("fields") != assigned[layer]):
            raise ResultValidationError(f"invalid {layer} layer schema")
        path = root / LAYER_FILES[layer]
        if (not path.is_file() or path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256")):
            raise ResultValidationError(f"missing or modified {layer} layer: {path}")
        if layer in PROFILER_LAYERS and not set(assigned[layer]) - set(KEY_FIELDS):
            raise ResultValidationError(f"empty profiler schema: {layer}")
        columns, rows = read_result_csv(path)
        if columns != assigned[layer] or len(rows) != record.get("row_count"):
            raise ResultValidationError(f"invalid {layer} layer rows or columns")
        keys = [measurement_key(row) for row in rows]
        if layer == "summary":
            expected_keys = keys
            if len(keys) != manifest["row_count"]:
                raise ResultValidationError("summary row count mismatch")
            combined = {key: dict(row) for key, row in zip(keys, rows)}
        elif expected_keys is None:
            raise ResultValidationError("summary layer must be first")
        elif layer not in PROFILER_LAYERS and keys != expected_keys:
            raise ResultValidationError(f"measurement order mismatch in {layer}")
        for key, row in zip(keys, rows):
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
    root, destination = Path(root), Path(destination)
    if destination.resolve() in {(root / name).resolve() for name in LAYER_FILES.values()}:
        raise ResultValidationError("cannot overwrite a result layer with a wide export")
    if destination.exists():
        raise ResultValidationError(f"refusing to overwrite existing CSV: {destination}")
    fields, rows = read_result_layers(root)
    raw = _csv_bytes(fields, rows)
    atomic_write(destination, lambda stream: stream.write(raw.decode("utf-8")))
    return len(rows)
