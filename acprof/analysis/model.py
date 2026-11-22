"""只读 CSV → 长表和配置汇总；保留历史字段，不初始化采集器或绘图库。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from decimal import InvalidOperation
from pathlib import Path
from typing import Any

from acprof.analysis.audit import require_consistent_run_identity
from acprof.analysis.comparison import load_comparison_snapshot
from acprof.analysis.conditions import PURPOSES, PreparedComparison
from acprof.analysis.lifecycle import LifecycleConflict, unique_startup_observations
from acprof.analysis.workload import window_request_weight
from acprof.gpu_mode import normalize_gpu_mode
from acprof.platform import recorded_identity
from acprof.quality import QUALITY_FIELDS, read_quality
from acprof.results.csv import (
    KEY_FIELDS,
    MISSING_CSV_TEXT,
    ResultValidationError,
    is_measured_success,
    measurement_key,
    numeric_coordinate_key,
    read_result_csv_snapshot,
    recorded_run_id,
    verify_result_source,
)
from acprof.results.layout import expand_result_sources, resolve_result_file
from acprof.results.measurement_plan import decimal_coordinate, decimal_key_text
from acprof.results.metadata import ResultMetadataSnapshot
from acprof.results.metrics import ANALYSIS_METRICS, METRICS, VIEW_METRICS, supported_csv_header
from acprof.results.profile import (
    USABLE_QUALITIES,
    metric_observation,
    observation_contract,
    observation_signature,
)

_DEVICE_MODES = {"CPU": "off", "GPU": "on"}

DIMENSIONS = ("run_id", "model", "runtime", "device", "cpu", "memory", "gpu", "concurrency",
              "environment_class", "task", "input_case", "experiment_batch",
              "runtime_kind", "runtime_id", "resource_scope")
SUMMARY_FIELDS = ("config_id", "model", "runtime", "device", "cpu", "memory", "gpu",
                  *[name for name in VIEW_METRICS if ANALYSIS_METRICS[name].summary], "status",
                  "run_status", "measurement_status", "comparability", *QUALITY_FIELDS)
DEFAULT_MATRIX = ("latency_app_p95_s", "throughput_samples_per_s", "container_mem_usage_peak_bytes",
                  "observed_energy_per_request_j", "cpu_ipc", "cold_start_s")


def number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _text(value, default="unknown"):
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return default
    return str(value).strip()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()[:16]


def _exact_coordinate(raw, display):
    """Keep extra identity evidence only when a numeric display loses recorded digits."""
    if display is not None and display >= 0 and raw != str(display):
        recorded = decimal_coordinate(raw)
        if recorded > 0 and recorded != decimal_coordinate(str(display)):
            return decimal_key_text(recorded)
    return None


def _report_window_coordinate(row, name, *, recorded):
    if name not in row:
        return None
    raw = row[name]
    if not recorded and raw.lower() in MISSING_CSV_TEXT:
        return None
    try:
        return numeric_coordinate_key(name, raw)
    except ResultValidationError:
        if recorded:
            raise
        # Historical descriptive charts accepted absent and nonnumeric labels.
        # Finite protocol coordinates still have to satisfy their integer domain.
        try:
            coordinate = decimal_coordinate(raw)
        except InvalidOperation:
            return None
        if coordinate.is_finite():
            raise
        return None


def _report_window_coordinates(row, *, run_id, recorded_source, complete_key):
    if (recorded_source and complete_key) or (run_id is not None and "repeat_idx" in row):
        mode = row.get("gpu_mode")
        if mode and mode not in ("on", "off"):
            # Descriptive exports use GPU/CPU aliases. Preserve the original cells.
            row = {**row, "gpu_mode": normalize_gpu_mode(mode)}
        coordinates = measurement_key(row)
        return coordinates, coordinates[-2], coordinates[-1]
    recorded = recorded_source or run_id is not None
    warmup = _report_window_coordinate(row, "warmup", recorded=recorded)
    repeat = _report_window_coordinate(row, "repeat_idx", recorded=recorded)
    return None, warmup, repeat


def _report_device(row) -> str:
    if mode := _text(row.get("gpu_mode"), default=""):
        return "GPU" if normalize_gpu_mode(mode) == "on" else "CPU"
    # Historical descriptive tables can carry only a free-form device label.
    label = _text(row.get("device"))
    canonical = label.upper()
    return canonical if canonical in _DEVICE_MODES else label


def _identity(row, meta, run_id, batch, *, coordinates):
    environment = recorded_identity(meta)["environment_class"]
    recorded = _text(row.get("environment_class"))
    if recorded not in {"native_linux", "wsl2", "vm", "cloud", "container_host"}:
        recorded = "unknown"
    if environment != "unknown" and recorded not in {"unknown", environment}:
        raise ValueError("inconsistent environment_class between CSV and metadata")
    device = _report_device(row)
    axes = {name: number(row.get(name)) for name in ("cpu_cores", "mem_cap_gb", "input_scale")}
    if coordinates is not None:
        for field, key in zip(KEY_FIELDS, coordinates, strict=True):
            if field not in axes or key == "inherited":
                continue
            display = axes[field]
            if display is None or display <= 0:
                raise ResultValidationError(f"{field} is outside the positive report axis range: {row[field]!r}")
    scale = axes["input_scale"]
    input_case = {
        "scale": _exact_coordinate(row.get("input_scale"), scale) or scale, "type": meta.get("input_scale_type", "unknown"),
        "task_param": _text(row.get("task_param")),
        "batch_size": number(row.get("batch_size", meta.get("batch_size"))),
        "input_units": number(row.get("input_units_per_request")),
    }
    if name := _text(row.get("input_case"), default=""):
        input_case["name"] = name
    identity = dict(zip(DIMENSIONS, (
        run_id,
        _text(row.get("model") or row.get("model_name") or meta.get("model_name")),
        _text(row.get("runtime") or row.get("runtime_backend") or meta.get("runtime_backend")),
        device, axes["cpu_cores"], axes["mem_cap_gb"],
        _text(row.get("gpu_device_uuid") or meta.get("gpu")) if device == "GPU" else "n/a",
        number(row.get("concurrency", meta.get("concurrency"))),
        recorded if recorded != "unknown" else environment,
        _text(row.get("task") or meta.get("pipeline_tag") or meta.get("task_family")),
        _canonical(input_case), _text(row.get("experiment_batch"), batch),
        *(_text(row.get(key) or meta.get(key)) for key in ("runtime_kind", "runtime_id", "resource_scope")),
    ), strict=True))
    exact = {axis: key for axis, field in (("cpu", "cpu_cores"), ("memory", "mem_cap_gb"))
             if (key := _exact_coordinate(row.get(field), identity[axis])) is not None}
    if exact:
        identity["exact_resources"] = exact
    return identity


def _window_energy(entry: dict) -> tuple[dict[str, float | None], dict[str, dict]]:
    names = ("cpu_energy_total_j", "gpu_energy_total_j") if entry["device"] == "GPU" else ("cpu_energy_total_j",)
    row = entry["row"]
    components = {name: metric_observation(row, name) for name in names}
    contracts = {name: observation_contract(item) for name, item in components.items()}
    # Some CSVs record the GPU method without recording collector quality or cadence.
    method = _text(row.get("gpu_energy_source"))
    if "gpu_energy_total_j" in contracts and method != "unknown":
        contracts["gpu_energy_total_j"]["energy_method"] = method
    quality = "complete"
    reason = ""
    for name, item in components.items():
        if item["quality"] not in USABLE_QUALITIES:
            quality, reason = item["quality"], f"{name}: {item['reason']}"
            break
        if item["quality"] == "unknown":
            quality, reason = "unknown", "component_quality_unknown"
    if entry["device"] not in {"CPU", "GPU"}:
        quality, reason = "unknown", "unknown_device"
    derived = {"components": contracts, "quality": quality, "reason": reason}
    energy = _derive_energy(entry["values"], entry["device"], quality, entry["request_weight"])
    observed = {name: {**derived, "quality": "unavailable", "reason": "missing_window_value"}
                if value is None and quality == "complete" else derived for name, value in energy.items()}
    return energy, {**components, **observed}


def _derive_energy(values: dict, device: str, quality: str, weight: float | None) -> dict[str, float | None]:
    cpu, gpu = values.get("cpu_energy_total_j"), values.get("gpu_energy_total_j")
    energy = None
    if quality in USABLE_QUALITIES and cpu is not None and cpu >= 0:
        if device == "CPU":
            energy = cpu
        elif device == "GPU" and gpu is not None and gpu >= 0:
            energy = cpu + gpu
    energy = number(energy)
    total = number(energy * weight) if energy is not None and weight is not None else None
    return {"observed_energy_per_request_j": energy, "observed_energy_j": total}


def _prepare_energy_windows(rows: list[dict], formal: dict[int, dict]) -> dict[str, dict[int, dict]]:
    observations: dict[str, dict[int, dict]] = defaultdict(dict)
    for index, entry in enumerate(rows):
        values, evidence = _window_energy(entry)
        entry["values"].update(values)
        if index in formal:
            for name, item in evidence.items():
                if name in entry["values"]:
                    observations[name][index] = item
    return observations


def _evidence(name, row):
    if _text(row.get("result_origin")) == "inferred_not_measured":
        return "inferred_not_measured"
    source = ANALYSIS_METRICS[name].source
    if source in {"derived", "derived_package_gpu", "rapl_cgroup_attribution"} or "_est_" in name:
        return "derived"
    return "measured"


def _usable_windows(entries, name, observations: dict[int, dict] | None) -> tuple[list[tuple[dict, float]], set[str]]:
    valid: list[tuple[dict, float]] = []
    signatures: set[str] = set()
    for index, entry in enumerate(entries):
        numeric = number(entry["values"].get(name))
        if not entry["eligible"] or numeric is None:
            continue
        if observations is not None or name in METRICS:
            observation = observations[index] if observations is not None else metric_observation(entry["row"], name)
            if observation["quality"] not in USABLE_QUALITIES:
                continue
            signatures.add(observation_signature(observation))
        valid.append((entry, numeric))
    return valid, signatures


def _summarize(entries, name, observations: dict[int, dict] | None = None):
    metric = ANALYSIS_METRICS[name]
    valid, signatures = _usable_windows(entries, name, observations)
    if metric.aggregation == "request_weighted":
        valid = [(entry, value) for entry, value in valid if entry["request_weight"] is not None]
    n_windows = len(valid)
    if metric.aggregation == "lifecycle":
        try:
            startups = unique_startup_observations(
                (entry["row"].get("cold_start_started_at"), {name: lifecycle_value})
                for entry, lifecycle_value in valid
            )
        except LifecycleConflict:
            return {"value": None, "n": 0, "aggregation": "lifecycle_conflict", "missing": 0}
        values = [startup[name] for startup in startups]
    else:
        values = [value for _, value in valid]
    value: float | None = None
    missing = sum(entry["eligible"] for entry in entries) - n_windows
    if values and len(signatures) <= 1:
        if metric.aggregation == "sum":
            # A partial sum would masquerade as the energy of all successful windows.
            value = sum(values) if not missing else None
        elif metric.aggregation == "max":
            value = max(values)
        elif metric.aggregation == "request_weighted":
            weight = sum(entry["request_weight"] for entry, _ in valid)
            if not missing and math.isfinite(weight):
                value = sum(v * entry["request_weight"] for entry, v in valid) / weight
        else:
            value = statistics.fmean(values)
    return {"value": number(value), "n": len(values), "missing": missing,
            "aggregation": metric.aggregation}


@dataclass
class AnalysisModel:
    sources: list[dict]
    raw_rows: list[dict]
    configs: list[dict]

    @property
    def records(self):
        """Normalized long form; raw values remain available even for excluded rows."""
        records = []
        for entry in self.raw_rows:
            _, energy = _window_energy(entry)
            identity = {key: entry[key] for key in (*DIMENSIONS, *QUALITY_FIELDS, "config_id",
                                                     "eligible", "source", "source_row")}
            for name, value in entry["values"].items():
                observation = energy.get(name)
                if observation is None and name in METRICS:
                    observation = metric_observation(entry["row"], name)
                records.append({**identity, "metric": name, "value": value, "unit": ANALYSIS_METRICS[name].unit,
                                "evidence": _evidence(name, entry["row"]), "observation": observation})
        return records

    @property
    def summary(self):
        return [{key: config["metrics"][key]["value"] if key in config["metrics"] else config[key]
                 for key in SUMMARY_FIELDS} for config in self.configs]

    def to_dict(self):
        return {"schema_version": 1, "sources": self.sources, "raw_rows": self.raw_rows,
                "records": self.records, "configs": self.configs, "summary": self.summary}


def verify_analysis_sources(model: AnalysisModel) -> None:
    """Reject changed result or parsed metadata evidence before publication."""
    for source in model.sources:
        verify_result_source(Path(source["path"]), source["sha256"])
        for path, digest in source.get("metadata_sha256", {}).items():
            verify_result_source(Path(path), digest)


def _validated_csv_fields(path: Path, raw_fields: list[str]) -> tuple[bool, tuple[str, ...], set[str]]:
    fields = [name.strip() for name in raw_fields]
    field_names = set(fields)
    if not fields or any(not name for name in fields) or len(fields) != len(field_names):
        raise ValueError(f"missing or duplicate CSV columns: {path}")
    numeric_fields = tuple(
        name for name, metric in ANALYSIS_METRICS.items()
        if metric.kind == "number" and name in field_names
    )
    return fields != raw_fields, numeric_fields, field_names


def _build_analysis_configs(
    groups: dict[str, list[dict[str, Any]]],
    comparisons: dict[str, PreparedComparison],
) -> list[dict[str, Any]]:
    configs = []
    for config_id, rows in groups.items():
        formal = {index: entry for index, entry in enumerate(rows) if number(entry["row"].get("warmup")) == 0}
        valid = sum(r["eligible"] for r in rows)
        inferred = any(_text(entry["row"].get("result_origin")) == "inferred_not_measured" for entry in formal.values())
        status = ("ok" if valid and valid == len(formal) else "partial" if valid else
                  "inferred_not_measured" if inferred else "failed" if formal else "no_formal_windows")
        config = {key: rows[0][key] for key in DIMENSIONS}
        if "exact_resources" in rows[0]:
            config["exact_resources"] = rows[0]["exact_resources"]
        config.update({key: rows[0][key] for key in (*QUALITY_FIELDS, "run_status", "measurement_status", "missing_metrics")})
        try:
            case = measurement_key(rows[0]["row"])[:4]
        except ValueError:
            case = None
        config["comparison_profiles"] = {purpose: comparisons[rows[0]["source"]].profile(
            purpose=purpose, case=case) for purpose in PURPOSES}
        config["comparability"] = config["comparison_profiles"]["same-hardware"]["status"]
        summaries, provenance = {}, {}
        energy = _prepare_energy_windows(rows, formal)
        for name in VIEW_METRICS:
            observations = energy.get(name)
            if observations is None and name in METRICS and any(name in entry["row"] for entry in formal.values()):
                observations = {index: metric_observation(entry["row"], name) for index, entry in formal.items()}
            if observations:
                provenance[name] = list({_canonical(item): item for item in observations.values()}.values())
            summaries[name] = _summarize(rows, name, observations)
        config["metric_observations"] = provenance
        config.update(config_id=config_id, status=status, valid_windows=valid,
                      input_scale=number(rows[0]["row"].get("input_scale")),
                      excluded_rows=len(rows) - valid, source=rows[0]["source"],
                      metrics=summaries)
        configs.append(config)
    return configs


def load_analysis(sources) -> AnalysisModel:
    """Load authoritative experiment layers without writing a wide CSV."""
    snapshots, groups, seen_paths = [], defaultdict(list), set()
    entries: list[dict[str, Any]] = []
    seen_keys: dict[object, tuple[str, str]] = {}
    comparisons = {}
    for source in expand_result_sources(sources):
        path = resolve_result_file(source).resolve()
        if path in seen_paths:
            raise ValueError(f"duplicate CSV input: {path}")
        seen_paths.add(path)
        metadata_snapshot = ResultMetadataSnapshot(path)
        result_snapshot = read_result_csv_snapshot(path, validate_measurements=False)
        normalize_fields, numeric_fields, fields = _validated_csv_fields(path, result_snapshot.fields)
        digest = result_snapshot.sha256
        meta, state = (metadata_snapshot.read(name, label="analysis metadata").payload
                       if metadata_snapshot.exists(name) else {}
                       for name in ("static_meta.json", "run_state.json"))
        run_id = recorded_run_id(state.get("run_id"))
        complete_key = fields.issuperset(KEY_FIELDS)
        # Reports allow reordered columns and extensions on an exported protocol table.
        recorded_source = (path.name == "result_layers.json" or "status" in state or "runtime" in state
                           or supported_csv_header([name for name in METRICS if name in fields]))
        snapshot = load_comparison_snapshot(path, result_snapshot=result_snapshot,
                                            metadata_snapshot=metadata_snapshot,
                                            verify_result_snapshot=False)
        require_consistent_run_identity(snapshot["audit"])
        comparisons[str(path)] = PreparedComparison(snapshot)
        evidence = {"run_status": state.get("status", "unknown"),
                    **{key: snapshot[key] for key in ("measurement_status", "missing_metrics", *QUALITY_FIELDS)}}
        snapshots.append({"path": str(path), "display_name": path.parent.name, "sha256": digest, "run_id": run_id or "unknown",
                          "run_state": state.get("status", "unknown"), "metadata": meta,
                          "audit": snapshot["audit"], **evidence})
        device_quality = {}
        first_entry = len(entries)
        for line, original in enumerate(result_snapshot.rows, 2):
            raw = ({name.strip(): value for name, value in original.items()}
                   if normalize_fields else original)
            row = {key: value.strip() for key, value in raw.items()}
            row_run_id = recorded_run_id(row.get("run_id")) or run_id
            try:
                coordinates, warmup, repeat = _report_window_coordinates(
                    row, run_id=row_run_id, recorded_source=recorded_source, complete_key=complete_key)
                identity = _identity(row, meta, row_run_id or "unknown", "unknown", coordinates=coordinates)
            except ValueError as error:
                raise ResultValidationError(f"{path}:{line}: {error}") from error
            # A batch label and artifact location are presentation, not recorded
            # experimental conditions. Moving a result must not create a run.
            stable_identity = {key: value for key, value in identity.items() if key != "experiment_batch"}
            source_id = "source-" + digest[:16]
            config_id = (identity["run_id"] if identity["run_id"] != "unknown" else source_id) + ":" + _digest(stable_identity)
            if repeat is not None:
                key = ((row_run_id, coordinates) if row_run_id is not None and coordinates is not None
                       else (config_id, warmup, repeat))
                signature = _digest({"row": row, "identity": stable_identity})
                if key in seen_keys:
                    previous_signature, previous = seen_keys[key]
                    kind = "duplicate" if signature == previous_signature else "conflicting"
                    raise ValueError(f"{kind} measurement for run {identity['run_id']}: {path}:{line}; first seen {previous}")
                seen_keys[key] = (signature, f"{path}:{line}")
            eligible = is_measured_success(row) and warmup == "0"
            # Unclassified devices retain the source-wide quality evidence.
            mode = _DEVICE_MODES.get(identity["device"])
            if mode not in device_quality:
                device_quality[mode] = read_quality(path, device=mode, metadata_snapshot=metadata_snapshot)
            values = {name: None if type(value := row[name]) is str and value in MISSING_CSV_TEXT else number(value)
                      for name in numeric_fields}
            entry = {**identity, "config_id": config_id, "source": str(path), "source_row": line,
                     "request_weight": window_request_weight(row.get("repeat_in_window"), values.get("repeat_in_window")),
                     "row": raw, "eligible": eligible, "values": values,
                     **evidence, **device_quality[mode]}
            groups[config_id].append(entry)
            entries.append(entry)
        if len(entries) == first_entry:
            raise ValueError(f"CSV has no measurement rows: {path}")
        try:
            verify_result_source(path, digest)
            fingerprints = metadata_snapshot.observed_fingerprints()
            snapshots[-1]["metadata_sha256"] = {str(artifact): expected for artifact, expected in fingerprints.items()}
        except (OSError, ValueError, csv.Error) as error:
            raise ValueError(f"CSV changed during analysis: {path}: {error}") from error
    if not snapshots:
        raise ValueError("at least one result CSV is required")
    return AnalysisModel(snapshots, entries, _build_analysis_configs(groups, comparisons))
