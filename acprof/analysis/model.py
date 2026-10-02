"""只读 CSV → 长表和配置汇总；保留历史字段，不初始化采集器或绘图库。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from acprof.artifact_layout import ArtifactLayout
from acprof.metric_registry import ANALYSIS_METRICS, VIEW_METRICS
from acprof.platform import recorded_identity

DIMENSIONS = ("run_id", "model", "runtime", "device", "cpu", "memory", "gpu", "concurrency",
              "environment_class", "task", "input_case", "experiment_batch")
SUMMARY_FIELDS = ("config_id", "model", "runtime", "device", "cpu", "memory", "gpu",
                  *[name for name in VIEW_METRICS if ANALYSIS_METRICS[name].summary], "status")
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


def _json(path):
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=lambda _: None)
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _identity(row, meta, run_id, batch):
    environment = recorded_identity(meta)["environment_class"]
    recorded = _text(row.get("environment_class"))
    if recorded not in {"native_linux", "wsl2", "vm", "cloud", "container_host"}:
        recorded = "unknown"
    if environment != "unknown" and recorded not in {"unknown", environment}:
        raise ValueError("inconsistent environment_class between CSV and metadata")
    mode = _text(row.get("gpu_mode")).lower()
    device = {"off": "CPU", "on": "GPU"}.get(mode, _text(row.get("device")))
    scale = number(row.get("input_scale"))
    input_case = row.get("input_case") or _canonical({
        "scale": scale, "type": meta.get("input_scale_type", "unknown"),
        "task_param": _text(row.get("task_param")),
        "batch_size": number(row.get("batch_size", meta.get("batch_size"))),
        "input_units": number(row.get("input_units_per_request")),
    })
    return dict(zip(DIMENSIONS, (
        _text(row.get("run_id"), run_id),
        _text(row.get("model") or row.get("model_name") or meta.get("model_name")),
        _text(row.get("runtime") or row.get("runtime_backend") or meta.get("runtime_backend")),
        device, number(row.get("cpu_cores")), number(row.get("mem_cap_gb")),
        _text(row.get("gpu_device_uuid") or meta.get("gpu")) if device == "GPU" else "n/a",
        number(row.get("concurrency", meta.get("concurrency"))),
        recorded if recorded != "unknown" else environment,
        _text(row.get("task") or meta.get("pipeline_tag") or meta.get("task_family")),
        _text(input_case), _text(row.get("experiment_batch"), batch),
    )))


def _values(row, device):
    values = {name: number(row.get(name)) for name in ANALYSIS_METRICS
              if ANALYSIS_METRICS[name].kind == "number" and name in row}
    cpu, gpu, count = (number(row.get(name)) for name in
                       ("cpu_energy_total_j", "gpu_energy_total_j", "repeat_in_window"))
    energy = None
    if cpu is not None and cpu >= 0:
        if device == "CPU":
            energy = cpu
        elif device == "GPU" and gpu is not None and gpu >= 0:
            energy = cpu + gpu
    energy = number(energy)
    values["observed_energy_per_request_j"] = energy
    values["observed_energy_j"] = number(energy * count) if energy is not None and count and count > 0 else None
    return values


def _evidence(name, row):
    if _text(row.get("result_origin")) == "inferred_not_measured":
        return "inferred_not_measured"
    source = ANALYSIS_METRICS[name].source
    if source in {"derived", "derived_package_gpu", "rapl_cgroup_attribution"} or "_est_" in name:
        return "derived"
    return "measured"


def _summarize(entries, name):
    metric = ANALYSIS_METRICS[name]
    valid: list[tuple[dict, float]] = []
    for entry in entries:
        numeric = number(entry["values"].get(name))
        if entry["eligible"] and numeric is not None:
            valid.append((entry, numeric))
    n_windows = len(valid)
    if metric.aggregation == "lifecycle":
        lifecycles = {}
        for entry, value in valid:
            # Without startup identity only one consistent reused value is defensible.
            key = _text(entry["row"].get("cold_start_started_at"))
            if key in lifecycles and lifecycles[key] != value:
                return {"value": None, "n": 0, "aggregation": "lifecycle_conflict", "missing": 0}
            lifecycles[key] = value
        values = list(lifecycles.values())
    else:
        values = [value for _, value in valid]
    value = None
    missing = sum(entry["eligible"] for entry in entries) - n_windows
    if values:
        if metric.aggregation == "sum":
            # A partial sum would masquerade as the energy of all successful windows.
            value = sum(values) if not missing else None
        elif metric.aggregation == "max":
            value = max(values)
        elif metric.aggregation == "request_weighted":
            weights: list[float] = []
            for entry, _ in valid:
                weight = number(entry["row"].get("repeat_in_window"))
                if weight is not None and weight > 0:
                    weights.append(weight)
            if not missing and len(weights) == len(values):
                value = sum(v * w for v, w in zip(values, weights)) / sum(weights)
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
        return [{**{key: entry[key] for key in DIMENSIONS}, "config_id": entry["config_id"],
                 "metric": name, "value": value, "unit": ANALYSIS_METRICS[name].unit,
                 "evidence": _evidence(name, entry["row"]), "eligible": entry["eligible"],
                 "source": entry["source"], "source_row": entry["source_row"]}
                for entry in self.raw_rows for name, value in entry["values"].items()]

    @property
    def summary(self):
        return [{key: config["metrics"][key]["value"] if key in config["metrics"] else config[key]
                 for key in SUMMARY_FIELDS} for config in self.configs]

    def to_dict(self):
        return {"schema_version": 1, "sources": self.sources, "raw_rows": self.raw_rows,
                "records": self.records, "configs": self.configs, "summary": self.summary}


def load_analysis(sources) -> AnalysisModel:
    """Accept current/legacy CSV without rewriting, renaming or guessing retired metrics."""
    snapshots, entries, groups, seen_paths, seen_keys = [], [], defaultdict(list), set(), set()
    for source in sources:
        path = Path(source).expanduser().resolve()
        layout = ArtifactLayout.discover(path) if path.is_dir() else ArtifactLayout.from_csv(path)
        path = layout.result_csv if path.is_dir() else path
        if path in seen_paths:
            raise ValueError(f"duplicate CSV input: {path}")
        seen_paths.add(path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        meta = _json(layout.path("static_meta.json"))
        state = _json(layout.path("run_state.json"))
        run_id = _text(state.get("run_id"), "legacy-" + digest[:16])
        snapshots.append({"path": str(path), "sha256": digest, "run_id": run_id,
                          "run_state": state.get("status", "unknown"), "metadata": meta})
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), strict=True)
        fields = [name.strip() for name in (reader.fieldnames or [])]
        if not fields or any(not name for name in fields) or len(fields) != len(set(fields)):
            raise ValueError(f"missing or duplicate CSV columns: {path}")
        reader.fieldnames = fields
        first_entry = len(entries)
        for line, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"malformed CSV row: {path}:{line}")
            row = {key: value.strip() for key, value in raw.items()}
            identity = _identity(row, meta, run_id, path.parent.name)
            config_id = identity["run_id"] + ":" + _digest(identity)
            repeat = number(row.get("repeat_idx"))
            warmup = number(row.get("warmup"))
            if repeat is not None:
                key = (config_id, warmup, repeat)
                if key in seen_keys:
                    raise ValueError(f"duplicate measurement: {path}:{line}")
                seen_keys.add(key)
            eligible = (row.get("status", "").lower() == "ok" and warmup == 0
                        and row.get("result_origin") != "inferred_not_measured")
            entry = {**identity, "config_id": config_id, "source": str(path), "source_row": line,
                     "row": dict(raw), "eligible": eligible, "values": _values(row, identity["device"])}
            groups[config_id].append(entry)
            entries.append(entry)
        if len(entries) == first_entry:
            raise ValueError(f"CSV has no measurement rows: {path}")
    if not snapshots:
        raise ValueError("at least one result CSV is required")
    configs = []
    for config_id, rows in groups.items():
        formal = [r for r in rows if number(r["row"].get("warmup")) == 0]
        valid = sum(r["eligible"] for r in rows)
        inferred = any(_text(r["row"].get("result_origin")) == "inferred_not_measured" for r in formal)
        status = ("ok" if valid and valid == len(formal) else "partial" if valid else
                  "inferred_not_measured" if inferred else "failed" if formal else "no_formal_windows")
        config = {key: rows[0][key] for key in DIMENSIONS}
        config.update(config_id=config_id, status=status, valid_windows=valid,
                      excluded_rows=len(rows) - valid, source=rows[0]["source"],
                      metrics={name: _summarize(rows, name) for name in VIEW_METRICS})
        configs.append(config)
    return AnalysisModel(snapshots, entries, configs)
