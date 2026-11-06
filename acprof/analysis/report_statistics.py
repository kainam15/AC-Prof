"""Reuse window-statistics JSON in HTML, or estimate from its analysis snapshot."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from acprof.analysis.audit import MissingReasons, number
from acprof.analysis.model import AnalysisModel, verify_analysis_sources
from acprof.analysis.statistics_report import DEFAULT_STATISTICS_METRICS
from acprof.analysis.uncertainty import WINDOW_FILTER, summarize_windows, validate_window_options
from acprof.artifacts import read_json_object
from acprof.quality import QUALITY_FIELDS
from acprof.results.csv import ResultValidationError, measurement_key
from acprof.results.layout import ArtifactLayout
from acprof.results.metrics import METRICS

_PARAMETERS = ("confidence", "resamples", "seed", "block_size", "method", "resampling_unit", "filter", "assumption")
_IDENTITY = ("cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "environment_class",
             "runtime_kind", "runtime_id", "resource_scope")
_SUMMARY = (*_IDENTITY, "metric", "unit", "n_windows", "missing_windows", "mean", "std", "reason")
_AUDIT = (*QUALITY_FIELDS, "run_status", "measurement_status")
_MAX_CACHE_BYTES = 32 * 1024 * 1024
MAX_EMBEDDED_WINDOW_BYTES = 8 * 1024 * 1024


def _candidates(source: Path) -> list[Path]:
    layout = ArtifactLayout.from_csv(source)
    candidates = []
    for directory in dict.fromkeys((layout.path("analysis"), layout.contained("analysis"), layout.root)):
        try:
            # iterdir preserves permission/storage errors that glob can suppress.
            paths = list(directory.iterdir())
        except FileNotFoundError:
            continue
        for path in paths:
            if path.name.startswith("window-statistics-") and path.suffix == ".json":
                try:
                    candidates.append((path.stat().st_mtime_ns, path.name, path))
                except FileNotFoundError:
                    continue
    return [item[2] for item in sorted(candidates, reverse=True)]


def _excluded(entries: list[dict]) -> dict[str, int]:
    reasons: Counter[str] = Counter()
    for entry in entries:
        row = entry["row"]
        if number(row.get("warmup")) == 1:
            reasons["warmup"] += 1
        elif number(row.get("warmup")) != 0:
            reasons["invalid_window_identity"] += 1
        elif str(row.get("result_origin", "")).strip() == "inferred_not_measured":
            reasons["inferred_not_measured"] += 1
        elif not entry["eligible"]:
            reasons[str(row.get("status", "unknown")).strip().lower() or "unknown"] += 1
    return dict(reasons)


def _rows_by_identity(rows: list[dict]) -> dict[tuple, list[dict]]:
    defaults = {"environment_class": "unknown", "runtime_kind": "docker", "runtime_id": "",
                "resource_scope": "container_cgroup"}
    indexed = defaultdict(list)
    for row in rows:
        identity = tuple(number(row.get(key)) if key in {"cpu_cores", "mem_cap_gb", "input_scale"}
                         else row.get(key, defaults.get(key)) for key in _IDENTITY)
        indexed[identity].append(row)
    return indexed


def _unavailable_groups(metrics, reason, detail="") -> list[dict]:
    return [{"metric": metric, "unit": METRICS[metric].unit, "n_windows": 0, "missing_windows": 0,
             "mean": None, "std": None, "ci_low": None, "ci_high": None, "reason": reason,
             "detail": detail} for metric in metrics]


def _config_summary(entries, source, metrics, options, *, include_intervals, include_inputs=False):
    rows = [{key: value.strip() for key, value in entry["row"].items()}
            for entry in entries if entry["eligible"]]
    try:
        for row in rows:
            measurement_key(row)
    except ResultValidationError as error:
        # Historical chart inputs may lack window identity; never invent one.
        groups = _unavailable_groups(metrics, "invalid_window_identity", str(error))
    else:
        report = summarize_windows(rows, metrics, **options, include_intervals=include_intervals)
        groups = report["groups"] or _unavailable_groups(metrics, "no_successful_windows")
    excluded = _excluded(entries)
    resolver = MissingReasons(source["metadata"], environment_class=entries[0]["environment_class"])
    indexed = _rows_by_identity(rows)
    ordered_by_identity: dict[tuple, list[dict]] = {}
    for group in groups:
        reasons: Counter[str] = Counter()
        metric = group["metric"]
        identity = tuple(group.get(key) for key in _IDENTITY)
        matching = indexed.get(identity, [])
        for row in matching:
            if number(row.get(metric)) is None:
                reasons[resolver.for_row(row)(metric)] += 1
        group.update(config_id=entries[0]["config_id"], missing_reasons=dict(reasons), excluded_windows=excluded)
        if include_inputs:
            if identity not in ordered_by_identity:
                ordered_by_identity[identity] = sorted(matching, key=lambda sample: float(sample["repeat_idx"]))
            ordered = ordered_by_identity[identity]
            group.update(values=[number(row.get(metric)) for row in ordered],
                         repeat_indices=[float(row["repeat_idx"]) for row in ordered])
    return groups


def window_input_chunks(configs, source, metrics):
    """One canonical JSON stream for embedded inputs and optional offline import files."""
    header = {"kind": "acprof_window_inputs", "schema_version": 1, "result_sha256": source["sha256"]}
    def encode(value):
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    yield encode(header)[:-1] + ',"groups":['
    first = True
    for entries in configs.values():
        for group in _config_summary(entries, source, metrics, {}, include_intervals=False, include_inputs=True):
            yield ("" if first else ",") + encode(group)
            first = False
    yield "]}"


def _window_inputs(configs, source, metrics, budget):
    available = set(metrics) | set(DEFAULT_STATISTICS_METRICS)
    for entries in configs.values():
        for entry in entries:
            available.update(entry["row"])
    catalog = {name: {"label": metric.label, "unit": metric.unit} for name, metric in METRICS.items()
               if name in available and metric.kind == "number"
               and metric.window in {"request_window", "matched_control"}}
    digest, size = hashlib.sha256(), 0
    fragments: list[str] = []
    embedded = True
    for chunk in window_input_chunks(configs, source, catalog):
        encoded = chunk.encode("ascii")
        digest.update(encoded)
        size += len(encoded)
        if size > budget[0]:
            embedded = False
            fragments.clear()
        elif embedded:
            fragments.append(chunk)
    result = {"schema_version": 1, "status": "embedded" if embedded else "external_required",
              "metrics": catalog, "sha256": digest.hexdigest(), "bytes": size}
    if embedded:
        budget[0] -= size
        result["groups"] = json.loads("".join(fragments))["groups"]
    return result


def write_window_inputs(model, report, stream):
    """Regenerate oversized inputs from the same verified snapshot, without source writes."""
    source = next(source for source in model.sources if source["path"] == report["source"])
    configs = defaultdict(list)
    for entry in model.raw_rows:
        if entry["source"] == source["path"]:
            configs[entry["config_id"]].append(entry)
    digest = hashlib.sha256()
    for chunk in window_input_chunks(configs, source, report["recalculation"]["metrics"]):
        digest.update(chunk.encode("ascii"))
        stream.write(chunk)
    if digest.hexdigest() != report["recalculation"]["sha256"]:
        raise ValueError("window input snapshot changed")
    verify_analysis_sources(model)


def _summaries(configs, source, metrics, options, *, include_intervals):
    report = summarize_windows([], metrics, **options, include_intervals=False)
    for entries in configs.values():
        report["groups"].extend(_config_summary(entries, source, metrics, options, include_intervals=include_intervals))
    return report


def _cache_settings(report: dict, *, confidence=.95):
    if (type(report.get("schema_version")) is not int or report.get("schema_version") != 1
            or report.get("confidence") != confidence
            or report.get("method") != "circular_block_percentile_bootstrap"
            or report.get("resampling_unit") != "csv_request_window"
            or report.get("filter") not in {"status=ok and warmup=0", WINDOW_FILTER}):
        raise ValueError("incompatible_statistics_contract")
    groups = report.get("groups")
    if not isinstance(groups, list) or not groups or any(not isinstance(group, dict) for group in groups):
        raise ValueError("invalid_statistics_groups")
    metrics = list(dict.fromkeys(group.get("metric") for group in groups))
    if any(not isinstance(metric, str) for metric in metrics):
        raise ValueError("invalid_statistics_metrics")
    options = {key: report.get(key) for key in ("confidence", "resamples", "seed", "block_size")}
    if any(type(options[key]) is not int for key in ("resamples", "seed", "block_size")):
        raise ValueError("invalid_bootstrap_parameters")
    validate_window_options(metrics, **{key: options[key] for key in ("confidence", "resamples", "block_size")})
    return metrics, options


def _group_key(group):
    return tuple(group.get(key) for key in (*_IDENTITY, "metric"))


def _validate_interval(group, expected):
    if not set((*_SUMMARY, "ci_low", "ci_high")).issubset(group):
        raise ValueError("incomplete_statistics_group")
    if any(type(group[key]) is not int for key in ("n_windows", "missing_windows")):
        raise ValueError("invalid_window_count")
    if any(value is not None and type(value) not in (int, float) for value in (group["mean"], group["std"])):
        raise ValueError("invalid_statistics_value")
    low, high = group.get("ci_low"), group.get("ci_high")
    if expected["reason"]:
        valid = low is None and high is None
    else:
        valid = all(type(value) in (int, float) and math.isfinite(value) for value in (low, high))
        valid = valid and low <= high
    if not valid:
        raise ValueError("invalid_confidence_interval")


def _reuse(report, source, configs, requested_metrics, requested_options):
    if report.get("result_sha256") != source["sha256"]:
        raise ValueError("source_changed")
    if any(report.get(key) != source[key] for key in _AUDIT):
        raise ValueError("audit_changed")
    metrics, options = _cache_settings(report, confidence=requested_options.get("confidence", .95))
    if (requested_metrics is not None and set(metrics) != set(requested_metrics)
            or any(options.get(key) != value for key, value in requested_options.items())):
        raise ValueError("statistics_parameters_changed")
    expected = _summaries(configs, source, metrics, options, include_intervals=False)
    expected_groups = expected["groups"]
    keys = [_group_key(group) for group in expected_groups]
    cached = {_group_key(group): group for group in report["groups"]}
    if len(set(keys)) != len(keys) or len(cached) != len(report["groups"]) or set(keys) != cached.keys():
        raise ValueError("configuration_groups_changed")
    for group in expected_groups:
        saved = cached[_group_key(group)]
        if any(saved.get(key) != group.get(key) for key in _SUMMARY):
            raise ValueError("statistics_summary_changed")
        _validate_interval(saved, group)
        group.update(ci_low=saved["ci_low"], ci_high=saved["ci_high"])
    return expected


def _source_statistics(source, configs, metrics, options, budget, input_metrics):
    diagnostics = []
    report_path = None
    for candidate in _candidates(Path(source["path"])):
        try:
            cached = read_json_object(candidate, label="window statistics", bounded=False,
                                      byte_budget=[_MAX_CACHE_BYTES])
            report = _reuse(cached, source, configs, metrics, options)
        except (FileNotFoundError, IsADirectoryError, ValueError, TypeError) as error:
            diagnostics.append({"path": str(candidate), "reason": str(error)})
            continue
        report_path = str(candidate)
        break
    else:
        report = _summaries(configs, source, metrics or DEFAULT_STATISTICS_METRICS, options, include_intervals=True)
    return {"source": source["path"], "result_sha256": source["sha256"],
            "origin": "cached" if report_path else "computed", "report_path": report_path,
            "engine": "python", "rng": "python_random_mt19937",
            "recalculation": _window_inputs(configs, source, [*input_metrics, *(g["metric"] for g in report["groups"])], budget),
            "cache_diagnostics": diagnostics,
            "parameters": {key: report[key] for key in _PARAMETERS}, "groups": report["groups"],
            "audit": {**source.get("audit", {}), **{key: source[key] for key in _AUDIT}}}


def build_report_statistics(model: AnalysisModel, *, metrics=None, **options) -> list[dict]:
    """Keep estimates per HTML configuration and sources read-only, including on cache misses."""
    verify_analysis_sources(model)
    validate_window_options(metrics or DEFAULT_STATISTICS_METRICS,
                            **{key: value for key, value in options.items() if key != "seed"})
    entries_by_source: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for entry in model.raw_rows:
        entries_by_source[entry["source"]][entry["config_id"]].append(entry)
    budget = [MAX_EMBEDDED_WINDOW_BYTES]
    input_metrics = {key for entry in model.raw_rows for key in entry["row"]}
    result = [_source_statistics(source, entries_by_source[source["path"]], metrics, options, budget, input_metrics)
              for source in model.sources]
    verify_analysis_sources(model)
    return result
