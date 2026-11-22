"""Compare run means, resampling independent runs instead of individual requests."""
from __future__ import annotations

import csv
import math
import random
import statistics
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acprof.analysis.audit import require_consistent_run_identity
from acprof.analysis.changes import metric_change
from acprof.analysis.comparison import compare_snapshots, load_comparison_snapshot
from acprof.analysis.conditions import ALLOWED_RESOURCE_DIMENSIONS
from acprof.analysis.coordinates import COORDINATE_IDENTITY, project_coordinates
from acprof.analysis.uncertainty import metric_values_and_signature, validate_window_options
from acprof.artifacts import file_sha256
from acprof.hardware_conditions import conditions_path
from acprof.quality import QUALITY_FIELDS, combine_quality
from acprof.results import is_inferred_result
from acprof.results.csv import (
    ResultCsvSnapshot,
    is_measured_success,
    measurement_key,
    read_result_csv_snapshot,
    recorded_run_id,
    result_row_status,
    verify_result_source,
)
from acprof.results.layout import expand_result_sources, resolve_result_file
from acprof.results.metadata import ResultMetadataSnapshot
from acprof.results.metrics import METRICS


def _verify_source(artifact: Path, expected: str | None) -> None:
    try:
        verify_result_source(artifact, expected)
    except (OSError, ValueError, csv.Error) as error:
        raise ValueError(f"experiment changed during comparison: {artifact}: {error}") from error


def _interval(values: list[float], confidence: float):
    ordered = sorted(values)

    def quantile(p):
        position = (len(ordered) - 1) * p
        lo = math.floor(position)
        fraction = position - lo
        return ordered[lo] * (1 - fraction) + ordered[min(lo + 1, len(ordered) - 1)] * fraction

    tail = (1 - confidence) / 2
    return [quantile(tail), quantile(1 - tail)]


def _bootstrap(left, right, *, confidence, resamples, seed):
    rng = random.Random(seed)
    differences, ratios = [], []
    for _ in range(resamples):
        a = statistics.fmean(rng.choices(left, k=len(left)))
        b = statistics.fmean(rng.choices(right, k=len(right)))
        differences.append(b - a)
        if a > 0:
            ratios.append(b / a)
    return (_interval(differences, confidence),
            _interval(ratios, confidence) if len(ratios) == resamples else None)


def _aggregate_evidence(records):
    statuses = {record["run_status"] for record in records}
    measurement = {record["measurement_status"] for record in records}
    return {**combine_quality(records), "run_status": next(iter(statuses)) if len(statuses) == 1 else "mixed" if statuses else "unknown",
            "measurement_status": ("incomplete" if "incomplete" in measurement else
                                   "complete" if measurement == {"complete"} else "unknown")}


_FINGERPRINTED_METADATA_FILES = (
    "run_state.json",
    "input_scale_plan.json",
    "static_meta.json",
    "hardware_conditions.json",
    "quality_checks.json",
    "runtime_validation.json",
    "model_resolution.json",
    "runtime_failures.json",
)


def _record_experiment_fingerprints(
    metadata_snapshot: ResultMetadataSnapshot,
    result_snapshot: ResultCsvSnapshot,
    fingerprints: dict[Path, str | None],
) -> None:
    layout = metadata_snapshot.layout
    fingerprints[metadata_snapshot.result_path] = result_snapshot.sha256
    if result_snapshot.path.name == "result_layers.json":
        fingerprints[result_snapshot.path] = result_snapshot.sha256
    # Capture identity before reading any state used to select runs.
    for name in _FINGERPRINTED_METADATA_FILES:
        artifact = conditions_path(layout) if name == "hardware_conditions.json" else layout.path(name)
        logical_name = str(artifact.relative_to(layout.root)) if name == "hardware_conditions.json" else name
        fingerprints[artifact] = file_sha256(artifact) if metadata_snapshot.exists(logical_name) else None


@dataclass(frozen=True)
class _RunMetric:
    summary: dict[str, Any]
    signature: str | None


def _collect_run_case_groups(
    rows: list[dict[str, str]],
    *,
    run_id: str,
    complete: bool,
    purpose: str,
    metrics: list[str],
) -> dict[tuple[str, ...], _RunMetric]:
    groups = {}
    by_case = defaultdict(list)
    for row in rows:
        key = measurement_key(row)
        if key[4] == "0":
            by_case[key[:4]].append(row)
    scaling_cases = set()
    for case, windows in by_case.items():
        comparison_case = case
        if purpose == "resource-scaling":
            if case[2:] in scaling_cases:
                raise ValueError("resource-scaling requires one resource configuration per device/input in each run")
            scaling_cases.add(case[2:])
            comparison_case = ("0", "0", *case[2:])
        # Window classification is shared by every metric in this resource case.
        measured = [row for row in windows if is_measured_success(row)]
        failed_count = sum(result_row_status(row) != "ok" for row in windows)
        inferred_count = sum(is_inferred_result(row) for row in windows)
        resources = project_coordinates({"cpu_cores": case[0], "mem_cap_gb": case[1]})
        for metric in metrics:
            window_values, signature = metric_values_and_signature(measured, metric)
            groups[(*comparison_case, metric)] = _RunMetric({
                "run_id": run_id,
                "mean": statistics.fmean(window_values) if window_values and complete else None,
                "resource_coordinates": resources,
                "valid_windows": len(window_values),
                "failed_windows": failed_count,
                "missing_windows": len(measured) - len(window_values),
                "inferred_windows": inferred_count,
                "eligible": complete,
            }, signature)
    return groups


def _summarize_group_metric(
    case_metric: tuple[str, ...],
    sides: dict[str, list[_RunMetric]],
    *,
    sources: dict[str, list],
    status: str,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    _, _, gpu, scale, metric = case_metric
    coordinates = {side: records[0].summary["resource_coordinates"] if records else None for side, records in sides.items()}
    # Resource-scaling can change quotas between sides. The group describes its first recorded side.
    resources = coordinates["left"] or coordinates["right"]
    if resources is None:
        raise ValueError("comparison case has no recorded resource coordinates")
    inputs = project_coordinates({"input_scale": scale})
    axes = {**resources, **inputs}
    if "exact_coordinates" in inputs and "exact_coordinates" in resources:
        axes["exact_coordinates"] = {**resources["exact_coordinates"], **inputs["exact_coordinates"]}
    result: dict[str, Any] = {
        **axes,
        "gpu_mode": gpu, "resource_coordinates": coordinates,
        "metric": metric, "unit": METRICS[metric].unit,
        "difference": None, "ratio": None, "percent": None, "performance_ratio": None,
        "difference_ci": None, "ratio_ci": None, "reason": "",
    }
    signatures = {run.signature for runs in sides.values() for run in runs}
    result["comparability"] = "not comparable"
    values: dict[str, list[float]] = {}
    for side in ("left", "right"):
        records = [run.summary for run in sides[side]]
        values[side] = [record["mean"] for record in records if record["mean"] is not None]
        result[side] = {
            **_aggregate_evidence(records),
            "n_runs": len(values[side]),
            "requested_runs": len(sources[side]),
            "missing_runs": len(sources[side]) - len(values[side]),
            "mean": statistics.fmean(values[side]) if values[side] else None,
            "std": statistics.stdev(values[side]) if len(values[side]) > 1 else None,
            "failed_windows": sum(record["failed_windows"] for record in records),
            "missing_windows": sum(record["missing_windows"] for record in records),
            "inferred_windows": sum(record["inferred_windows"] for record in records),
            "runs": records,
        }
    if status != "compatible":
        result["reason"] = "conditions_" + status
    elif not values["left"] or not values["right"]:
        result["reason"] = "missing_comparable_runs"
    elif len(signatures) != 1 or None in signatures:
        result["reason"] = "not comparable"
    else:
        result["comparability"] = "comparable"
        a, b = result["left"]["mean"], result["right"]["mean"]
        changes = metric_change(a, b, direction=METRICS[metric].direction)
        result.update(difference=changes["delta"], ratio=changes["ratio"], percent=changes["percent"],
                      performance_ratio=changes["performance_ratio"])
        if min(len(values["left"]), len(values["right"])) < 3:
            result["reason"] = "insufficient_independent_runs"
        else:
            result["difference_ci"], result["ratio_ci"] = _bootstrap(
                values["left"], values["right"], confidence=confidence, resamples=resamples, seed=seed
            )
            if result["ratio_ci"] is None:
                result["reason"] = "nonpositive_baseline_for_ratio"
    return result


@dataclass(frozen=True)
class _ExperimentSnapshot:
    run_id: str
    result_path: Path
    status: Any
    groups: dict[tuple[str, ...], _RunMetric]
    comparison: dict[str, Any]
    fingerprints: dict[Path, str | None]


def _unique_run_id(run_id: object, seen: set[str]) -> str:
    run_id = recorded_run_id(run_id) if isinstance(run_id, str) else None
    if run_id is None:
        raise ValueError("independent comparison requires recorded run IDs")
    if run_id in seen:
        raise ValueError(f"duplicate independent run: {run_id}")
    return run_id


class ExperimentComparisonSession:
    """Reuse run summaries within one report; each comparison rechecks its sources."""

    def __init__(self) -> None:
        self._snapshots: dict[tuple[Path, tuple[str, ...], bool], _ExperimentSnapshot] = {}

    def _load(self, source: Path, run_ids: set[str], *, metrics: list[str], purpose: str) -> _ExperimentSnapshot:
        key = (source, tuple(metrics), purpose == "resource-scaling")
        if cached := self._snapshots.get(key):
            _unique_run_id(cached.run_id, run_ids)
            return cached
        metadata_snapshot = ResultMetadataSnapshot(source)
        result_snapshot = read_result_csv_snapshot(source)
        fingerprints: dict[Path, str | None] = {}
        _record_experiment_fingerprints(metadata_snapshot, result_snapshot, fingerprints)
        state = metadata_snapshot.read("run_state.json", label="run state").payload
        run_id = _unique_run_id(state.get("run_id"), run_ids)
        current_snapshot = load_comparison_snapshot(source, result_snapshot=result_snapshot,
                                                    metadata_snapshot=metadata_snapshot,
                                                    verify_result_snapshot=False)
        try:
            metadata_snapshot.verify_observed_sources(fingerprints)
        except ValueError as error:
            raise ValueError(f"experiment changed during comparison: {error}") from error
        require_consistent_run_identity(current_snapshot["audit"])
        groups = _collect_run_case_groups(
            result_snapshot.rows, run_id=run_id, complete=state.get("status") == "complete",
            purpose=purpose, metrics=metrics,
        )
        loaded = _ExperimentSnapshot(run_id, result_snapshot.path, state.get("status"), groups,
                                     current_snapshot, fingerprints)
        self._snapshots[key] = loaded
        return loaded

    def compare(self, left, right, *, metrics, purpose="same-hardware",
                confidence=0.95, resamples=5000, seed=0):
        # Reuse the metric/window validation contract; no profiler or lifecycle reuse values.
        validate_window_options(metrics, confidence=confidence, resamples=resamples)
        if not left or not right:
            raise ValueError("both comparison groups require at least one experiment")
        sources = {"left": expand_result_sources(left), "right": expand_result_sources(right)}
        experiments: dict[str, list[dict[str, Any]]] = {}
        run_ids: set[str] = set()
        result_files: set[Path] = set()
        fingerprints: dict[Path, str | None] = {}
        groups: defaultdict[tuple[str, ...], dict[str, list[_RunMetric]]] = defaultdict(
            lambda: {"left": [], "right": []}
        )
        checks: list[dict[str, Any]] = []
        reference_snapshot = None
        for side, paths in sources.items():
            experiments[side] = []
            group_snapshot = None
            for path in paths:
                selected = resolve_result_file(path)
                # An explicit CSV symlink uses the metadata beside the link.
                source = selected.parent.resolve() / selected.name
                loaded = self._load(source, run_ids, metrics=metrics, purpose=purpose)
                run_id = loaded.run_id
                run_ids.add(run_id)
                if loaded.result_path in result_files:
                    raise ValueError(f"duplicate independent result source: {loaded.result_path}")
                result_files.add(loaded.result_path)
                complete = loaded.status == "complete"
                fingerprints.update(loaded.fingerprints)
                # Returned comparison dictionaries must not expose the reusable evidence.
                current_snapshot = deepcopy(loaded.comparison)
                if reference_snapshot is None:
                    reference_snapshot = current_snapshot
                if group_snapshot is None:
                    group_snapshot = current_snapshot
                cross = compare_snapshots(reference_snapshot, current_snapshot, purpose=purpose)
                snapshot = cross["experiments"]["right"]
                evidence = {key: snapshot[key] for key in (*QUALITY_FIELDS, "run_status", "measurement_status")}
                experiments[side].append({"run_id": run_id, "source": str(source),
                                          "state": loaded.status, "eligible": complete, **evidence})
                checks.append({"left_run_id": next(iter(experiments["left"]))["run_id"],
                               "right_run_id": run_id, **cross})
                within = compare_snapshots(group_snapshot, current_snapshot, purpose="same-hardware")
                checks.append({**within, "group": side, "within_group": True, "run_id": run_id})
                if within and within["expected_differences"]:
                    checks.append({"status": "incompatible", "run_id": run_id,
                                   "reason": "replicates_change_experiment_identity",
                                   "differences": within["expected_differences"]})
                for case_metric, run in deepcopy(loaded.groups).items():
                    run.summary.update(evidence)
                    groups[case_metric][side].append(run)
        statuses = {check["status"] for check in checks}
        status = "incompatible" if "incompatible" in statuses else "unknown" if "unknown" in statuses else "compatible"
        output = [
            _summarize_group_metric(
                case_metric, sides, sources=sources,
                status=status, confidence=confidence,
                resamples=resamples, seed=seed,
            )
            for case_metric, sides in sorted(groups.items())
        ]
        for artifact, digest in fingerprints.items():
            _verify_source(artifact, digest)
        return {"schema_version": 1, "kind": "independent_experiment_comparison", "status": status,
                "coordinate_identity": COORDINATE_IDENTITY,
                "allowed_resource_dimensions": ALLOWED_RESOURCE_DIMENSIONS if purpose == "resource-scaling" else [],
                "purpose": purpose, "confidence": confidence, "resamples": resamples, "seed": seed,
                "resampling_unit": "independent_run_mean", "method": "unpaired_percentile_bootstrap",
                "direction": "right_minus_left_and_right_divided_by_left", "groups": output,
                "experiments": experiments, "condition_checks": checks,
                "quality": {side: combine_quality(records) for side, records in experiments.items()},
                "limitations": ["distinct run IDs and independent restarts are required; independence is an experimental assumption",
                                "resource-scaling allows CPU and memory quotas; simultaneous changes do not isolate either effect",
                                "successful windows are averaged equally within a run; run means are then weighted equally",
                                "intervals describe run variability, not continuous isolation, thermal equivalence or model quality",
                                "failed and missing windows are excluded and reported; success-conditioned estimates may be biased"],
                "source_sha256": {str(path): digest for path, digest in fingerprints.items()}}


def compare_experiments(left, right, *, metrics, purpose="same-hardware",
                        confidence=0.95, resamples=5000, seed=0):
    """Compare one pair of groups using a fresh source-reading session."""
    return ExperimentComparisonSession().compare(
        left, right, metrics=metrics, purpose=purpose,
        confidence=confidence, resamples=resamples, seed=seed,
    )
