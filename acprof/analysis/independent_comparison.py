"""Compare run means, resampling independent runs instead of individual requests."""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from acprof.analysis.audit import number
from acprof.analysis.comparison import compare_results
from acprof.analysis.conditions import ALLOWED_RESOURCE_DIMENSIONS
from acprof.analysis.uncertainty import summarize_windows
from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import file_sha256, read_json_object
from acprof.hardware_conditions import conditions_path
from acprof.metric_registry import METRICS
from acprof.quality import QUALITY_FIELDS, combine_quality
from acprof.result_csv import measurement_key, read_result_csv


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


def compare_experiments(left, right, *, metrics, purpose="same-hardware",
                        confidence=0.95, resamples=5000, seed=0):
    # Reuse the metric/window validation contract; no profiler or lifecycle reuse values.
    summarize_windows([], metrics, confidence=confidence, resamples=resamples, seed=seed)
    if not left or not right:
        raise ValueError("both comparison groups require at least one experiment")
    sources = {"left": list(left), "right": list(right)}
    experiments, run_ids, fingerprints = {}, set(), {}
    groups = defaultdict(lambda: {"left": [], "right": []})
    checks = []
    for side, paths in sources.items():
        experiments[side] = []
        for path in paths:
            source = Path(path)
            layout = ArtifactLayout.discover(source) if source.is_dir() else ArtifactLayout.from_csv(source)
            state = read_json_object(layout.path("run_state.json"), label="run state")
            run_id = state.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("independent comparison requires recorded run IDs")
            if run_id in run_ids:
                raise ValueError(f"duplicate independent run: {run_id}")
            run_ids.add(run_id)
            # Detect concurrent changes to the rows and their condition evidence.
            for name in ("result_layers.json", "run_state.json", "input_scale_plan.json",
                         "static_meta.json", "hardware_conditions.json", "quality_checks.json", "runtime_validation.json"):
                artifact = (layout.result_csv if name == "result_layers.json" else
                            conditions_path(layout) if name == "hardware_conditions.json" else layout.path(name))
                fingerprints[artifact] = file_sha256(artifact) if artifact.exists() else None
            _, rows = read_result_csv(layout.result_csv)
            summarize_windows(rows, metrics, confidence=confidence, resamples=1)
            complete = state.get("status") == "complete"
            cross = compare_results(left[0], path, purpose=purpose)
            snapshot = cross["experiments"]["right"]
            evidence = {key: snapshot[key] for key in (*QUALITY_FIELDS, "run_status", "measurement_status")}
            experiments[side].append({"run_id": run_id, "source": str(source.resolve()),
                                      "state": state.get("status"), "eligible": complete, **evidence})
            checks.append({"left_run_id": next(iter(experiments["left"]))["run_id"],
                           "right_run_id": run_id, **cross})
            within = compare_results(paths[0], path, purpose="same-hardware")
            checks.append({**within, "group": side, "within_group": True, "run_id": run_id})
            if within and within["expected_differences"]:
                checks.append({"status": "incompatible", "run_id": run_id,
                               "reason": "replicates_change_experiment_identity",
                               "differences": within["expected_differences"]})
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
                for metric in metrics:
                    window_values: list[float] = []
                    for row in windows:
                        value = number(row.get(metric))
                        if row.get("status") == "ok" and value is not None:
                            window_values.append(value)
                    groups[(*comparison_case, metric)][side].append({
                        "run_id": run_id, "mean": statistics.fmean(window_values) if window_values and complete else None,
                        "resource_coordinates": {"cpu_cores": float(case[0]), "mem_cap_gb": float(case[1])},
                        "valid_windows": len(window_values), "failed_windows": sum(row.get("status") != "ok" for row in windows),
                        "missing_windows": sum(row.get("status") == "ok" and number(row.get(metric)) is None for row in windows),
                        "eligible": complete, **evidence,
                    })
    statuses = {check["status"] for check in checks}
    status = "incompatible" if "incompatible" in statuses else "unknown" if "unknown" in statuses else "compatible"
    output = []
    for (cpu, mem, gpu, scale, metric), sides in sorted(groups.items()):
        coordinates = {side: records[0]["resource_coordinates"] if records else None for side, records in sides.items()}
        baseline_coordinates = coordinates["left"] or coordinates["right"]
        if purpose == "resource-scaling":
            cpu, mem = baseline_coordinates["cpu_cores"], baseline_coordinates["mem_cap_gb"]
        result: dict[str, Any] = {"cpu_cores": float(cpu), "mem_cap_gb": float(mem), "gpu_mode": gpu,
                  "resource_coordinates": coordinates,
                  "input_scale": float(scale), "metric": metric, "unit": METRICS[metric].unit,
                  "difference": None, "ratio": None, "difference_ci": None, "ratio_ci": None, "reason": ""}
        blocked = any(check.get("metric_comparability", {}).get(metric, {}).get("status") == "not comparable"
                      for check in checks)
        result["comparability"] = "not comparable" if blocked or status != "compatible" else "comparable"
        values = {}
        for side in ("left", "right"):
            records = sides[side]
            values[side] = [record["mean"] for record in records if record["mean"] is not None]
            result[side] = {**_aggregate_evidence(records), "n_runs": len(values[side]), "requested_runs": len(sources[side]),
                            "missing_runs": len(sources[side]) - len(values[side]),
                            "mean": statistics.fmean(values[side]) if values[side] else None,
                            "std": statistics.stdev(values[side]) if len(values[side]) > 1 else None,
                            "failed_windows": sum(record["failed_windows"] for record in records),
                            "missing_windows": sum(record["missing_windows"] for record in records), "runs": records}
        if blocked:
            result["reason"] = "not comparable"
        elif status != "compatible":
            result["reason"] = "conditions_" + status
        elif not values["left"] or not values["right"]:
            result["reason"] = "missing_comparable_runs"
        else:
            a, b = result["left"]["mean"], result["right"]["mean"]
            result.update(difference=b - a, ratio=b / a if a > 0 else None)
            if min(len(values["left"]), len(values["right"])) < 3:
                result["reason"] = "insufficient_independent_runs"
            else:
                result["difference_ci"], result["ratio_ci"] = _bootstrap(
                    values["left"], values["right"], confidence=confidence, resamples=resamples, seed=seed)
                if result["ratio_ci"] is None:
                    result["reason"] = "nonpositive_baseline_for_ratio"
        output.append(result)
    for artifact, digest in fingerprints.items():
        current = file_sha256(artifact) if artifact.exists() else None
        if current != digest:
            raise ValueError(f"experiment changed during comparison: {artifact}")
    return {"schema_version": 1, "kind": "independent_experiment_comparison", "status": status,
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
