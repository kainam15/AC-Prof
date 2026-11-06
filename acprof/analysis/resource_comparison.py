"""Descriptive baseline values computed in Python for every offline consumer."""
from __future__ import annotations

from acprof.analysis.conditions import (
    PURPOSES,
    canonical,
    compare_profiles,
    compare_workload_profiles,
)
from acprof.analysis.model import AnalysisModel, number
from acprof.results.metrics import ANALYSIS_METRICS

REPORT_PURPOSES = ("within-experiment", *PURPOSES)


def metric_change(reference, current, *, direction="neutral") -> dict:
    """Right minus/divided by left; a performance ratio above one is better."""
    a, b = number(reference), number(current)
    result = {"delta": None, "percent": None, "ratio": None, "performance_ratio": None, "reason": "missing"}
    if a is None or b is None:
        return result
    result.update(delta=number(b - a), percent=number(100 * ((b - a) / abs(a))) if a else None,
                  ratio=number(b / a) if a > 0 else None,
                  performance_ratio=(number(a / b) if direction == "lower" and a > 0 and b > 0 else
                                     number(b / a) if direction == "higher" and a > 0 and b >= 0 else None),
                  reason="" if a > 0 else "nonpositive_baseline_for_ratio")
    return result


def resource_qualification(left: dict, right: dict) -> dict:
    """Same-source descriptions tolerate absent legacy evidence, never known conflicts."""
    reasons = [name for name in ("source", "run_id", "model", "runtime", "task", "input_case",
                                "concurrency", "environment_class", "runtime_kind", "runtime_id", "resource_scope")
               if left.get(name) != right.get(name)]
    if left.get("input_scale") is None or right.get("input_scale") is None:
        reasons.append("input_scale_missing")
    profiles = [config["comparison_profiles"]["resource-scaling"] for config in (left, right)]
    checks = compare_profiles(*profiles)["checks"]
    # Resource/device changes are the subject of this description. They do not
    # prove equivalent hardware or establish an isolated causal effect.
    reasons.extend(name for name, status in checks.items() if status == "incompatible"
                   and not name.startswith(("hardware_", "runtime_threads", "resources.", "actual_workload", "invalid_artifacts")))
    workloads = [profile.get("workload") for profile in profiles]
    if all(workload and len(workload) == 1 for workload in workloads):
        normalized = [{"input": next(iter(workload.values()))} for workload in workloads]
        if compare_workload_profiles(*normalized) == "incompatible":
            reasons.append("actual_workload")
    changes = [name for name in ("cpu", "memory") if left[name] != right[name]]
    if (left["device"], left["gpu"]) != (right["device"], right["gpu"]):
        changes.append("gpu")
    return {"status": "incompatible" if reasons else "descriptive", "reasons": sorted(set(reasons)),
            "changed_resources": changes, "causal_attribution": False,
            "scope": "within_source_successful_windows; not_independent_runs"}


def build_baseline_comparisons(model: AnalysisModel) -> dict:
    """Reuse summaries and purpose profiles; never re-estimate window statistics."""
    observed = {entry["config_id"] for entry in model.raw_rows}
    configs = [config for config in model.configs if config["config_id"] in observed]
    result = {}
    qualifications = {}
    verdicts: list[dict] = []
    verdict_ids: dict[str, int] = {}

    def intern(verdict):
        key = canonical(verdict)
        if key not in verdict_ids:
            verdict_ids[key] = len(verdicts)
            verdicts.append(verdict)
        return verdict_ids[key]
    for baseline in configs:
        pairs = {}
        for config in configs:
            resource = resource_qualification(baseline, config)
            policies = {}
            for purpose in PURPOSES:
                left, right = (item["comparison_profiles"][purpose] for item in (baseline, config))
                key = (purpose, left["cohort"], right["cohort"])
                if key not in qualifications:
                    check = compare_profiles(left, right)
                    qualifications[key] = {name: check[name] for name in ("status", "reasons")}
                policies[purpose] = qualifications[key]
            comparable = resource["status"] == "descriptive" or any(
                check["status"] == "compatible" for check in policies.values())
            metrics = {}
            if comparable:
                for name, metric in config["metrics"].items():
                    reference = baseline["metrics"][name]["value"]
                    if reference is not None and metric["value"] is not None:
                        metrics[name] = metric_change(reference, metric["value"], direction=ANALYSIS_METRICS[name].direction)
            pairs[config["config_id"]] = {"resource": intern(resource),
                                         "policies": {name: intern(check) for name, check in policies.items()},
                                         "metrics": metrics}
        result[baseline["config_id"]] = pairs
    return {"pairs": result, "qualifications": verdicts}
