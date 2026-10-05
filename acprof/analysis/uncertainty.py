"""按测量窗口聚合的不确定性；独立 profiler 与冷启动复用值不参与。"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from concurrent.futures import CancelledError

from acprof.analysis.precision import assess_mean_precision, validate_precision_target
from acprof.metric_registry import METRICS
from acprof.result_csv import measurement_key


def summarize_windows(rows, metrics, *, confidence=0.95, resamples=5000, seed=0, block_size=1,
                      include_intervals=True, cancelled=lambda: False, precision_target=None):
    if not 0 < confidence < 1 or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("confidence 必须在 0 与 1 之间，resamples 必须为正整数")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size 必须为正整数")
    if precision_target is not None:
        precision_target = validate_precision_target(precision_target)
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("请选择不重复的数值指标")
    for name in metrics:
        metric = METRICS.get(name)
        if metric is None or metric.kind != "number" or metric.window not in {"request_window", "matched_control"}:
            raise ValueError(f"{name} 不属于独立测量窗口；不对复用的 profiler/生命周期值计算区间")
    cases, keys = defaultdict(list), set()
    excluded = defaultdict(int)
    for row in rows:
        if cancelled():
            raise CancelledError()
        key = measurement_key(row)
        environment = row.get("environment_class", "unknown")
        if (environment, key) in keys:
            raise ValueError(f"duplicate measurement: {key}")
        keys.add((environment, key))
        if key[4] == "0":
            case = (*key[:4], environment)
            if str(row.get("status", "")).strip().lower() == "ok":
                cases[case].append(row)
            elif precision_target is not None:
                excluded[case] += 1
    groups = []
    for case in sorted(cases):
        ordered = sorted(cases[case], key=lambda row: float(row["repeat_idx"]))
        gaps = ((block_size > 1 or precision_target is not None)
                and any(float(b["repeat_idx"]) != float(a["repeat_idx"]) + 1
                        for a, b in zip(ordered, ordered[1:])))
        for name in metrics:
            if cancelled():
                raise CancelledError()
            values = []
            for row in ordered:
                try:
                    value = float(row.get(name, "nan"))
                except (ValueError, TypeError):
                    value = float("nan")
                if math.isfinite(value):
                    values.append(value)
            count = len(values)
            result = {"cpu_cores": float(case[0]), "mem_cap_gb": float(case[1]), "gpu_mode": case[2],
                      "environment_class": case[4],
                      "input_scale": float(case[3]), "metric": name, "unit": METRICS[name].unit,
                      "n_windows": count, "missing_windows": len(ordered) - count,
                      "mean": statistics.fmean(values) if values else None,
                      "std": statistics.stdev(values) if count > 1 else None,
                      "ci_low": None, "ci_high": None, "reason": ""}
            if count < max(3, 3 * block_size):
                result["reason"] = "insufficient_windows"
            elif block_size > 1 and count != len(ordered):
                result["reason"] = "missing_windows_break_blocks"
            elif block_size > 1 and gaps:
                result["reason"] = "nonconsecutive_windows"
            elif include_intervals:
                result["ci_low"], result["ci_high"] = bootstrap_mean_interval(
                    values, confidence=confidence, resamples=resamples, seed=seed, block_size=block_size)
            if precision_target is not None:
                reason = result["reason"]
                if not reason:
                    if result["missing_windows"]:
                        reason = "missing_windows"
                    elif excluded[case]:
                        reason = "excluded_windows"
                    elif gaps:
                        reason = "nonconsecutive_windows"
                    elif any(value < 0 for value in values):
                        reason = "negative_values"
                    elif resamples < 2:
                        reason = "insufficient_resamples"
                    elif (result["ci_low"] is not None and result["ci_low"] == result["ci_high"]
                          and any(value != values[0] for value in values)):
                        reason = "degenerate_interval"
                result.update(assess_mean_precision(
                    result["mean"], result["ci_low"], result["ci_high"],
                    target=precision_target, unavailable_reason=reason))
                result["precision_excluded_windows"] = excluded[case]
            groups.append(result)
    report = {"schema_version": 1, "confidence": confidence, "resamples": resamples, "seed": seed,
              "block_size": block_size, "method": "circular_block_percentile_bootstrap",
              "resampling_unit": "csv_request_window", "filter": "status=ok and warmup=0",
              "assumption": "窗口（或选定连续块）之间可视为独立；区间仅描述本实验内变异", "groups": groups}
    if precision_target is not None:
        report["precision"] = {
            "target_relative_half_width": precision_target, "unit": "ratio",
            "formula": "(ci_high - ci_low) / (2 * mean)",
            "scope": "within_run_observed_windows",
            "interpretation": (
                "Descriptive user-selected interval-width target under the existing bootstrap "
                "assumptions; not experiment completeness, independent-run evidence or a stopping rule."
            ),
        }
    return report


def bootstrap_mean_interval(values, *, confidence=0.95, resamples=5000, seed=0, block_size=1):
    """有限窗口均值的 percentile 区间；不把单窗口当作可估计区间。"""
    if not 0 < confidence < 1 or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("invalid bootstrap settings")
    if not isinstance(block_size, int) or block_size < 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("invalid bootstrap values or block size")
    count = len(values)
    if count < max(3, 3 * block_size):
        return None, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        # 循环移动块；block_size=1 即普通 percentile bootstrap。
        sampled = []
        while len(sampled) < count:
            start = rng.randrange(count)
            sampled.extend(values[(start + offset) % count] for offset in range(block_size))
        estimates.append(statistics.fmean(sampled[:count]))
    estimates.sort()

    def quantile(probability):
        index = (len(estimates) - 1) * probability
        lower = math.floor(index)
        fraction = index - lower
        return estimates[lower] * (1 - fraction) + estimates[min(lower + 1, len(estimates) - 1)] * fraction

    tail = (1 - confidence) / 2
    return quantile(tail), quantile(1 - tail)
