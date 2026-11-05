"""按测量窗口聚合的不确定性；独立 profiler 与冷启动复用值不参与。"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from concurrent.futures import CancelledError

from acprof.analysis.precision import assess_mean_precision, validate_precision_target
from acprof.results.csv import measurement_key
from acprof.results.metrics import METRICS


def validate_window_options(metrics, *, confidence=0.95, resamples=5000, block_size=1,
                            precision_target=None):
    """Validate shared metric and statistical options without computing a report."""
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
    return precision_target


def _collect_window_cases(rows, *, cancelled, precision_target):
    cases, keys = defaultdict(list), set()
    excluded: defaultdict[tuple[str, ...], int] = defaultdict(int)
    for row in rows:
        if cancelled():
            raise CancelledError()
        key = measurement_key(row)
        environment = row.get("environment_class", "unknown")
        runtime = (
            row.get("runtime_kind", "docker"),
            row.get("runtime_id", ""),
            row.get("resource_scope", "container_cgroup"),
        )
        if (environment, runtime, key) in keys:
            raise ValueError(f"duplicate measurement: {key}")
        keys.add((environment, runtime, key))
        if key[4] == "0":
            case = (*key[:4], environment, *runtime)
            if str(row.get("status", "")).strip().lower() == "ok":
                cases[case].append(row)
            elif precision_target is not None:
                excluded[case] += 1
    return cases, excluded


def _finite_metric_values(ordered, name: str) -> list[float]:
    values = []
    for row in ordered:
        try:
            value = float(row.get(name, "nan"))
        except (ValueError, TypeError):
            value = float("nan")
        if math.isfinite(value):
            values.append(value)
    return values


def _precision_unavailable_reason(
    result: dict,
    values: list[float],
    *,
    excluded_count: int,
    gaps: bool,
    resamples: int,
) -> str:
    if result["reason"]:
        return result["reason"]
    if result["missing_windows"]:
        return "missing_windows"
    if excluded_count:
        return "excluded_windows"
    if gaps:
        return "nonconsecutive_windows"
    if any(value < 0 for value in values):
        return "negative_values"
    if resamples < 2:
        return "insufficient_resamples"
    if (
        result["ci_low"] is not None
        and result["ci_low"] == result["ci_high"]
        and any(value != values[0] for value in values)
    ):
        return "degenerate_interval"
    return ""


def _summarize_case_metric(
    case: tuple[str, ...],
    ordered: list[dict],
    name: str,
    *,
    gaps: bool,
    excluded_count: int,
    confidence: float,
    resamples: int,
    seed: int,
    block_size: int,
    include_intervals: bool,
    precision_target: float | None,
) -> dict:
    values = _finite_metric_values(ordered, name)
    count = len(values)
    mean_val: float | None = statistics.fmean(values) if values else None
    ci_low: float | None = None
    ci_high: float | None = None
    result: dict[str, object] = {
        "cpu_cores": None if case[0] == "inherited" else float(case[0]),
        "mem_cap_gb": None if case[1] == "inherited" else float(case[1]),
        "gpu_mode": case[2],
        "environment_class": case[4],
        "runtime_kind": case[5],
        "runtime_id": case[6],
        "resource_scope": case[7],
        "input_scale": float(case[3]),
        "metric": name,
        "unit": METRICS[name].unit,
        "n_windows": count,
        "missing_windows": len(ordered) - count,
        "mean": mean_val,
        "std": statistics.stdev(values) if count > 1 else None,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "reason": "",
    }
    if count < max(3, 3 * block_size):
        result["reason"] = "insufficient_windows"
    elif block_size > 1 and count != len(ordered):
        result["reason"] = "missing_windows_break_blocks"
    elif block_size > 1 and gaps:
        result["reason"] = "nonconsecutive_windows"
    elif include_intervals:
        ci_low, ci_high = bootstrap_mean_interval(
            values, confidence=confidence, resamples=resamples, seed=seed, block_size=block_size
        )
        result["ci_low"], result["ci_high"] = ci_low, ci_high
    if precision_target is not None:
        reason = _precision_unavailable_reason(
            result, values, excluded_count=excluded_count, gaps=gaps, resamples=resamples
        )
        result.update(assess_mean_precision(
            mean_val, ci_low, ci_high,
            target=precision_target, unavailable_reason=reason,
        ))
        result["precision_excluded_windows"] = excluded_count
    return result


def summarize_windows(rows, metrics, *, confidence=0.95, resamples=5000, seed=0, block_size=1,
                      include_intervals=True, cancelled=lambda: False, precision_target=None):
    precision_target = validate_window_options(metrics, confidence=confidence, resamples=resamples,
                                              block_size=block_size, precision_target=precision_target)
    cases, excluded = _collect_window_cases(rows, cancelled=cancelled, precision_target=precision_target)
    groups = []
    for case in sorted(cases):
        ordered = sorted(cases[case], key=lambda row: float(row["repeat_idx"]))
        gaps = ((block_size > 1 or precision_target is not None)
                and any(float(b["repeat_idx"]) != float(a["repeat_idx"]) + 1
                        for a, b in zip(ordered, ordered[1:], strict=False)))
        for name in metrics:
            if cancelled():
                raise CancelledError()
            groups.append(_summarize_case_metric(
                case, ordered, name,
                gaps=gaps, excluded_count=excluded[case],
                confidence=confidence, resamples=resamples, seed=seed,
                block_size=block_size, include_intervals=include_intervals,
                precision_target=precision_target,
            ))
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

    def block_samples():
        # 每个块沿用一次随机起点；最后一块只贡献剩余窗口。
        for consumed in range(0, count, block_size):
            start = rng.randrange(count)
            for offset in range(min(block_size, count - consumed)):
                yield values[(start + offset) % count]

    estimates = []
    for _ in range(resamples):
        # 循环移动块；block_size=1 即普通 percentile bootstrap。
        sampled = ((values[rng.randrange(count)] for _ in range(count))
                   if block_size == 1 else block_samples())
        # 与 fmean 相同的求和与除法；样本数已知，无需再次计数。
        estimates.append(math.fsum(sampled) / count)
    estimates.sort()

    def quantile(probability):
        index = (len(estimates) - 1) * probability
        lower = math.floor(index)
        fraction = index - lower
        return estimates[lower] * (1 - fraction) + estimates[min(lower + 1, len(estimates) - 1)] * fraction

    tail = (1 - confidence) / 2
    return quantile(tail), quantile(1 - tail)
