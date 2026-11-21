"""按测量窗口聚合的不确定性；独立 profiler 与冷启动复用值不参与。"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from decimal import MAX_EMAX, MIN_EMIN, ROUND_DOWN, Context, Decimal
from operator import itemgetter

from acprof.analysis.audit import number
from acprof.analysis.precision import assess_mean_precision, validate_precision_target
from acprof.cancellation import cancellable, check_cancelled
from acprof.results.csv import (
    is_measured_success,
    measurement_key,
    recorded_run_id,
    require_matching_run_id,
    run_id_column,
)
from acprof.results.measurement_plan import decimal_coordinate, decimal_key_text
from acprof.results.metrics import METRICS
from acprof.results.profile import USABLE_QUALITIES, metric_observation, observation_signature

WINDOW_FILTER = "status=ok and warmup=0 and result_origin!=inferred_not_measured"
_ORDINAL_GAP = Context(prec=1, rounding=ROUND_DOWN, Emin=MIN_EMIN, Emax=MAX_EMAX)
_MAX_SAFE_INTEGER = 2**53 - 1
STATISTIC_AXES = ("cpu_cores", "mem_cap_gb", "input_scale")


def ordered_windows(rows: Iterable[dict]) -> list[tuple[Decimal, dict]]:
    """Pair validated ordinals with their rows for exact numeric ordering."""
    return sorted(((decimal_coordinate(row["repeat_idx"]), row) for row in rows), key=itemgetter(0))


def portable_repeat_index(index: Decimal) -> float | str:
    """Preserve the original JSON encoding within the binary64 safe integer range."""
    return float(index) if index <= _MAX_SAFE_INTEGER else decimal_key_text(index)


def window_case_key(row: dict, key: tuple[str, ...]) -> tuple[str, ...]:
    return (*key[:4], row.get("environment_class", "unknown"), row.get("runtime_kind", "docker"),
            row.get("runtime_id", ""), row.get("resource_scope", "container_cgroup"))


def _case_coordinates(case: tuple[str, ...]) -> dict:
    result: dict = {"cpu_cores": None, "mem_cap_gb": None, "gpu_mode": case[2],
                    "environment_class": case[4], "runtime_kind": case[5], "runtime_id": case[6],
                    "resource_scope": case[7], "input_scale": None}
    exact = {}
    for axis, raw in zip(STATISTIC_AXES, (case[0], case[1], case[3]), strict=True):
        if raw != "inherited":
            display = float(raw)
            result[axis] = display if math.isfinite(display) else None
            if decimal_coordinate(raw) != decimal_coordinate(str(display)):
                exact[axis] = raw
    if exact:
        result["exact_coordinates"] = exact
    return result


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
    run_id = None
    identity_column = "run_id"
    for line, row in enumerate(cancellable(rows, cancelled), 2):
        if line == 2:
            identity_column = run_id_column(row) or "run_id"
        if observed := recorded_run_id(row.get(identity_column)):
            if run_id is not None and run_id != observed:
                require_matching_run_id(run_id, observed, source=f"within-run window statistics row {line}")
            run_id = observed
        key = measurement_key(row)
        case = window_case_key(row, key)
        identity = (*case, *key[4:])
        if identity in keys:
            raise ValueError(f"duplicate measurement: {key}")
        keys.add(identity)
        if key[4] == "0":
            if is_measured_success(row):
                cases[case].append(row)
            elif precision_target is not None:
                excluded[case] += 1
    return cases, excluded


def finite_metric_values(ordered, name: str) -> list[float]:
    """Collect usable readings for a registered metric with one observation contract."""
    values = []
    signatures = set()
    for row in ordered:
        value = number(row.get(name))
        if value is None:
            continue
        observation = metric_observation(row, name)
        if observation['quality'] in USABLE_QUALITIES:
            values.append(value)
            signatures.add(observation_signature(observation))
    return values if len(signatures) <= 1 else []


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
    coordinates: dict,
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
    cancelled: Callable[[], bool] | None,
) -> dict:
    values = finite_metric_values(cancellable(ordered, cancelled), name)
    count = len(values)
    mean_val: float | None = statistics.fmean(values) if values else None
    ci_low: float | None = None
    ci_high: float | None = None
    result: dict[str, object] = {
        **coordinates,
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
                      include_intervals=True, cancelled=None, precision_target=None):
    precision_target = validate_window_options(metrics, confidence=confidence, resamples=resamples,
                                              block_size=block_size, precision_target=precision_target)
    cases, excluded = _collect_window_cases(rows, cancelled=cancelled, precision_target=precision_target)
    groups = []
    # Excluded formal windows still identify a case for optional precision assessment.
    for case in cancellable(sorted(cases.keys() | excluded.keys()), cancelled):
        indexed = ordered_windows(cancellable(cases[case], cancelled))
        ordered = [row for _, row in indexed]
        coordinates = _case_coordinates(case)
        if indexed and indexed[-1][0] > _MAX_SAFE_INTEGER:
            coordinates["repeat_index_order"] = "decimal_v1"
        # A positive integer gap rounds down to one significant digit as 1 only
        # when it equals 1. This avoids expanding compact ordinals such as 1E+100000000.
        gaps = ((block_size > 1 or precision_target is not None)
                and any(_ORDINAL_GAP.subtract(b, a) != 1
                        for (a, _), (b, _) in zip(indexed, indexed[1:], strict=False)))
        for name in metrics:
            check_cancelled(cancelled)
            groups.append(_summarize_case_metric(
                coordinates, ordered, name,
                gaps=gaps, excluded_count=excluded[case],
                confidence=confidence, resamples=resamples, seed=seed,
                block_size=block_size, include_intervals=include_intervals,
                precision_target=precision_target, cancelled=cancelled,
            ))
    report = {"schema_version": 1, "confidence": confidence, "resamples": resamples, "seed": seed,
              "block_size": block_size, "method": "circular_block_percentile_bootstrap",
              "resampling_unit": "csv_request_window", "filter": WINDOW_FILTER,
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
    check_cancelled(cancelled)
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
