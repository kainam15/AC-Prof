"""Request-window latency statistics shared by the HTTP client and packet merge."""
from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable


def latency_distribution_metrics(
    prefix: str, latencies: Iterable[float], *, slow_latency_threshold_s: float | None = None,
) -> dict[str, float]:
    values = [value for value in latencies if math.isfinite(value)]
    count = len(values)
    # Keep reductions in acquisition order before sorting the private sample list.
    mean = sum(values) / count if count else math.nan
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / count) if count >= 2 else math.nan
    maximum = max(values, default=math.nan)
    values.sort()

    def nearest_rank(percentile: int) -> float:
        return values[math.ceil((percentile / 100.0) * count) - 1] if count else math.nan

    p50, p95 = nearest_rank(50), nearest_rank(95)
    slow_ratio = math.nan
    if (count and slow_latency_threshold_s is not None
            and math.isfinite(slow_latency_threshold_s) and slow_latency_threshold_s > 0):
        slow_ratio = (count - bisect_right(values, slow_latency_threshold_s)) / count
    return {
        f"{prefix}_request_count": float(count),
        f"{prefix}_p50_s": p50,
        f"{prefix}_p90_s": nearest_rank(90),
        f"{prefix}_p95_s": p95,
        f"{prefix}_tail_ratio": p95 / p50 if p50 > 0 else math.nan,
        f"{prefix}_std_s": std,
        f"{prefix}_cv": std / mean if math.isfinite(std) and math.isfinite(mean) and mean > 0 else math.nan,
        f"{prefix}_iqr_s": nearest_rank(75) - nearest_rank(25) if count >= 2 else math.nan,
        f"{prefix}_max_s": maximum,
        f"{prefix}_slow_ratio": slow_ratio,
    }
