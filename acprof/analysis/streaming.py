from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

STREAMING_LATENCY_METRICS = ("ttft_s", "tpot_s", "itl_s", "output_e2e_s", "e2e_s")


def distribution(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)

    def percentile(p: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * p
        low = math.floor(position)
        return ordered[low] + (
            ordered[min(low + 1, len(ordered) - 1)] - ordered[low]
        ) * (position - low)

    return {
        "mean": statistics.fmean(ordered) if ordered else None,
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "sample_count": len(ordered),
    }


def streaming_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    succeeded = [row for row in records if row["status"] == "ok"]
    summary: dict[str, Any] = {
        "time_unit": "seconds",
        "percentile_method": "linear",
        "excluded_requests": len(records) - len(succeeded),
        "failed_requests": sum(row["status"] == "error" for row in records),
        "cancelled_requests": sum(row["status"] == "cancelled" for row in records),
        "dropped_requests": sum(row["status"] == "dropped" for row in records),
    }
    for name in ("ttft_s", "tpot_s", "output_e2e_s", "e2e_s"):
        values = [row[name] for row in succeeded if row.get(name) is not None]
        summary[name] = {
            **distribution(values),
            "sample_unit": "request",
            "unit": "seconds_per_token" if name == "tpot_s" else "seconds",
            "missing_requests": len(succeeded) - len(values),
        }
    for field, name, unit in (
        ("itl_s", "itl_s", "token_interval"),
        ("output_event_intervals_s", "output_event_interval_s", "output_event_interval"),
    ):
        values = [value for row in succeeded for value in (row.get(field) or [])]
        summary[name] = {
            **distribution(values),
            "sample_unit": unit,
            "unit": "seconds",
            "requests_with_samples": sum(bool(row.get(field)) for row in succeeded),
            "missing_requests": sum(not row.get(field) for row in succeeded),
        }
    summary["output_token_count_sources"] = dict(Counter(
        row.get("output_token_count_source", "unavailable") for row in succeeded
    ))
    summary["token_count_errors"] = sum(bool(row.get("token_count_error")) for row in succeeded)
    summary["errors"] = dict(Counter(
        row.get("error_code", row["status"]) for row in records if row["status"] != "ok"
    ))
    return summary
