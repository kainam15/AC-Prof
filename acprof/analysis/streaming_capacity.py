"""Pure single-point capacity aggregation for streaming load results."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from acprof.analysis.streaming import nonnegative_latency

SLO_METRICS = ("ttft_s", "tpot_s", "e2e_s")
_REQUEST_STATUSES = {"ok", "error", "dropped", "cancelled"}


def normalize_slo_thresholds(thresholds: Mapping[str, float | None] | None) -> dict[str, float | None]:
    values = dict(thresholds or {})
    if set(values) - set(SLO_METRICS):
        raise ValueError("unsupported SLO metric")
    normalized: dict[str, float | None] = {}
    for name in SLO_METRICS:
        value = values.get(name)
        if value is None:
            normalized[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} SLO threshold must be finite and positive")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} SLO threshold must be finite and positive")
        normalized[name] = float(value)
    return normalized


def _duration(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("capacity duration_s must be finite and positive")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("capacity duration_s must be finite and positive")
    return float(value)


def _token_count(row: Mapping[str, Any], field: str) -> int | None:
    value = row.get(field)
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{field} must be a nonnegative integer or null")
    return value


def _slo_summary(
    succeeded: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float | None], duration_s: float,
) -> tuple[dict[str, Any], float | None]:
    configured = {name: value for name, value in thresholds.items() if value is not None}
    counts = {"satisfied": 0, "violated": 0, "unevaluable": 0}
    per_metric: dict[str, dict[str, Any]] = {
        name: {
            "threshold_s": thresholds[name],
            "configured": thresholds[name] is not None,
            "satisfied": 0,
            "violated": 0,
            "unevaluable": 0,
        }
        for name in SLO_METRICS
    }
    for row in succeeded:
        violated = False
        unevaluable = False
        for name, threshold in configured.items():
            raw = row.get(name)
            value = nonnegative_latency(raw, name) if raw is not None else None
            if value is None:
                unevaluable = True
                per_metric[name]["unevaluable"] += 1
            elif value > threshold:
                violated = True
                per_metric[name]["violated"] += 1
            else:
                per_metric[name]["satisfied"] += 1
        if not configured:
            continue
        status = "violated" if violated else "unevaluable" if unevaluable else "satisfied"
        counts[status] += 1
    summary = {
        "configured": bool(configured),
        "thresholds_s": dict(thresholds),
        "counts": counts,
        "per_metric": per_metric,
    }
    goodput = counts["satisfied"] / duration_s if configured else None
    return summary, goodput


def _token_component(
    succeeded: Sequence[Mapping[str, Any]], duration_s: float, field: str,
) -> dict[str, Any]:
    values = [_token_count(row, field) for row in succeeded]
    known = [value for value in values if value is not None]
    covered = len(known)
    complete = bool(succeeded) and covered == len(succeeded)
    observed = sum(known) if covered else None
    return {
        "tokens": sum(known) if complete else None,
        "tokens_per_s": sum(known) / duration_s if complete else None,
        "observed_tokens": observed,
        "observed_tokens_per_s": observed / duration_s if observed is not None else None,
        "covered_requests": covered,
        "successful_requests": len(succeeded),
        "coverage_ratio": covered / len(succeeded) if succeeded else None,
    }


def _total_token_component(
    succeeded: Sequence[Mapping[str, Any]], duration_s: float,
) -> dict[str, Any]:
    pairs = [
        (input_tokens, output_tokens)
        for row in succeeded
        if (input_tokens := _token_count(row, "input_tokens")) is not None
        and (output_tokens := _token_count(row, "output_tokens")) is not None
    ]
    known = [input_tokens + output_tokens for input_tokens, output_tokens in pairs]
    covered = len(known)
    complete = bool(succeeded) and covered == len(succeeded)
    observed = sum(known) if covered else None
    return {
        "tokens": sum(known) if complete else None,
        "tokens_per_s": sum(known) / duration_s if complete else None,
        "observed_tokens": observed,
        "observed_tokens_per_s": observed / duration_s if observed is not None else None,
        "covered_requests": covered,
        "successful_requests": len(succeeded),
        "coverage_ratio": covered / len(succeeded) if succeeded else None,
    }


def _source_counts(succeeded: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    values = []
    for row in succeeded:
        value = row.get(field, "unavailable")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be nonempty text")
        values.append(value)
    return dict(Counter(values))


def _total_scope_counts(succeeded: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    values = []
    for row in succeeded:
        input_scope = row.get("input_token_count_scope", "unavailable")
        output_scope = row.get("token_count_scope", "unavailable")
        if any(not isinstance(value, str) or not value for value in (input_scope, output_scope)):
            raise ValueError("total token scope components must be nonempty text")
        values.append(f"{input_scope}+{output_scope}")
    return dict(Counter(values))


def build_capacity(
    requests: Sequence[Mapping[str, Any]], duration_s: Any, client_concurrency: int,
    slo_thresholds: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    duration = _duration(duration_s)
    if type(client_concurrency) is not int or client_concurrency < 1:
        raise ValueError("client_concurrency must be a positive integer")
    if any(not isinstance(row, Mapping) or row.get("status") not in _REQUEST_STATUSES for row in requests):
        raise ValueError("capacity requests must have a known terminal status")
    counts = Counter(row["status"] for row in requests)
    succeeded = [row for row in requests if row["status"] == "ok"]
    thresholds = normalize_slo_thresholds(slo_thresholds)
    slo, goodput = _slo_summary(succeeded, thresholds, duration)
    return {
        "schema_version": 1,
        "duration_s": duration,
        "client_concurrency": client_concurrency,
        "counts": {
            "offered": len(requests),
            "sent": sum(row.get("sent_s") is not None for row in requests),
            "succeeded": len(succeeded),
            "failed": counts["error"],
            "dropped": counts["dropped"],
            "cancelled": counts["cancelled"],
        },
        "successful_requests_per_s": len(succeeded) / duration,
        "goodput_requests_per_s": goodput,
        "slo": slo,
        "token_throughput": {
            "input": _token_component(succeeded, duration, "input_tokens"),
            "output": _token_component(succeeded, duration, "output_tokens"),
            "total": _total_token_component(succeeded, duration),
            "scope_request_counts": {
                "input": _source_counts(succeeded, "input_token_count_scope"),
                "output": _source_counts(succeeded, "token_count_scope"),
                "total": _total_scope_counts(succeeded),
            },
            "source_request_counts": {
                "input": _source_counts(succeeded, "input_token_count_source"),
                "output": _source_counts(succeeded, "output_token_count_source"),
            },
        },
    }
