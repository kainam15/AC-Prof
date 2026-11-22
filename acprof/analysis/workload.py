"""Decode window summaries and aggregate their actual-request evidence per case."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from decimal import InvalidOperation
from fractions import Fraction

from acprof.analysis.conditions import canonical
from acprof.artifacts import loads_finite_json, reject_isolated_surrogates, require_schema_version
from acprof.results.csv import MISSING_CSV_TEXT, measurement_key, result_row_status
from acprof.results.measurement_plan import decimal_coordinate


def _variant_count(variant: object, index: int) -> int:
    path = f"workload_contract.variants[{index}]"
    if not isinstance(variant, dict):
        raise ValueError(f"{path}: expected an object")
    count = variant.get("count")
    if type(count) is not int or count <= 0:
        raise ValueError(f"{path}.count: expected a positive integer")
    contract = variant.get("contract")
    if contract is not None:
        require_schema_version(contract, 1, f"{path}.contract")
        if contract.get("task") is not None and not isinstance(contract["task"], str):
            raise ValueError(f"{path}.contract.task: expected a task name or null")
    return count


def parse_workload_summary(raw: object) -> dict | None:
    """Return a valid summary, including unknown requests, or None for an absent CSV value."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("workload_contract: expected a JSON text cell")
    if raw.strip().lower() in MISSING_CSV_TEXT:
        return None
    try:
        summary = loads_finite_json(raw)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"invalid workload_contract JSON: {error}") from error
    require_schema_version(summary, 1, "workload_contract")
    total, variants = summary.get("request_count"), summary.get("variants")
    if type(total) is not int or total < 0 or not isinstance(variants, list):
        raise ValueError("workload_contract 缺少合法 request_count/variants")
    if sum(_variant_count(item, index) for index, item in enumerate(variants)) != total:
        raise ValueError("workload_contract 的 variants 计数与 request_count 不一致")
    if not raw.isascii() or "\\u" in raw:
        reject_isolated_surrogates(summary)
    return summary


def _window_requests(raw: object) -> int | None:
    """Use an exact whole number from the measured CSV column when a summary is absent."""
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in MISSING_CSV_TEXT):
        return None
    try:
        value = decimal_coordinate(str(raw))
    except InvalidOperation:
        return None
    if value.is_finite() and math.isfinite(value) and value >= 0 and value == value.to_integral_value():
        return int(value)
    return None


def window_request_weight(raw: str | None, projected: float | None) -> float | None:
    """Accept a report weight only if the original CSV count survives numeric projection."""
    if projected is None or projected <= 0 or not raw or not raw.isascii() or "_" in raw:
        return None
    return projected if _window_requests(raw) == projected else None


def _case_evidence(counts: Counter[str], uncounted_windows: int) -> dict:
    observed = sum(counts.values())
    variants = [
        {"contract": json.loads(contract), "count": count,
         "proportion": None if uncounted_windows else str(Fraction(count, observed))}
        for contract, count in sorted(counts.items())
    ]
    case = {"request_count": None if uncounted_windows else observed, "variants": variants}
    if uncounted_windows:
        case.update(observed_request_count=observed, uncounted_windows=uncounted_windows)
    return case


def merge_workload_cases(cases: dict[str, dict]) -> dict:
    """Count requests across resource coordinates while retaining each source case."""
    counts: Counter[str] = Counter()
    uncounted = 0
    for case in cases.values():
        uncounted += case.get("uncounted_windows", 0)
        for variant in case["variants"]:
            counts[canonical(variant["contract"])] += variant["count"]
    return {**_case_evidence(counts, uncounted), "resource_cases": cases}


def actual_workload(rows: list[dict]) -> dict | None:
    """Pool measured windows while retaining unknown contracts and uncounted windows."""
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    uncounted: Counter[str] = Counter()
    for row in rows:
        row_key = measurement_key(row)
        if result_row_status(row) not in {"ok", "warn"} or row_key[4] != "0":
            continue
        key = canonical(row_key[:4])
        summary = parse_workload_summary(row.get("workload_contract"))
        if summary is None:
            completed = _window_requests(row.get("repeat_in_window"))
            if completed is None:
                uncounted[key] += 1
                continue
            case_counts = counts[key]
            if completed:
                case_counts["null"] += completed
            continue
        case_counts = counts[key]
        for variant in summary["variants"]:
            contract = variant.get("contract")
            if contract is not None:
                # Runtime parameters describe backend identity, not per-request work.
                contract = {name: item for name, item in contract.items() if name != "runtime"}
            case_counts[canonical(contract)] += variant["count"]
    if not counts:
        return None
    return {key: _case_evidence(counts[key], uncounted[key]) for key in sorted(counts.keys() | uncounted.keys())}
