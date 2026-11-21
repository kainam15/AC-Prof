"""Decode actual-request evidence recorded in a result CSV window."""
from __future__ import annotations

from acprof.artifacts import loads_finite_json, reject_isolated_surrogates, require_schema_version
from acprof.results.csv import MISSING_CSV_TEXT


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
