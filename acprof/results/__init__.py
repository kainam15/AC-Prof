"""Measurement result contracts: metrics, CSV identity, layers, and artifact paths."""

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation


UNRESOLVED_TIMEOUT_ATTEMPT = "client_request_timeout: planned_request_attempted=unknown"


def is_unresolved_timeout_attempt(error: str) -> bool:
    """Match the leading attribution field in a normalized client timeout diagnostic."""
    return error == UNRESOLVED_TIMEOUT_ATTEMPT or error.startswith(UNRESOLVED_TIMEOUT_ATTEMPT + ";")


def is_inferred_result(row: Mapping[str, object]) -> bool:
    """Identify a recorded placeholder whose planned request was never measured."""
    return str(row.get("result_origin", "")).strip() == "inferred_not_measured"


def is_observed_formal_row(row: Mapping[str, object]) -> bool:
    """Accept formal CSV windows and legacy collection flags, excluding placeholders."""
    if is_inferred_result(row):
        return False
    flag = str(row.get("warmup", "0")).strip().lower()
    if flag in {"0", "false"}:
        return True
    try:
        number = Decimal(flag)
    except InvalidOperation:
        return False
    return number.is_finite() and number.is_zero()
