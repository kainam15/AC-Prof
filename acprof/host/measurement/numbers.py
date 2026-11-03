"""Numeric conversion shared by request reduction and result serialization."""
from __future__ import annotations

import math
from typing import Any


def resource_text(value: int | None) -> str:
    return "nan" if value is None else str(value)


def to_float_or_nan(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "nan"
    return f"{value:.6f}"


def finite_positive(value: Any) -> bool:
    number = to_float_or_nan(value)
    return math.isfinite(number) and number > 0.0


def per_positive_denominator(value: Any, denominator: Any) -> float:
    numerator = to_float_or_nan(value)
    divisor = to_float_or_nan(denominator)
    if math.isfinite(numerator) and math.isfinite(divisor) and divisor > 0.0:
        return numerator / divisor
    return float("nan")
