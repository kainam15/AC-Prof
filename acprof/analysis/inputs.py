"""Shared scalar input contracts for numerical analysis and plotting."""

import math


def resource_coordinate(value: object) -> float:
    """Keep parsed resource identity; compact integral labels without truncation."""
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"resource coordinate must be finite and positive: {value!r}")
    return int(number) if number.is_integer() else number
