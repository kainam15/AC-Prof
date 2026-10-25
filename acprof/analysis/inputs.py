"""Shared scalar input contracts for numerical analysis and plotting."""

import math


def normalize_gpu_mode(value: object) -> str:
    """Accept the documented analysis aliases, but never guess an unknown mode."""
    normalized = str(value).strip().lower()
    if normalized in {"on", "1", "1.0", "true", "yes", "gpu"}:
        return "on"
    if normalized in {"off", "0", "0.0", "false", "no", "cpu"}:
        return "off"
    raise ValueError(f"unsupported gpu_mode value: {value!r}")


def resource_coordinate(value: object) -> float:
    """Keep parsed resource identity; compact integral labels without truncation."""
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"resource coordinate must be finite and positive: {value!r}")
    return int(number) if number.is_integer() else number
