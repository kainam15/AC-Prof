"""Optional descriptive precision of an existing within-run mean interval."""
from __future__ import annotations

import math


def validate_precision_target(target: float) -> float:
    """A target is a finite positive ratio, not a percentage or a stopping rule."""
    try:
        valid = (not isinstance(target, bool) and isinstance(target, (int, float))
                 and math.isfinite(target) and target > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("precision_target must be a finite positive ratio (0.05 = 5%)")
    return float(target)


def assess_mean_precision(mean: float | None, ci_low: float | None, ci_high: float | None, *,
                          target: float, unavailable_reason: str = "") -> dict[str, str | float | None]:
    """Assess interval width only; callers supply data-validity exclusions.

    The interval and mean retain the original metric unit. Their ratio is
    dimensionless. It is half the interval width, not the larger distance from
    an asymmetric interval endpoint to the mean.
    """
    target = validate_precision_target(target)
    result: dict[str, str | float | None] = {
        "precision_status": "not_assessable", "relative_ci_half_width": None,
        "precision_reason": unavailable_reason,
    }
    if unavailable_reason:
        return result
    if mean is None or not math.isfinite(mean):
        result["precision_reason"] = "mean_unavailable"
    elif mean <= 0:
        result["precision_reason"] = "nonpositive_mean"
    elif ci_low is None or ci_high is None:
        result["precision_reason"] = "interval_unavailable"
    elif not math.isfinite(ci_low) or not math.isfinite(ci_high) or ci_low > ci_high:
        result["precision_reason"] = "invalid_interval"
    else:
        # Halve before subtracting so finite opposite-sign endpoints cannot
        # overflow the interval width before a representable ratio is computed.
        ratio = (ci_high / 2 - ci_low / 2) / mean
        if not math.isfinite(ratio):
            result["precision_reason"] = "nonfinite_relative_width"
        else:
            result.update(precision_status="met" if ratio <= target else "not_met",
                          relative_ci_half_width=ratio)
    return result
