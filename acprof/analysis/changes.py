"""Finite descriptive changes shared by resource and independent-run reports."""
from acprof.analysis.audit import number


def metric_change(reference, current, *, direction="neutral") -> dict:
    """Right minus/divided by left; a performance ratio above one is better."""
    a, b = number(reference), number(current)
    result = {"delta": None, "percent": None, "ratio": None, "performance_ratio": None, "reason": "missing"}
    if a is None or b is None:
        return result
    result.update(delta=number(b - a), percent=number(100 * ((b - a) / abs(a))) if a else None,
                  ratio=number(b / a) if a > 0 else None,
                  performance_ratio=(number(a / b) if direction == "lower" and a > 0 and b > 0 else
                                     number(b / a) if direction == "higher" and a > 0 and b >= 0 else None),
                  reason="" if a > 0 else "nonpositive_baseline_for_ratio")
    return result
