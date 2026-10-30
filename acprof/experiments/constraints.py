"""Pure, lossless numeric constraints shared by configuration entry points."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Literal


class ConstraintViolation(ValueError):
    """A stable reason code; presentation layers provide field-specific copy."""

    def __init__(self, code: Literal["integer", "number", "finite", "minimum"], reason: str):
        self.code = code
        super().__init__(reason)


def integer(value: object, *, allow_text: bool = False, allow_integral_float: bool = True) -> int:
    if isinstance(value, bool):
        raise ConstraintViolation("integer", "must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if allow_integral_float and isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if allow_text and isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ConstraintViolation("integer", "must be an integer") from exc
    raise ConstraintViolation("integer", "must be an integer")


def finite_number(value: object, *, allow_text: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (Integral, float, str) if allow_text else (Integral, float)):
        raise ConstraintViolation("number", "must be a number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ConstraintViolation("number", "must be a finite number") from exc
    if not math.isfinite(number):
        raise ConstraintViolation("finite", "must be finite")
    return number


@dataclass(frozen=True)
class NumericConstraint:
    whole: bool = False
    minimum: int | float = 0
    exclusive: bool = False

    def parse(self, value: object, *, allow_text: bool = False,
              allow_integral_float: bool = True) -> int | float:
        number = (integer(value, allow_text=allow_text, allow_integral_float=allow_integral_float)
                  if self.whole else finite_number(value, allow_text=allow_text))
        if number < self.minimum or self.exclusive and number == self.minimum:
            comparison = "greater than" if self.exclusive else "at least"
            raise ConstraintViolation("minimum", f"must be {comparison} {self.minimum}")
        return number


MEASUREMENT_NUMBERS: dict[str, NumericConstraint] = {
    "batch_size": NumericConstraint(whole=True, minimum=1),
    "warmup": NumericConstraint(whole=True),
    "repeat": NumericConstraint(whole=True),
    "repeat_in_window": NumericConstraint(whole=True),
    "auto_warmup_requests": NumericConstraint(whole=True),
    "repeat_window_seconds": NumericConstraint(exclusive=True),
    "request_timeout_seconds": NumericConstraint(exclusive=True),
    "sample_hz": NumericConstraint(exclusive=True),
    "raw_sample_limit": NumericConstraint(whole=True, minimum=2),
    "idle_seconds": NumericConstraint(),
    "idle_cooldown_seconds": NumericConstraint(),
}

# Direct measurement may intentionally run zero windows for preparation/cleanup.
# A user-requested run must collect at least one window.
RUN_NUMBERS = {**MEASUREMENT_NUMBERS, "repeat": NumericConstraint(whole=True, minimum=1)}
PROFILER_NUMBERS = dict.fromkeys((
    "advisor_repeat", "torch_profiler_repeat", "ncu_repeat", "massif_repeat", "nsys_repeat",
    "compute_profile_cpus", "compute_profile_mem", "massif_reference_cpu", "massif_reference_mem",
    "nsys_reference_cpu", "nsys_reference_mem",
), NumericConstraint(whole=True, minimum=1))
