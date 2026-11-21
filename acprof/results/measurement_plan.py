"""Exact measurement coordinates and coverage of recorded window plans."""
from __future__ import annotations

from collections.abc import Callable, Collection, Sequence, Set
from dataclasses import dataclass
from decimal import (
    MAX_EMAX,
    MAX_PREC,
    MIN_EMIN,
    Context,
    Decimal,
    Inexact,
    InvalidOperation,
    Overflow,
)

from acprof.cancellation import cancellable, check_cancelled
from acprof.experiments.constraints import RUN_NUMBERS, finite_number

MeasurementKey = tuple[str, ...]
# Identity reduction must preserve every digit accepted by the Decimal constructor.
_KEY_CONTEXT = Context(prec=MAX_PREC, Emin=MIN_EMIN, Emax=MAX_EMAX, capitals=1, clamp=0,
                       traps=[InvalidOperation, Inexact, Overflow])


def decimal_coordinate(text: str) -> Decimal:
    """Parse a coordinate independently of the caller's Decimal context."""
    return Decimal(text, context=_KEY_CONTEXT)


def decimal_key_text(number: Decimal) -> str:
    """Canonical spelling of a validated finite measurement coordinate."""
    return "0" if number.is_zero() else _KEY_CONTEXT.to_sci_string(_KEY_CONTEXT.normalize(number))


def _require_unique(values: Sequence[object], field: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field}: duplicate axis values")


def _axis_values(raw: object, field: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field}: expected a non-empty comma-separated list")
    return [value.strip() for value in raw.split(",")]


def _resource_limits(raw: object, field: str) -> tuple[int | None, ...]:
    if raw == "auto":
        return (None,)
    items = _axis_values(raw, field)
    try:
        values = tuple(int(value) for value in items)
    except ValueError as error:
        raise ValueError(f"{field}: expected positive integer limits or auto") from error
    if any(value <= 0 for value in values):
        raise ValueError(f"{field}: expected positive integer limits or auto")
    _require_unique(values, field)
    return values


def _window_count(raw: object, name: str) -> int:
    try:
        return int(RUN_NUMBERS[name].parse(raw, allow_integral_float=False))
    except ValueError as error:
        raise ValueError(f"options.{name}: {error}") from error


def recorded_scales(raw: object) -> tuple[int | float, ...]:
    field = "runtime.planned.scales"
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field}: expected a non-empty list of finite positive numbers")
    for index, value in enumerate(raw):
        try:
            positive = finite_number(value) > 0
        except ValueError as error:
            raise ValueError(f"{field}[{index}]: {error}") from error
        if not positive:
            raise ValueError(f"{field}[{index}]: expected a positive scale")
    # Compare the decimal coordinates that CSV rows can actually distinguish.
    _require_unique(tuple(decimal_coordinate(str(value)) for value in raw), field)
    return tuple(raw)


def _axis_keys(values: Sequence[int | float | None], cancelled: Callable[[], bool] | None) -> set[str]:
    return {"inherited" if value is None else decimal_key_text(decimal_coordinate(str(value)))
            for value in cancellable(values, cancelled)}


@dataclass(frozen=True)
class RecordedMeasurementOptions:
    cpus: tuple[int | None, ...]
    mems: tuple[int | None, ...]
    gpus: tuple[str, ...]
    warmup: int
    repeat: int

    @classmethod
    def parse(cls, options: object) -> RecordedMeasurementOptions:
        if not isinstance(options, dict):
            raise ValueError("options: expected an object containing the recorded measurement options")
        gpus = tuple(_axis_values(options.get("gpus"), "options.gpus"))
        if any(mode not in {"off", "on"} for mode in gpus):
            raise ValueError("options.gpus: expected off or on")
        _require_unique(gpus, "options.gpus")
        return cls(
            cpus=_resource_limits(options.get("cpus"), "options.cpus"),
            mems=_resource_limits(options.get("mems"), "options.mems"),
            gpus=gpus,
            warmup=_window_count(options.get("warmup"), "warmup"),
            repeat=_window_count(options.get("repeat"), "repeat"),
        )

    def coverage(self, scales: Sequence[int | float], actual: Set[MeasurementKey], *,
                 cancelled: Callable[[], bool] | None = None) -> dict[str, int]:
        return RecordedMeasurementPlan(self, tuple(scales)).coverage(actual, cancelled=cancelled)


@dataclass(frozen=True)
class RecordedMeasurementPlan:
    """Validated coordinates for a whole experiment, without enumerating its windows."""

    options: RecordedMeasurementOptions
    scales: tuple[int | float, ...]

    def coverage(self, actual: Collection[MeasurementKey], *,
                 cancelled: Callable[[], bool] | None = None) -> dict[str, int]:
        """Count coverage of distinct keys already checked by the result reader."""
        options = self.options
        cpus = _axis_keys(options.cpus, cancelled)
        mems = _axis_keys(options.mems, cancelled)
        inputs = _axis_keys(self.scales, cancelled)
        expected = len(cpus) * len(mems) * len(options.gpus) * len(inputs) * (options.warmup + options.repeat)
        matched = sum(
            cpu in cpus and mem in mems and gpu in options.gpus and scale in inputs
            and Decimal(index) < (options.warmup if warmup == "1" else options.repeat)
            for cpu, mem, gpu, scale, warmup, index in cancellable(actual, cancelled)
        )
        check_cancelled(cancelled)
        return {"expected": expected, "actual": len(actual), "missing": expected - matched,
                "unexpected": len(actual) - matched}
