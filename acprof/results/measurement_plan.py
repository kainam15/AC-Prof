"""Decode recorded window plans shared by result audit and run recovery."""
from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import dataclass
from decimal import Decimal

from acprof.experiments.constraints import RUN_NUMBERS, finite_number
from acprof.results.csv import MeasurementKey, decimal_key_text


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
    _require_unique(tuple(Decimal(str(value)) for value in raw), field)
    return tuple(raw)


def _axis_keys(values: Sequence[int | float | None]) -> set[str]:
    return {"inherited" if value is None else decimal_key_text(Decimal(str(value))) for value in values}


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

    def coverage(self, scales: Sequence[int | float], actual: Set[MeasurementKey]) -> dict[str, int]:
        """Count coverage using validated scales and distinct keys produced by measurement_key."""
        cpus, mems, inputs = _axis_keys(self.cpus), _axis_keys(self.mems), _axis_keys(scales)
        expected = len(cpus) * len(mems) * len(self.gpus) * len(inputs) * (self.warmup + self.repeat)
        matched = sum(
            cpu in cpus and mem in mems and gpu in self.gpus and scale in inputs
            and Decimal(index) < (self.warmup if warmup == "1" else self.repeat)
            for cpu, mem, gpu, scale, warmup, index in actual
        )
        return {"expected": expected, "actual": len(actual), "missing": expected - matched,
                "unexpected": len(actual) - matched}
