"""Input scale parsing, serialization, and deterministic candidates."""

import math

from acprof.config import SCALING_DIMENSIONS

AUTO_INPUT_SCALE_COUNT = 6


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def format_scale_value(scale: float) -> str:
    value = float(scale)
    if value.is_integer():
        return str(int(value))
    return str(value)


def serialize_input_scales(scales: list[float]) -> str:
    return ",".join(format_scale_value(scale) for scale in scales)


def resolve_input_scales(task_family: str, input_scales: str | None = None) -> list[float]:
    if input_scales:
        return sorted(set(_parse_float_list(input_scales)))

    scaling_cfg = SCALING_DIMENSIONS.get(task_family)
    if scaling_cfg:
        return sorted(set(float(value) for value in scaling_cfg.values))
    return [1.0]


def integer_auto_scales(max_value: int, count: int = AUTO_INPUT_SCALE_COUNT) -> list[float]:
    if max_value < count:
        raise RuntimeError(
            f"cannot generate {count} unique integer input scales from max_value={max_value}"
        )

    scales = [max(1, math.floor(max_value * idx / count)) for idx in range(1, count + 1)]
    scales[-1] = int(max_value)
    if len(set(scales)) != count:
        raise RuntimeError(
            f"failed to derive {count} unique integer input scales from max_value={max_value}: {scales}"
        )
    return [float(value) for value in scales]


def float_auto_scales(max_value: float, count: int = AUTO_INPUT_SCALE_COUNT) -> list[float]:
    if max_value <= 0:
        raise RuntimeError(f"invalid max_value for float scales: {max_value}")
    scales = [round(float(max_value) * idx / count, 6) for idx in range(1, count + 1)]
    scales[-1] = round(float(max_value), 6)
    return scales
