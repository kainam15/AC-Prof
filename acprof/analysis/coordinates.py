"""Lossless measurement axes for statistics and independent comparison reports."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from acprof.experiments.constraints import format_input_scale
from acprof.results.csv import numeric_coordinate_key
from acprof.results.measurement_plan import decimal_coordinate

RESOURCE_AXES = ("cpu_cores", "mem_cap_gb")
STATISTIC_AXES = (*RESOURCE_AXES, "input_scale")
COORDINATE_IDENTITY = "recorded_decimal_v1"


def window_case_key(row: dict, key: tuple[str, ...]) -> tuple[str, ...]:
    return (*key[:4], row.get("environment_class", "unknown"), row.get("runtime_kind", "docker"),
            row.get("runtime_id", ""), row.get("resource_scope", "container_cgroup"))


def project_coordinates(axes: Mapping[str, str]) -> dict:
    """Keep JSON numbers when possible and retain any decimal lost by their projection."""
    result: dict = {}
    exact = {}
    for field, raw in axes.items():
        if raw == "inherited":
            result[field] = None
            continue
        display = float(raw)
        result[field] = display if math.isfinite(display) else None
        if decimal_coordinate(raw) != decimal_coordinate(str(display)):
            exact[field] = raw
    if exact:
        result["exact_coordinates"] = exact
    return result


def _exact_label(field: str, raw: object, display: object) -> str:
    if not isinstance(raw, str) or numeric_coordinate_key(field, raw) != raw:
        raise ValueError("invalid_statistics_coordinates")
    projected = float(raw)
    if not math.isfinite(projected):
        if display is not None:
            raise ValueError("invalid_statistics_coordinates")
    else:
        expected = decimal_coordinate(str(projected))
        if (type(display) not in (int, float) or decimal_coordinate(str(display)) != expected
                or decimal_coordinate(raw) == expected):
            raise ValueError("invalid_statistics_coordinates")
    return raw


def report_coordinate_labels(record: Mapping, fields: Sequence[str] = STATISTIC_AXES) -> dict[str, str]:
    """Validate the display numbers against any exact axes before rendering them."""
    exact = record.get("exact_coordinates", {})
    if not isinstance(exact, dict) or exact.keys() - set(fields):
        raise ValueError("invalid_statistics_coordinates")
    labels = {}
    for field in fields:
        if field not in record:
            raise ValueError("invalid_statistics_coordinates")
        value = record[field]
        if field in exact:
            labels[field] = _exact_label(field, exact[field], value)
        elif value is None and field in RESOURCE_AXES:
            labels[field] = "inherited"
        elif type(value) in (int, float) and value > 0 and (isinstance(value, int) or math.isfinite(value)):
            labels[field] = format_input_scale(value)
        else:
            raise ValueError("invalid_statistics_coordinates")
    return labels
