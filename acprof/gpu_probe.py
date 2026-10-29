"""Lightweight GPU probe evidence; no hardware imports or probing at import time."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypedDict, TypeGuard

from acprof.config_constraints import integer

ProbeStatus = Literal["available", "unavailable", "unknown", "not_applicable"]


class ProbeField(TypedDict):
    value: str | int | None
    status: ProbeStatus
    source: str | None


class ProbeError(TypedDict):
    source: str
    stage: str
    field: str | None
    error_type: str
    message: str


class GpuProbeReport(TypedDict):
    schema_version: int
    device: dict[str, Any]
    fields: dict[str, ProbeField]
    errors: list[ProbeError]


def _is_probe_field(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("status") in ("available", "unavailable", "unknown", "not_applicable")
        and "value" in value
        and not isinstance(value["value"], bool)
        and isinstance(value["value"], str | int | type(None))
        and "source" in value
        and isinstance(value["source"], str | type(None))
    )


def _is_probe_error(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        all(isinstance(value.get(key), str) for key in ("source", "stage", "error_type", "message"))
        and "field" in value
        and isinstance(value["field"], str | type(None))
    )


def is_gpu_probe_report(value: object) -> TypeGuard[GpuProbeReport]:
    """Validate the JSON boundary before treating transported reports as typed evidence."""
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        return False
    fields, errors = value.get("fields"), value.get("errors")
    return (
        value["schema_version"] == 1
        and isinstance(value.get("device"), dict)
        and isinstance(fields, dict)
        and isinstance(errors, list)
        and all(isinstance(key, str) and _is_probe_field(field) for key, field in fields.items())
        and all(_is_probe_error(error) for error in errors)
    )


def gpu_probe_report(
    names: Iterable[str], *, status: ProbeStatus = "unavailable"
) -> GpuProbeReport:
    return {
        "schema_version": 1,
        "device": {},
        "fields": {name: {"value": None, "status": status, "source": None} for name in names},
        "errors": [],
    }


def record_gpu_probe_error(
    report: GpuProbeReport,
    source: str,
    stage: str,
    error: BaseException,
    *,
    field: str | None = None,
) -> None:
    report["errors"].append(
        {
            "source": source,
            "stage": stage,
            "field": field,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    )


def set_gpu_probe_value(report: GpuProbeReport, field: str, value: str | int, source: str) -> None:
    report["fields"][field] = {"value": value, "status": "available", "source": source}


def gpu_compute_value(field: str, value: object) -> str | int:
    """Validate finite, nonzero legacy or current compute metadata scalars."""
    if field == "gpu_compute_capability":
        if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*\.[0-9]+", value):
            raise ValueError("invalid GPU compute capability")
        return value
    if field != "gpu_sm_count":
        raise ValueError(f"unknown GPU compute field: {field}")
    count = integer(value)
    if count < 1:
        raise ValueError("gpu_sm_count must be positive")
    return count


def require_gpu_identity(
    device: Mapping[str, Any],
    *,
    expected_uuid: str | None = None,
    expected_index: int | None = None,
) -> str:
    """A pinned UUID is authoritative; its current host index may change."""
    uuid = device.get("uuid")
    if not isinstance(uuid, str) or not uuid.startswith("GPU-") or any(c in uuid for c in ",\n\r "):
        raise ValueError("GPU metadata requires one physical GPU UUID")
    if expected_uuid is not None and uuid.lower() != expected_uuid.lower():
        raise ValueError(f"GPU UUID mismatch: expected {expected_uuid}, received {uuid}")
    if (
        expected_uuid is None
        and expected_index is not None
        and device.get("index") != expected_index
    ):
        raise ValueError(
            f"GPU index mismatch: expected {expected_index}, received {device.get('index')}"
        )
    return uuid
