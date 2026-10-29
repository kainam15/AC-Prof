"""One preparation-time NVML session, UUID-bound fallback and explicit errors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from acprof.gpu_probe import (
    GpuProbeReport,
    gpu_probe_report,
    record_gpu_probe_error,
    require_gpu_identity,
    set_gpu_probe_value,
)
from acprof.host.gpu_device import resolve_gpu_device, selected_gpu_device
from acprof.input_budget import positive_integer

_FIELDS = ("name", "memory_total_bytes")


def _name(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if (
        not isinstance(value, str)
        or not value.strip()
        or value.strip().lower() in {"unknown", "n/a"}
    ):
        raise ValueError("GPU name is unavailable")
    return value.strip()


def _copy_values(report: GpuProbeReport, values: Mapping[str, Any], source: str) -> None:
    for name in _FIELDS:
        if report["fields"][name]["status"] == "available" or name not in values:
            continue
        try:
            value = (
                _name(values[name])
                if name == "name"
                else positive_integer(values[name], name, allow_integral_float=False)
            )
        except (ValueError, TypeError) as exc:
            record_gpu_probe_error(report, source, "query", exc, field=name)
        else:
            set_gpu_probe_value(report, name, value, source)


def _query_nvml(report: GpuProbeReport, device_index: int) -> None:
    initialized = False
    stage = "import"
    try:
        import pynvml

        stage = "init"
        pynvml.nvmlInit()
        initialized = True
        stage = "resolve"
        expected_uuid = report["device"].get("uuid")
        handle = (
            pynvml.nvmlDeviceGetHandleByUUID(expected_uuid)
            if expected_uuid
            else pynvml.nvmlDeviceGetHandleByIndex(device_index)
        )
        uuid = _name(pynvml.nvmlDeviceGetUUID(handle))
        if expected_uuid is None:
            report["device"]["uuid"] = uuid
        require_gpu_identity({"uuid": uuid}, expected_uuid=expected_uuid)
        for name, query in (
            ("index", lambda: int(pynvml.nvmlDeviceGetIndex(handle))),
            ("pci_bus_id", lambda: _name(pynvml.nvmlDeviceGetPciInfo(handle).busId)),
        ):
            try:
                report["device"][name] = query()
            except Exception as exc:  # noqa: BLE001 - preserve optional driver failures as evidence
                record_gpu_probe_error(report, "nvml", "query", exc, field=name)
        for name, query in (
            ("name", lambda: pynvml.nvmlDeviceGetName(handle)),
            ("memory_total_bytes", lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).total),
        ):
            if report["fields"][name]["status"] == "available":
                continue
            try:
                _copy_values(report, {name: query()}, "nvml")
            except Exception as exc:  # noqa: BLE001 - field failures do not erase other measurements
                record_gpu_probe_error(report, "nvml", "query", exc, field=name)
    except Exception as exc:  # noqa: BLE001 - optional backend failure is returned with its stage
        record_gpu_probe_error(report, "nvml", stage, exc)
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as exc:  # noqa: BLE001 - cleanup evidence must not mask a query failure
                record_gpu_probe_error(report, "nvml", "shutdown", exc)


def probe_gpu_metadata(
    *, gpu_device: Mapping[str, Any] | None = None, device_index: int = 0
) -> GpuProbeReport:
    """Reuse selected-device facts, otherwise query one physical GPU outside collection."""
    report = gpu_probe_report(_FIELDS)
    selected = dict(selected_gpu_device() if gpu_device is None else gpu_device)
    if selected:
        report["device"] = {
            key: value
            for key, value in selected.items()
            if key in {"uuid", "index", "pci_bus_id", "identity_source"}
        }
        try:
            if selected.get("status") == "unavailable":
                raise ValueError(
                    str(selected.get("error") or "selected CUDA device is unavailable")
                )
            require_gpu_identity(selected)
        except ValueError as exc:
            record_gpu_probe_error(report, "selected_device", "resolve", exc)
            return report
        _copy_values(report, selected, "selected_device")
    if all(field["status"] == "available" for field in report["fields"].values()):
        return report
    _query_nvml(report, device_index)
    if any(field["status"] != "available" for field in report["fields"].values()):
        uuid = report["device"].get("uuid")
        try:
            # resolve_gpu_device performs one smi query, never an NVML session.
            fallback = resolve_gpu_device(str(uuid or device_index))
            require_gpu_identity(
                fallback, expected_uuid=uuid, expected_index=None if uuid else device_index
            )
        except Exception as exc:  # noqa: BLE001 - keep fallback provenance alongside NVML failure
            record_gpu_probe_error(report, "nvidia-smi", "resolve", exc)
        else:
            report["device"].update({key: fallback[key] for key in ("uuid", "index", "pci_bus_id")})
            _copy_values(report, fallback, "nvidia-smi")
    return report
