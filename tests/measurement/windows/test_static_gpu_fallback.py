"""Static GPU metadata preserves identity and every attempted backend failure."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host.metadata import gpu as gpu_metadata


@pytest.fixture
def gpu_backends(monkeypatch):
    handle = object()
    device = {
        "uuid": "GPU-one",
        "index": 1,
        "pci_bus_id": "00000000:01:00.0",
        "name": "GPU One",
        "memory_total_bytes": 8192 * 1024**2,
    }
    nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetHandleByIndex=Mock(return_value=handle),
        nvmlDeviceGetHandleByUUID=Mock(return_value=handle),
        nvmlDeviceGetUUID=Mock(return_value=device["uuid"]),
        nvmlDeviceGetIndex=Mock(return_value=device["index"]),
        nvmlDeviceGetPciInfo=Mock(
            return_value=SimpleNamespace(busId=device["pci_bus_id"].encode())
        ),
        nvmlDeviceGetName=Mock(return_value=device["name"].encode()),
        nvmlDeviceGetMemoryInfo=Mock(
            return_value=SimpleNamespace(total=device["memory_total_bytes"])
        ),
    )
    fallback = Mock(side_effect=AssertionError("unexpected smi query"))
    monkeypatch.setitem(__import__("sys").modules, "pynvml", nvml)
    monkeypatch.setattr(gpu_metadata, "resolve_gpu_device", fallback)
    monkeypatch.setattr(gpu_metadata, "selected_gpu_device", lambda: {})
    return nvml, fallback, device


def test_one_nvml_session_supplies_both_fields(gpu_backends):
    nvml, fallback, device = gpu_backends
    report = gpu_metadata.probe_gpu_metadata(device_index=1)
    assert report["device"] == {key: device[key] for key in ("uuid", "index", "pci_bus_id")}
    assert report["fields"]["name"] == {"value": "GPU One", "status": "available", "source": "nvml"}
    assert report["fields"]["memory_total_bytes"]["value"] == 8192 * 1024**2
    assert report["errors"] == []
    nvml.nvmlInit.assert_called_once_with()
    nvml.nvmlShutdown.assert_called_once_with()
    nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(1)
    nvml.nvmlDeviceGetHandleByUUID.assert_not_called()
    fallback.assert_not_called()


def test_selected_device_is_reused_without_another_probe(gpu_backends):
    nvml, fallback, device = gpu_backends
    report = gpu_metadata.probe_gpu_metadata(gpu_device=device)
    assert all(field["source"] == "selected_device" for field in report["fields"].values())
    nvml.nvmlInit.assert_not_called()
    fallback.assert_not_called()


def test_pinned_uuid_survives_host_index_reordering(gpu_backends):
    nvml, fallback, device = gpu_backends
    report = gpu_metadata.probe_gpu_metadata(gpu_device={"uuid": device["uuid"], "index": 99})
    assert report["device"]["index"] == 1
    assert report["fields"]["name"]["status"] == "available"
    nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-one")
    nvml.nvmlDeviceGetHandleByIndex.assert_not_called()
    fallback.assert_not_called()


def test_successful_fallback_keeps_init_failure_and_skips_shutdown(gpu_backends):
    nvml, fallback, device = gpu_backends
    nvml.nvmlInit.side_effect = RuntimeError("NVML init denied")
    fallback.side_effect = None
    fallback.return_value = device
    report = gpu_metadata.probe_gpu_metadata(device_index=1)
    assert report["fields"]["name"]["source"] == "nvidia-smi"
    assert report["errors"] == [
        {
            "source": "nvml",
            "stage": "init",
            "field": None,
            "error_type": "RuntimeError",
            "message": "NVML init denied",
        }
    ]
    fallback.assert_called_once_with("1")
    nvml.nvmlShutdown.assert_not_called()


def test_partial_failure_only_fills_missing_field_using_same_uuid(gpu_backends):
    nvml, fallback, device = gpu_backends
    nvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("memory query denied")
    fallback.side_effect = None
    fallback.return_value = {**device, "index": 2, "name": "Do not replace NVML name"}
    report = gpu_metadata.probe_gpu_metadata(device_index=1)
    assert report["fields"]["name"] == {"value": "GPU One", "status": "available", "source": "nvml"}
    assert report["fields"]["memory_total_bytes"]["source"] == "nvidia-smi"
    assert report["errors"][0]["field"] == "memory_total_bytes"
    fallback.assert_called_once_with("GPU-one")
    nvml.nvmlShutdown.assert_called_once_with()


def test_shutdown_failure_preserves_successful_values(gpu_backends):
    nvml, fallback, device = gpu_backends
    nvml.nvmlShutdown.side_effect = RuntimeError("shutdown failed")
    report = gpu_metadata.probe_gpu_metadata(device_index=1)
    assert all(field["status"] == "available" for field in report["fields"].values())
    assert report["errors"][0]["stage"] == "shutdown"
    fallback.assert_not_called()


def test_all_failures_remain_visible_without_zero_substitution(gpu_backends):
    nvml, fallback, device = gpu_backends
    nvml.nvmlDeviceGetName.side_effect = RuntimeError("name denied")
    nvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("memory denied")
    nvml.nvmlShutdown.side_effect = RuntimeError("shutdown denied")
    fallback.side_effect = RuntimeError("smi denied")
    report = gpu_metadata.probe_gpu_metadata(device_index=1)
    assert len(report["errors"]) == 4
    assert [error["message"] for error in report["errors"]] == [
        "name denied",
        "memory denied",
        "shutdown denied",
        "smi denied",
    ]
    assert all(
        field["value"] is None and field["status"] == "unavailable"
        for field in report["fields"].values()
    )


def test_uuid_mismatch_never_accepts_another_cards_values(gpu_backends):
    nvml, fallback, device = gpu_backends
    nvml.nvmlDeviceGetUUID.return_value = "GPU-wrong"
    fallback.side_effect = None
    fallback.return_value = {**device, "uuid": "GPU-wrong"}
    report = gpu_metadata.probe_gpu_metadata(gpu_device={"uuid": "GPU-one", "index": 99})
    assert len(report["errors"]) == 2
    assert all("UUID mismatch" in error["message"] for error in report["errors"])
    assert all(field["value"] is None for field in report["fields"].values())
    fallback.assert_called_once_with("GPU-one")
    nvml.nvmlDeviceGetName.assert_not_called()
    nvml.nvmlDeviceGetHandleByIndex.assert_not_called()


@pytest.mark.parametrize(
    "selected",
    [
        {"uuid": "", "status": "unavailable", "error": "CUDA device hidden", "name": "stale GPU"},
        {"index": 1, "name": "unattributed GPU", "memory_total_bytes": 100},
    ],
)
def test_unresolved_selected_device_never_falls_back_to_default_gpu(gpu_backends, selected):
    nvml, fallback, device = gpu_backends
    report = gpu_metadata.probe_gpu_metadata(gpu_device=selected)
    assert report["errors"][0]["source"] == "selected_device"
    assert all(field["value"] is None for field in report["fields"].values())
    nvml.nvmlInit.assert_not_called()
    fallback.assert_not_called()


def test_import_failure_is_distinct_from_initialization_failure(gpu_backends, monkeypatch):
    nvml, fallback, device = gpu_backends
    monkeypatch.setitem(__import__("sys").modules, "pynvml", None)
    fallback.side_effect = None
    fallback.return_value = device
    report = gpu_metadata.probe_gpu_metadata(device_index=1)
    assert report["errors"][0]["stage"] == "import"
    assert report["fields"]["name"]["status"] == "available"


def test_unsupported_mig_identity_does_not_revert_to_index(gpu_backends):
    nvml, fallback, device = gpu_backends
    nvml.nvmlDeviceGetUUID.return_value = "MIG-instance"
    fallback.side_effect = ValueError("physical GPU required")
    report = gpu_metadata.probe_gpu_metadata(device_index=1)
    assert report["fields"]["name"]["value"] is None
    fallback.assert_called_once_with("MIG-instance")


def test_null_memory_result_has_a_query_error_before_fallback(gpu_backends):
    nvml, fallback, device = gpu_backends
    nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(total=None)
    fallback.side_effect = None
    fallback.return_value = device
    report = gpu_metadata.probe_gpu_metadata(device_index=1)
    assert report["errors"][0]["field"] == "memory_total_bytes"
    assert report["errors"][0]["stage"] == "query"
    assert report["fields"]["memory_total_bytes"]["value"] == device["memory_total_bytes"]
