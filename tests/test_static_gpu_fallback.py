"""GPU index fallback must never misattribute another device's identity."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host import machine_metadata


@pytest.mark.parametrize(
    ("device_index", "expected"),
    [(0, "GPU Zero"), (1, "GPU One"), (2, "unknown"), (-1, "unknown")],
)
def test_nvidia_smi_name_fallback_respects_gpu_index(device_index, expected):
    unavailable_nvml = SimpleNamespace(nvmlInit=lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))
    with (
        patch.dict("sys.modules", {"pynvml": unavailable_nvml}),
        patch("acprof.host.machine_metadata.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch(
            "acprof.host.machine_metadata.host_command.run_command",
            return_value=SimpleNamespace(returncode=0, stdout="GPU Zero\nGPU One\n"),
        ),
    ):
        assert machine_metadata.get_gpu_name(device_index) == expected


@pytest.mark.parametrize(
    ("device_index", "expected"),
    [(0, 8192 * 1024**2), (1, 16384 * 1024**2), (2, None), (-1, None)],
)
def test_nvidia_smi_memory_fallback_respects_gpu_index(device_index, expected):
    unavailable_nvml = SimpleNamespace(nvmlInit=lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))
    with (
        patch.dict("sys.modules", {"pynvml": unavailable_nvml}),
        patch("acprof.host.machine_metadata.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch(
            "acprof.host.machine_metadata.host_command.run_command",
            return_value=SimpleNamespace(returncode=0, stdout="8192\n16384\n"),
        ),
    ):
        assert machine_metadata.get_gpu_mem_total_bytes(device_index) == expected
