"""Resolve CUDA's visible device in a short helper, independently of NVML access."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from acprof.system.command import run_command

_CUDA_PROBE = '''
import ctypes as c, json
try:
    cuda = c.CDLL("libcuda.so.1")
    def check(code):
        if code: raise RuntimeError("CUDA driver error " + str(code))
    check(cuda.cuInit(0))
    count = c.c_int(); check(cuda.cuDeviceGetCount(c.byref(count)))
    if count.value < 1: raise RuntimeError("No visible CUDA device")
    device = c.c_int(); check(cuda.cuDeviceGet(c.byref(device), 0))
    name = c.create_string_buffer(256); check(cuda.cuDeviceGetName(name, 256, device))
    uuid = (c.c_ubyte * 16)(); check(cuda.cuDeviceGetUuid(c.byref(uuid), device))
    memory = c.c_size_t(); check(cuda.cuDeviceTotalMem_v2(c.byref(memory), device))
    value = bytes(uuid).hex()
    uuid_text = "GPU-" + "-".join((value[:8], value[8:12], value[12:16], value[16:20], value[20:]))
    print(json.dumps(dict(status="available", uuid=uuid_text, name=name.value.decode(),
                         memory_total_bytes=memory.value, visible_count=count.value)))
except Exception as exc:
    print(json.dumps(dict(status="unavailable", error=str(exc))))
'''


def process_gpu_device(requested=None) -> dict:
    if requested is None:
        requested = os.environ.get("ACPROF_GPU_DEVICE") or os.environ.get("DEVICE_INDEX")
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = [v.strip() for v in original.split(",")] if original is not None else None
    if requested is None:
        selected = visible[0] if visible else "0"
    elif str(requested).isdigit():
        ordinal = int(requested)
        if visible is not None and ordinal >= len(visible):
            raise ValueError("--gpu-device exceeds CUDA_VISIBLE_DEVICES; Process uses visible ordinals")
        selected = visible[ordinal] if visible is not None else str(ordinal)
    else:
        selected = str(requested)
        if visible is not None and selected not in visible:
            raise ValueError("explicit GPU UUID is outside CUDA_VISIBLE_DEVICES")
    if not selected or selected == "-1":
        raise ValueError("GPU requested but CUDA_VISIBLE_DEVICES hides every device")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": selected}
    try:
        result = run_command([sys.executable, "-I", "-c", _CUDA_PROBE], env=env,
                                capture_output=True, text=True, timeout=15, check=True)
        observed = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        observed = {"status": "unavailable", "error": str(exc)}
    return {"index": 0, "uuid": "", "name": "unknown", "memory_total_bytes": None,
            **observed, "cuda_visible_devices": selected, "original_cuda_visible_devices": original,
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "identity_source": "cuda_driver_visible_ordinal", "sharing": "unknown",
            "telemetry_scope": "device"}
