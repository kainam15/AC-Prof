"""Restore recorded execution conditions outside measurement windows."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from typing import Mapping

from acprof.cpu_affinity import normalize_cpu_set, parse_cpu_set
from acprof.host.gpu_device import gpu_device_scope, pin_gpu_device
from acprof.runtime_settings import RUNTIME_ENV_NAMES


MEASUREMENT_ENV_NAMES = (
    "AUTO_WARMUP_REQUESTS", "SLOW_LATENCY_THRESHOLD_S", "IDLE_DEBUG_TRACE_INTERVAL_S",
    "DEVICE_INDEX", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
)
DEVICE_ENV_NAMES = ("ACPROF_GPU_DEVICE", "GPU_DEVICE_UUID", "NVIDIA_VISIBLE_DEVICES")


def measurement_environment() -> dict:
    return {**{name: os.environ.get(name) for name in MEASUREMENT_ENV_NAMES},
            **{name: os.environ[name] for name in RUNTIME_ENV_NAMES if name in os.environ}}


@contextmanager
def source_runtime_environment(recorded: Mapping):
    """Clear unrecorded overrides and restore the caller's environment on every exit."""
    names = (*MEASUREMENT_ENV_NAMES, *RUNTIME_ENV_NAMES, *DEVICE_ENV_NAMES)
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            value = recorded.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True)
class ExecutionConditions:
    gpu_mode: str
    gpu_uuid: str
    cpuset_cpus: str
    request_timeout_seconds: float
    environment: dict

    @classmethod
    def from_options(cls, options: Mapping, *, gpu: str) -> "ExecutionConditions":
        if gpu not in {"off", "on"} or gpu not in options.get("gpus", "").split(","):
            raise ValueError("source does not include the requested GPU mode")
        environment = options.get("measurement_environment")
        if not isinstance(environment, dict):
            raise ValueError("source measurement_environment is missing")
        uuid = str(options.get("gpu_device", "")) if gpu == "on" else ""
        if gpu == "on" and not uuid.startswith("GPU-"):
            raise ValueError("source GPU UUID is missing; refusing index-based replay")
        timeout = float(options["request_timeout_seconds"])
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("source request timeout must be finite and positive")
        return cls(gpu, uuid, normalize_cpu_set(options.get("cpuset_cpus", "")), timeout, dict(environment))

    @property
    def container_options(self) -> dict:
        return {"cpuset_cpus": self.cpuset_cpus, "request_timeout_seconds": self.request_timeout_seconds}

    @contextmanager
    def activate(self):
        with source_runtime_environment(self.environment), gpu_device_scope():
            requested = parse_cpu_set(self.cpuset_cpus)
            if requested and not requested <= os.sched_getaffinity(0):
                raise ValueError("source CPU affinity is unavailable on this host")
            device = {}
            if self.gpu_mode == "on":
                device = pin_gpu_device(self.gpu_uuid)
                if device["uuid"] != self.gpu_uuid:
                    raise ValueError("resolved GPU UUID differs from the source")
                os.environ.update(ACPROF_GPU_DEVICE=self.gpu_uuid, GPU_DEVICE_UUID=self.gpu_uuid,
                                  DEVICE_INDEX=str(device["index"]))
            yield device
