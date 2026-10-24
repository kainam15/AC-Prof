"""Structural monitor contracts; ownership and prerequisite discovery are separate."""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeVar

from acprof.monitors.sampling import SamplingPolicy

if TYPE_CHECKING:
    from acprof.monitors.energy_cpu import CPUEnergyMonitor, CPUEnergyResult, CPUSample
    from acprof.monitors.energy_nvml import GPUEnergyMonitor
    from acprof.monitors.gpu_metrics import EnergyResult
    from acprof.monitors.perf_mips import PerfMIPSMonitor
    from acprof.monitors.resource_metrics import ResourceUsageResult, ResourceUsageSample
    from acprof.monitors.resource_usage import ResourceUsageMonitor

Result = TypeVar("Result", covariant=True)


class MonitorLifecycle(Protocol):
    def start(self) -> None: ...
    def close(self) -> None: ...


class PreparedMonitor(Protocol):
    def prepare(self) -> None: ...


class SampleMonitor(MonitorLifecycle, Protocol[Result]):
    def stop(self) -> Result: ...


class InstructionMonitor(MonitorLifecycle, Protocol[Result]):
    def stop(self, repeat_in_window: int, latency_app_s: float) -> Result: ...


class CPUStopResult(NamedTuple):
    metrics: CPUEnergyResult
    error: str
    samples: list[CPUSample]


class GPUStopResult(NamedTuple):
    metrics: EnergyResult
    device_name: str
    error: str
    samples: list[tuple[float, float]]


class ResourceStopResult(NamedTuple):
    metrics: ResourceUsageResult
    error: str
    samples: list[ResourceUsageSample]


class CPUCollectorFactory(Protocol):
    def __call__(self, *, sample_hz: float, idle_seconds: float, container_name: str,
                 dram_energy: str, sampling_policy: SamplingPolicy,
                 raw_sample_limit: int) -> CPUEnergyMonitor: ...


class GPUCollectorFactory(Protocol):
    def __call__(self, *, sample_hz: float, idle_seconds: float, device_index: int,
                 device_uuid: str, sampling_policy: SamplingPolicy,
                 raw_sample_limit: int) -> GPUEnergyMonitor: ...


class ResourceCollectorFactory(Protocol):
    def __call__(self, *, sample_hz: float, container_name: str, cpu_cores: float,
                 mem_cap_gb: float, use_gpu: bool, device_index: int, device_uuid: str,
                 sampling_policy: SamplingPolicy, raw_sample_limit: int) -> ResourceUsageMonitor: ...


class InstructionCollectorFactory(Protocol):
    def __call__(self, container_name: str) -> PerfMIPSMonitor: ...
