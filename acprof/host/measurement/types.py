"""Explicit inputs, observations and completed results for one measurement window."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from acprof.monitors.interfaces import (
    CPUCollectorFactory,
    GPUCollectorFactory,
    InstructionCollectorFactory,
    ResourceCollectorFactory,
)

if TYPE_CHECKING:
    from acprof.monitors.energy_cpu import CPUEnergyMonitor, CPUEnergyResult, CPUSample
    from acprof.monitors.energy_nvml import GPUEnergyMonitor
    from acprof.monitors.gpu_metrics import EnergyResult
    from acprof.monitors.perf_mips import MIPSResult
    from acprof.monitors.resource_metrics import ResourceUsageResult, ResourceUsageSample


@dataclass(frozen=True)
class WindowSpec:
    scale_entry: dict[str, Any]
    request_limit: int
    warmup: int
    repeat_idx: int
    scale: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", float(self.scale_entry["input_scale"]))

    def group_id(self, case_name: str) -> str:
        phase = "w" if self.warmup else "r"
        return f"{case_name}_{self.scale_entry['scale_label']}_{phase}{self.repeat_idx}"


@dataclass(frozen=True)
class MonitorDependencies:
    """Collector factories are the replaceable hardware adapter boundary."""

    gpu: GPUCollectorFactory | None = None
    cpu: CPUCollectorFactory | None = None
    resource: ResourceCollectorFactory | None = None
    instructions: InstructionCollectorFactory | None = None


@dataclass
class WindowObservations:
    """Keep partial observations available when startup, cleanup or publication fails."""

    gpu_monitor: GPUEnergyMonitor | None = None
    cpu_monitor: CPUEnergyMonitor | None = None
    resource_result: ResourceUsageResult | None = None
    gpu_result: EnergyResult | None = None
    cpu_result: CPUEnergyResult | None = None
    mips_result: MIPSResult | None = None
    resource_error: str = ""
    gpu_error: str = ""
    cpu_error: str = ""
    gpu_samples: list[tuple[float, float]] = field(default_factory=list)
    cpu_samples: list[CPUSample] = field(default_factory=list)
    resource_samples: list[ResourceUsageSample] = field(default_factory=list)
    collector_evidence: dict[str, Any] = field(default_factory=dict)
    request_record: dict[str, Any] | None = None
    gpu_idle_measured_at: str = "nan"
    cpu_idle_measured_at: str = "nan"
    gpu_idle_trace: dict[str, Any] = field(default_factory=dict)
    idle_trace: dict[str, Any] = field(default_factory=dict)
    idle_debug_snapshot: dict[str, Any] | None = None
    gpu_idle_debug_snapshot: dict[str, Any] | None = None


@dataclass(frozen=True)
class WindowResult:
    """A completed row and its provenance, ready for the outer publication loop."""

    row: dict[str, Any]
    idle_diag_record: dict[str, Any] | None
    sniff_group_id: str
    telemetry_evidence: dict[str, Any] | None = None
    request_record: dict[str, Any] | None = None
