"""Explicit inputs, observations and completed results for one measurement window."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    energy_mod: Any = None
    cpu_energy_mod: Any = None
    resource_usage_mod: Any = None
    perf_mips_mod: Any = None


@dataclass
class WindowObservations:
    """Keep partial observations available when startup, cleanup or publication fails."""

    gpu_monitor: Any = None
    cpu_monitor: Any = None
    resource_result: Any = None
    gpu_result: Any = None
    cpu_result: Any = None
    mips_result: Any = None
    resource_error: str = ""
    gpu_error: str = ""
    cpu_error: str = ""
    gpu_samples: list[tuple[float, float]] = field(default_factory=list)
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
