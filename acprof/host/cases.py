"""Typed requests for resource matrices and individual measurement cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from acprof.config import (
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo

if TYPE_CHECKING:
    from acprof.host.run_state import RunState


@dataclass(frozen=True)
class MatrixProgress:
    """One safely completed resource case in a matrix sweep."""

    completed_cases: int
    total_cases: int
    cpu: int
    mem: int
    gpu: str
    result_csv: str | None


@dataclass(frozen=True)
class CaseRequest:
    """Identity, resources, and locations for one container execution."""

    task_info: TaskInfo
    cpu: int
    mem: int
    gpu: str
    image_info: ImageInfo
    output_dir: str
    project_dir: str
    cpuset_cpus: str = ""


@dataclass(frozen=True)
class CaseWorkload:
    """Request counts and deadlines, independent of host collector settings."""

    batch_size: int = 1
    warmup: int = 2
    repeat: int = 5
    repeat_in_window: int = DEFAULT_REPEAT_IN_WINDOW
    repeat_window_seconds: float = DEFAULT_REPEAT_WINDOW_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS


@dataclass(frozen=True)
class CaseMeasurement:
    """Collector policy for a case; no mutable execution state."""

    sample_hz: float = 20.0
    idle_seconds: float = DEFAULT_IDLE_SECONDS
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS
    idle_debug: bool = False
    sniff_iface: str = "docker0"
    require_packet_latency: bool = True
    profiling_mode: str = "full"
    dram_energy: str = "auto"
    sampling_policy: str = "burst"
    raw_sample_limit: int = 100000


@dataclass(frozen=True)
class CasePlans:
    """Previously prepared plans consumed by the measured client."""

    input_scales: str | None = None
    input_scale_plan_file: str | None = None
    compute_profile_plan_file: str | None = None
    execution_profile_plan_file: str | None = None
    input_scale_order: list[float] | None = None


@dataclass(frozen=True)
class MatrixRequest:
    """Frozen inputs shared by every case in one resource matrix."""

    task_info: TaskInfo
    image_info: ImageInfo
    cpu_list: list[int]
    mem_list: list[int]
    gpu_list: list[str]
    output_dir: str
    project_dir: str
    workload: CaseWorkload = CaseWorkload()
    measurement: CaseMeasurement = CaseMeasurement()
    plans: CasePlans = CasePlans()
    progress_callback: Callable[[MatrixProgress], None] | None = None
    prune_startup_oom: bool = False
    run_state: RunState | None = None
    matrix_order: str = "seeded"
    matrix_seed: int = 0
    cpuset_cpus: str = ""

    def case_request(self, cpu: int, mem: int, gpu: str) -> CaseRequest:
        return CaseRequest(
            task_info=self.task_info,
            cpu=cpu,
            mem=mem,
            gpu=gpu,
            image_info=self.image_info,
            output_dir=self.output_dir,
            project_dir=self.project_dir,
            cpuset_cpus=self.cpuset_cpus,
        )

    def case_plans(self, input_scale_order: list[float]) -> CasePlans:
        return replace(self.plans, input_scale_order=list(input_scale_order))
