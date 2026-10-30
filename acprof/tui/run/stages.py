"""Stable TUI process stage identities, independent of translated labels."""
from __future__ import annotations

from enum import Enum

from acprof.messages import Message
from acprof.tui.i18n import message_id


class Stage(Enum):
    WAITING = "waiting"
    STARTING = "starting"
    CLEANUP_INCOMPLETE = "cleanup_incomplete"
    VERIFYING_ARTIFACT = "verifying_artifact"
    TASK_UNSUPPORTED = "task_unsupported"
    RUNTIME_VALIDATION = "runtime_validation"
    INTERFACE_RESOLUTION = "interface_resolution"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    ENVIRONMENT_PREFLIGHT = "environment_preflight"
    PREPARING_IMAGE = "preparing_image"
    PREPARING_INPUT = "preparing_input"
    CONFIRMING_INTERFACE = "confirming_interface"
    PREPARING_ENVIRONMENT = "preparing_environment"
    PREPARING_MODEL = "preparing_model"
    STARTING_PROBE_CONTAINER = "starting_probe_container"
    PREPARING_MEMORY_PROBE = "preparing_memory_probe"
    SCANNING_MEMORY = "scanning_memory"
    MEMORY_FOUND = "memory_found"
    MEMORY_FEASIBILITY = "memory_feasibility"
    PROBE_COMPLETED = "probe_completed"
    PROBE_FAILED = "probe_failed"
    PREPARING_MATRIX = "preparing_matrix"
    STARTING_CONTAINER = "starting_container"
    MAX_SCALE_PROBE = "max_scale_probe"
    BUILDING_IMAGE = "building_image"
    PLANNING_INPUT = "planning_input"
    COMPUTE_PROFILING = "compute_profiling"
    EXECUTION_PROFILING = "execution_profiling"
    SERVER_READY = "server_ready"
    MEASURING = "measuring"
    CLEANING_CASE = "cleaning_case"
    CASE_COMPLETED = "case_completed"
    OOM_PRUNED = "oom_pruned"
    MERGING_RESULTS = "merging_results"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    PARTIAL = "partial"
    TERMINATED = "terminated"

    @property
    def label(self) -> Message:
        """Localized only by the presentation layer; safe for control flow."""
        return message_id(_LABELS[self])


_LABELS: dict[Stage, str] = {
    Stage.WAITING: "run.stage.waiting",
    Stage.STARTING: "run.stage.starting",
    Stage.CLEANUP_INCOMPLETE: "run.stage.cleanup_incomplete",
    Stage.VERIFYING_ARTIFACT: "run.stage.verifying_artifacts",
    Stage.TASK_UNSUPPORTED: "run.stage.task_unsupported",
    Stage.RUNTIME_VALIDATION: "run.stage.runtime_validation",
    Stage.INTERFACE_RESOLUTION: "run.stage.interface_resolution",
    Stage.DEPENDENCY_RESOLUTION: "run.stage.dependency_resolution",
    Stage.ENVIRONMENT_PREFLIGHT: "run.stage.environment_preflight",
    Stage.PREPARING_IMAGE: "run.stage.preparing_image",
    Stage.PREPARING_INPUT: "run.stage.preparing_inputs",
    Stage.CONFIRMING_INTERFACE: "run.stage.confirming_interface",
    Stage.PREPARING_ENVIRONMENT: "run.stage.preparing_environment",
    Stage.PREPARING_MODEL: "run.stage.preparing_model",
    Stage.STARTING_PROBE_CONTAINER: "run.stage.starting_probe_container",
    Stage.PREPARING_MEMORY_PROBE: "run.stage.preparing_memory_probe",
    Stage.SCANNING_MEMORY: "run.stage.scanning_memory",
    Stage.MEMORY_FOUND: "run.stage.memory_found",
    Stage.MEMORY_FEASIBILITY: "run.stage.memory_feasibility",
    Stage.PROBE_COMPLETED: "run.stage.probe_completed",
    Stage.PROBE_FAILED: "run.stage.probe_failed",
    Stage.PREPARING_MATRIX: "run.stage.preparing_matrix",
    Stage.STARTING_CONTAINER: "run.stage.starting_service",
    Stage.MAX_SCALE_PROBE: "run.stage.max_scale_probe",
    Stage.BUILDING_IMAGE: "run.stage.building_image",
    Stage.PLANNING_INPUT: "run.stage.planning_inputs",
    Stage.COMPUTE_PROFILING: "run.stage.compute_profiling",
    Stage.EXECUTION_PROFILING: "run.stage.execution_profiling",
    Stage.SERVER_READY: "run.stage.server_ready",
    Stage.MEASURING: "run.stage.measuring",
    Stage.CLEANING_CASE: "run.stage.cleaning_case",
    Stage.CASE_COMPLETED: "run.stage.case_completed",
    Stage.OOM_PRUNED: "run.stage.oom_pruned",
    Stage.MERGING_RESULTS: "run.stage.merging_results",
    Stage.COMPLETED: "run.stage.completed",
    Stage.FAILED: "common.state.failed",
    Stage.STOPPED: "run.stage.stopped",
    Stage.PARTIAL: "run.stage.partially_completed",
    Stage.TERMINATED: "run.stage.terminated",
}
