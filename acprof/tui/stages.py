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
    Stage.WAITING: "ui.0692",
    Stage.STARTING: "ui.0764",
    Stage.CLEANUP_INCOMPLETE: "ui.0258",
    Stage.VERIFYING_ARTIFACT: "ui.0196",
    Stage.TASK_UNSUPPORTED: "ui.0585",
    Stage.RUNTIME_VALIDATION: "ui.0237",
    Stage.INTERFACE_RESOLUTION: "ui.0233",
    Stage.DEPENDENCY_RESOLUTION: "ui.0238",
    Stage.ENVIRONMENT_PREFLIGHT: "ui.0234",
    Stage.PREPARING_IMAGE: "ui.0235",
    Stage.PREPARING_INPUT: "ui.0236",
    Stage.CONFIRMING_INTERFACE: "ui.0053",
    Stage.PREPARING_ENVIRONMENT: "ui.0054",
    Stage.PREPARING_MODEL: "ui.0055",
    Stage.STARTING_PROBE_CONTAINER: "ui.0863",
    Stage.PREPARING_MEMORY_PROBE: "ui.0865",
    Stage.SCANNING_MEMORY: "ui.0867",
    Stage.MEMORY_FOUND: "ui.0769",
    Stage.MEMORY_FEASIBILITY: "ui.0874",
    Stage.PROBE_COMPLETED: "ui.0768",
    Stage.PROBE_FAILED: "ui.0772",
    Stage.PREPARING_MATRIX: "ui.0877",
    Stage.STARTING_CONTAINER: "ui.0776",
    Stage.MAX_SCALE_PROBE: "ui.0775",
    Stage.BUILDING_IMAGE: "ui.0777",
    Stage.PLANNING_INPUT: "ui.0778",
    Stage.COMPUTE_PROFILING: "ui.0881",
    Stage.EXECUTION_PROFILING: "ui.0882",
    Stage.SERVER_READY: "ui.0779",
    Stage.MEASURING: "ui.0774",
    Stage.CLEANING_CASE: "ui.0780",
    Stage.CASE_COMPLETED: "ui.0770",
    Stage.OOM_PRUNED: "ui.0781",
    Stage.MERGING_RESULTS: "ui.0884",
    Stage.COMPLETED: "ui.0767",
    Stage.FAILED: "ui.0771",
    Stage.STOPPED: "ui.0195",
    Stage.PARTIAL: "ui.0194",
    Stage.TERMINATED: "ui.0773",
}
