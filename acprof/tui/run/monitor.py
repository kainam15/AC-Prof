"""Derive monitor visibility from existing progress; never drive run lifecycle."""
from __future__ import annotations

from enum import Enum

from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.stages import Stage


class MonitorPhase(Enum):
    PREPARATION = "preparation"
    VALIDATION = "validation"
    MEASUREMENT = "measurement"
    FINISHED = "finished"


def monitor_phase(snapshot: ProgressSnapshot) -> MonitorPhase:
    stage = snapshot.stage
    if stage in {
        Stage.RUNTIME_VALIDATION, Stage.PREPARING_INPUT, Stage.PLANNING_INPUT,
        Stage.SERVER_READY, Stage.STARTING_PROBE_CONTAINER, Stage.PREPARING_MEMORY_PROBE,
        Stage.SCANNING_MEMORY, Stage.MEMORY_FOUND, Stage.MEMORY_FEASIBILITY,
        Stage.MAX_SCALE_PROBE, Stage.PROBE_COMPLETED, Stage.PROBE_FAILED,
    }:
        return MonitorPhase.VALIDATION
    if stage in {
        Stage.PREPARING_MATRIX, Stage.STARTING_CONTAINER, Stage.MEASURING,
        Stage.CLEANING_CASE, Stage.CASE_COMPLETED, Stage.OOM_PRUNED,
        Stage.MERGING_RESULTS, Stage.VERIFYING_ARTIFACT,
        Stage.COMPUTE_PROFILING, Stage.EXECUTION_PROFILING,
    }:
        return MonitorPhase.MEASUREMENT
    if stage in {
        Stage.COMPLETED, Stage.FAILED, Stage.PARTIAL, Stage.STOPPED,
        Stage.TERMINATED, Stage.CLEANUP_INCOMPLETE, Stage.TASK_UNSUPPORTED,
    }:
        return (MonitorPhase.MEASUREMENT if snapshot.total_cases and snapshot.probe_result is None
                else MonitorPhase.FINISHED)
    return MonitorPhase.PREPARATION


def visible_statuses(snapshot: ProgressSnapshot) -> tuple[tuple[str, str], ...]:
    phase = monitor_phase(snapshot)
    items: tuple[tuple[str, str], ...]
    if phase is MonitorPhase.MEASUREMENT:
        items = ()
    elif phase in {MonitorPhase.VALIDATION, MonitorPhase.FINISHED}:
        items = (("run.monitor.interface_prefix", snapshot.interface_status), ("run.monitor.runtime_prefix", snapshot.runtime_status))
    else:
        items = (("run.monitor.interface_prefix", snapshot.interface_status),) if snapshot.interface_status != "passed" else ()
    return tuple((label, status) for label, status in items if status != "not_started")
