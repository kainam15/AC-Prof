"""Pure monitor-header projections of authoritative run progress snapshots."""
from __future__ import annotations

from dataclasses import dataclass

from acprof.tui.i18n import message_id
from acprof.tui.run.monitor import MonitorPhase, monitor_phase, visible_statuses
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.stages import Stage
from acprof.tui.ui.view_models import FactRow, StatusSummary, ViewSection, status_presentation

# Display vocabulary only; execution and result validation remain the owners of Stage.
_STAGE_STATES = {
    Stage.WAITING: "pending",
    Stage.SERVER_READY: "ready",
    Stage.COMPLETED: "complete",
    Stage.PROBE_COMPLETED: "complete",
    Stage.MEMORY_FOUND: "complete",
    Stage.CASE_COMPLETED: "complete",
    Stage.FAILED: "failed",
    Stage.CLEANUP_INCOMPLETE: "failed",
    Stage.PROBE_FAILED: "failed",
    Stage.TASK_UNSUPPORTED: "failed",
    Stage.TERMINATED: "failed",
    Stage.STOPPED: "failed",
    Stage.PARTIAL: "partial",
    Stage.OOM_PRUNED: "limited",
}


@dataclass(frozen=True, slots=True)
class MonitorHeaderView:
    summary: StatusSummary
    cases: ViewSection | None
    resources: ViewSection | None
    checks: ViewSection | None
    details: ViewSection | None
    progress_total: int | None
    progress_completed: int


def _summary(snapshot: ProgressSnapshot) -> StatusSummary:
    state = status_presentation(_STAGE_STATES.get(snapshot.stage, "running"))
    headline = snapshot.stage.label
    if snapshot.warnings or snapshot.errors:
        headline = message_id("run.monitor.stage_with_counts", headline, snapshot.warnings, snapshot.errors)
    return StatusSummary(state.level, headline)


def _resource_value(raw: str, *, memory: bool = False, gpu: bool = False) -> str:
    if raw in ("", "-", "unknown"):
        return message_id("common.value.unknown")
    if raw == "inherited":
        return message_id("run.resources.inherited_limits")
    if gpu and raw in {"on", "off"}:
        return message_id("common.option.on") if raw == "on" else message_id("common.option.off")
    return f"{raw}GB" if memory else raw


def _resource_section(snapshot: ProgressSnapshot) -> ViewSection:
    return ViewSection(message_id("run.monitor.resources_label"), (
        FactRow("CPU", _resource_value(snapshot.cpu), detail=snapshot.cpu, copyable=True),
        FactRow(message_id("results.view.memory"), _resource_value(snapshot.mem, memory=True),
                detail=snapshot.mem, copyable=True),
        FactRow("GPU", _resource_value(snapshot.gpu, gpu=True), detail=snapshot.gpu, copyable=True),
    ))


def _check_row(label: str, raw: str) -> FactRow:
    if raw == "waiting":
        return FactRow(message_id(label), message_id("run.status.needs_confirmation"),
                       status="pending", detail=raw)
    return FactRow(message_id(label), status_presentation(raw).label, status=raw, detail=raw)


def _check_section(snapshot: ProgressSnapshot) -> ViewSection | None:
    rows = tuple(_check_row(label, raw) for label, raw in visible_statuses(snapshot))
    return ViewSection(message_id("run.monitor.checks_title"), rows) if rows else None


def _detail_section(snapshot: ProgressSnapshot) -> ViewSection | None:
    rows = []
    if snapshot.detail and (
        snapshot.stage is Stage.WAITING or snapshot.detail != message_id("run.monitor.not_started")
    ):
        rows.append(FactRow(message_id("common.action.details"), snapshot.detail,
                            detail=snapshot.detail, copyable=True))
    if snapshot.warnings:
        rows.append(FactRow(message_id("run.monitor.warning_count"), str(snapshot.warnings), status="warning"))
    if snapshot.errors:
        rows.append(FactRow(message_id("run.monitor.error_count"), str(snapshot.errors), status="failed"))
    return ViewSection(message_id("run.monitor.details_title"), tuple(rows), collapsed=True) if rows else None


def monitor_header_view(snapshot: ProgressSnapshot) -> MonitorHeaderView:
    """Project known facts without clocks, I/O, inferred readiness or new run state."""
    measurement = monitor_phase(snapshot) is MonitorPhase.MEASUREMENT
    cases = None
    total = None
    completed = 0
    if measurement and snapshot.total_cases > 0:
        current = str(snapshot.current_case) if snapshot.current_case else message_id("common.value.not_applicable")
        cases = ViewSection(message_id("run.monitor.cases_title"), (
            FactRow(message_id("run.monitor.current_case_label"), current),
            FactRow(message_id("run.monitor.processed_label"), f"{snapshot.completed_cases}/{snapshot.total_cases}"),
        ))
        # These are processed cases, including failures and skips, not valid windows.
        # Keep inconsistent counts visible as facts without inventing a bounded ratio.
        if 0 <= snapshot.completed_cases <= snapshot.total_cases:
            total, completed = snapshot.total_cases, snapshot.completed_cases
    resources = _resource_section(snapshot) if measurement and snapshot.current_case > 0 else None
    return MonitorHeaderView(
        summary=_summary(snapshot), cases=cases, resources=resources,
        checks=_check_section(snapshot), details=_detail_section(snapshot),
        progress_total=total, progress_completed=completed,
    )
