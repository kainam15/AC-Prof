"""Pure projections of cached environment checks, without probes or launch gates."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from acprof.messages import join_messages
from acprof.tui.i18n import message_id
from acprof.tui.ui.view_models import (
    CalloutContent,
    FactRow,
    StatusPresentation,
    StatusSummary,
    ViewAction,
    ViewSection,
    status_presentation,
)

if TYPE_CHECKING:
    from acprof.tui.run.diagnostics import PreflightCheck
    from acprof.tui.run.global_status import StatusItem


LABELS = {
    "environment": "environment", "docker": "docker", "cgroup": "cgroup", "uv": "uv",
    "process_resources": "process_resources", "latency": "latency", "throughput": "throughput",
    "container_cpu": "cgroup_cpu", "container_memory": "cgroup_memory", "process_cpu": "process_cpu",
    "process_memory": "process_memory", "process_runtime": "process_runtime", "ncu": "ncu", "nsys": "nsys",
    "intel_advisor": "advisor", "torch": "torch", "massif": "massif",
    "cpu_energy": "rapl", "cpu_instructions": "perf", "capture_interface": "capture",
    "capture_permissions": "capture_permissions",
    "gpu_device": "gpu", "gpu_memory": "gpu_memory", "gpu_utilization": "gpu_utilization", "gpu_power": "gpu_power",
    "profiler": "profiler", "check_error": "check_error",
}
REMEDIES = {
    "environment": "platform", "docker": "docker", "cgroup": "cgroup", "uv": "uv",
    "process_resources": "resources", "cpu_energy": "rapl", "cpu_instructions": "permissions",
    "tcpdump": "capture_tools", "tshark": "capture_tools", "capture_permissions": "permissions",
    "capture_interface": "interface", "gpu_device": "gpu", "gpu_memory": "gpu", "gpu_utilization": "gpu",
    "gpu_power": "gpu", "process_cpu": "process_metrics", "process_memory": "process_metrics",
    "ncu": "profiler", "nsys": "profiler", "intel_advisor": "profiler", "torch": "profiler",
    "massif": "profiler", "profiler": "profiler",
}
CAPABILITY_SECTIONS = {
    "runtime": ("environment", "docker", "cgroup", "uv", "process_runtime", "process_resources"),
    "cpu_memory": ("process_cpu", "process_memory", "container_cpu", "container_memory", "cpu_energy", "cpu_instructions"),
    "gpu": ("gpu_device", "gpu_memory", "gpu_utilization", "gpu_power"),
    "network": ("tcpdump", "tshark", "capture_permissions", "capture_interface"),
    "profilers": ("ncu", "nsys", "intel_advisor", "torch", "massif", "profiler"),
}

# Normalize only presentation aliases; recorded statuses still decide grouping.
_DISPLAY_STATES = {
    "ok": ("ok", "run.environment.item.ok"),
    "fail": ("failed", "run.environment.item.fail"),
    "error": ("failed", "run.environment.item.error"),
    "warn": ("limited", "run.environment.item.warn"),
    "pending": ("pending", "run.environment.item.pending"),
    "declared": ("unverified", "run.environment.item.declared"),
    "not_requested": ("unknown", "run.environment.item.not_requested"),
    "checking": ("checking", "structured.status.checking"),
    "stale": ("stale", "run.environment.state.stale"),
}
_NOTES = {
    "ok": "run.environment.unverified",
    "declared": "run.environment.note.declared",
    "pending": "run.environment.note.pending",
    "not_requested": "run.environment.note.not_requested",
    "checking": "run.preflight.checking_notice",
    "stale": "run.status.stale",
}


@dataclass(frozen=True, slots=True)
class EnvironmentCheckView:
    callout: CalloutContent | None = None
    facts: ViewSection | None = None


def item_status(row: PreflightCheck) -> str:
    return "error" if row.capability_status == "error" and row.status in {"ok", "fail", "error"} else row.status


def check_group(row: PreflightCheck) -> str:
    return {"ok": "success", "fail": "error", "error": "error", "warn": "warning",
            "not_requested": "not_requested"}.get(item_status(row), "pending")


def check_priority(row: PreflightCheck) -> int:
    """Keep the existing order, with future states beside unverified checks."""
    return {"fail": 0, "error": 0, "warn": 1, "ok": 2, "declared": 3, "pending": 3,
            "not_requested": 4}.get(item_status(row), 3)


def check_label(row: PreflightCheck) -> str:
    return message_id("run.environment.label." + LABELS[row.check_id]) if row.check_id in LABELS else row.label


def check_presentation(row: PreflightCheck) -> StatusPresentation:
    """Reuse shared symbols while naming machine availability, not launch gates."""
    alias, label = _DISPLAY_STATES.get(item_status(row), ("unknown", "common.value.unknown"))
    presentation = status_presentation(alias)
    return StatusPresentation(presentation.level, presentation.symbol, message_id(label))


def global_presentation(item: StatusItem) -> StatusPresentation:
    if item.group == 'error':
        return StatusPresentation('error', '×', message_id('run.status.blocked'))
    if item.check.status == 'ok':
        return status_presentation('passed')
    if item.check.status == 'pending':
        return status_presentation('pending')
    if item.check.status == 'not_requested':
        return StatusPresentation('neutral', '-', message_id('run.environment.group.not_requested'))
    return check_presentation(item.check)


def global_check_view(item: StatusItem, *, technical_detail: str, cause: str,
                      check_allowed: bool, navigate_allowed: bool) -> EnvironmentCheckView:
    """Describe current impact separately from cached diagnostic provenance."""
    row = item.check
    view = check_view(row, technical_detail=technical_detail, check_allowed=check_allowed)
    if view.callout is not None:
        presentation = global_presentation(item)
        problem = join_messages(' · ', (check_label(row), presentation.label, cause))
        next_step = (message_id('run.plan.invalid_next') if row.check_id.startswith('config_')
                     else message_id('run.status.repeat_next') if row.check_id == 'quality_repeat'
                     else message_id('run.resources.next') if item.target == 'resources'
                     else view.callout.next_step)
        return EnvironmentCheckView(callout=replace(
            view.callout, problem=problem, next_step=next_step,
            impact=(message_id('run.status.blocked_impact') if item.blocks
                    else message_id('run.plan.insufficient_windows') if row.check_id == 'quality_repeat'
                    else view.callout.impact),
            actions=(ViewAction('environment-retry', message_id('run.environment.retry'), disabled=not check_allowed),
                     *((ViewAction('status-open', message_id('run.status.open'), primary=True, disabled=not navigate_allowed),)
                       if item.target else ())),
        ))
    if view.facts is not None:
        note = (message_id('run.status.not_requested') if row.status == 'not_requested'
                else message_id('run.status.stale') if row.status == 'stale'
                else message_id('run.preflight.checking_notice') if row.status == 'checking'
                else message_id('run.preflight.configuration_changed') if row.status == 'pending'
                else message_id('run.resources.inherited') if row.resource_diagnostic and row.resource_diagnostic.code == 'inherited'
                else cause or view.facts.rows[1].value)
        return EnvironmentCheckView(facts=replace(view.facts, rows=(
            replace(view.facts.rows[0], value=global_presentation(item).label), replace(view.facts.rows[1], value=note),
        )))
    return view


def check_view(row: PreflightCheck, *, technical_detail: str, check_allowed: bool) -> EnvironmentCheckView:
    """Project cached evidence; the caller owns translation and credential redaction."""
    status = item_status(row)
    presentation = check_presentation(row)
    if status in {"warn", "fail", "error"}:
        problem = join_messages(" ", (
            presentation.symbol,
            message_id("run.environment.capability_line", check_label(row), presentation.label),
        ))
        if row.capability_status in {"permission_denied", "unsupported"}:
            problem = join_messages(" · ", (
                problem, message_id("run.environment.reason." + row.capability_status),
            ))
        section = next((name for name, identifiers in CAPABILITY_SECTIONS.items()
                        if row.check_id in identifiers), "other")
        impact = "check_error" if status == "error" else section
        return EnvironmentCheckView(callout=CalloutContent(
            presentation.level,
            problem,
            message_id("run.environment.impact." + impact),
            message_id("run.environment.remedy." + REMEDIES.get(row.check_id, "generic")),
            detail=technical_detail,
            actions=(ViewAction("environment-retry", message_id("run.environment.retry"),
                                primary=True, disabled=not check_allowed),),
        ))
    alias = _DISPLAY_STATES.get(status, ("unknown", "common.value.unknown"))[0]
    return EnvironmentCheckView(facts=ViewSection(
        check_label(row),
        (
            FactRow(message_id("common.label.status"), presentation.label, status=alias),
            FactRow(message_id("common.field.note"), message_id(_NOTES.get(status, "run.environment.note.unknown"))),
        ),
        detail_rows=(FactRow(message_id("structured.detail.title"), technical_detail,
                             detail=technical_detail, copyable=True),),
    ))


def group_summary(rows: Sequence[PreflightCheck], *, group: str, state: str) -> StatusSummary:
    """Describe one inventory group without deriving configuration or time validity."""
    selected = tuple(row for row in rows if check_group(row) == group)
    group_key = group if group in {"error", "warning", "success", "pending"} else "pending"
    headline: str = message_id("run.environment.group_count", message_id("run.environment.group." + group_key), len(selected))
    if group == "error" and selected:
        parts = []
        for status in ("fail", "error"):
            count = sum(item_status(row) == status for row in selected)
            if count:
                label: str = message_id("run.environment.item.fail" if status == "fail" else "run.environment.item.error")
                parts.append(message_id("run.environment.group_count", label, count))
        headline = join_messages(" · ", parts)
    elif group == "pending" and selected:
        statuses = {item_status(row) for row in selected}
        if len(statuses) == 1 and next(iter(statuses)) in {"not_requested", "declared"}:
            label = check_presentation(selected[0]).label
            headline = message_id("run.environment.group_count", label, len(selected))
    state_labels = {
        "checking": ("info", "run.environment.state.checking"),
        "error": ("error", "run.environment.state.error"),
        "stale": ("neutral", "run.environment.state.stale"),
    }
    if state in state_labels:
        level, key = state_labels[state]
        return StatusSummary(level if selected else "neutral", join_messages(" · ", (message_id(key), headline)))
    if state not in {"ready", "limited"}:
        return StatusSummary("neutral", join_messages(" · ", (message_id("common.value.unknown"), headline)))
    if not selected:
        return StatusSummary("neutral", headline)
    return StatusSummary({"error": "error", "warning": "warning", "success": "success"}.get(group, "neutral"), headline)
