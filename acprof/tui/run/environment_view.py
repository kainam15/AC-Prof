"""Pure projections of cached environment checks, without probes or launch gates."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from acprof.capabilities import PROFILER_PRIMARY_FIELDS, REQUIRED_MEASUREMENT_FIELDS
from acprof.messages import join_messages
from acprof.tui.i18n import message_id
from acprof.tui.run.source_groups import SourceSummary, source_group
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
    from acprof.platform import Environment
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
SOURCE_IMPACTS = {
    'base': 'runtime', 'docker': 'runtime', 'process': 'runtime', 'nvml': 'gpu',
    'rapl': 'cpu_memory', 'perf': 'cpu_memory', 'network': 'network', 'profilers': 'profilers',
}
COMPACT_LABELS = {
    'uv': 'uv', 'process_runtime': 'supervision', 'process_resources': 'resources',
    'process_cpu': 'cpu', 'process_memory': 'memory', 'container_cpu': 'cpu', 'container_memory': 'memory',
    'gpu_device': 'device', 'gpu_memory': 'memory', 'gpu_utilization': 'utilization', 'gpu_power': 'power',
    'cpu_energy': 'energy', 'cpu_instructions': 'instructions',
}
# Reuse post-run primary requirements; Advisor is post-hoc only and its
# legacy primary field is not part of the current CSV protocol.
CHECK_FIELDS = {
    **REQUIRED_MEASUREMENT_FIELDS,
    'gpu_memory': ('gpu_mem_used_avg_bytes', 'gpu_mem_used_peak_bytes'),
    'gpu_utilization': ('gpu_util_avg_pct',),
    **{('torch' if tool == 'torch_profiler_eager' else tool): (field,)
       for tool, field in PROFILER_PRIMARY_FIELDS.items() if tool != 'intel_advisor'},
    **dict.fromkeys(('tcpdump', 'tshark', 'capture_permissions', 'capture_interface'),
                    REQUIRED_MEASUREMENT_FIELDS['packet_latency']),
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
                      check_allowed: bool, navigate_allowed: bool, environment: Environment | None = None) -> EnvironmentCheckView:
    """Describe current impact separately from cached diagnostic provenance."""
    row = item.check
    view = check_view(row, technical_detail=technical_detail, check_allowed=check_allowed)
    if view.callout is not None:
        presentation = global_presentation(item)
        reason = row.reason_code or row.capability_status
        reason_label = (message_id('run.environment.reason.' + reason)
                        if reason in {'permission_denied', 'unsupported', 'not_installed', 'dependency_unavailable', 'error'} else '')
        problem = join_messages(' · ', tuple(part for part in (check_label(row), presentation.label, reason_label, cause) if part))
        next_step = (message_id('run.plan.invalid_next') if row.check_id.startswith('config_')
                     else message_id('run.status.repeat_next') if row.check_id == 'quality_repeat'
                     else message_id('run.resources.next') if item.target == 'resources'
                     else row.docker_diagnostic.next_step if row.docker_diagnostic else
                     message_id('run.docker_repair.next') if row.check_id == 'docker'
                     else repair_guidance(row, environment=environment))
        return EnvironmentCheckView(callout=replace(
            view.callout, problem=problem, next_step=next_step,
            impact=(message_id('run.status.blocked_impact') if item.blocks
                    else message_id('run.source.unrequested_impact') if item.requested is False
                    else message_id('run.docker_repair.impact.optional') if row.check_id == 'docker'
                    else message_id('run.plan.insufficient_windows') if row.check_id == 'quality_repeat'
                    else view.callout.impact),
            actions=(ViewAction('environment-retry', message_id('run.environment.retry'), disabled=not check_allowed),
                     *((ViewAction('status-open', message_id('run.docker_repair.open' if row.check_id == 'docker' else 'run.status.open'), primary=True, disabled=not navigate_allowed),)
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
            FactRow(message_id("run.source.requested"), request_label(item.requested)),
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
        reason = row.reason_code or row.capability_status
        if reason in {"permission_denied", "unsupported", "not_installed", "dependency_unavailable"}:
            problem = join_messages(" · ", (
                problem, message_id("run.environment.reason." + reason),
            ))
        section = SOURCE_IMPACTS.get(source_group(row.check_id), "other")
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


def compact_check_label(row: PreflightCheck) -> str:
    key = COMPACT_LABELS.get(row.check_id)
    return message_id('run.source.check.' + key) if key else check_label(row)


def request_label(requested: bool | None) -> str:
    return message_id('run.source.request.' + ('yes' if requested else 'no' if requested is False else 'unknown'))


def source_title(source: SourceSummary) -> str:
    label = message_id('structured.status.checking' if source.state == 'checking'
                       else 'run.environment.group.' + source.state)
    symbol = {'error': '×', 'warning': '!', 'success': '✓', 'pending': '…', 'checking': '…', 'not_requested': '-'}[source.state]
    title = join_messages(' · ', (message_id('run.source.' + source.group_id), join_messages(' ', (symbol, label))))
    if source.issue_count:
        title = join_messages(' · ', (title, message_id('run.source.issue_count', source.issue_count)))
    return title


def source_overview(source: SourceSummary) -> str:
    lines = [message_id('run.source.scope', len(source.items), sum(item.requested is True for item in source.items),
                        sum(item.requested is None for item in source.items))]
    for issue in source.issues:
        if issue.group not in {'error', 'warning'}:
            continue
        affected = join_messages(' / ', tuple(compact_check_label(item.check) for item in issue.items))
        lines.append(message_id('run.source.issue', message_id('run.environment.group.' + issue.group), affected))
    risks = tuple(dict.fromkeys(field for item in source.items
                               if item.requested is True and item.group in {'error', 'warning'}
                               for field in CHECK_FIELDS.get(item.check.check_id, ())))
    if risks:
        lines.append(message_id('run.source.risk_fields', ', '.join(risks)))
    lines.append(message_id('run.source.actual'))
    return join_messages('\n', lines)


def repair_guidance(row: PreflightCheck, *, environment: Environment | None = None) -> str:
    """Platform-specific instructions only; never install, probe or change a gate."""
    base = message_id('run.environment.remedy.' + REMEDIES.get(row.check_id, 'generic'))
    if row.reason_code != 'not_installed':
        return base
    tool = row.check_id
    key = ('capture' if tool in {'tcpdump', 'tshark'} else tool)
    if key not in {'capture', 'uv', 'ncu', 'nsys', 'intel_advisor', 'torch', 'massif'}:
        return base
    scope = 'wsl' if environment is not None and environment.environment == 'wsl2' else 'linux'
    if environment is not None and environment.system and environment.system != 'Linux':
        scope = 'unsupported'
    return join_messages('\n', (base, message_id('run.source.install.' + scope), message_id('run.source.install.' + key)))
