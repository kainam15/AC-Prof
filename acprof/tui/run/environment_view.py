"""Pure projections of cached environment checks, without probes or launch gates."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from acprof.capabilities import PROFILER_PRIMARY_FIELDS, REQUIRED_MEASUREMENT_FIELDS
from acprof.host.preparation.preparation_log import redact_text
from acprof.messages import join_messages
from acprof.tui.i18n import message_id
from acprof.tui.run.resource_view import resource_recovery_messages
from acprof.tui.run.source_groups import SourceSummary, source_group
from acprof.tui.ui.view_models import (
    CalloutContent,
    FactRow,
    StatusPresentation,
    StatusSummary,
    ViewAction,
    ViewSection,
    status_presentation,
    status_symbol,
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
    "cpu_energy": "rapl", "dram_energy": "dram_energy", "cpu_instructions": "perf", "capture_interface": "capture",
    "packet_latency": "packet_latency", "cgroup_metrics": "cgroup_metrics",
    "capture_permissions": "capture_permissions",
    "gpu_device": "gpu", "gpu_memory": "gpu_memory", "gpu_utilization": "gpu_utilization", "gpu_power": "gpu_power",
    "profiler": "profiler", "check_error": "check_error",
}
REMEDIES = {
    "environment": "platform", "docker": "docker", "cgroup": "cgroup", "uv": "uv",
    "process_resources": "resources", "cpu_energy": "rapl", "dram_energy": "rapl", "cpu_instructions": "permissions",
    "packet_latency": "capture_tools", "cgroup_metrics": "cgroup",
    "tcpdump": "capture_tools", "tshark": "capture_tools", "capture_permissions": "permissions",
    "capture_interface": "interface", "gpu_device": "gpu", "gpu_memory": "gpu", "gpu_utilization": "gpu",
    "gpu_power": "gpu", "process_cpu": "process_metrics", "process_memory": "process_metrics",
    "ncu": "profiler", "nsys": "profiler", "intel_advisor": "profiler", "torch": "profiler",
    "massif": "profiler", "profiler": "profiler",
}
SOURCE_IMPACTS = {
    'base': 'runtime', 'docker': 'runtime', 'cgroup': 'cpu_memory', 'process': 'runtime', 'nvml': 'gpu',
    'rapl': 'cpu_memory', 'perf': 'cpu_memory', 'network': 'network', 'profilers': 'profilers',
}
COMPACT_LABELS = {
    'uv': 'uv', 'process_runtime': 'supervision', 'process_resources': 'resources',
    'process_cpu': 'cpu', 'process_memory': 'memory', 'container_cpu': 'cpu', 'container_memory': 'memory',
    'gpu_device': 'device', 'gpu_memory': 'memory', 'gpu_utilization': 'utilization', 'gpu_power': 'power',
    'cpu_instructions': 'instructions',
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
    "ok": ("ok", "common.state.passed"),
    "fail": ("failed", "run.environment.item.fail"),
    "error": ("failed", "run.environment.state.error"),
    "warn": ("warn", "run.environment.item.fail"),
    "pending": ("pending", "run.runtime.pending_selection"),
    "declared": ("unverified", "run.environment.item.declared"),
    "not_requested": ("unknown", "run.environment.group.not_requested"),
    "checking": ("checking", "structured.status.checking"),
    "stale": ("stale", "run.environment.state.stale"),
}
_REASON_LABELS = {
    'not_installed': 'common.state.not_installed',
    'dependency_unavailable': 'run.environment.reason.dependency_unavailable',
    'permission_denied': 'run.environment.reason.permission_denied',
    'unsupported': 'run.environment.reason.unsupported',
    'error': 'run.environment.state.error',
    'partial': 'run.environment.state.limited',
    'not_applicable': 'common.value.not_applicable',
    'tool_missing': 'run.environment.reason.tool_missing',
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


def _reason(row: PreflightCheck) -> str:
    if row.reason_code:
        return row.reason_code
    if row.docker_diagnostic:
        return {
            'cli_missing': 'not_installed', 'socket_permission': 'permission_denied',
            'context_unsupported': 'unsupported', 'desktop_unsupported': 'unsupported',
            'unsupported_environment': 'unsupported', 'check_failed': 'error',
        }.get(row.docker_diagnostic.code, row.docker_diagnostic.code)
    if row.resource_diagnostic and row.resource_diagnostic.code in _REASON_LABELS:
        return row.resource_diagnostic.code
    return row.capability_status


def check_presentation(row: PreflightCheck) -> StatusPresentation:
    """The evidence names the problem; the recorded status supplies its severity."""
    status = item_status(row)
    alias, label = _DISPLAY_STATES.get(status, ("unknown", "common.value.unknown"))
    if status in {"warn", "fail", "error"}:
        reason = _reason(row)
        if row.check_id.startswith('config_'):
            label = 'run.plan.invalid'
        elif row.check_id == 'quality_repeat':
            label = 'run.plan.flow_check_limited'
        elif row.docker_diagnostic and reason in {'cli_not_on_path', 'daemon_unreachable', 'context_unavailable'}:
            label = 'run.environment.reason.' + reason
        else:
            label = _REASON_LABELS.get(reason, label)
    return replace(status_presentation(alias), label=message_id(label))


def check_title(row: PreflightCheck, presentation: StatusPresentation) -> str:
    return message_id("run.environment.capability_line", check_label(row), presentation.label)


def global_check_view(item: StatusItem, *, technical_detail: str, tr: Callable[[str], str],
                      check_allowed: bool, navigate_allowed: bool, environment: Environment | None = None) -> EnvironmentCheckView:
    """Describe current impact separately from cached diagnostic provenance."""
    row = item.check
    view = check_view(row, technical_detail=technical_detail, check_allowed=check_allowed, environment=environment)
    cause = next((line.strip() for line in tr(row.detail).splitlines() if line.strip()), '')
    if row.capability_status:
        cause = cause.removeprefix(row.capability_status + ': ').removeprefix(row.capability_status + ' · ')
    if row.affected_fields:
        cause = cause.removesuffix(' · ' + ', '.join(row.affected_fields))
    cause = redact_text(cause, limit=180)
    if view.callout is not None:
        problem = view.callout.problem
        if cause and cause.casefold() != tr(check_presentation(row).label).casefold():
            problem = join_messages('\n', (problem, cause))
        next_step = (message_id('run.plan.invalid_next') if row.check_id.startswith('config_')
                     else message_id('run.status.repeat_next') if row.check_id == 'quality_repeat'
                     else resource_recovery_messages(row.resource_diagnostic.code)[1]
                     if item.target == 'resources' and row.resource_diagnostic else message_id('run.resources.next')
                     if item.target == 'resources'
                     else row.docker_diagnostic.next_step if row.docker_diagnostic else
                     message_id('run.docker_repair.next') if row.check_id == 'docker'
                     else view.callout.next_step)
        impact = (message_id('run.status.blocked_impact') if item.blocks
                  else message_id('run.docker_repair.impact.optional') if row.check_id == 'docker'
                  else message_id('run.source.unrequested_impact') if item.requested is False
                  else message_id('run.plan.insufficient_windows') if row.check_id == 'quality_repeat'
                  else view.callout.impact)
        fields = row.affected_fields or CHECK_FIELDS.get(row.check_id)
        if fields:
            impact = join_messages('\n', (impact, message_id('run.source.related_fields', ', '.join(fields))))
        if row.check_id == 'quality_repeat' and item.target:
            actions = (
                ViewAction('plan-set-repeat-3', message_id('run.plan.set_repeat_3'),
                           primary=True, disabled=not navigate_allowed),
                ViewAction('status-open', message_id('run.status.open'), disabled=not navigate_allowed),
            )
        else:
            actions = (ViewAction('environment-retry', message_id('run.environment.retry'), disabled=not check_allowed),
                       *((ViewAction('status-open', message_id('run.docker_repair.open' if row.check_id == 'docker' else 'run.status.open'), primary=True, disabled=not navigate_allowed),)
                         if item.target else ()))
        return EnvironmentCheckView(callout=replace(
            view.callout, problem=problem, next_step=next_step, impact=impact, actions=actions,
        ))
    if view.facts is not None:
        note = (message_id('run.status.not_requested') if row.status == 'not_requested'
                else message_id('run.status.stale') if row.status == 'stale'
                else message_id('run.preflight.checking_notice') if row.status == 'checking'
                else message_id('run.preflight.configuration_changed') if row.status == 'pending'
                else message_id('run.resources.inherited') if row.resource_diagnostic and row.resource_diagnostic.code == 'inherited'
                else cause or view.facts.rows[1].value)
        return EnvironmentCheckView(facts=replace(view.facts, rows=(
            view.facts.rows[0], replace(view.facts.rows[1], value=note),
            FactRow(message_id("run.source.requested"), request_label(item.requested)),
        )))
    return view


def check_view(row: PreflightCheck, *, technical_detail: str, check_allowed: bool,
               environment: Environment | None = None) -> EnvironmentCheckView:
    """Project cached evidence; the caller owns translation and credential redaction."""
    status = item_status(row)
    presentation = check_presentation(row)
    if status in {"warn", "fail", "error"}:
        problem = join_messages(" ", (presentation.symbol, check_title(row, presentation)))
        section = SOURCE_IMPACTS.get(source_group(row.check_id), "other")
        impact = "check_error" if status == "error" or _reason(row) == "error" else section
        return EnvironmentCheckView(callout=CalloutContent(
            presentation.level,
            problem,
            message_id("run.environment.impact." + impact),
            repair_guidance(row, environment=environment),
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
        counts: dict[str, int] = {}
        for row in selected:
            label = check_presentation(row).label
            counts[label] = counts.get(label, 0) + 1
        headline = join_messages(" · ", (message_id("run.environment.group_count", label, count)
                                         for label, count in counts.items()))
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


def source_status_segments(source: SourceSummary) -> tuple[tuple[str, str], ...]:
    """Count visible checks by status; global issue deduplication stays separate."""
    counts: dict[str, int] = {}
    for item in source.items:
        counts[item.group] = counts.get(item.group, 0) + 1
    levels = {'error': 'error', 'warning': 'warning', 'success': 'success',
              'pending': 'info', 'not_requested': 'neutral'}
    return tuple((group, f'{status_symbol(level)} {counts[group]}')
                 for group, level in levels.items() if counts.get(group))


def source_title(source: SourceSummary) -> str:
    return join_messages(' · ', (message_id('run.source.' + source.group_id),
                                  *(label for _, label in source_status_segments(source))))


def source_overview(source: SourceSummary) -> str:
    lines = [message_id('run.source.scope', len(source.items), sum(item.requested is True for item in source.items),
                        sum(item.requested is None for item in source.items))]
    for issue in source.issues:
        if issue.group not in {'error', 'warning'}:
            continue
        affected = join_messages(' / ', tuple(compact_check_label(item.check) for item in issue.items))
        lines.append(message_id('run.source.issue', check_presentation(issue.primary.check).label, affected))
    risks = tuple(dict.fromkeys(field for item in source.items
                               if item.requested is True and item.group in {'error', 'warning'}
                               for field in item.check.affected_fields or CHECK_FIELDS.get(item.check.check_id, ())))
    if risks:
        lines.append(message_id('run.source.risk_fields', ', '.join(risks)))
    lines.append(message_id('run.source.actual'))
    return join_messages('\n', lines)


def repair_guidance(row: PreflightCheck, *, environment: Environment | None = None) -> str:
    """Platform-specific instructions only; never install, probe or change a gate."""
    base = message_id('run.environment.remedy.' + REMEDIES.get(row.check_id, 'generic'))
    reason = _reason(row)
    if row.check_id == 'cpu_instructions' and reason in {'tool_missing', 'not_installed'}:
        return message_id('run.source.install.perf')
    if reason == 'dependency_unavailable':
        if row.root_cause == 'profiler_docker_unavailable':
            return message_id('run.environment.remedy.docker_dependency')
        key = {('capture_permissions', 'tcpdump'): 'tcpdump_dependency',
               ('capture_permissions', 'getcap'): 'getcap_dependency',
               ('capture_interface', 'ip'): 'ip_dependency',
               ('gpu_device', 'nvidia-smi'): 'nvidia_smi_dependency'}.get((row.check_id, row.source), 'dependency')
        return message_id('run.environment.remedy.' + key)
    if reason == 'permission_denied':
        key = {'cpu_energy': 'rapl_access', 'dram_energy': 'rapl_access', 'cpu_instructions': 'permissions',
               'capture_permissions': 'permissions', 'packet_latency': 'permissions',
               'process_resources': 'resources'}.get(row.check_id, 'access')
        return message_id('run.environment.remedy.' + key)
    if reason == 'unsupported':
        if row.source == 'platform_policy':
            return message_id('run.environment.remedy.native_hardware')
        if source_group(row.check_id) == 'nvml':
            return message_id('run.environment.remedy.nvml_support')
        return message_id('run.environment.remedy.unsupported')
    if reason != 'not_installed':
        return base
    tool = row.check_id
    key = ('capture' if tool in {'tcpdump', 'tshark'} else tool)
    if key not in {'capture', 'uv', 'ncu', 'nsys', 'intel_advisor', 'torch', 'massif'}:
        return base
    scope = 'wsl' if environment is not None and environment.environment == 'wsl2' else 'linux'
    if environment is not None and environment.system and environment.system != 'Linux':
        scope = 'unsupported'
    if key in {'torch', 'massif'}:
        return message_id('run.source.install.' + key)
    return join_messages('\n', (message_id('run.source.install.' + scope), message_id('run.source.install.' + key)))
