"""One immutable projection for global diagnostics and collection admission.

Inputs are cached evidence and current form validation only. This module never
probes the host, repairs permissions, or owns a second copy of controller state.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from acprof.capabilities import measurement_requested
from acprof.experiments.configuration import ConfigIssue, RunConfig
from acprof.messages import join_messages
from acprof.tui.i18n import message_id
from acprof.tui.run.diagnostics import PreflightCheck

GROUPS = ('error', 'warning', 'success', 'pending', 'not_requested')
_GROUP = {'fail': 'error', 'error': 'error', 'warn': 'warning', 'ok': 'success',
          'not_requested': 'not_requested'}
_TARGETS = {
    'process_resources': 'resources', 'cpu_energy': 'permissions',
    'cpu_instructions': 'permissions', 'capture_permissions': 'permissions',
    'capture_interface': 'field:sniff_iface', 'runtime': 'field:runtime',
    'runtime_auto': 'field:runtime', 'process_runtime': 'field:runtime',
    'ncu': 'field:compute_profile_tool', 'torch': 'field:compute_profile_tool',
    'massif': 'field:execution_profile_tool', 'nsys': 'field:execution_profile_tool',
    'gpu_device': 'field:gpus', 'gpu_memory': 'field:gpus',
    'gpu_utilization': 'field:gpus', 'gpu_power': 'field:gpus',
}


@dataclass(frozen=True, slots=True)
class StatusItem:
    check: PreflightCheck
    required: bool = False
    target: str = 'environment'

    @property
    def group(self) -> str:
        return _GROUP.get(self.check.status, 'pending')

    @property
    def blocks(self) -> bool:
        return self.required and self.group in {'error', 'pending'}


@dataclass(frozen=True, slots=True)
class GlobalStatus:
    items: tuple[StatusItem, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {group: sum(item.group == group for item in self.items) for group in GROUPS}

    @property
    def blockers(self) -> tuple[StatusItem, ...]:
        return tuple(item for item in self.items if item.blocks)

    @property
    def can_start(self) -> bool:
        return not self.blockers

    @property
    def state(self) -> str:
        if any(item.check.check_id.startswith('config_') and item.group == 'error' for item in self.items):
            return 'incomplete'
        if any(item.group == 'error' for item in self.blockers):
            return 'failed'
        if any(item.check.status == 'checking' for item in self.blockers):
            return 'checking'
        if self.blockers:
            return 'pending'
        return 'limited' if self.counts['warning'] else 'ready'


def _requested(identifier: str, config: RunConfig, runtime: str) -> bool:
    gpu = 'on' in config.gpus.split(',')
    if identifier in {'ncu', 'torch'}:
        return config.compute_profile_tool in {identifier, 'both'}
    if identifier == 'intel_advisor':
        return False  # Post-hoc tool; no experiment configuration requests it.
    if identifier in {'nsys', 'massif'}:
        return config.execution_profile_tool in {identifier, 'both'}
    if identifier in {'docker', 'cgroup', 'container_cpu', 'container_memory'}:
        return runtime == 'docker'
    if identifier in {'uv', 'process_runtime', 'process_resources', 'process_cpu', 'process_memory'}:
        return runtime == 'process'
    if identifier in {'tcpdump', 'tshark', 'capture_permissions', 'capture_interface'}:
        return runtime == 'docker' and measurement_requested(config.profiling_mode, 'packet_latency')
    if identifier in {'cpu_energy', 'cpu_instructions', 'gpu_power'}:
        return runtime == 'docker' and measurement_requested(config.profiling_mode, identifier, gpu=gpu)
    if identifier in {'gpu_device', 'gpu_memory', 'gpu_utilization'}:
        return gpu
    return True


def build_status(
    config: RunConfig, *, issues: Sequence[ConfigIssue], validation_pending: bool,
    checks: Sequence[PreflightCheck], inventory: Sequence[PreflightCheck],
    completed: bool, current: bool, running: bool, inventory_running: bool,
    error: str = '', inventory_error: str = '', cleanup_error: str = '',
    warnings: Sequence[ConfigIssue] = (),
) -> GlobalStatus:
    """Replace by root ID, then classify for the current request, never by text.

    Current preflight owns required capability results. Inventory-only failures
    are advisory; unrequested tools are neutral. Old preflight results become
    pending, and cannot overwrite current form errors or fresh inventory evidence.
    """
    rows: dict[str, StatusItem] = {}
    runtime = config.runtime
    if runtime == 'auto':
        runtime = next((row.runtime_kind for row in checks if row.runtime_kind), '') if current else ''

    def put(row: PreflightCheck, *, required: bool = False, target: str = '') -> None:
        identifier = row.check_id
        if not identifier:
            # Legacy producers without IDs can only be deduplicated by their
            # exact label; no fuzzy exception-text classification.
            identifier = next((key for key, item in rows.items() if item.check.label == row.label),
                              f'check_{len(rows)}')
            row = replace(row, check_id=identifier)
        rows[identifier] = StatusItem(row, required, target or _TARGETS.get(identifier, 'environment'))

    for row in inventory:
        if row.check_id == 'capture_permissions' and row.source == 'tcpdump' and row.status == 'fail':
            # scan_environment emits this typed dependency when the tool is
            # absent, not when an installed executable lacks capabilities.
            row = replace(row, check_id='tcpdump')
        if not _requested(row.check_id, config, runtime):
            status = 'not_requested'
        elif inventory_running:
            status = 'checking'
        elif row.status in {'fail', 'error'} or row.capability_status == 'error':
            status = 'warn'
        else:
            status = row.status
        put(replace(row, status=status))

    for row in checks:
        # Form validation below is the authority for config errors, even when
        # the worker validated an older configuration.
        if row.check_id.startswith('config_'):
            continue
        if not current and not _requested(row.check_id, config, runtime):
            status = 'not_requested'
        elif running:
            status = 'checking'
        elif not completed:
            status = 'pending'
        elif not current:
            status = 'stale'
        elif row.status == 'not_requested':
            status = 'not_requested'
        elif row.capability_status == 'error' and row.status != 'warn':
            status = 'fail'
        else:
            status = row.status
        put(replace(row, status=status), required=status != 'not_requested')

    if validation_pending:
        put(PreflightCheck(message_id('run.status.configuration'), 'pending',
                           message_id('run.preflight.validation_pending'), check_id='configuration'),
            required=True, target='field:model')
    elif issues:
        grouped: dict[str | None, list[str]] = {}
        for issue in issues:
            grouped.setdefault(issue.field, []).append(issue.reason)
        for field, reasons in grouped.items():
            put(PreflightCheck(message_id('run.status.config_field', field or 'configuration'), 'fail',
                               join_messages('；', tuple(dict.fromkeys(reasons))),
                               check_id='config_' + (field or 'other'), source='configuration'),
                required=True, target='field:' + (field or 'model'))
    else:
        put(PreflightCheck(message_id('run.status.configuration'), 'ok',
                           message_id('run.status.configuration_passed'), check_id='configuration'),
            target='field:model')
        for warning in warnings:
            put(PreflightCheck(message_id('run.plan.evidence_label'), 'warn', warning.reason,
                               check_id='quality_' + (warning.field or 'other'), source='configuration'),
                target='field:' + (warning.field or 'model'))

    if running or not completed or not current:
        status = 'checking' if running else 'pending'
        put(PreflightCheck(message_id('run.status.preflight'), status,
                           message_id('run.preflight.checking_notice' if running else 'run.preflight.configuration_changed'),
                           check_id='preflight'), required=True)
    elif error:
        put(PreflightCheck(message_id('run.environment.label.check_error'), 'fail', error,
                           check_id='check_error'), required=True)
    if inventory_error:
        put(PreflightCheck(message_id('run.status.inventory_error'), 'checking' if inventory_running else 'warn', inventory_error,
                           check_id='inventory_error', source='scan_environment'))
    if cleanup_error:
        # This survives configuration changes until an explicit successful
        # cleanup/recheck. It is never downgraded to a stale informational row.
        put(PreflightCheck(message_id('run.status.cleanup_error'), 'fail', cleanup_error,
                           check_id='cleanup_error'), required=True)
        if error == cleanup_error:
            rows.pop('check_error', None)
        if inventory_error == cleanup_error:
            rows.pop('inventory_error', None)
    return GlobalStatus(tuple(sorted(rows.values(), key=lambda item: GROUPS.index(item.group))))
