"""Stable source ownership and pure summaries of the current status snapshot.

Add a collector's check IDs here. Diagnostic provenance (``check.source``)
must never decide ownership. Unknown IDs stay visible in Other checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acprof.tui.run.global_status import GlobalStatus, StatusIssue, StatusItem


SOURCE_CHECKS = {
    'base': ('environment', 'runtime', 'runtime_auto'),
    'docker': ('docker',),
    'cgroup': ('cgroup', 'container_cpu', 'container_memory', 'cgroup_metrics'),
    'process': ('uv', 'process_runtime', 'process_resources', 'process_cpu', 'process_memory', 'process_io', 'process_threads', 'process_fds', 'process_ctx'),
    'nvml': ('gpu_device', 'gpu_memory', 'gpu_utilization', 'gpu_power', 'gpu_temperature', 'gpu_pstate', 'gpu_memory_clock', 'gpu_sm_clock'),
    'rapl': ('cpu_energy', 'dram_energy'),
    'perf': ('cpu_instructions',),
    'network': ('tcpdump', 'tshark', 'capture_interface', 'capture_permissions', 'packet_latency'),
    'profilers': ('ncu', 'nsys', 'intel_advisor', 'torch', 'massif', 'profiler'),
    'configuration': ('configuration',),
    'other': ('preflight', 'check_error', 'inventory_error', 'cleanup_error', 'latency', 'throughput'),
}
CHECK_SOURCES = {identifier: group for group, identifiers in SOURCE_CHECKS.items() for identifier in identifiers}


def source_group(check_id: str) -> str:
    if check_id.startswith(('config_', 'quality_')):
        return 'configuration'
    return CHECK_SOURCES.get(check_id, 'other')


@dataclass(frozen=True, slots=True)
class SourceSummary:
    group_id: str
    items: tuple[StatusItem, ...]
    issues: tuple[StatusIssue, ...]

    @property
    def state(self) -> str:
        groups = {item.group for item in self.items}
        for group in ('error', 'warning'):
            if group in groups:
                return group
        if any(item.check.status == 'checking' for item in self.items):
            return 'checking'
        if 'pending' in groups:
            return 'pending'
        return 'success' if 'success' in groups else 'not_requested'

    @property
    def issue_count(self) -> int:
        return sum(issue.group in {'error', 'warning'} for issue in self.issues)

    @property
    def collapsed(self) -> bool:
        return self.state not in {'error', 'warning'}


def source_summaries(snapshot: GlobalStatus, *, group: str | None = None) -> tuple[SourceSummary, ...]:
    """Project each status category separately while preserving full cached evidence."""
    issues = snapshot.issues
    result = []
    for name in SOURCE_CHECKS:
        children = tuple(item for item in snapshot.items
                         if source_group(item.check.check_id) == name and (group is None or item.group == group))
        entries = tuple(issue for issue in issues
                        if source_group(issue.primary.check.check_id) == name
                        and (group is None or issue.group == group))
        # A source is visible only when it has checks in the selected category.
        # GlobalStatus.counts still deduplicates issues; these titles count checks.
        if children:
            result.append(SourceSummary(name, children, entries))
    return tuple(result)
