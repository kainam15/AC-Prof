"""Pure reductions for low-frequency observed process-tree counters."""
from __future__ import annotations

import math

BOUNDARY_COUNTERS = {
    "process_io_read_bytes": "io_read_bytes", "process_io_write_bytes": "io_write_bytes",
    "process_io_read_ops": "io_read_ops", "process_io_write_ops": "io_write_ops",
    "process_leader_voluntary_ctx_switches": "voluntary_ctx",
    "process_leader_involuntary_ctx_switches": "involuntary_ctx",
    "process_minor_faults": "minor_faults", "process_major_faults": "major_faults",
}
END_GAUGES = {"process_threads_end": "threads", "process_fds_end": "fds",
              "process_swap_bytes_end": "swap_bytes"}


def counter_decreased(samples: list[dict], field: str) -> bool:
    return any(isinstance(old := before['members'][key].get(field), (int, float))
               and isinstance(new := member.get(field), (int, float)) and new < old
               for before, after in zip(samples, samples[1:], strict=False)
               for key, member in after['members'].items() if key in before['members'])


def complete_members(sample: dict, field: str) -> bool:
    members = sample["members"]
    return bool(members) and all(
        isinstance(member.get(field), (int, float)) and math.isfinite(member[field])
        and member[field] >= 0 for member in members.values())


def boundary_metrics(samples: list[dict]) -> tuple[dict[str, float], dict[str, str]]:
    """Never charge lifetime counters of a newly observed process to a window."""
    values: dict[str, float] = {}
    errors: dict[str, str] = {}
    if len(samples) < 2 or not all(sample.get("boundary") for sample in (samples[0], samples[-1])):
        return values, errors
    first, last = samples[0], samples[-1]
    if last['members']:
        values['process_count_end'] = float(len(last['members']))
    for name, field in END_GAUGES.items():
        if complete_members(last, field):
            values[name] = sum(member[field] for member in last['members'].values())
    stable = bool(first['members']) and all(sample['members'].keys() == first['members'].keys() for sample in samples)
    for name, field in BOUNDARY_COUNTERS.items():
        if not stable:
            errors[name] = "process_membership_changed"
        elif all(complete_members(sample, field) for sample in (first, last)):
            deltas = [last['members'][key][field] - before[field] for key, before in first['members'].items()]
            if any(delta < 0 for delta in deltas):
                errors[name] = "counter_decreased"
            else:
                values[name] = sum(deltas)
    if stable and all(complete_members(sample, 'rss') for sample in (first, last)):
        values['process_rss_growth_bytes'] = (sum(m['rss'] for m in last['members'].values())
                                             - sum(m['rss'] for m in first['members'].values()))
    return values, errors
