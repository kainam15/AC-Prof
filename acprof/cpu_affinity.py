"""Canonical optional Linux CPU sets; parsing does not inspect or change the host."""
from __future__ import annotations

import re


def parse_cpu_set(value: str) -> set[int]:
    if not value.strip():
        return set()
    cpus: set[int] = set()
    for part in value.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:-(\d+))?\s*", part)
        if match is None:
            raise ValueError("CPU set must contain CPU IDs or ranges, e.g. 0-3,8")
        start, end = int(match[1]), int(match[2] or match[1])
        if start > end or end > 1048575:
            raise ValueError("invalid CPU set range")
        cpus.update(range(start, end + 1))
    return cpus


def normalize_cpu_set(value: str) -> str:
    values = sorted(parse_cpu_set(value))
    groups = []
    if not values:
        return ""
    start = end = values[0]
    for cpu_id in values[1:]:
        if cpu_id == end + 1:
            end = cpu_id
        else:
            groups.append(str(start) if start == end else f"{start}-{end}")
            start = end = cpu_id
    groups.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(groups)


def expanded_cpu_ids(value: str) -> str:
    """Encode the explicit IDs understood by the standalone Process launcher."""
    return ",".join(map(str, sorted(parse_cpu_set(value))))


def thread_masks_within(affinities: list[str] | None, requested: str) -> bool:
    """Require evidence for every observed thread, allowing workers to pin themselves more narrowly."""
    allowed = parse_cpu_set(requested)
    if not isinstance(affinities, list) or not affinities or not allowed:
        return False
    for mask in affinities:
        if not isinstance(mask, str):
            return False
        try:
            cpus = parse_cpu_set(mask)
        except ValueError:
            return False
        if not cpus or not cpus <= allowed:
            return False
    return True
