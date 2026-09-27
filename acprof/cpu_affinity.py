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
    for value in values[1:]:
        if value == end + 1:
            end = value
        else:
            groups.append(str(start) if start == end else f"{start}-{end}")
            start = end = value
    groups.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(groups)
