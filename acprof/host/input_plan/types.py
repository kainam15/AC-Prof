"""Typed input scale planning results."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlannedInputScales:
    scales: list[float]
    source: str
    plan_file: str | None = None
    workload: dict[str, Any] = field(default_factory=dict)
    plan_sha256: str = ""
