"""Read-only profiler availability for a selected completed experiment.

Reuse posthoc's own applicability and completion semantics. This inspection
never starts a profiler, probes the host, or modifies measurement artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.posthoc.backfill import csv_tool_complete, row_tool_complete
from acprof.host.posthoc.context import (
    SUPPORTED_TOOLS,
    TOOL_ERROR_FIELD,
    TOOL_GPU_MODES,
    load_result_context,
)

ToolState = Literal["complete", "partial", "failed", "missing", "inapplicable"]


@dataclass(frozen=True)
class ProfilerStatus:
    tool: str
    state: ToolState
    completed: int
    total: int
    errors: int

    @property
    def needs_collection(self) -> bool:
        return self.state in {"partial", "failed", "missing"}


@dataclass(frozen=True)
class ProfilerOverview:
    result_dir: Path
    tools: tuple[ProfilerStatus, ...]

    @property
    def pending(self) -> int:
        return sum(status.needs_collection for status in self.tools)

    @property
    def applicable(self) -> int:
        return sum(status.state != "inapplicable" for status in self.tools)

    @property
    def complete(self) -> int:
        return sum(status.state == "complete" for status in self.tools)

    @property
    def suggested_tools(self) -> tuple[str, ...]:
        # Recommend compute coverage only; costly execution timelines are opt-in.
        return tuple(status.tool for status in self.tools
                     if status.needs_collection and status.tool in {"torch", "ncu"})


def inspect_profile_result(source: Path) -> ProfilerOverview:
    """Inspect only an experiment directory or its layered manifest.

    Standalone CSVs may be plotted but lack immutable profiler provenance.
    """
    path = source.expanduser()
    if path.is_dir():
        directory = path
    elif path.name == "result_layers.json":
        directory = path.parent
    else:
        raise ValueError("仅支持分层实验目录或 result_layers.json；独立 CSV 不能补采")
    context = load_result_context(directory)
    statuses: list[ProfilerStatus] = []
    for tool in SUPPORTED_TOOLS:
        rows = tuple(row for row in context.rows
                     if normalize_gpu_mode(row.get("gpu_mode")) in TOOL_GPU_MODES[tool])
        if not rows:
            statuses.append(ProfilerStatus(tool, "inapplicable", 0, 0, 0))
            continue
        completed = sum(row_tool_complete(row, tool) for row in rows)
        errors = sum(bool(str(row.get(TOOL_ERROR_FIELD[tool]) or "").strip()) for row in rows)
        state: ToolState = (
            "complete" if csv_tool_complete(context, tool)
            else "failed" if errors
            else "partial" if completed
            else "missing"
        )
        statuses.append(ProfilerStatus(tool, state, completed, len(rows), errors))
    return ProfilerOverview(context.result_dir, tuple(statuses))
