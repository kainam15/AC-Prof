"""Read-only profiler availability for a selected completed experiment.

Reuse posthoc's own applicability and completion semantics. This inspection
never starts a profiler, probes the host, or modifies measurement artifacts.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from acprof.cancellation import cancellable, check_cancelled
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.posthoc.backfill import row_tool_complete
from acprof.host.posthoc.context import (
    SUPPORTED_TOOLS,
    TOOL_ERROR_FIELD,
    TOOL_GPU_MODES,
    load_result_context,
)
from acprof.result_csv import ResultCsvSnapshot, csv_diagnostic_text

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


def inspect_profile_result(source: Path, *, result_snapshot: ResultCsvSnapshot | None = None,
                            cancelled: Callable[[], bool] | None = None) -> ProfilerOverview:
    """Inspect only an experiment directory or its layered manifest.

    Standalone CSVs may be plotted but lack immutable profiler provenance.
    A caller supplying a snapshot owns source verification after inspection.
    """
    check_cancelled(cancelled)
    path = source.expanduser()
    if path.is_dir():
        directory = path
    elif path.name == "result_layers.json":
        directory = path.parent
    else:
        raise ValueError("仅支持分层实验目录或 result_layers.json；独立 CSV 不能补采")
    context = load_result_context(directory, result_snapshot=result_snapshot, cancelled=cancelled)
    totals = dict.fromkeys(SUPPORTED_TOOLS, 0)
    completed = dict.fromkeys(SUPPORTED_TOOLS, 0)
    errors = dict.fromkeys(SUPPORTED_TOOLS, 0)
    for row in cancellable(context.rows, cancelled):
        mode = normalize_gpu_mode(row.get("gpu_mode"))
        for tool in cancellable(SUPPORTED_TOOLS, cancelled):
            if mode not in TOOL_GPU_MODES[tool]:
                continue
            totals[tool] += 1
            completed[tool] += row_tool_complete(row, tool)
            errors[tool] += bool(csv_diagnostic_text(row.get(TOOL_ERROR_FIELD[tool])))
    statuses: list[ProfilerStatus] = []
    for tool in SUPPORTED_TOOLS:
        state: ToolState = (
            "inapplicable" if not totals[tool]
            else "complete" if completed[tool] == totals[tool]
            else "failed" if errors[tool]
            else "partial" if completed[tool]
            else "missing"
        )
        statuses.append(ProfilerStatus(tool, state, completed[tool], totals[tool], errors[tool]))
    check_cancelled(cancelled)
    return ProfilerOverview(context.result_dir, tuple(statuses))
