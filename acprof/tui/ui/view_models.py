"""Immutable, presentation-only data for shared TUI views.

These values do not own run state, permissions, readers or timers. CatalogMessage
values remain untranslated until a widget renders them in the selected language.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo

from rich.cells import cell_len

from acprof.tui.i18n import message_id


@dataclass(frozen=True, slots=True)
class ViewAction:
    id: str
    label: str
    primary: bool = False
    disabled: bool = False


@dataclass(frozen=True, slots=True)
class StatusSummary:
    level: str
    headline: str
    actions: tuple[ViewAction, ...] = ()


@dataclass(frozen=True, slots=True)
class FactRow:
    label: str
    value: str
    status: str | None = None
    detail: str = ""
    copyable: bool = False


@dataclass(frozen=True, slots=True)
class ViewSection:
    title: str
    rows: tuple[FactRow, ...]
    detail_rows: tuple[FactRow, ...] = ()
    collapsed: bool = False


@dataclass(frozen=True, slots=True)
class CalloutContent:
    level: str
    problem: str
    impact: str
    next_step: str
    detail: str = ""
    actions: tuple[ViewAction, ...] = ()


@dataclass(frozen=True, slots=True)
class StatusPresentation:
    level: str
    symbol: str
    label: str


_SYMBOLS = {"success": "✓", "warning": "!", "error": "×", "info": "…", "neutral": "-"}

# Presentation aliases only: callers continue to use the recorded state for gates.
_STATES = {
    "complete": ("success", "results.overview.complete"),
    "ok": ("success", "common.state.passed"),
    "passed": ("success", "common.state.passed"),
    "ready": ("success", "run.environment.state.ready"),
    "interrupted": ("error", "results.overview.interrupted"),
    "failed": ("error", "common.state.failed"),
    "error": ("error", "common.state.failed"),
    "blocked": ("error", "run.environment.state.blocked"),
    "damaged": ("error", "results.overview.damaged"),
    "incomplete": ("warning", "structured.status.incomplete"),
    "limited": ("warning", "run.environment.item.warn"),
    "warning": ("warning", "common.state.warning"),
    "warn": ("warning", "common.state.warning"),
    "missing": ("warning", "results.overview.missing"),
    "cancelled": ("warning", "run.status.cancelled"),
    "running": ("info", "results.overview.running"),
    "preparing": ("info", "structured.status.preparing"),
    "checking": ("info", "structured.status.checking"),
    "pending": ("neutral", "structured.status.pending"),
    "unknown": ("neutral", "common.value.unknown"),
    "unverified": ("neutral", "common.state.unverified"),
    "stale": ("neutral", "run.environment.state.stale"),
    "not_applicable": ("neutral", "common.value.not_applicable"),
    "exists": ("neutral", "structured.status.present"),
    "not_generated": ("neutral", "results.overview.not_generated"),
}

# Exact typed reason codes, never a classifier over exception text.
_REASONS = {
    "unsupported_run_state": ("error", "state_format"),
    "artifact_layout_unsupported": ("error", "artifact_format"),
    "recorded_evidence_invalid": ("error", "evidence"),
    "resource_limit": ("warning", "resources"),
    "input_budget_exceeded": ("warning", "input_budget"),
    "access_denied": ("error", "access"),
    "runtime_task_unsupported": ("error", "task"),
    "runtime_dependency_missing": ("error", "dependency_missing"),
    "runtime_dependency_incompatible": ("error", "dependency_incompatible"),
    "request_timeout": ("warning", "timeout"),
    "runtime_initialization_failed": ("error", "initialization"),
    "inference_failed": ("error", "inference"),
}


def status_symbol(level: str) -> str:
    """Keep unknown display levels neutral, never implicitly successful."""
    return _SYMBOLS.get(level, _SYMBOLS["neutral"])


def status_presentation(raw: object) -> StatusPresentation:
    """Translate a known state; future or absent enum values remain unknown."""
    state = raw.strip().lower() if isinstance(raw, str) else "unknown"
    level, key = _STATES.get(state, _STATES["unknown"])
    return StatusPresentation(level, status_symbol(level), message_id(key))


def format_value(value: object) -> str:
    """Preserve real zero and structured messages while labelling missing values."""
    if value is None or isinstance(value, str) and value in ("", "unknown"):
        return message_id("common.value.unknown")
    return value if isinstance(value, str) else str(value)


def format_local_timestamp(raw: object, tz: tzinfo | None = None) -> str:
    """Match experiment-picker local time, at seconds precision.

    A recorded offset is converted to the host timezone (or an explicit test
    timezone). A naive historical timestamp has no source offset to infer and
    keeps its recorded wall-clock time. Callers retain the raw value in details.
    """
    if not isinstance(raw, str):
        return message_id("common.value.unknown")
    try:
        timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(tz)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OverflowError, OSError):
        return message_id("common.value.unknown")


def _cell_groups(text: str) -> list[str]:
    """Keep combining marks and joined glyphs with their preceding character."""
    groups: list[str] = []
    for character in text:
        if groups and (cell_len(character) == 0 or groups[-1].endswith("\u200d")):
            groups[-1] += character
        else:
            groups.append(character)
    return groups


def _take_cells(groups: list[str], budget: int) -> str:
    selected: list[str] = []
    used = 0
    for group in groups:
        width = cell_len(group)
        if used + width > budget:
            break
        selected.append(group)
        used += width
    return "".join(selected)


def middle_ellipsis(text: str, max_cells: int) -> str:
    """Fit a value to terminal cells without modifying the stored full value."""
    if max_cells <= 0:
        return ""
    if cell_len(text) <= max_cells:
        return text
    if max_cells == 1:
        return "…"
    groups = _cell_groups(text)
    prefix = _take_cells(groups, max_cells // 2)
    suffix_groups = list(reversed(groups))
    suffix = _take_cells(suffix_groups, max_cells - cell_len(prefix) - 1)
    # Reverse groups, not code points, to retain combining marks in each glyph.
    suffix = "".join(reversed(_cell_groups(suffix)))
    return prefix + "…" + suffix


def reason_presentation(code: str, detail: str = "") -> CalloutContent:
    """Explain an exact known reason and preserve every byte of technical text."""
    level, key = _REASONS.get(code, ("warning", "unknown"))
    prefix = "structured.reason." + key
    raw = "\n".join(value for value in (code, detail) if value)
    return CalloutContent(
        level,
        message_id(prefix + ".problem"),
        message_id(prefix + ".impact"),
        message_id(prefix + ".next"),
        detail=raw,
    )
