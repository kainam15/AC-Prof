"""Pure backfill presentation using the existing profiler applicability rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from acprof.tui.i18n import message_id
from acprof.tui.results.profile_status import ProfilerOverview, tool_status_message
from acprof.tui.ui.view_models import (
    CalloutContent,
    FactRow,
    StatusSummary,
    ViewAction,
    ViewSection,
)

BackfillPhase = Literal["empty", "checking", "cancelled", "stale", "unavailable", "error"]

_PHASES: dict[BackfillPhase, tuple[str, str, str, str, str]] = {
    "empty": (
        "neutral", message_id("profiling.view.empty"), message_id("profiling.view.empty_problem"),
        message_id("profiling.view.empty_impact"), message_id("profiling.view.choose_next"),
    ),
    "checking": (
        "info", message_id("profiling.view.checking"), message_id("profiling.view.checking_problem"),
        message_id("profiling.view.checking_impact"), message_id("profiling.view.checking_next"),
    ),
    "cancelled": (
        "warning", message_id("profiling.view.cancelled"), message_id("profiling.view.cancelled_problem"),
        message_id("profiling.view.unverified_impact"), message_id("profiling.view.retry_next"),
    ),
    "stale": (
        "neutral", message_id("profiling.view.stale"), message_id("profiling.view.stale_problem"),
        message_id("profiling.view.unverified_impact"), message_id("profiling.view.retry_next"),
    ),
    "unavailable": (
        "error", message_id("profiling.view.unavailable"), message_id("profiling.view.unavailable_problem"),
        message_id("profiling.view.unavailable_impact"), message_id("profiling.view.retry_next"),
    ),
    "error": (
        "error", message_id("profiling.view.error"), message_id("profiling.view.error_problem"),
        message_id("profiling.view.unverified_impact"), message_id("profiling.view.retry_next"),
    ),
}


@dataclass(frozen=True, slots=True)
class BackfillView:
    summary: StatusSummary
    callout: CalloutContent | None
    sections: tuple[ViewSection, ...]


def _recovery_actions(
    *, choose_allowed: bool, refresh_allowed: bool, choose_first: bool = False,
    only_choose: bool = False,
) -> tuple[ViewAction, ...]:
    choose = ViewAction(
        "backfill-choose", message_id("results.picker.title"),
        primary=choose_first or only_choose, disabled=not choose_allowed,
    )
    if only_choose:
        return (choose,)
    refresh = ViewAction(
        "backfill-refresh", message_id("run.environment.retry"),
        primary=not choose_first, disabled=not refresh_allowed,
    )
    return (choose, refresh) if choose_first else (refresh, choose)


def _sections(overview: ProfilerOverview) -> tuple[ViewSection, ...]:
    return (
        ViewSection(
            message_id("profiling.backfill.tools"),
            tuple(
                FactRow(status.tool, tool_status_message(status), status=status.state)
                for status in overview.tools
            ),
        ),
        ViewSection(
            message_id("profiling.view.source"),
            (
                FactRow(
                    message_id("results.view.directory"),
                    str(overview.result_dir),
                    copyable=True,
                ),
            ),
            collapsed=True,
        ),
    )


def _checked_view(
    overview: ProfilerOverview, *, profile_allowed: bool, choose_allowed: bool,
) -> BackfillView:
    sections = _sections(overview)
    if not overview.applicable:
        return BackfillView(
            StatusSummary("neutral", message_id("profiling.view.inapplicable")),
            CalloutContent(
                "neutral",
                message_id("profiling.view.inapplicable_problem"),
                message_id("profiling.view.inapplicable_impact"),
                message_id("profiling.view.choose_next"),
                actions=_recovery_actions(
                    choose_allowed=choose_allowed, refresh_allowed=False, only_choose=True,
                ),
            ),
            sections,
        )
    if not overview.pending:
        return BackfillView(
            StatusSummary(
                "success", message_id("profiling.view.complete", overview.complete, overview.applicable),
            ),
            None,
            sections,
        )
    if not profile_allowed:
        return BackfillView(
            StatusSummary("warning", message_id("profiling.view.busy", overview.pending)),
            CalloutContent(
                "warning",
                message_id("profiling.view.busy_problem"),
                message_id("profiling.view.busy_impact"),
                message_id("profiling.view.busy_next"),
            ),
            sections,
        )
    return BackfillView(
        StatusSummary(
            "warning", message_id("profiling.view.ready", overview.pending, overview.applicable),
        ),
        None,
        sections,
    )


def _unchecked_view(
    phase: BackfillPhase, *, detail: str, source: str, standalone_csv: bool,
    choose_allowed: bool, refresh_allowed: bool,
) -> BackfillView:
    level, headline, problem, impact, next_step = _PHASES[phase]
    known_csv = phase == "unavailable" and standalone_csv
    if known_csv:
        problem = message_id("profiling.view.csv_problem")
        impact = message_id("profiling.view.csv_impact")
        next_step = message_id("profiling.view.choose_next")
    actions = () if phase == "checking" else _recovery_actions(
        choose_allowed=choose_allowed,
        refresh_allowed=refresh_allowed,
        choose_first=known_csv,
        only_choose=phase == "empty",
    )
    return BackfillView(
        StatusSummary(level, headline),
        CalloutContent(
            level, problem, impact, next_step,
            detail="\n\n".join(value for value in (source, detail) if value),
            actions=actions,
        ),
        (),
    )


def backfill_view(
    overview: ProfilerOverview | None, *, phase: BackfillPhase = "empty",
    detail: str = "", source: str = "", standalone_csv: bool = False,
    profile_allowed: bool = True, choose_allowed: bool = True, refresh_allowed: bool = True,
) -> BackfillView:
    """Project supplied evidence without reading files or starting work.

    Tool totals count applicable CSV records, not cases or formal windows.
    Only the caller may identify an inspected standalone CSV; raw errors never
    determine source type. Successful inspection supersedes lifecycle hints.
    """
    if overview is not None:
        return _checked_view(
            overview, profile_allowed=profile_allowed, choose_allowed=choose_allowed,
        )
    return _unchecked_view(
        phase, detail=detail, source=source, standalone_csv=standalone_csv,
        choose_allowed=choose_allowed, refresh_allowed=refresh_allowed,
    )
