"""Pure backfill summaries preserve collection gates and raw evidence."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.tui.i18n import translate
from acprof.tui.results.backfill_view import backfill_view
from acprof.tui.results.profile_status import (
    ProfilerOverview,
    ProfilerStatus,
    tool_status_message,
)
from acprof.tui.ui.view_models import status_presentation


@pytest.fixture
def mixed_overview():
    return ProfilerOverview(
        Path("/实验结果/很长的中文模型名称/[revision]/分层实验"),
        (
            ProfilerStatus("torch", "complete", 7, 7, 0),
            ProfilerStatus("ncu", "partial", 2, 7, 0),
            ProfilerStatus("nsys", "failed", 0, 7, 3),
            ProfilerStatus("massif", "inapplicable", 0, 0, 0),
        ),
    )


@pytest.mark.parametrize(
    ("phase", "level", "headline", "primary"),
    [
        ("empty", "neutral", "Choose an experiment", "backfill-choose"),
        ("checking", "info", "Checking backfill eligibility", None),
        ("cancelled", "warning", "Backfill check cancelled", "backfill-refresh"),
        ("stale", "neutral", "needs a new check", "backfill-refresh"),
        ("unavailable", "error", "Backfill unavailable", "backfill-refresh"),
        ("error", "error", "Backfill check failed", "backfill-refresh"),
    ],
)
def test_lifecycle_states_keep_evidence_out_of_the_first_screen(phase, level, headline, primary):
    source = "  /实验结果/[模型]/" + "中文目录-" * 20 + "/result_layers.json  "
    detail = "  [red] PermissionError: 拒绝访问 [/red]\n  exact trailing space  \n"
    view = backfill_view(None, phase=phase, source=source, detail=detail)

    assert view.summary.level == level
    assert headline in translate(view.summary.headline, "en")
    assert view.sections == ()
    assert view.summary.actions == ()
    assert view.callout is not None
    assert view.callout.level == level
    assert view.callout.detail == source + "\n\n" + detail
    assert isinstance(view.callout.detail, str)
    visible = "\n".join(
        translate(value, "en")
        for value in (
            view.summary.headline,
            view.callout.problem,
            view.callout.impact,
            view.callout.next_step,
        )
    )
    assert source not in visible
    assert "PermissionError" not in visible
    assert "[red]" not in visible
    assert len(view.callout.actions) <= 2
    assert [action.id for action in view.callout.actions if action.primary] == (
        [] if primary is None else [primary]
    )


@pytest.mark.parametrize("phase", ["empty", "cancelled", "stale", "unavailable", "error"])
@pytest.mark.parametrize(("choose_allowed", "refresh_allowed"), [(True, True), (False, True), (True, False), (False, False)])
def test_recovery_buttons_obey_caller_gates(phase, choose_allowed, refresh_allowed):
    view = backfill_view(
        None, phase=phase, choose_allowed=choose_allowed, refresh_allowed=refresh_allowed,
    )
    assert view.callout is not None
    actions = {action.id: action for action in view.callout.actions}
    assert actions["backfill-choose"].disabled is not choose_allowed
    if phase == "empty":
        assert list(actions) == ["backfill-choose"]
    else:
        assert actions["backfill-refresh"].disabled is not refresh_allowed
        assert list(actions) == ["backfill-refresh", "backfill-choose"]


def test_known_standalone_csv_has_a_specific_explanation_and_choose_first():
    detail = "仅支持分层实验目录或 result_layers.json；独立 CSV 不能补采"
    view = backfill_view(
        None, phase="unavailable", standalone_csv=True, detail=detail,
        source="/结果/[独立].csv", choose_allowed=False, refresh_allowed=False,
    )
    assert view.callout is not None
    assert translate(view.callout.problem, "en") == "Standalone CSV lacks backfill provenance"
    assert translate(view.callout.next_step, "en") == "Choose a layered experiment directory."
    assert view.callout.detail == "/结果/[独立].csv\n\n" + detail
    assert [action.id for action in view.callout.actions] == ["backfill-choose", "backfill-refresh"]
    assert [action.primary for action in view.callout.actions] == [True, False]
    assert all(action.disabled for action in view.callout.actions)


@pytest.mark.parametrize("phase", ["unavailable", "error"])
def test_unknown_exceptions_and_csv_suffixes_do_not_guess_a_source_type(phase):
    raw = "仅支持分层实验目录；独立 CSV 不能补采 [AccessDenied]"
    view = backfill_view(None, phase=phase, source="/result.csv", detail=raw)
    assert view.callout is not None
    assert "Standalone CSV" not in translate(view.callout.problem, "en")
    if phase == "error":
        assert "cannot be confirmed" in translate(view.callout.impact, "en")
    else:
        assert translate(view.callout.impact, "en") == "Backfill cannot start from this result."
    assert view.callout.detail.endswith(raw)
    assert view.callout.actions[0].id == "backfill-refresh"


def test_read_error_cannot_be_relabelled_as_a_confirmed_unavailable_csv():
    view = backfill_view(None, phase="error", standalone_csv=True, detail="OSError")
    assert view.callout is not None
    assert translate(view.callout.problem, "en") == "Could not read or check the result"
    assert view.callout.actions[0].id == "backfill-refresh"


@pytest.mark.parametrize(("source", "detail"), [("", " \n  "), ("  /目录/[原文]  ", ""), ("", "")])
def test_technical_detail_keeps_whitespace_and_markup_characters_verbatim(source, detail):
    view = backfill_view(None, phase="error", source=source, detail=detail)
    assert view.callout is not None
    assert view.callout.detail == (source or detail)


def test_pending_tools_and_csv_record_counts_are_distinct(mixed_overview):
    view = backfill_view(mixed_overview)
    assert view.summary.level == "warning"
    assert translate(view.summary.headline, "en") == "Ready to backfill · 2/3 tools pending"
    assert view.summary.actions == ()
    assert view.callout is None
    tools, source = view.sections
    assert translate(tools.title, "en") == "Backfill tools"
    assert [row.label for row in tools.rows] == ["torch", "ncu", "nsys", "massif"]
    assert [translate(row.value, "en") for row in tools.rows] == [
        "Complete 7/7", "Partial 2/7", "3 failed · 0/7 complete", "Not applicable",
    ]
    assert [row.status for row in tools.rows] == ["complete", "partial", "failed", "inapplicable"]
    assert source.collapsed
    assert len(source.rows) == 1
    assert source.rows[0].value == str(mixed_overview.result_dir)
    assert source.rows[0].copyable


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ProfilerStatus("torch", "complete", 5, 5, 0), "Complete 5/5"),
        (ProfilerStatus("ncu", "partial", 2, 5, 0), "Partial 2/5"),
        (ProfilerStatus("nsys", "failed", 1, 5, 2), "2 failed · 1/5 complete"),
        (ProfilerStatus("massif", "missing", 0, 5, 0), "Not collected"),
        (ProfilerStatus("massif", "inapplicable", 0, 0, 0), "Not applicable"),
    ],
)
def test_tool_status_messages_retain_existing_record_count_semantics(status, expected):
    assert translate(tool_status_message(status), "en") == expected


def test_complete_means_all_applicable_tools_complete(mixed_overview):
    overview = replace(
        mixed_overview,
        tools=(
            ProfilerStatus("torch", "complete", 7, 7, 0),
            ProfilerStatus("massif", "inapplicable", 0, 0, 0),
        ),
    )
    view = backfill_view(overview, profile_allowed=False)
    assert view.summary.level == "success"
    assert translate(view.summary.headline, "en") == "1/1 tools complete · No backfill needed"
    assert view.summary.actions == ()
    assert view.callout is None


@pytest.mark.parametrize("tools", [(), (ProfilerStatus("massif", "inapplicable", 0, 0, 0),)])
def test_no_applicable_tools_is_neutral_not_zero_of_zero_success(tools):
    view = backfill_view(ProfilerOverview(Path("/实验"), tools), choose_allowed=False)
    assert view.summary.level == "neutral"
    assert "No backfill tools apply" in translate(view.summary.headline, "en")
    assert "0/0" not in view.summary.headline
    assert view.callout is not None
    assert [action.id for action in view.callout.actions] == ["backfill-choose"]
    assert view.callout.actions[0].disabled


def test_operation_lock_explains_waiting_without_changing_profiler_evidence(mixed_overview):
    ready = backfill_view(mixed_overview)
    busy = backfill_view(mixed_overview, profile_allowed=False)
    assert busy.sections == ready.sections
    assert busy.summary.level == "warning"
    assert "2 tools pending" in translate(busy.summary.headline, "en")
    assert busy.callout is not None
    assert translate(busy.callout.problem, "en") == "Another operation is running"
    assert busy.summary.actions == busy.callout.actions == ()
    assert mixed_overview.pending == 2


def test_completed_inspection_overrides_stale_lifecycle_hints(mixed_overview):
    view = backfill_view(
        mixed_overview, phase="error", standalone_csv=True,
        detail="previous failure", source="/previous.csv",
    )
    assert view == backfill_view(mixed_overview)


@pytest.mark.parametrize(
    ("state", "level", "symbol", "label"),
    [("partial", "warning", "!", "Incomplete"), ("inapplicable", "neutral", "-", "Not applicable")],
)
def test_profiler_state_aliases_use_the_shared_status_vocabulary(state, level, symbol, label):
    presentation = status_presentation(state)
    assert presentation.level == level
    assert presentation.symbol == symbol
    assert translate(presentation.label, "en") == label


def test_projection_never_reopens_result_files_or_rechecks_applicability(mixed_overview):
    forbidden = AssertionError("presentation must use supplied evidence")
    with (
        patch.object(Path, "open", side_effect=forbidden),
        patch.object(Path, "stat", side_effect=forbidden),
        patch.object(Path, "is_dir", side_effect=forbidden),
        patch.object(Path, "resolve", side_effect=forbidden),
        patch("acprof.tui.results.profile_status.inspect_profile_result", side_effect=forbidden),
    ):
        ready = backfill_view(mixed_overview)
        unavailable = backfill_view(None, phase="unavailable", source="/missing/result.csv")
    assert ready.sections[1].rows[0].value == str(mixed_overview.result_dir)
    assert unavailable.callout is not None
