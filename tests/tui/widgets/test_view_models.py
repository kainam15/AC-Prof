"""Pure presentation contracts; no Pilot, readers, timers or collection."""
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import UTC, timedelta, timezone

import pytest
from rich.cells import cell_len

from acprof.tui.i18n import message_id, translate
from acprof.tui.ui.view_models import (
    CalloutContent,
    FactRow,
    StatusSummary,
    ViewAction,
    ViewSection,
    format_local_timestamp,
    format_value,
    middle_ellipsis,
    reason_presentation,
    status_presentation,
    status_symbol,
)


@pytest.mark.parametrize(("raw", "level", "symbol", "english"), [
    ("complete", "success", "✓", "Completed"),
    ("interrupted", "error", "×", "Interrupted"),
    ("failed", "error", "×", "Failed"),
    ("incomplete", "warning", "!", "Incomplete"),
    ("preparing", "info", "…", "Preparing"),
    ("running", "info", "…", "Running"),
    ("unknown", "neutral", "-", "Unknown"),
    ("unverified", "neutral", "-", "Unverified"),
    ("not_applicable", "neutral", "-", "Not applicable"),
    ("exists", "neutral", "-", "Artifact created"),
])
def test_status_preserves_distinct_user_facing_semantics(raw, level, symbol, english):
    result = status_presentation(raw)
    assert (result.level, result.symbol) == (level, symbol)
    assert translate(result.label, "en") == english
    assert translate(result.label, "zh") != raw


@pytest.mark.parametrize("raw", [None, "", 0, "future_backend_state"])
def test_unrecognized_states_never_appear_as_raw_enums_or_success(raw):
    result = status_presentation(raw)
    assert (result.level, result.symbol) == ("neutral", "-")
    assert translate(result.label, "zh") == "未知"
    assert translate(result.label, "en") == "Unknown"
    assert status_symbol("unrecognized_level") == "-"


@pytest.mark.parametrize("value", [None, "", "unknown"])
def test_missing_fact_values_remain_unknown_in_both_languages(value):
    value = format_value(value)
    assert translate(value, "zh") == "未知"
    assert translate(value, "en") == "Unknown"


def test_zero_and_catalog_messages_are_not_converted_to_unknown_or_flattened():
    assert format_value(0) == "0"
    assert format_value(0.0) == "0.0"
    assert format_value("0 B") == "0 B"
    label = message_id("common.action.retry")
    assert format_value(label) is label
    assert translate(format_value(label), "en") == "Retry"


@pytest.mark.parametrize(("source", "tz", "expected"), [
    ("2026-11-08T17:59:59.987Z", timezone(timedelta(hours=8)), "2026-11-09 01:59:59"),
    ("2026-11-08T03:04:05-03:00", UTC, "2026-11-08 06:04:05"),
    ("2026-11-08T21:03:04.999", timezone(timedelta(hours=8)), "2026-11-08 21:03:04"),
])
def test_timestamp_offsets_and_seconds_precision(source, tz, expected):
    assert format_local_timestamp(source, tz) == expected


@pytest.mark.parametrize("source", [None, "", "unknown", "not-a-timestamp", 0])
def test_missing_or_invalid_timestamps_do_not_invent_dates(source):
    assert translate(format_local_timestamp(source), "en") == "Unknown"


@pytest.mark.parametrize(("source", "limit", "expected"), [
    ("abcdefghi", 7, "abc…ghi"),
    ("中文路径文件", 7, "中…文件"),
    ("abc", 3, "abc"),
    ("中文", 4, "中文"),
    ("anything", 1, "…"),
    ("anything", 0, ""),
    ("anything", -1, ""),
    ("abcdefghi", 2, "a…"),
])
def test_middle_ellipsis_uses_terminal_cells_and_retains_both_ends(source, limit, expected):
    result = middle_ellipsis(source, limit)
    assert result == expected
    assert cell_len(result) <= max(0, limit)


def test_middle_ellipsis_keeps_combining_marks_with_their_characters():
    source = "e\u0301abcdefgA\u0301"
    assert middle_ellipsis(source, 5) == "e\u0301a…gA\u0301"
    assert cell_len(middle_ellipsis(source, 5)) == 5


def test_fact_rows_keep_full_copyable_values_and_unmodified_technical_details():
    raw_path = "/home/用户/研究/模型结果/2026-11-08/result_layers.json"
    raw_detail = "  ValueError('[incomplete]')\n/home/用户/研究/model.py:31\n"
    row = FactRow(message_id("common.field.model"), raw_path, detail=raw_detail, copyable=True)
    section = ViewSection("Identity", (), detail_rows=(row,), collapsed=True)
    assert section.detail_rows[0].value == raw_path
    assert section.detail_rows[0].detail == raw_detail
    with pytest.raises(FrozenInstanceError):
        row.value = middle_ellipsis(raw_path, 12)
    assert row.value == raw_path


def test_shared_views_hold_translatable_labels_and_explicit_action_state():
    action = ViewAction("retry", message_id("common.action.retry"), primary=True, disabled=True)
    summary = StatusSummary("warning", message_id("common.state.warning"), (action,))
    callout = CalloutContent("warning", "Problem", "Impact", "Next step", actions=(action,))
    assert translate(summary.headline, "en") == "Warning"
    assert translate(callout.actions[0].label, "en") == "Retry"
    assert callout.actions[0].disabled


def test_reason_presentation_maps_only_typed_codes_and_preserves_raw_evidence():
    detail = "  PermissionError: 无权访问 /home/用户/data\nRuntimeError('OOM')\n"
    result = reason_presentation("access_denied", detail)
    assert result.level == "error"
    assert translate(result.problem, "en") == "Access to a required resource was denied."
    assert result.detail == "access_denied\n" + detail
    assert "OOM" not in str(result.problem)

    unknown = reason_presentation("new_reason_2027", detail)
    assert unknown.level == "warning"
    assert translate(unknown.impact, "en") == "The available information does not establish the specific cause."
    assert unknown.detail == "new_reason_2027\n" + detail
    assert "new_reason_2027" not in str(unknown.problem)
    assert reason_presentation("", detail).detail == detail


def test_view_models_do_not_import_textual_in_a_fresh_interpreter():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import acprof.tui.ui.view_models; "
         "assert not any(name == 'textual' or name.startswith('textual.') for name in sys.modules)"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
