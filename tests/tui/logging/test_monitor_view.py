"""Monitor projections preserve progress meaning without UI workers or clocks."""
import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.tui.i18n import translate
from acprof.tui.run.monitor_view import monitor_header_view
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.stages import Stage


def test_waiting_has_no_fabricated_progress_or_resource_values():
    view = monitor_header_view(ProgressSnapshot())

    assert view.summary.level == "neutral"
    assert translate(view.summary.headline, "zh") == "等待"
    assert translate(view.summary.headline, "en") == "Waiting"
    assert view.summary.actions == ()
    assert view.cases is view.resources is view.checks is None
    assert view.progress_total is None
    assert view.progress_completed == 0
    assert view.details.collapsed
    assert translate(view.details.rows[0].value, "en") == "Not started"


@pytest.mark.parametrize(("stage", "level"), [
    (Stage.SERVER_READY, "success"), (Stage.COMPLETED, "success"),
    (Stage.PROBE_COMPLETED, "success"), (Stage.MEMORY_FOUND, "success"),
    (Stage.CASE_COMPLETED, "success"), (Stage.FAILED, "error"),
    (Stage.CLEANUP_INCOMPLETE, "error"), (Stage.PROBE_FAILED, "error"),
    (Stage.TASK_UNSUPPORTED, "error"), (Stage.STOPPED, "error"),
    (Stage.TERMINATED, "error"), (Stage.PARTIAL, "warning"),
    (Stage.OOM_PRUNED, "warning"), (Stage.VERIFYING_ARTIFACT, "info"),
    (Stage.MEASURING, "info"), (Stage.MAX_SCALE_PROBE, "info"),
    (Stage.STARTING, "info"), (Stage.RUNTIME_VALIDATION, "info"),
    (Stage.BUILDING_IMAGE, "info"),
])
def test_stage_feedback_comes_from_the_stage_not_warning_or_error_counters(stage, level):
    view = monitor_header_view(ProgressSnapshot(stage=stage, warnings=3, errors=2))

    assert view.summary.level == level
    assert translate(view.summary.headline, "zh") == f"{translate(stage.label, 'zh')} · 3 警告 / 2 错误"
    assert translate(view.summary.headline, "en") == f"{translate(stage.label, 'en')} · Warn 3 / Err 2"
    assert view.summary.actions == ()


def test_current_case_does_not_advance_processed_progress_or_mutate_snapshot():
    snapshot = ProgressSnapshot(
        stage=Stage.MEASURING, current_case=2, completed_cases=1, total_cases=4,
        cpu="4", mem="8", gpu="on", measurement_active=True,
        detail="raw collection detail", failure={"evidence": {"attempt": 7}},
    )
    before = copy.deepcopy(snapshot)
    view = monitor_header_view(snapshot)

    assert snapshot == before
    assert [row.value for row in view.cases.rows] == ["2", "1/4"]
    assert [translate(row.label, "en") for row in view.cases.rows] == ["Current case", "Processed"]
    assert view.progress_total == 4
    assert view.progress_completed == 1
    assert [translate(row.value, "en") for row in view.resources.rows] == ["4", "8GB", "On"]
    assert [row.detail for row in view.resources.rows] == ["4", "8", "on"]
    assert all(row.copyable and row.status is None for row in view.resources.rows)
    assert view.checks is None
    assert view.details.rows[0].value == "raw collection detail"


def test_planned_matrix_has_zero_progress_but_no_selected_resources():
    view = monitor_header_view(ProgressSnapshot(stage=Stage.PREPARING_MATRIX, total_cases=4))

    assert view.progress_total == 4
    assert view.progress_completed == 0
    assert translate(view.cases.rows[0].value, "en") == "Not applicable"
    assert view.cases.rows[1].value == "0/4"
    assert view.resources is None
    assert view.summary.level == "info"


@pytest.mark.parametrize(("total", "completed"), [(0, 0), (-1, 0), (4, -1), (4, 5)])
def test_unknown_or_inconsistent_progress_never_invents_a_bounded_percentage(total, completed):
    view = monitor_header_view(ProgressSnapshot(
        stage=Stage.MEASURING, current_case=2, total_cases=total, completed_cases=completed,
    ))

    assert view.progress_total is None
    assert view.progress_completed == 0
    if total > 0:
        assert view.cases.rows[1].value == f"{completed}/{total}"
    else:
        assert view.cases is None
    assert view.resources is not None


@pytest.mark.parametrize("stage", [Stage.COMPLETED, Stage.PARTIAL, Stage.FAILED, Stage.STOPPED])
def test_terminal_states_retain_last_case_and_resource_facts(stage):
    view = monitor_header_view(ProgressSnapshot(
        stage=stage, current_case=4, completed_cases=4, total_cases=4, cpu="2", mem="1", gpu="off",
    ))

    assert [row.value for row in view.cases.rows] == ["4", "4/4"]
    assert view.progress_total == view.progress_completed == 4
    assert [translate(row.value, "en") for row in view.resources.rows] == ["2", "1GB", "Off"]
    assert view.checks is None


def test_processed_all_cases_still_waits_for_artifact_verification():
    view = monitor_header_view(ProgressSnapshot(
        stage=Stage.VERIFYING_ARTIFACT, current_case=4, completed_cases=4, total_cases=4,
    ))

    assert view.progress_total == view.progress_completed == 4
    assert view.summary.level == "info"
    assert translate(view.summary.headline, "en") == "Verifying artifacts"


@pytest.mark.parametrize("stage", [
    Stage.RUNTIME_VALIDATION, Stage.PREPARING_MEMORY_PROBE, Stage.SCANNING_MEMORY,
    Stage.MAX_SCALE_PROBE, Stage.PROBE_COMPLETED, Stage.PROBE_FAILED,
])
def test_probe_and_validation_counts_do_not_become_formal_case_progress(stage):
    view = monitor_header_view(ProgressSnapshot(
        stage=stage, current_case=2, completed_cases=1, total_cases=5,
        cpu="4", mem="8", gpu="on", measurement_active=True,
        interface_status="passed", runtime_status="running",
    ))

    assert view.cases is view.resources is None
    assert view.progress_total is None
    assert view.progress_completed == 0
    assert len(view.checks.rows) == 2


@pytest.mark.parametrize(("cpu", "memory", "gpu", "expected"), [
    ("inherited", "inherited", "off", ("Inherited limits", "Inherited limits", "Off")),
    ("0", "0", "on", ("0", "0GB", "On")),
    ("-", "-", "-", ("Unknown", "Unknown", "Unknown")),
    ("", "", "", ("Unknown", "Unknown", "Unknown")),
    ("0-3", "0.5", "future [literal]", ("0-3", "0.5GB", "future [literal]")),
])
def test_resources_keep_inheritance_zero_unknown_and_raw_values_separate(cpu, memory, gpu, expected):
    view = monitor_header_view(ProgressSnapshot(
        stage=Stage.MEASURING, current_case=1, total_cases=2, cpu=cpu, mem=memory, gpu=gpu,
    ))

    rows = view.resources.rows
    assert tuple(translate(row.value, "en") for row in rows) == expected
    assert tuple(row.detail for row in rows) == (cpu, memory, gpu)
    assert all(row.status is None for row in rows)


def test_gpu_off_is_localized_without_claiming_runtime_success():
    view = monitor_header_view(ProgressSnapshot(stage=Stage.MEASURING, current_case=1, gpu="off"))
    row = view.resources.rows[2]

    assert translate(row.value, "zh") == "关闭"
    assert translate(row.value, "en") == "Off"
    assert row.detail == "off"
    assert row.status is None


@pytest.mark.parametrize(("raw", "display", "status"), [
    ("waiting", "Needs confirmation", "pending"), ("passed", "Passed", "passed"),
    ("failed", "Failed", "failed"), ("cancelled", "Cancelled", "cancelled"),
    ("running", "Running", "running"), ("future_state", "Unknown", "future_state"),
])
def test_checks_localize_states_and_keep_raw_state_in_details(raw, display, status):
    view = monitor_header_view(ProgressSnapshot(stage=Stage.RUNTIME_VALIDATION, interface_status=raw))

    assert len(view.checks.rows) == 1
    row = view.checks.rows[0]
    assert translate(row.label, "en") == "Interface resolution"
    assert translate(row.value, "en") == display
    assert row.status == status
    assert row.detail == raw


def test_preparation_checks_keep_existing_phase_visibility():
    snapshot = ProgressSnapshot(
        stage=Stage.PREPARING_IMAGE, interface_status="passed", runtime_status="failed",
    )
    assert monitor_header_view(snapshot).checks is None
    runtime = monitor_header_view(replace(snapshot, stage=Stage.RUNTIME_VALIDATION))
    assert [translate(row.label, "en") for row in runtime.checks.rows] == [
        "Interface resolution", "Runtime validation",
    ]
    measurement = monitor_header_view(replace(snapshot, stage=Stage.MEASURING))
    assert measurement.checks is None


def test_long_raw_details_preserve_whitespace_paths_and_literal_markup():
    raw = "  original: /tmp/中文 path/[file]\r\n[bold] <literal> " * 500
    view = monitor_header_view(ProgressSnapshot(stage=Stage.FAILED, detail=raw))

    assert view.details.collapsed
    assert len(view.details.rows) == 1
    row = view.details.rows[0]
    assert row.value == raw
    assert row.detail == raw
    assert row.copyable
    assert row.status is None


@pytest.mark.parametrize(("warnings", "errors", "labels", "values"), [
    (2, 0, ["Warnings"], ["2"]), (0, 3, ["Errors"], ["3"]),
    (2, 3, ["Warnings", "Errors"], ["2", "3"]),
])
def test_nonzero_counts_remain_available_in_collapsed_details(warnings, errors, labels, values):
    view = monitor_header_view(ProgressSnapshot(
        stage=Stage.COMPLETED, warnings=warnings, errors=errors,
    ))

    assert view.details.collapsed
    assert [translate(row.label, "en") for row in view.details.rows] == labels
    assert [row.value for row in view.details.rows] == values
    assert view.summary.level == "success"


def test_placeholder_and_zero_counts_do_not_create_obsolete_detail_rows():
    view = monitor_header_view(ProgressSnapshot(stage=Stage.STARTING))

    assert view.details is None
    assert translate(view.summary.headline, "en") == "Starting"


def test_projection_neither_reads_time_nor_opens_a_detail_path():
    snapshot = ProgressSnapshot(
        stage=Stage.MEASURING, current_case=2, completed_cases=1, total_cases=4,
        measurement_active=True, detail="/tmp/raw-error.txt",
    )
    with patch("time.monotonic", side_effect=AssertionError("clock access")), \
            patch.object(Path, "read_text", side_effect=AssertionError("file read")):
        view = monitor_header_view(snapshot)

    assert view.progress_completed == 1
    assert view.details.rows[0].value == "/tmp/raw-error.txt"


def test_view_and_sections_are_immutable():
    view = monitor_header_view(ProgressSnapshot(stage=Stage.PREPARING_MATRIX, total_cases=4))

    with pytest.raises(FrozenInstanceError):
        view.progress_total = 8
    with pytest.raises(FrozenInstanceError):
        view.cases.rows = ()
