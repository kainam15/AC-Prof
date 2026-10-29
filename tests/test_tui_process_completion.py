"""Unit tests for pure TUI process completion state transitions."""
from __future__ import annotations

from pathlib import Path

from acprof.tui.i18n import translate
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.run_results import (
    NotificationSpec,
    RunResult,
    evaluate_process_completion,
)
from acprof.tui.stages import Stage


def test_run_success_completion_and_csv_publication():
    run_result = RunResult(
        belongs_to_attempt=True,
        complete=True,
        partial=False,
        result_csv="/tmp/exp/result.csv",
        completed_cases=4,
        total_cases=4,
        new_cases=4,
        retained_dir="/tmp/exp",
    )
    snapshot = ProgressSnapshot(
        stage=Stage.VERIFYING_ARTIFACT,
        total_cases=4,
        measurement_status="running",
    )
    completion = evaluate_process_completion(
        "run",
        0,
        snapshot,
        ProgressSnapshot(),
        run_result=run_result,
    )
    assert completion.launch_error == ""
    assert completion.log_messages == ()
    assert completion.primary_notice is None
    assert completion.current_csv == "/tmp/exp/result.csv"
    assert completion.summary_text == ""
    assert completion.final_snapshot.stage is Stage.COMPLETED
    assert completion.final_snapshot.measurement_status == "passed"
    assert not completion.final_snapshot.measurement_active
    assert completion.final_snapshot.completed_cases == 4
    assert completion.final_snapshot.total_cases == 4
    assert completion.run_notice is not None
    assert completion.run_notice.severity == "information"


def test_run_without_new_csv_computes_unverified_or_empty_summary():
    unverified = evaluate_process_completion(
        "run",
        1,
        None,
        ProgressSnapshot(total_cases=2),
        run_result=RunResult(belongs_to_attempt=False, detail="corrupt state"),
    )
    assert unverified.current_csv == ""
    assert "corrupt state" in translate(unverified.summary_text, "en")
    assert unverified.final_snapshot.stage is Stage.FAILED
    assert unverified.final_snapshot.measurement_status == "failed"
    assert unverified.run_notice is not None
    assert unverified.run_notice.severity == "error"

    empty_attempt = evaluate_process_completion(
        "run",
        1,
        None,
        ProgressSnapshot(total_cases=2),
        run_result=RunResult(belongs_to_attempt=True, new_cases=0),
    )
    assert empty_attempt.current_csv == ""
    assert "0" in str(empty_attempt.summary_text)


def test_run_unsupported_task_preserves_stage_and_remedies():
    snapshot = ProgressSnapshot(
        stage=Stage.TASK_UNSUPPORTED,
        detail="Unsupported task: video-classification",
        measurement_active=True,
        measurement_status="running",
    )
    completion = evaluate_process_completion(
        "run",
        2,
        snapshot,
        ProgressSnapshot(),
    )
    assert completion.unsupported_task
    assert len(completion.log_messages) == 2
    assert completion.log_messages[0] == "Unsupported task: video-classification"
    assert completion.primary_notice == NotificationSpec(
        "Unsupported task: video-classification",
        severity="error",
        title="任务不支持",
        timeout=12,
    )
    assert completion.run_notice is None
    assert completion.final_snapshot.stage is Stage.TASK_UNSUPPORTED
    assert not completion.final_snapshot.measurement_active
    assert completion.final_snapshot.measurement_status == "failed"


def test_probe_requires_structured_completion_event_and_summary():
    incomplete_snapshot = ProgressSnapshot(stage=Stage.SCANNING_MEMORY)
    completion = evaluate_process_completion(
        "probe",
        0,
        incomplete_snapshot,
        ProgressSnapshot(),
    )
    assert "missing structured probe result" in translate(completion.launch_error, "en")
    assert completion.primary_notice is not None
    assert completion.primary_notice.severity == "error"
    assert completion.final_snapshot.stage is Stage.PROBE_FAILED
    assert not completion.final_snapshot.measurement_active

    valid_snapshot = ProgressSnapshot(
        stage=Stage.PROBE_COMPLETED,
        probe_summary="/tmp/probe_summary.json",
        probe_result={"status": "ok"},
    )
    ok_completion = evaluate_process_completion(
        "probe",
        0,
        valid_snapshot,
        ProgressSnapshot(),
    )
    assert ok_completion.launch_error == ""
    assert ok_completion.final_snapshot.stage is Stage.PROBE_COMPLETED
    assert ok_completion.primary_notice == NotificationSpec(
        "任务已完成",
        severity="information",
        timeout=5,
    )


def test_probe_cancellation_transitions_to_terminated():
    snapshot = ProgressSnapshot(
        stage=Stage.SCANNING_MEMORY,
        measurement_active=True,
        measurement_status="running",
    )
    completion = evaluate_process_completion(
        "probe",
        130,
        snapshot,
        ProgressSnapshot(),
        stop_requested=True,
    )
    assert completion.final_snapshot.stage is Stage.TERMINATED
    assert completion.final_snapshot.measurement_status == "cancelled"
    assert not completion.final_snapshot.measurement_active
    assert completion.primary_notice is not None
    assert completion.primary_notice.severity == "warning"


def test_stats_requires_valid_in_memory_result(tmp_path: Path):
    missing = evaluate_process_completion(
        "stats",
        0,
        None,
        ProgressSnapshot(),
        analysis_ready=False,
    )
    assert missing.launch_error == "分析进程未返回当前选择的有效统计数据"
    assert missing.final_snapshot.stage is Stage.FAILED
    assert missing.primary_notice is not None
    assert missing.primary_notice.severity == "error"

    ready = evaluate_process_completion(
        "stats",
        0,
        None,
        ProgressSnapshot(),
        analysis_ready=True,
    )
    assert ready.launch_error == ""
    assert ready.final_snapshot.stage is Stage.COMPLETED
    assert ready.primary_notice is not None
    assert list(tmp_path.iterdir()) == []


def test_compare_accepts_exit_code_one_with_valid_in_memory_result(tmp_path: Path):
    ready = evaluate_process_completion(
        "compare",
        1,
        None,
        ProgressSnapshot(),
        analysis_ready=True,
    )
    assert ready.comparison_ready
    assert ready.launch_error == ""
    assert ready.final_snapshot.stage is Stage.COMPLETED
    assert ready.primary_notice is not None
    assert ready.primary_notice.severity == "information"

    missing_on_zero = evaluate_process_completion(
        "compare",
        0,
        None,
        ProgressSnapshot(),
        analysis_ready=False,
    )
    assert not missing_on_zero.comparison_ready
    assert missing_on_zero.launch_error == "分析进程未返回当前选择的有效统计数据"
    assert list(tmp_path.iterdir()) == []
    assert missing_on_zero.final_snapshot.stage is Stage.FAILED


def test_html_report_error_and_cancellation_use_dedicated_notices():
    failed = evaluate_process_completion(
        "html-report",
        1,
        None,
        ProgressSnapshot(),
        launch_error="render failure",
    )
    assert failed.primary_notice is not None
    assert failed.primary_notice.severity == "error"
    assert translate(failed.primary_notice.title, "en") != ""
    assert failed.final_snapshot.stage is Stage.FAILED

    cancelled = evaluate_process_completion(
        "html-report",
        130,
        None,
        ProgressSnapshot(),
        stop_requested=True,
    )
    assert cancelled.primary_notice is not None
    assert cancelled.primary_notice.severity == "warning"
    assert cancelled.final_snapshot.stage is Stage.TERMINATED
