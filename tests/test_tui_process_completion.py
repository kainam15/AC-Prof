from acprof.tui.process_completion import build_process_completion
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.run_results import RunResult
from acprof.tui.stages import Stage


def _completion(kind, returncode=0, snapshot=None, launch_error="", **overrides):
    options = {
        "stopped": False,
        "latest_snapshot": ProgressSnapshot(),
        "run_result": RunResult(),
        "stats_report_path": None,
        "stats_report_reused": False,
        "comparison_report_exists": False,
    }
    options.update(overrides)
    return build_process_completion(
        kind,
        returncode,
        snapshot,
        launch_error,
        **options,
    )


def test_compare_exit_one_with_persisted_report_is_a_completed_comparison():
    result = _completion("compare", 1, comparison_report_exists=True)

    assert result.comparison_ready
    assert result.snapshot.stage is Stage.COMPLETED
    assert result.notice is not None
    assert result.notice.severity == "information"
    assert result.launch_error == ""


def test_compare_zero_exit_without_report_is_not_success():
    result = _completion("compare")

    assert not result.comparison_ready
    assert result.snapshot.stage is Stage.FAILED
    assert result.notice is not None
    assert result.notice.severity == "error"
    assert result.launch_error == "比较进程未返回报告路径"


def test_run_zero_exit_without_result_evidence_is_not_success():
    result = _completion("run")

    assert result.snapshot.stage is Stage.FAILED
    assert result.notice is None
    assert result.run_notice is not None
    assert result.run_notice.severity == "error"
    assert result.summary_text is not None


def test_partial_run_preserves_new_case_count_and_detail():
    result = _completion(
        "run",
        2,
        launch_error="collection failed",
        run_result=RunResult(
            belongs_to_attempt=True,
            result_csv="results/run/result.csv",
            completed_cases=2,
            total_cases=4,
            new_cases=2,
            detail="two cases failed",
            retained_dir="results/run",
        ),
    )

    assert result.snapshot.stage is Stage.PARTIAL
    assert result.snapshot.completed_cases == 2
    assert result.snapshot.total_cases == 4
    assert not result.snapshot.measurement_active
    assert result.current_csv == "results/run/result.csv"
    assert result.run_notice is not None
    assert result.run_notice.severity == "warning"
    assert "two cases failed" in result.run_notice.message


def test_cancelled_measurement_restores_terminal_snapshot_state():
    result = _completion(
        "run",
        -2,
        snapshot=ProgressSnapshot(measurement_active=True, measurement_status="running"),
        stopped=True,
        run_result=RunResult(belongs_to_attempt=True),
    )

    assert result.snapshot.stage is Stage.STOPPED
    assert not result.snapshot.measurement_active
    assert result.snapshot.measurement_status == "cancelled"


def test_reused_stats_report_uses_path_notice():
    result = _completion(
        "stats",
        stats_report_path="results/stats.json",
        stats_report_reused=True,
    )

    assert result.snapshot.stage is Stage.COMPLETED
    assert result.notice is not None
    assert "results/stats.json" in result.notice.message
    assert result.notice.timeout == 8


def test_unsupported_task_keeps_specific_notification_and_failed_snapshot():
    result = _completion(
        "run",
        2,
        snapshot=ProgressSnapshot(stage=Stage.TASK_UNSUPPORTED, detail="unsupported fixture"),
    )

    assert result.snapshot.stage is Stage.TASK_UNSUPPORTED
    assert result.snapshot.measurement_status == "failed"
    assert result.log_messages == ("unsupported fixture", result.log_messages[1])
    assert result.notice is not None
    assert result.notice.title == "任务不支持"


def test_probe_zero_exit_without_structured_result_is_not_success():
    result = _completion("probe")

    assert result.snapshot.stage is Stage.PROBE_FAILED
    assert result.notice is not None
    assert result.notice.severity == "error"
    assert "missing structured probe result" in result.launch_error
