"""Pure completion decisions for finished UI subprocesses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from acprof.messages import join_messages
from acprof.tui.i18n import message_id
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.run_results import RunResult
from acprof.tui.stages import Stage


@dataclass(frozen=True)
class CompletionNotice:
    message: str
    title: str = ""
    severity: str = "information"
    timeout: int = 5


@dataclass(frozen=True)
class ProcessCompletion:
    launch_error: str
    comparison_ready: bool
    snapshot: ProgressSnapshot
    log_messages: tuple[str, ...] = ()
    notice: CompletionNotice | None = None
    run_notice: CompletionNotice | None = None
    current_csv: str = ""
    summary_text: str | None = None


def _early_presentation(
    kind: str,
    returncode: int,
    launch_error: str,
    *,
    stopped: bool,
    comparison_ready: bool,
    unsupported_task: bool,
    snapshot: ProgressSnapshot | None,
    stats_report_path: str | Path | None,
    stats_report_reused: bool,
) -> tuple[tuple[str, ...], CompletionNotice | None]:
    if launch_error:
        return (message_id("ui.0786", launch_error),), CompletionNotice(
            launch_error,
            title=message_id("ui.0986") if kind == "html-report" else "任务启动失败",
            severity="error",
            timeout=8,
        )
    if comparison_ready:
        comparison_notice = message_id("ui.0488")
        return (comparison_notice,), CompletionNotice(comparison_notice, timeout=8)
    if returncode == 0 and kind != "run" and not stopped:
        notice: str = (
            message_id("ui.0571", stats_report_path)
            if kind == "stats" and stats_report_reused
            else "任务已完成"
        )
        return (message_id("ui.0788", kind),), CompletionNotice(
            notice,
            timeout=8 if stats_report_reused else 5,
        )
    if stopped:
        return (message_id("ui.0790", returncode),), CompletionNotice(
            message_id("ui.0962") if kind == "html-report" else "任务已终止；部分 case 结果可能仍可续跑",
            severity="warning",
            timeout=7,
        )
    if unsupported_task and snapshot is not None:
        return (snapshot.detail, message_id("ui.0587")), CompletionNotice(
            snapshot.detail,
            title="任务不支持",
            severity="error",
            timeout=12,
        )
    if kind != "run":
        return (message_id("ui.0792", kind, returncode),), CompletionNotice(
            message_id("ui.0793", returncode), severity="error", timeout=8,
        )
    return (), None


def build_process_completion(
    kind: str,
    returncode: int,
    snapshot: ProgressSnapshot | None,
    launch_error: str,
    *,
    stopped: bool,
    latest_snapshot: ProgressSnapshot,
    run_result: RunResult,
    stats_report_path: str | Path | None,
    stats_report_reused: bool,
    comparison_report_exists: bool,
) -> ProcessCompletion:
    """Resolve evidence and final state without reading widgets or the filesystem."""
    if kind == "stats" and returncode == 0 and not launch_error and stats_report_path is None:
        launch_error = "统计进程未返回报告路径"
    if kind == "probe" and returncode == 0 and not launch_error and not stopped:
        if (
            snapshot is None
            or snapshot.stage is not Stage.PROBE_COMPLETED
            or not snapshot.probe_result
            or not snapshot.probe_summary
        ):
            launch_error = message_id("ui.0183", "missing structured probe result")
    comparison_ready = (
        kind == "compare"
        and returncode in {0, 1}
        and not launch_error
        and not stopped
        and comparison_report_exists
    )
    if (
        kind == "compare"
        and returncode == 0
        and not launch_error
        and not stopped
        and not comparison_ready
    ):
        launch_error = "比较进程未返回报告路径"
    if snapshot is not None and snapshot.measurement_status == "running":
        snapshot = replace(
            snapshot,
            measurement_status="cancelled" if stopped else "failed" if returncode else "passed",
        )

    unsupported_task = (
        snapshot is not None
        and snapshot.stage is Stage.TASK_UNSUPPORTED
        and returncode != 0
        and not stopped
        and not launch_error
    )
    log_messages, notice = _early_presentation(
        kind,
        returncode,
        launch_error,
        stopped=stopped,
        comparison_ready=comparison_ready,
        unsupported_task=unsupported_task,
        snapshot=snapshot,
        stats_report_path=stats_report_path,
        stats_report_reused=stats_report_reused,
    )

    current_csv = ""
    summary_text = None
    run_notice = None
    if unsupported_task and snapshot is not None:
        final_snapshot = replace(snapshot, measurement_active=False, measurement_status="failed")
    elif kind == "run":
        stage = run_result.stage(returncode, stopped, launch_error)
        final_state = snapshot or latest_snapshot
        detail: str = message_id(
            "ui.0191",
            run_result.completed_cases,
            run_result.total_cases or final_state.total_cases,
        )
        if run_result.detail:
            detail = join_messages("\n", (detail, run_result.detail))
        if run_result.new_cases and run_result.retained_dir:
            detail = join_messages("\n", (detail, message_id("ui.0184", run_result.retained_dir)))
        if run_result.complete and not run_result.result_csv and not run_result.new_cases:
            detail = join_messages("\n", (detail, message_id("ui.0185")))
        final_snapshot = replace(
            final_state,
            stage=stage,
            detail=detail,
            completed_cases=run_result.completed_cases,
            total_cases=run_result.total_cases or final_state.total_cases,
            measurement_active=False,
            measurement_status=(
                "passed" if stage is Stage.COMPLETED else "cancelled" if stopped else "failed"
            ),
        )
        current_csv = run_result.result_csv
        if not current_csv:
            summary_text = (
                message_id("ui.0183", run_result.detail)
                if run_result.detail and not run_result.belongs_to_attempt
                else message_id("ui.0190", run_result.new_cases)
            )
        run_notice = CompletionNotice(
            message_id("ui.0193", stage.label, detail),
            severity=(
                "information"
                if stage is Stage.COMPLETED
                else "warning"
                if stage in {Stage.PARTIAL, Stage.STOPPED}
                else "error"
            ),
        )
    elif kind == "probe":
        final_snapshot = snapshot or latest_snapshot
        if launch_error or (returncode != 0 and not stopped):
            detail = launch_error or (
                final_snapshot.detail
                if final_snapshot.stage is Stage.PROBE_FAILED
                else message_id("ui.0796", returncode)
            )
            final_snapshot = replace(
                final_snapshot,
                stage=Stage.PROBE_FAILED,
                detail=detail,
                measurement_active=False,
            )
        elif stopped:
            final_snapshot = replace(
                final_snapshot,
                stage=Stage.TERMINATED,
                detail="用户终止了最大尺度探测",
                measurement_active=False,
            )
    else:
        if launch_error or (returncode != 0 and not stopped and not comparison_ready):
            stage = Stage.FAILED
            detail = launch_error or message_id("ui.0799", kind, returncode)
        elif stopped:
            stage = Stage.TERMINATED
            detail = message_id("ui.0800", kind)
        else:
            stage = Stage.COMPLETED
            detail = message_id("ui.0801", kind)
        final_snapshot = ProgressSnapshot(stage=stage, detail=detail)

    return ProcessCompletion(
        launch_error=launch_error,
        comparison_ready=comparison_ready,
        snapshot=final_snapshot,
        log_messages=log_messages,
        notice=notice,
        run_notice=run_notice,
        current_csv=current_csv,
        summary_text=summary_text,
    )
