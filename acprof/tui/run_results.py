"""本次子进程与既有产物协议的只读关联，不写入新的运行协议。"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from acprof.analysis.audit import audit_result
from acprof.artifact_layout import ArtifactLayout
from acprof.host.run_state import RunStateError, load_run_state
from acprof.messages import Message, join_messages
from acprof.tui.i18n import message_id
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.stages import Stage

MAX_CAPABILITY_REPORT_BYTES = 4 * 1024 * 1024


def inspect_probe_result(snapshot, config, project_dir: Path, pid: int, started_at_ns: int) -> None:
    """Verify a current probe's existing summary after its subprocess has exited."""
    from acprof.host.largest_scale_probe import PROBE_SUMMARY_NAME, read_probe_summary

    if snapshot is None or snapshot.stage is not Stage.PROBE_COMPLETED or not snapshot.probe_result:
        raise ValueError("Probe exited without a successful structured completion event")
    if not snapshot.probe_summary or config is None:
        raise ValueError("Probe exited without its summary path or launch configuration")
    path = Path(snapshot.probe_summary).resolve()
    root = ArtifactLayout.discover(config.result_dir(project_dir)).path("probes").resolve()
    if path.name != PROBE_SUMMARY_NAME or path.parent.parent != root:
        raise ValueError("Probe summary is outside the current experiment's probe directory")
    # create_probe_output_dir already records the child PID in this immutable
    # directory name. No new identity sidecar or historical file is rewritten.
    parts = path.parent.name.split("_")
    if len(parts) not in (4, 5) or parts[:2] != ["largest", "scale"] or parts[3] != str(pid):
        raise ValueError("Probe summary does not belong to the current subprocess")
    if path.stat().st_mtime_ns < started_at_ns:
        raise ValueError("Probe summary predates this subprocess launch")
    summary = read_probe_summary(path)
    if summary["model"]["id"] != config.model:
        raise ValueError("Probe summary model identity differs from the launch configuration")
    expected = {
        "status": summary["status"], "cpu": summary["resource"]["cpu_cores"],
        "mem": summary["resource"]["mem_gb"], "gpu": summary["resource"]["gpu_mode"],
        "input_scale": summary["input"]["planned_scale"],
        "request_s": summary["timing"].get("request_s"),
        "ready_plus_request_s": summary["timing"].get("ready_plus_request_s"),
        "cold_start_s": summary["cold_start"].get("total_s"),
    }
    if any(snapshot.probe_result.get(key) != value for key, value in expected.items()):
        raise ValueError("Probe completion event disagrees with the persisted summary")


def _finite_json_number(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(message_id('ui.0559'))
    return value


def _read_capability_report(path: Path) -> dict:
    with path.open("rb") as stream:
        content = stream.read(MAX_CAPABILITY_REPORT_BYTES + 1)
    if len(content) > MAX_CAPABILITY_REPORT_BYTES:
        raise ValueError(message_id('ui.0560'))
    try:
        value = json.loads(content, parse_float=_finite_json_number,
                           parse_constant=_finite_json_number)
    except json.JSONDecodeError as exc:
        raise ValueError(message_id('ui.0557')) from exc
    if not isinstance(value, dict):
        raise ValueError(message_id('ui.0558'))
    return value


def _stamp(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class RunArtifacts:
    directory: Path
    state: dict = field(default_factory=dict)
    csv_stamp: tuple[int, int, int, int] | None = None
    state_error: str = ""
    captured_at_ns: int = field(default_factory=time.time_ns)
    task_children: dict[str, RunArtifacts] = field(default_factory=dict)

    @classmethod
    def read(cls, directory: Path, *, capture_state_error: bool = False) -> RunArtifacts:
        layout = ArtifactLayout.discover(directory)
        path = layout.path("run_state.json")
        state = {}
        state_error = ""
        if path.is_file():
            try:
                state = load_run_state(directory)
            except RunStateError as exc:
                if not capture_state_error:
                    raise
                state_error = exc.args[0] if exc.args and isinstance(exc.args[0], Message) else str(exc)
        children = {str(child.relative_to(directory)): cls.read(child, capture_state_error=True)
                    for child in directory.glob("tasks/*/*")
                    if child.is_dir() and not child.is_symlink()
                    and child.resolve().is_relative_to(directory.resolve())}
        return cls(directory, state, _stamp(layout.result_csv), state_error, task_children=children)


@dataclass(frozen=True)
class RunResult:
    belongs_to_attempt: bool = False
    complete: bool = False
    partial: bool = False
    result_csv: str = ""
    completed_cases: int = 0
    total_cases: int = 0
    new_cases: int = 0
    detail: str = ""
    retained_dir: str = ""

    def stage(self, returncode: int, stopped: bool, error: str) -> Stage:
        if stopped:
            return Stage.STOPPED
        if error:
            return Stage.PARTIAL if self.result_csv or self.new_cases else Stage.FAILED
        if self.complete and not returncode:
            return Stage.PARTIAL if self.partial else Stage.COMPLETED
        return Stage.PARTIAL if self.result_csv or self.new_cases else Stage.FAILED


def inspect_run_result(before: RunArtifacts, pid: int) -> RunResult:
    """Run after process exit; PID plus appended attempt separates resume/history."""
    group = inspect_task_collection(before, pid)
    if group is not None:
        return group
    after = RunArtifacts.read(before.directory, capture_state_error=True)
    layout = ArtifactLayout.discover(before.directory)
    if after.state_error:
        details: list[str] = [message_id('ui.0567', after.state_error)]
        if after.csv_stamp and after.csv_stamp != before.csv_stamp:
            details.append(message_id('ui.0568', layout.result_csv))
        return RunResult(detail=join_messages("; ", details), retained_dir=str(layout.root))
    state, previous = after.state, before.state
    attempts = state.get("attempts", [])
    if (not state.get("run_id") or not isinstance(attempts, list) or not attempts
            or not isinstance(attempts[-1], dict) or attempts[-1].get("pid") != pid
            or not attempts[-1].get("ended_at")
            or (state.get("run_id") == previous.get("run_id")
                and len(attempts) <= len(previous.get("attempts", [])))):
        return RunResult()
    cases = state.get("cases", {})
    old_cases = previous.get("cases", {}) if state.get("run_id") == previous.get("run_id") else {}
    completed = {name: record for name, record in cases.items() if record.get("status") == "complete"}
    new_cases = sum(record != old_cases.get(name) for name, record in completed.items())
    options = state.get("options", {})
    total = 1
    for name in ("cpus", "mems", "gpus"):
        total *= len([part for part in str(options.get(name, "")).split(",") if part])
    current_csv = str(layout.result_csv) if after.csv_stamp and after.csv_stamp != before.csv_stamp else ""
    audit = audit_result(layout.result_csv) if after.csv_stamp else {}
    coverage = audit.get("coverage") or {}
    complete = (state.get("status") == "complete" and audit.get("valid", False)
                and bool(coverage) and coverage.get("missing") == coverage.get("unexpected") == 0)
    counts = audit.get("counts", {})
    partial = state.get("outcome") == "partial" or bool(counts.get("error") or counts.get("warn"))
    capability_path = layout.path("capability_report.json")
    capability = {}
    capability_issue = ""
    if capability_path.is_file():
        try:
            capability = _read_capability_report(capability_path)
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            reason = exc.args[0] if exc.args and isinstance(exc.args[0], Message) else str(exc)
            capability_issue = message_id('ui.0556', capability_path.name, reason)
    partial |= capability.get("requested_measurements_complete") is not True
    details = [issue["message"] for issue in audit.get("issues", [])
               if issue.get("severity") == "error"]
    if capability_issue:
        details.append(capability_issue)
    return RunResult(True, complete, partial, current_csv, len(completed), total, new_cases,
                     join_messages("; ", details), str(layout.root))


def inspect_task_collection(before: RunArtifacts, pid: int) -> RunResult | None:
    """Verify every child result; a collection status alone cannot establish success."""
    from acprof.artifacts import read_json_object
    paths = before.directory / "metadata" / "task_runs"
    for path in paths.glob("*.json"):
        if path.stat().st_mtime_ns < before.captured_at_ns:
            continue
        group = read_json_object(path, label="task collection")
        if group.get("pid") != pid or group.get("started_at_ns", 0) < before.captured_at_ns:
            continue
        if group.get("schema_version") != 1 or group.get("run_id") != path.stem:
            raise ValueError("invalid task collection identity")
        results, details = [], []
        for task in group["tasks"]:
            child = before.directory / "tasks" / task["task_id"] / group["run_id"]
            if not child.resolve().is_relative_to(before.directory.resolve()):
                raise ValueError("task result outside collection")
            previous = before.task_children.get(str(child.relative_to(before.directory)),
                                                RunArtifacts(child, captured_at_ns=before.captured_at_ns))
            result = inspect_run_result(previous, pid)
            results.append(result)
            if task.get("error") or result.detail:
                details.append(f"{task['task_id']}: {task.get('error') or result.detail}")
        csvs = [result.result_csv for result in results if result.result_csv]
        complete = bool(results) and group.get("status") == "ok" and all(result.complete for result in results)
        return RunResult(True, complete, not complete or any(result.partial for result in results),
                         csvs[-1] if csvs else "", sum(result.completed_cases for result in results),
                         sum(result.total_cases for result in results), sum(result.new_cases for result in results),
                         "; ".join(details), str(before.directory))
    return None


@dataclass(frozen=True)
class NotificationSpec:
    message: str
    severity: Literal["information", "warning", "error"] = "information"
    title: str = ""
    timeout: int | None = None

    def kwargs(self) -> dict[str, Any]:
        options: dict[str, Any] = {"severity": self.severity}
        if self.title:
            options["title"] = self.title
        if self.timeout is not None:
            options["timeout"] = self.timeout
        return options


@dataclass(frozen=True)
class ProcessCompletion:
    launch_error: str
    comparison_ready: bool
    unsupported_task: bool
    log_messages: tuple[str, ...]
    primary_notice: NotificationSpec | None
    run_notice: NotificationSpec | None
    final_snapshot: ProgressSnapshot
    current_csv: str = ""
    summary_text: str = ""


def _completion_log_and_notice(
    kind: str,
    returncode: int,
    snapshot: ProgressSnapshot | None,
    *,
    launch_error: str,
    stop_requested: bool,
    comparison_ready: bool,
    unsupported_task: bool,
    stats_report_path: Path | None,
    stats_report_reused: bool,
) -> tuple[tuple[str, ...], NotificationSpec | None]:
    if launch_error:
        title = message_id('ui.0986') if kind == "html-report" else "任务启动失败"
        return (
            (message_id('ui.0786', launch_error),),
            NotificationSpec(launch_error, severity="error", title=title, timeout=8),
        )
    if comparison_ready:
        notice: str = message_id('ui.0488')
        return (
            (notice,),
            NotificationSpec(notice, severity="information", timeout=8),
        )
    if returncode == 0 and kind != "run" and not stop_requested:
        reused = kind == "stats" and stats_report_reused
        notice = message_id('ui.0571', stats_report_path) if reused else "任务已完成"
        return (
            (message_id('ui.0788', kind),),
            NotificationSpec(notice, severity="information", timeout=8 if stats_report_reused else 5),
        )
    if stop_requested:
        notice = (
            message_id('ui.0962')
            if kind == "html-report"
            else "任务已终止；部分 case 结果可能仍可续跑"
        )
        return (
            (message_id('ui.0790', returncode),),
            NotificationSpec(notice, severity="warning", timeout=7),
        )
    if unsupported_task and snapshot is not None:
        return (
            (snapshot.detail, message_id('ui.0587')),
            NotificationSpec(snapshot.detail, severity="error", title="任务不支持", timeout=12),
        )
    if kind != "run":
        return (
            (message_id('ui.0792', kind, returncode),),
            NotificationSpec(message_id('ui.0793', returncode), severity="error", timeout=8),
        )
    return (), None


def _evaluate_run_outcome(
    returncode: int,
    snapshot: ProgressSnapshot | None,
    latest_snapshot: ProgressSnapshot,
    *,
    launch_error: str,
    stop_requested: bool,
    run_result: RunResult,
) -> tuple[ProgressSnapshot, str, str, NotificationSpec]:
    stage = run_result.stage(returncode, stop_requested, launch_error)
    final_state = snapshot or latest_snapshot
    total_cases = run_result.total_cases or final_state.total_cases
    detail: str = message_id('ui.0191', run_result.completed_cases, total_cases)
    if run_result.detail:
        detail = join_messages("\n", (detail, run_result.detail))
    if run_result.new_cases and run_result.retained_dir:
        detail = join_messages("\n", (detail, message_id('ui.0184', run_result.retained_dir)))
    if run_result.complete and not run_result.result_csv and not run_result.new_cases:
        detail = join_messages("\n", (detail, message_id('ui.0185')))
    measurement_status = (
        "passed" if stage is Stage.COMPLETED else "cancelled" if stop_requested else "failed"
    )
    final_snapshot = replace(
        final_state,
        stage=stage,
        detail=detail,
        completed_cases=run_result.completed_cases,
        total_cases=total_cases,
        measurement_active=False,
        measurement_status=measurement_status,
    )
    current_csv = run_result.result_csv
    summary_text = ""
    if not current_csv:
        summary_text = (
            message_id('ui.0183', run_result.detail)
            if run_result.detail and not run_result.belongs_to_attempt
            else message_id('ui.0190', run_result.new_cases)
        )
    severity: Literal["information", "warning", "error"] = (
        "information"
        if stage is Stage.COMPLETED
        else "warning"
        if stage in {Stage.PARTIAL, Stage.STOPPED}
        else "error"
    )
    return (
        final_snapshot,
        current_csv,
        summary_text,
        NotificationSpec(message_id('ui.0193', stage.label, detail), severity=severity),
    )


def _evaluate_probe_snapshot(
    returncode: int,
    snapshot: ProgressSnapshot | None,
    latest_snapshot: ProgressSnapshot,
    *,
    launch_error: str,
    stop_requested: bool,
) -> ProgressSnapshot:
    final_state = snapshot or latest_snapshot
    if launch_error or (returncode != 0 and not stop_requested):
        detail = launch_error or (
            final_state.detail
            if final_state.stage is Stage.PROBE_FAILED
            else message_id('ui.0796', returncode)
        )
        return replace(
            final_state,
            stage=Stage.PROBE_FAILED,
            detail=detail,
            measurement_active=False,
        )
    if stop_requested:
        return replace(
            final_state,
            stage=Stage.TERMINATED,
            detail="用户终止了最大尺度探测",
            measurement_active=False,
        )
    return final_state


def _evaluate_auxiliary_snapshot(
    kind: str,
    returncode: int,
    *,
    launch_error: str,
    stop_requested: bool,
    comparison_ready: bool,
) -> ProgressSnapshot:
    if launch_error or (returncode != 0 and not stop_requested and not comparison_ready):
        return ProgressSnapshot(
            stage=Stage.FAILED,
            detail=launch_error or message_id('ui.0799', kind, returncode),
        )
    if stop_requested:
        return ProgressSnapshot(stage=Stage.TERMINATED, detail=message_id('ui.0800', kind))
    return ProgressSnapshot(stage=Stage.COMPLETED, detail=message_id('ui.0801', kind))


def evaluate_process_completion(
    kind: str,
    returncode: int,
    snapshot: ProgressSnapshot | None,
    latest_snapshot: ProgressSnapshot,
    *,
    launch_error: str = "",
    stop_requested: bool = False,
    run_result: RunResult = RunResult(),
    stats_report_path: Path | None = None,
    stats_report_reused: bool = False,
    comparison_report_path: Path | None = None,
) -> ProcessCompletion:
    """Compute the pure state transition, logs, and notifications on process exit."""
    if kind == "stats" and returncode == 0 and not launch_error and stats_report_path is None:
        launch_error = "统计进程未返回报告路径"
    if kind == "probe" and returncode == 0 and not launch_error and not stop_requested:
        if (snapshot is None or snapshot.stage is not Stage.PROBE_COMPLETED
                or not snapshot.probe_result or not snapshot.probe_summary):
            launch_error = message_id('ui.0183', "missing structured probe result")
    comparison_ready = (
        kind == "compare" and returncode in {0, 1} and not launch_error
        and not stop_requested and comparison_report_path is not None
        and comparison_report_path.is_file()
    )
    if kind == "compare" and returncode == 0 and not launch_error and not stop_requested and not comparison_ready:
        launch_error = "比较进程未返回报告路径"
    if snapshot is not None and snapshot.measurement_status == "running":
        snapshot = replace(
            snapshot,
            measurement_status="cancelled" if stop_requested else "failed" if returncode else "passed",
        )
    unsupported_task = (
        snapshot is not None and snapshot.stage is Stage.TASK_UNSUPPORTED
        and returncode != 0 and not stop_requested and not launch_error
    )
    log_messages, primary_notice = _completion_log_and_notice(
        kind,
        returncode,
        snapshot,
        launch_error=launch_error,
        stop_requested=stop_requested,
        comparison_ready=comparison_ready,
        unsupported_task=unsupported_task,
        stats_report_path=stats_report_path,
        stats_report_reused=stats_report_reused,
    )
    current_csv = ""
    summary_text = ""
    run_notice: NotificationSpec | None = None
    if unsupported_task and snapshot is not None:
        final_snapshot = replace(snapshot, measurement_active=False, measurement_status="failed")
    elif kind == "run":
        final_snapshot, current_csv, summary_text, run_notice = _evaluate_run_outcome(
            returncode,
            snapshot,
            latest_snapshot,
            launch_error=launch_error,
            stop_requested=stop_requested,
            run_result=run_result,
        )
    elif kind == "probe":
        final_snapshot = _evaluate_probe_snapshot(
            returncode,
            snapshot,
            latest_snapshot,
            launch_error=launch_error,
            stop_requested=stop_requested,
        )
    else:
        final_snapshot = _evaluate_auxiliary_snapshot(
            kind,
            returncode,
            launch_error=launch_error,
            stop_requested=stop_requested,
            comparison_ready=comparison_ready,
        )
    return ProcessCompletion(
        launch_error=launch_error,
        comparison_ready=comparison_ready,
        unsupported_task=unsupported_task,
        log_messages=log_messages,
        primary_notice=primary_notice,
        run_notice=run_notice,
        final_snapshot=final_snapshot,
        current_csv=current_csv,
        summary_text=summary_text,
    )
