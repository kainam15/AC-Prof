"""本次子进程与既有产物协议的只读关联，不写入新的运行协议。"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from acprof.analysis.audit import audit_result
from acprof.artifact_layout import ArtifactLayout
from acprof.host.run_state import RunStateError, load_run_state
from acprof.messages import Message, join_messages
from acprof.tui.i18n import message_id
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
