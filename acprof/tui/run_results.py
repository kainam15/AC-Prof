"""本次子进程与既有产物协议的只读关联，不写入新的运行协议。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from acprof.analysis.audit import audit_result
from acprof.artifact_layout import ArtifactLayout
from acprof.host.run_state import load_run_state
from acprof.messages import Message, join_messages, message

MAX_CAPABILITY_REPORT_BYTES = 4 * 1024 * 1024


def _finite_json_number(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(message("包含非有限数值"))
    return value


def _read_capability_report(path: Path) -> dict:
    with path.open("rb") as stream:
        content = stream.read(MAX_CAPABILITY_REPORT_BYTES + 1)
    if len(content) > MAX_CAPABILITY_REPORT_BYTES:
        raise ValueError(message("超过 4 MiB 读取上限"))
    try:
        value = json.loads(content, parse_float=_finite_json_number,
                           parse_constant=_finite_json_number)
    except json.JSONDecodeError as exc:
        raise ValueError(message("JSON 格式无效")) from exc
    if not isinstance(value, dict):
        raise ValueError(message("最外层必须是 JSON 对象"))
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

    @classmethod
    def read(cls, directory: Path) -> RunArtifacts:
        layout = ArtifactLayout.discover(directory)
        path = layout.path("run_state.json")
        state = load_run_state(directory) if path.is_file() else {}
        return cls(directory, state, _stamp(layout.result_csv))


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

    def stage(self, returncode: int, stopped: bool, error: str) -> str:
        if stopped:
            return "已停止"
        if error:
            return "部分完成" if self.result_csv or self.new_cases else "失败"
        if self.complete and not returncode:
            return "部分完成" if self.partial else "已完成"
        return "部分完成" if self.result_csv or self.new_cases else "失败"


def inspect_run_result(before: RunArtifacts, pid: int) -> RunResult:
    """Run after process exit; PID plus appended attempt separates resume/history."""
    after = RunArtifacts.read(before.directory)
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
    layout = ArtifactLayout.discover(before.directory)
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
            capability_issue = message("能力报告 {0} 无效：{1}", capability_path.name, reason)
    partial |= capability.get("requested_measurements_complete") is not True
    details = [issue["message"] for issue in audit.get("issues", [])
               if issue.get("severity") == "error"]
    if capability_issue:
        details.append(capability_issue)
    return RunResult(True, complete, partial, current_csv, len(completed), total, new_cases,
                     join_messages("; ", details), str(layout.root))
