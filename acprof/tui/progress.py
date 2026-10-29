"""结构化采集进度及正式测量窗口状态；普通日志不参与控制。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any

from acprof.messages import message
from acprof.preparation_events import parse_event as parse_preparation_event
from acprof.progress_events import parse_event
from acprof.tui.i18n import message_id
from acprof.tui.presentation import UNKNOWN
from acprof.tui.stages import Stage

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
def _format_probe_duration(value: float | None) -> str:
    return f"{value:.3f}s" if value is not None and math.isfinite(value) else UNKNOWN


@dataclass(frozen=True)
class ProgressSnapshot:
    stage: Stage = Stage.WAITING
    detail: str = message_id('run.monitor.not_started')
    current_case: int = 0
    completed_cases: int = 0
    total_cases: int = 0
    cpu: str = "-"
    mem: str = "-"
    gpu: str = "-"
    measurement_active: bool = False
    warnings: int = 0
    errors: int = 0
    final_csv: str = ""
    probe_summary: str = ""
    probe_result: dict | None = None
    interface_status: str = "not_started"
    runtime_status: str = "not_started"
    measurement_status: str = "not_started"
    failure: dict | None = None


class RunProgressTracker:
    """Translate validated control records into low-frequency UI state."""

    def __init__(self) -> None:
        self._case_id = ""
        self.snapshot = ProgressSnapshot()

    def feed(self, raw_line: str) -> ProgressSnapshot:
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        from acprof.failures import FAILURE_PREFIX, Failure
        if line.startswith(FAILURE_PREFIX):
            import json
            try:
                failure = Failure(**json.loads(line[len(FAILURE_PREFIX):]))
            except (ValueError, TypeError):
                return self.snapshot
            detail = (message_id('run.unsupported_task.log_hint', failure.evidence["task"])
                      if failure.reason_code == "runtime_task_unsupported" and "task" in failure.evidence
                      else f"{failure.reason_code}: {failure.detail}")
            self.snapshot = replace(self.snapshot,
                stage=Stage.TASK_UNSUPPORTED if failure.stage in {"preflight", "dependency_preflight", "precision_preflight"} else Stage.RUNTIME_VALIDATION,
                detail=detail, measurement_active=False,
                errors=max(1, self.snapshot.errors), failure=failure.to_dict())
            return self.snapshot
        preparation = parse_preparation_event(line)
        if preparation is not None:
            if self.snapshot.measurement_active:
                raise ValueError("preparation request inside measurement window")
            stage, status = preparation["stage"], preparation["status"]
            labels = {"resolution": Stage.INTERFACE_RESOLUTION, "dependencies": Stage.DEPENDENCY_RESOLUTION, "preflight": Stage.ENVIRONMENT_PREFLIGHT, "image": Stage.PREPARING_IMAGE,
                      "input": Stage.PREPARING_INPUT, "runtime": Stage.RUNTIME_VALIDATION, "interface": Stage.CONFIRMING_INTERFACE,
                      "environment": Stage.PREPARING_ENVIRONMENT, "model": Stage.PREPARING_MODEL,
                      "compute_profile": Stage.COMPUTE_PROFILING, "execution_profile": Stage.EXECUTION_PROFILING}
            preparation_updates = {"stage": labels[stage], "detail": preparation.get("request", {}).get("detail", "")}
            if status == "failed":
                preparation_updates["errors"] = self.snapshot.errors + 1
            if stage in {"resolution", "interface"}:
                preparation_updates["interface_status"] = status
            elif stage == "runtime":
                preparation_updates["runtime_status"] = status
            self.snapshot = replace(self.snapshot, **preparation_updates)
            return self.snapshot
        event = parse_event(line)
        if event is not None:
            return self._feed_event(event)
        # Human-readable output is diagnostic only. Wording or localization
        # cannot change identity, progress, measurement boundaries or artifacts.
        updates: dict[str, Any] = {}
        if "[WARN]" in line or "[warning]" in line.lower():
            updates["warnings"] = self.snapshot.warnings + 1
        if "[ERROR]" in line or line.startswith("Traceback"):
            updates["errors"] = self.snapshot.errors + 1
        if updates:
            self.snapshot = replace(self.snapshot, **updates)
        return self.snapshot

    def _feed_event(self, event: dict) -> ProgressSnapshot:
        name = event["event"]
        case_id = event.get("case_id", "")
        state = self.snapshot
        updates: dict[str, Any] = {}
        if name == "matrix_planned":
            updates.update(stage=Stage.PREPARING_MATRIX, detail=message_id('run.plan.resources_ready'),
                           total_cases=event["total_cases"])
        elif name == "case_selected":
            if state.measurement_active:
                raise ValueError("case_selected while measurement is still active")
            self._case_id = case_id
            updates.update(stage=Stage.STARTING_CONTAINER,
                detail=message_id('run.monitor.preparing_case', event["current"], event["total"]),
                current_case=event["current"], total_cases=event["total"],
                completed_cases=max(state.completed_cases, event["current"] - 1),
                cpu="inherited" if event["cpu"] is None else str(event["cpu"]),
                mem="inherited" if event["mem"] is None else str(event["mem"]), gpu=event["gpu"])
        elif name == "case_started":
            if state.measurement_active:
                raise ValueError("case_started while measurement is still active")
            self._case_id = case_id
            stage = Stage.MERGING_RESULTS if event.get("phase") == "postprocessing" else Stage.STARTING_CONTAINER
            updates.update(stage=stage, measurement_active=False)
        elif name == "results_merging":
            updates.update(stage=Stage.MERGING_RESULTS, measurement_active=False)
        elif name == "result_published":
            # A path is an association hint. ProcessWorkerActions still verifies
            # persisted run/attempt identity and audits the published result.
            updates["final_csv"] = event["final_csv"]
        elif name == "run_finished":
            status = event["status"]
            updates.update(stage={"ok": Stage.VERIFYING_ARTIFACT, "no_results": Stage.PARTIAL,
                                  "failed": Stage.FAILED, "cancelled": Stage.STOPPED}[status],
                           measurement_active=False)
        elif name == "probe_summary":
            updates["probe_summary"] = event["path"]
        elif name == "probe_started":
            if state.measurement_active:
                raise ValueError("probe_started while measurement is still active")
            self._case_id = case_id
            candidates = event["memory_candidates"]
            updates.update(stage=Stage.PREPARING_MEMORY_PROBE,
                detail=message_id('run.probe.memory_candidates', ",".join(map(str, candidates)), f'{event["input_scale"]:g}'),
                current_case=0, completed_cases=0, total_cases=len(candidates),
                cpu=str(event["cpu"]), mem="-", gpu=event["gpu"], measurement_active=False)
        elif case_id == self._case_id:
            if name == "measurement_started":
                updates.update(stage=Stage.MEASURING, detail=message_id('run.monitor.collecting'),
                               measurement_active=True, measurement_status="running")
            elif name == "measurement_stopped":
                updates.update(stage=Stage.CLEANING_CASE, measurement_active=False)
            elif name in {"case_finished", "case_skipped"}:
                status = event["status"]
                stage = (Stage.OOM_PRUNED if status == "oom_pruned" else
                         Stage.CASE_COMPLETED if status in {"ok", "completed"} else Stage.FAILED)
                updates.update(stage=stage, completed_cases=max(state.completed_cases, state.current_case),
                               measurement_active=False)
                self._case_id = ""
            elif name == "probe_memory_started":
                updates.update(stage=Stage.SCANNING_MEMORY,
                    detail=message_id('run.probe.trying_memory', event["mem"], f'{event["input_scale"]:g}'),
                    current_case=event["current"], total_cases=event["total"],
                    completed_cases=max(state.completed_cases, event["current"] - 1),
                    cpu=str(event["cpu"]), mem=str(event["mem"]), gpu=event["gpu"], measurement_active=False)
            elif name == "probe_request_started":
                updates.update(stage=Stage.MAX_SCALE_PROBE, detail=message_id('run.probe.request_running', state.mem),
                               measurement_active=True)
            elif name == "probe_request_stopped":
                updates.update(measurement_active=False)
            elif name == "probe_memory_finished":
                status = event["status"]
                labels = {"ok": message_id('run.probe.viable_memory'), "startup_oom": message_id('run.probe.startup_oom'),
                          "runtime_oom": message_id('run.probe.inference_oom'), "cuda_oom": message_id('run.probe.cuda_oom'),
                          "timeout": message_id('common.error.request_timeout'), "error": message_id('run.stage.probe_failed')}
                updates.update(stage=Stage.MEMORY_FOUND if status == "ok" else Stage.MEMORY_FEASIBILITY,
                    detail=message("{0}GB · {1}", event["mem"], labels.get(status, status)),
                    completed_cases=max(state.completed_cases, state.current_case), measurement_active=False)
            elif name == "probe_finished":
                status = event["status"]
                detail = (message_id('run.probe.success_summary', event["mem"], f'{event["input_scale"]:g}',
                    _format_probe_duration(event.get("request_s")),
                    _format_probe_duration(event.get("cold_start_s")),
                    _format_probe_duration(event.get("ready_plus_request_s"))) if status == "ok" else
                    message_id('run.probe.no_viable_memory', event["mem"], status))
                completed = max(1, state.current_case)
                updates.update(stage=Stage.PROBE_COMPLETED if status == "ok" else Stage.PROBE_FAILED,
                    detail=detail, current_case=completed, completed_cases=completed, total_cases=completed,
                    cpu=str(event["cpu"]), mem=str(event["mem"]), gpu=event["gpu"], measurement_active=False,
                    probe_result=event)
        if state.measurement_active and name not in {"measurement_stopped", "case_finished", "run_finished",
                                                     "probe_request_stopped", "probe_memory_finished"}:
            if name not in {"measurement_started", "probe_request_started"}:
                raise ValueError("control progress inside measurement window")
        if updates:
            self.snapshot = replace(state, **updates)
        return self.snapshot
