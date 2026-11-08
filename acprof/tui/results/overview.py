"""Lightweight experiment identity and artifacts; no measurement-row aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from acprof.experiments.arguments import flatten_run_options
from acprof.messages import join_messages
from acprof.results.layout import ArtifactLayout, resolve_result_file, resolve_single_result_source
from acprof.tui.experiments.catalog import ExperimentRecord, read_experiment_record
from acprof.tui.i18n import message_id
from acprof.tui.results.data_status import ExperimentDataStatus
from acprof.tui.ui.view_models import (
    CalloutContent,
    FactRow,
    StatusSummary,
    ViewAction,
    ViewSection,
    format_local_timestamp,
    format_value,
    reason_presentation,
    status_presentation,
)


@dataclass(frozen=True)
class ExperimentOverview:
    source: Path
    result_file: Path
    record: ExperimentRecord
    data_state: str
    metadata_state: str


def selection_path(source: Path) -> Path:
    """Migrate historical picker file paths, preserving standalone CSV inputs."""
    if source.name in {"result_layers.json", "result_all.csv"} and source.parent.is_dir():
        return source.parent
    return source


def read_experiment_overview(source: Path) -> ExperimentOverview:
    path = resolve_single_result_source(source)
    if not path.exists():
        raise FileNotFoundError(message_id("results.summary.path_missing"))
    if not path.is_dir() and path.name != "result_layers.json" and path.suffix.lower() != ".csv":
        raise ValueError(message_id("results.summary.unsupported_source"))
    result = resolve_result_file(path)
    root = path if path.is_dir() else path.parent
    record = read_experiment_record(root)
    layout = ArtifactLayout.discover(root)
    data = "exists" if result.is_file() and result.stat().st_size else "missing"
    meta = "damaged" if record.issues else "exists" if layout.path("static_meta.json").is_file() else "missing"
    return ExperimentOverview(source, result, record, data, meta)


@dataclass(frozen=True)
class OverviewView:
    summary: StatusSummary
    sections: tuple[ViewSection, ...]
    completed_cases: int
    total_cases: int | None
    callout: CalloutContent | None = None


def _case_progress(record: ExperimentRecord) -> tuple[int, int | None]:
    cases = record.state.get("cases")
    if not isinstance(cases, dict) or not cases:
        return 0, None
    return sum(isinstance(case, dict) and case.get("status") == "complete"
               for case in cases.values()), len(cases)


def _raw_reasons(record: ExperimentRecord, data: ExperimentDataStatus | None) -> str:
    reasons = [*record.issues]
    for failure in record.failures:
        reasons.append("\n".join(str(failure[key]) for key in ("reason_code", "detail")
                                 if failure.get(key) is not None))
    if data is not None:
        reasons.extend(str(issue.get("message", issue)) for issue in data.audit.get("issues", ()))
        if data.profiler_error:
            reasons.append(data.profiler_error)
    return "\n\n".join(reason for reason in reasons if reason)


def _overview_callout(overview: ExperimentOverview, data: ExperimentDataStatus | None,
                      data_error: str) -> CalloutContent | None:
    record = overview.record
    detail = _raw_reasons(record, data)
    if data_error:
        return CalloutContent("error", message_id("results.view.read_problem"),
            message_id("results.view.read_impact"), message_id("results.view.read_next"),
            "\n\n".join(item for item in (data_error, detail) if item))
    if record.issues:
        return CalloutContent("warning", message_id("results.view.metadata_problem"),
            message_id("results.view.metadata_impact"), message_id("results.view.read_next"), detail)
    if record.failures:
        first = record.failures[0]
        reason = reason_presentation(str(first.get("reason_code", "")),
                                     str(first.get("detail", "")))
        return CalloutContent(reason.level, reason.problem, reason.impact, reason.next_step, detail)
    if data is not None and data.formal_ok_rows == 0:
        return CalloutContent("warning", message_id("results.view.no_windows"),
            message_id("results.view.no_windows_impact"),
            message_id("results.view.resume_next" if record.has_recovery_state
                       else "results.view.new_run_next"), detail)
    if data is not None and not data.audit.get("valid", False):
        return CalloutContent("warning", message_id("results.view.audit_problem"),
            message_id("results.view.audit_impact"), message_id("results.view.read_next"), detail)
    return None


def _identity_section(record: ExperimentRecord, options: dict, task: dict) -> ViewSection:
    return ViewSection(message_id("results.view.identity"), (
        FactRow(message_id("results.view.model"), format_value(record.model_id), copyable=True),
        FactRow(message_id("results.view.task"),
                format_value(record.metadata.get("pipeline_tag") or task.get("pipeline_tag")
                             or options.get("task"))),
    ), (
        FactRow(message_id("results.view.revision"), format_value(record.revision), copyable=True),
    ))


def _run_section(record: ExperimentRecord, options: dict, runtime: dict) -> ViewSection:
    state, meta = record.state, record.metadata
    recorded_attempts = state.get("attempts")
    attempts = ([item for item in recorded_attempts if isinstance(item, dict)]
                if isinstance(recorded_attempts, (list, tuple)) else [])
    started = next((item.get("started_at") for item in attempts if item.get("started_at")), None)
    ended = state.get("completed_at") or (attempts[-1].get("ended_at") if attempts else None)
    environment = meta.get("measurement_environment") or meta.get("environment_class")
    if isinstance(environment, dict):
        environment = environment.get("environment_class") or environment.get("label")
    details = [FactRow(message_id("results.view.run_id"), format_value(record.run_id), copyable=True)]
    for key, value in (("created", record.created_at), ("started", started), ("ended", ended)):
        details.append(FactRow(message_id("results.view." + key), format_local_timestamp(value),
                               detail=str(value) if value not in (None, "", "unknown") else ""))
    return ViewSection(message_id("results.view.run"), (
        FactRow(message_id("results.view.runtime"),
                format_value(meta.get("runtime_kind") or runtime.get("kind"))),
        FactRow(message_id("results.view.environment"), format_value(environment)),
        FactRow(message_id("results.view.profile"), format_value(record.runtime_profile)),
        FactRow(message_id("results.view.mode"),
                format_value(meta.get("profiling_mode") or options.get("profiling_mode"))),
    ), tuple(details))


def _resources_section(options: dict) -> ViewSection:
    memory = options.get("mems")
    memory_text = (f"{memory} GiB" if memory not in (None, "", "unknown", "auto", "inherited")
                   else format_value(memory))
    return ViewSection(message_id("results.view.resources"), (
        FactRow("CPU", format_value(options.get("cpus"))),
        FactRow(message_id("results.view.memory"), memory_text),
        FactRow("GPU", format_value(options.get("gpus"))),
    ))


def _artifacts_section(overview: ExperimentOverview, data: ExperimentDataStatus | None,
                       data_unverified: bool) -> ViewSection:
    windows = (message_id("results.view.unverified") if data_unverified else
               message_id("results.view.checking") if data is None else str(data.formal_ok_rows))
    return ViewSection(message_id("results.view.artifacts"), (
        FactRow(message_id("results.view.result_files"), message_id(
            "results.view.files_present" if overview.data_state == "exists"
            else "results.view.files_missing")),
        FactRow(message_id("results.view.windows"), windows),
        FactRow(message_id("results.view.metadata"),
                message_id("results.overview." + overview.metadata_state)),
    ), (
        FactRow(message_id("results.view.directory"), str(overview.record.directory), copyable=True),
        FactRow(message_id("results.view.result_path"), str(overview.result_file), copyable=True),
    ))


def overview_view(overview: ExperimentOverview, data: ExperimentDataStatus | None = None, *,
                  data_error: str = "", data_cancelled: bool = False) -> OverviewView:
    """Project existing metadata/check results; never scan files or alter result semantics."""
    record = overview.record
    options = flatten_run_options(record.options)
    runtime = record.state.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    task = runtime.get("task")
    task = task if isinstance(task, dict) else {}
    completed, total = _case_progress(record)
    status = status_presentation(record.status)
    progress = message_id("results.view.case_count", completed, total) if total else message_id("results.view.cases_unknown")
    data_unverified = bool(data_error) or (data is None and data_cancelled)
    windows = (message_id("results.view.unverified") if data_unverified else
               message_id("results.view.checking") if data is None else
               message_id("results.view.no_windows") if data.formal_ok_rows == 0 else
               message_id("results.view.window_count", data.formal_ok_rows))
    callout = _overview_callout(overview, data, data_error)
    level = status.level
    if data_error:
        level = "error"
    elif callout is not None and level in {"success", "neutral", "info"}:
        level = callout.level
    recoverable = record.has_recovery_state and record.status not in {"running", "preparing"}
    actions = []
    if recoverable:
        actions.append(ViewAction("overview-resume", message_id("results.view.resume"), primary=True))
    elif data is not None and data.formal_ok_rows:
        actions.append(ViewAction("overview-statistics", message_id("results.view.statistics"), primary=True))
    if callout is not None:
        actions.append(ViewAction("overview-reason", message_id("results.view.reason")))
    summary = StatusSummary(level, join_messages(" · ", (status.label, progress, windows)), tuple(actions))
    return OverviewView(summary, (
        _identity_section(record, options, task),
        _run_section(record, options, runtime),
        _resources_section(options),
        _artifacts_section(overview, data, data_unverified),
    ), completed, total, callout)
