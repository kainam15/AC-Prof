"""Lightweight experiment identity and artifacts; no measurement-row aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from acprof.experiments.arguments import flatten_run_options
from acprof.messages import join_messages
from acprof.results.layout import ArtifactLayout, resolve_result_file, resolve_single_result_source
from acprof.tui.experiments.catalog import ExperimentRecord, read_experiment_record
from acprof.tui.i18n import message_id


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


def overview_text(overview: ExperimentOverview) -> str:
    record = overview.record
    state, meta = record.state, record.metadata
    options = flatten_run_options(record.options)
    runtime = state.get("runtime") or {}
    task = runtime.get("task") or {}
    unknown = message_id("results.overview.unknown")

    def value(item):
        return item if item is not None and item not in ("", "unknown") else unknown

    attempts = state.get("attempts") or []
    started = next((item.get("started_at") for item in attempts if item.get("started_at")), None)
    ended = state.get("completed_at") or (attempts[-1].get("ended_at") if attempts else None)
    cases = state.get("cases")
    progress = unknown
    if isinstance(cases, dict) and cases:
        progress = message_id("results.overview.case_progress", sum(
            item.get("status") == "complete" for item in cases.values() if isinstance(item, dict)), len(cases))
    status = record.status
    status_key = {"complete": "complete", "running": "running", "preparing": "running",
                  "failed": "failed", "error": "failed", "interrupted": "interrupted"}.get(status, "unknown")
    environment = meta.get("measurement_environment") or meta.get("environment_class")
    if isinstance(environment, dict):
        environment = environment.get("environment_class") or environment.get("label")
    memory = options.get("mems")
    memory = f"{memory} GiB" if memory and str(memory) not in {"auto", "inherited"} else value(memory)
    lines = [
        message_id("results.overview.identity", value(record.model_id), value(record.revision),
                   value(meta.get("pipeline_tag") or task.get("pipeline_tag") or options.get("task"))),
        message_id("results.overview.run", message_id("results.overview." + status_key), progress, value(record.run_id)),
        message_id("results.overview.runtime", value(meta.get("runtime_kind") or runtime.get("kind")),
                   value(environment), value(record.runtime_profile), value(meta.get("profiling_mode") or options.get("profiling_mode"))),
        message_id("results.overview.resources", value(options.get("cpus")), memory, value(options.get("gpus"))),
        message_id("results.overview.times", value(record.created_at), value(started), value(ended)),
        message_id("results.overview.artifacts", message_id("results.overview." + overview.data_state),
                   message_id("results.overview." + overview.metadata_state)),
    ]
    reasons = [*record.issues, *(": ".join(str(item[key]) for key in ("reason_code", "detail") if item.get(key))
                                or str(item) for item in record.failures)]
    if reasons:
        lines.append(message_id("results.overview.warnings", "; ".join(reasons[:3])))
    return join_messages("\n", lines)
