"""Preparation-only workload and cost summaries; no probes or model imports."""
from __future__ import annotations

from dataclasses import dataclass

from acprof.config import SCALING_DIMENSIONS
from acprof.experiments.configuration import ConfigIssue, RunConfig
from acprof.extensions import CATALOG
from acprof.gpu_mode import normalize_gpu_mode
from acprof.messages import join_messages
from acprof.tui.i18n import message_id
from acprof.tui.ui.view_models import CalloutContent, FactRow, StatusSummary, ViewSection


@dataclass(frozen=True)
class RunEstimate:
    cases: int
    scales: int | None
    warmup_windows: int | None
    formal_windows: int | None
    estimated_seconds: float | None


@dataclass(frozen=True, slots=True)
class PlanView:
    summary: StatusSummary
    section: ViewSection
    callout: CalloutContent | None = None


def input_identity(config: RunConfig) -> tuple:
    return tuple(getattr(config, name) for name in (
        "model", "model_source", "revision", "task", "tasks", "task_run_id", "task_family", "backend", "model_spec", "workload_spec", "batch_size",
        "input_scales", "input_scale_policy",
    ))


def input_unit(config: RunConfig, planned: dict | None = None) -> str:
    axis = (planned or {}).get("scale_type", "")
    if not axis and config.task:
        declarations = [item for item in CATALOG.extensions.values() if config.task in item.tasks]
        if declarations and all(item.input_plan.text_payload for item in declarations):
            axis = "seq_length"
        elif declarations and all(item.input_plan.audio_payload for item in declarations):
            axis = "duration_s"
    if not axis:
        family = config.task_family or CATALOG.task_families.get(config.task, "")
        if family in {"nlp", "cv", "timeseries"} and config.task != "table-question-answering":
            axis = SCALING_DIMENSIONS[family].param_name
    units = {"seq_length": "tokens", "duration_s": "s", "resolution_scale": "×224px",
             "resolution_px": "px", "context_length": message_id('common.unit.timesteps'), "table_rows": message_id('common.unit.rows'),
             "denoising_steps": message_id('common.unit.steps'), "frame_count": message_id('common.unit.frames')}
    return units.get(axis, axis or message_id('run.inputs.unit_pending'))


def estimate_run(config: RunConfig, planned: dict | None = None) -> RunEstimate:
    from acprof.application.resources import resource_matrix
    cpus, mems, gpus = resource_matrix(config.to_namespace())
    cases = len(cpus) * len(mems) * len(gpus)
    scales = len(planned["scales"]) if planned and planned.get("scales") else (
        len({float(value) for value in config.input_scales.split(",")}) if config.input_scales else
        1 if config.input_scale_policy == "minimal" else None)
    if scales is None:
        return RunEstimate(cases, None, None, None, None)
    warmup, formal = cases * scales * config.warmup, cases * scales * config.repeat
    # This is a labelled scenario estimate, not a timing measurement or deadline.
    request_seconds = config.repeat_window_seconds if config.repeat_in_window == 0 else config.repeat_in_window
    baseline = config.idle_seconds + config.idle_cooldown_seconds if config.profiling_mode == "full" else 0
    return RunEstimate(cases, scales, warmup, formal, (warmup + formal) * (request_seconds + baseline))


def _duration_text(seconds: float | None) -> str:
    if seconds is None:
        return message_id("common.state.pending")
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:g} s"


def _resource_value(value: str, runtime: str) -> str:
    if value != "auto":
        return value
    return message_id("run.resources.inherited_limits" if runtime == "process"
                      else "run.plan.automatic_resources")


def _gpu_value(value: str) -> str:
    modes = {normalize_gpu_mode(part) for part in value.split(",") if part.strip()}
    if len(modes) > 1:
        return "CPU + GPU"
    return message_id("run.resources.cpu_only" if "off" in modes else "run.resources.gpu_only")


def _plan_rows(config: RunConfig, planned: dict | None, estimate: RunEstimate) -> tuple[FactRow, ...]:
    pending = message_id("common.state.pending")
    runtime = {"docker": "Docker", "process": "Process",
               "auto": message_id("run.runtime.automatic")}.get(config.runtime, config.runtime)
    return (
        FactRow(message_id("run.runtime.label"), runtime),
        FactRow(message_id("run.plan.resources_label"), str(estimate.cases)),
        FactRow(message_id("run.plan.scales_label"),
                message_id("run.plan.scale_count_unit",
                           estimate.scales if estimate.scales is not None else pending,
                           input_unit(config, planned))),
        FactRow(message_id("run.plan.windows_label"),
                message_id("run.plan.window_totals",
                           estimate.warmup_windows if estimate.warmup_windows is not None else pending,
                           estimate.formal_windows if estimate.formal_windows is not None else pending)),
        FactRow(message_id("run.plan.per_condition_label"),
                message_id("run.plan.per_condition_value", config.repeat),
                status="limited" if config.repeat < 3 else None),
    )


def _plan_details(
    config: RunConfig, planned: dict | None, estimate: RunEstimate, *,
    download_budget: str, model_store: str,
) -> tuple[FactRow, ...]:
    scales = (", ".join(f"{value:g}" for value in planned["scales"])
              if planned and planned.get("scales") else config.input_scales)
    assumption = (message_id("run.plan.assumed_request_duration") if config.repeat_in_window
                  else message_id("run.plan.assumed_window_duration", config.repeat_window_seconds))
    profilers = (message_id("profiling.options.off_summary")
                 if config.compute_profile_tool == config.execution_profile_tool == "none"
                 else message_id("profiling.options.selected_summary",
                                 config.compute_profile_tool, config.execution_profile_tool))
    cache = model_store or config.model_store
    source = {"huggingface": "Hugging Face", "modelscope": "ModelScope"}.get(
        config.model_source, config.model_source,
    )
    details: tuple[FactRow, ...] = (
        FactRow(message_id("run.plan.evidence_label"),
                message_id("run.plan.insufficient_windows" if config.repeat < 3
                           else "run.plan.valid_counts_note")),
        FactRow(message_id("run.plan.scale_values_label"),
                scales or message_id("common.state.pending"), detail=scales, copyable=bool(scales)),
        FactRow(message_id("run.resources.cpu_list"), _resource_value(config.cpus, config.runtime),
                detail=config.cpus),
        FactRow(message_id("run.resources.memory_gb"), _resource_value(config.mems, config.runtime),
                detail=config.mems),
        FactRow(message_id("run.resources.gpu_mode"), _gpu_value(config.gpus), detail=config.gpus),
        FactRow(message_id("run.windows.request_timeout_seconds"), f"{config.request_timeout_seconds:g}"),
        FactRow(message_id("run.plan.profilers_label"), profilers),
        FactRow(message_id("run.plan.duration_label"),
                join_messages("\n", (
                    message_id("run.plan.duration_estimate", _duration_text(estimate.estimated_seconds), assumption),
                    message_id("run.plan.duration_exclusions"),
                ))),
        FactRow(message_id("downloads.budget.label"),
                download_budget or config.max_download or message_id("common.limit.unlimited")),
        FactRow(message_id("downloads.size.expected"), message_id("downloads.plan.pending_estimate")),
        FactRow(message_id("run.plan.strategy_label"), message_id("downloads.strategy.automatic_summary")),
        FactRow(message_id("models.source.label"), source, detail=config.model_source),
        FactRow(message_id("downloads.model_store.path_label"),
                cache or message_id("common.state.pending"), detail=cache, copyable=bool(cache)),
        FactRow(message_id("settings.experiment.output_directory"), config.output_dir,
                detail=config.output_dir, copyable=True),
    )
    if not config.input_scales:
        details += (FactRow(message_id("run.inputs.planning_mode"),
                            message_id("run.inputs.smallest_scale" if config.input_scale_policy == "minimal"
                                       else "run.inputs.automatic_range"),
                            detail=config.input_scale_policy),)
    if config.runtime == "process":
        details += (FactRow(message_id("run.plan.resource_rules_label"),
                            message_id("run.runtime.process_limits_note")),)
    if config.resume:
        details += (FactRow(message_id("run.plan.resume_label"),
                            message_id("run.resume.original_parameters")),)
    return details


def run_plan_view(
    config: RunConfig,
    planned: dict | None = None,
    *,
    download_budget: str = "",
    model_store: str = "",
) -> PlanView:
    """Project a validated configuration; callers resolve runtime and environment defaults."""
    estimate = estimate_run(config, planned)
    if config.repeat < 3:
        summary = StatusSummary("warning", message_id("run.plan.flow_check_limited"))
    else:
        summary = StatusSummary("neutral", message_id(
            "run.plan.pending_inputs" if estimate.scales is None else "run.plan.calculated",
        ))
    return PlanView(
        summary,
        ViewSection(
            message_id("run.plan.title"),
            _plan_rows(config, planned, estimate),
            detail_rows=_plan_details(config, planned, estimate,
                                      download_budget=download_budget, model_store=model_store),
        ),
    )


def invalid_plan_view(issues: tuple[ConfigIssue, ...]) -> PlanView:
    """Replace stale facts with validation feedback without interpreting the raw reasons."""
    detail = "\n".join(f"[{issue.field}] {issue.reason}" if issue.field else str(issue.reason)
                       for issue in issues)
    return PlanView(
        StatusSummary("error", message_id("run.plan.invalid")),
        ViewSection(message_id("run.plan.title"), ()),
        CalloutContent(
            "error",
            issues[0].reason if issues else message_id("run.plan.invalid"),
            message_id("run.plan.invalid_impact"),
            message_id("run.plan.invalid_next"),
            detail=detail,
        ),
    )
