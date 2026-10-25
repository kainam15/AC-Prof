"""Preparation-only workload and cost summaries; no probes or model imports."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from acprof.config import SCALING_DIMENSIONS
from acprof.experiment import RunConfig
from acprof.extensions import CATALOG
from acprof.messages import join_messages, message
from acprof.tui.i18n import message_id


@dataclass(frozen=True)
class RunEstimate:
    cases: int
    scales: int | None
    warmup_windows: int | None
    formal_windows: int | None
    estimated_seconds: float | None


def input_identity(config: RunConfig) -> tuple:
    return tuple(getattr(config, name) for name in (
        "model", "model_source", "revision", "task", "task_family", "backend", "model_spec", "workload_spec", "batch_size",
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
             "resolution_px": "px", "context_length": message_id('ui.0631'), "table_rows": message_id('ui.0632'),
             "denoising_steps": message_id('ui.0633'), "frame_count": message_id('ui.0634')}
    return units.get(axis, axis or message_id('ui.0629'))


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


def plan_summary(config: RunConfig, planned: dict | None = None) -> str:
    estimate = estimate_run(config, planned)
    unknown = message_id('ui.0630')
    counts = message_id('ui.0635',
        estimate.cases, estimate.scales if estimate.scales is not None else unknown, input_unit(config, planned),
        estimate.warmup_windows if estimate.warmup_windows is not None else unknown,
        estimate.formal_windows if estimate.formal_windows is not None else unknown)
    evidence = message("不足 3 个：仅验证流程，无置信区间" if config.repeat < 3 else "有效数以结果为准")
    per_condition = message_id('ui.0636', config.repeat, evidence)
    duration: str
    if estimate.estimated_seconds is None:
        duration = unknown
    elif estimate.estimated_seconds >= 3600:
        duration = f"{estimate.estimated_seconds / 3600:.1f} h"
    elif estimate.estimated_seconds >= 60:
        duration = f"{estimate.estimated_seconds / 60:.1f} min"
    else:
        duration = f"{estimate.estimated_seconds:g} s"
    assumption = message_id('ui.0639') if config.repeat_in_window else message_id('ui.0640', config.repeat_window_seconds)
    budget = config.max_download or os.environ.get("ACPROF_MAX_DOWNLOAD", "") or message_id('ui.0642')
    return join_messages("\n", (message_id('ui.2007', config.runtime), counts, per_condition,
                               message_id('ui.0641', duration, assumption),
                               message_id('ui.0643', budget)))


def preparation_details(config: RunConfig, planned: dict | None = None) -> str:
    cache = config.model_store or os.environ.get("ACPROF_MODEL_STORE", "") or str(Path.home() / ".cache/acprof/model-store")
    scales = ", ".join(f"{value:g}" for value in planned["scales"]) if planned else config.input_scales or message_id('ui.0630')
    return join_messages("\n", (plan_summary(config, planned),
        message_id('ui.0648', scales, input_unit(config, planned)),
        message_id('ui.0644'),
        message_id('ui.0007', cache, config.model_source),
        message_id('ui.0004'),
        message_id('ui.0646'),
        message_id('ui.0647') if config.resume else ""))
