"""Preparation-only workload and cost summaries; no probes or model imports."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from acprof.config import SCALING_DIMENSIONS
from acprof.experiment import RunConfig
from acprof.extensions import CATALOG
from acprof.messages import join_messages, message


@dataclass(frozen=True)
class RunEstimate:
    cases: int
    scales: int | None
    warmup_windows: int | None
    formal_windows: int | None
    estimated_seconds: float | None


def input_identity(config: RunConfig) -> tuple:
    return tuple(getattr(config, name) for name in (
        "model", "task", "task_family", "backend", "model_spec", "workload_spec", "batch_size",
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
             "resolution_px": "px", "context_length": message("时间步"), "table_rows": message("行"),
             "denoising_steps": message("步"), "frame_count": message("帧")}
    return units.get(axis, axis or message("单位待解析"))


def estimate_run(config: RunConfig, planned: dict | None = None) -> RunEstimate:
    cases = len(config.cpus.split(",")) * len(config.mems.split(",")) * len(config.gpus.split(","))
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
    unknown = message("待解析")
    counts = message("{0} 配置 · 输入 {1} 档 ({2}) · 预热 {3} / 正式 {4} 窗口",
        estimate.cases, estimate.scales if estimate.scales is not None else unknown, input_unit(config, planned),
        estimate.warmup_windows if estimate.warmup_windows is not None else unknown,
        estimate.formal_windows if estimate.formal_windows is not None else unknown)
    if estimate.estimated_seconds is None:
        duration = unknown
    elif estimate.estimated_seconds >= 3600:
        duration = f"{estimate.estimated_seconds / 3600:.1f} h"
    elif estimate.estimated_seconds >= 60:
        duration = f"{estimate.estimated_seconds / 60:.1f} min"
    else:
        duration = f"{estimate.estimated_seconds:g} s"
    assumption = message("假设每请求 1s") if config.repeat_in_window else message("按自动窗口 {0:g}s 估算", config.repeat_window_seconds)
    budget = config.max_download or os.environ.get("ACPROF_MAX_DOWNLOAD", "") or message("不设上限")
    return join_messages("\n", (counts, message("时间约 {0} · {1}，不含准备、校准和 profiler", duration, assumption),
                               message("下载预算：{0}", budget)))


def preparation_details(config: RunConfig, planned: dict | None = None) -> str:
    cache = config.model_store or os.environ.get("ACPROF_MODEL_STORE", "") or str(Path.home() / ".cache/acprof/model-store")
    scales = ", ".join(f"{value:g}" for value in planned["scales"]) if planned else config.input_scales or message("待解析")
    return join_messages("\n", (plan_summary(config, planned),
        message("输入：{0} {1}", scales, input_unit(config, planned)),
        message("预计下载：待解析；权重传输前显示实际计划与有效预算。"),
        message("缓存：{0} · 下载源：{1}", cache, config.download_mode),
        message("时间为假设估计；不包含下载、构建、服务启动和额外验证。"),
        message("续跑只接受原实验参数；更改参数需要新的输出目录。") if config.resume else ""))
