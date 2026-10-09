"""RunConfig form conversion and preset matching, independent of Textual widgets."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from acprof.experiment import PRESET_FIELDS, RunConfig
from acprof.tui.presentation import format_input_number

INPUT_FIELDS = {
    "model": "model",
    "revision": "revision",
    "task": "task",
    "backend": "backend",
    "cpus": "cpus",
    "cpuset-cpus": "cpuset_cpus",
    "mems": "mems",
    "input-scales": "input_scales",
    "workload-spec": "workload_spec",
    "model-spec": "model_spec",
    "max-download": "max_download",
    "model-store": "model_store",
    "model-store-max": "model_store_max",
    "output-dir": "output_dir",
    "batch-size": "batch_size",
    "warmup": "warmup",
    "repeat": "repeat",
    "repeat-in-window": "repeat_in_window",
    "repeat-window-seconds": "repeat_window_seconds",
    "request-timeout-seconds": "request_timeout_seconds",
    "sample-hz": "sample_hz",
    "idle-seconds": "idle_seconds",
    "idle-cooldown-seconds": "idle_cooldown_seconds",
    "sniff-iface": "sniff_iface",
}
SELECT_FIELDS = {
    "input-scale-policy": "input_scale_policy",
    "model-source": "model_source",
    "task-family": "task_family",
    "gpus": "gpus",
    "profiling-mode": "profiling_mode",
    "notify": "notify",
}
CHECKED_FIELDS = {
    "prune-startup-oom": "prune_startup_oom",
    "skip-build": "skip_build",
    "resume-run": "resume",
    "idle-debug": "idle_debug",
}

PROFILER_CHECKBOXES = ("torch-profiler", "ncu-profiler", "massif-profiler", "nsys-profiler")


def profiler_tools_from_checks(checks: Mapping[str, bool]) -> tuple[str, str]:
    """Convert four independent TUI choices to the persisted CLI tool modes."""
    def mode(first: str, second: str, first_mode: str, second_mode: str) -> str:
        selected_first, selected_second = checks[first], checks[second]
        if selected_first and selected_second:
            return "both"
        if selected_first:
            return first_mode
        if selected_second:
            return second_mode
        return "none"

    return (
        mode("torch-profiler", "ncu-profiler", "torch", "ncu"),
        mode("massif-profiler", "nsys-profiler", "massif", "nsys"),
    )


def collect_config(inputs: Mapping[str, str], selects: Mapping[str, str],
                   checks: Mapping[str, bool], *, project_dir: Path,
                   allow_empty_model: bool = False, extra_options: dict | None = None) -> RunConfig:
    compute_tool, execution_tool = profiler_tools_from_checks(checks)
    values = {**{field: inputs[key] for key, field in INPUT_FIELDS.items()},
              **{field: selects[key] for key, field in SELECT_FIELDS.items()},
              **{field: checks[key] for key, field in CHECKED_FIELDS.items()},
              "compute_profile_tool": compute_tool,
              "execution_profile_tool": execution_tool}
    config = RunConfig(**values, extra_options=dict(extra_options or {}))
    if allow_empty_model and not config.model:
        validated = replace(config, model="settings/default-model").validate(project_dir=project_dir)
        return replace(validated, model="")
    return config.validate(project_dir=project_dir)


def config_values(config: RunConfig) -> tuple[dict[str, str], dict[str, str], dict[str, bool]]:
    inputs = {key: format_input_number(value) if isinstance(value, (int, float)) else value
              for key, field in INPUT_FIELDS.items() for value in [getattr(config, field)]}
    selects = {key: getattr(config, field) for key, field in SELECT_FIELDS.items()}
    checks = {key: getattr(config, field) for key, field in CHECKED_FIELDS.items()}
    checks.update({
        "torch-profiler": config.compute_profile_tool in {"torch", "both"},
        "ncu-profiler": config.compute_profile_tool in {"ncu", "both"},
        "massif-profiler": config.execution_profile_tool in {"massif", "both"},
        "nsys-profiler": config.execution_profile_tool in {"nsys", "both"},
    })
    return inputs, selects, checks


PRESET_OPTIONS = (("自定义", "custom"), ("基础 CPU Smoke", "smoke"),
                  ("主矩阵（分析器关闭）", "main"))


def infer_preset(config: RunConfig) -> str:
    if matches_preset(config, "smoke"):
        return "smoke"
    if matches_preset(config, "main"):
        return "main"
    return "custom"


def matches_preset(config: RunConfig, preset: str) -> bool:
    if config.extra_options:
        return False
    if preset == "smoke":
        template = RunConfig.smoke(config.model)
    elif preset == "main":
        template = RunConfig.main_matrix(config.model)
    else:
        return preset == "custom"
    return all(getattr(config, name) == getattr(template, name) for name in PRESET_FIELDS)
