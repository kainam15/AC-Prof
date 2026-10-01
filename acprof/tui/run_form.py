"""RunConfig form conversion and preset matching, independent of Textual widgets."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from acprof.experiment import RunConfig
from acprof.tui.presentation import format_input_number

INPUT_FIELDS = {
    "model": "model",
    "task": "task",
    "backend": "backend",
    "cpus": "cpus",
    "cpuset-cpus": "cpuset_cpus",
    "mems": "mems",
    "input-scales": "input_scales",
    "workload-spec": "workload_spec",
    "model-spec": "model_spec",
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
    "task-family": "task_family",
    "gpus": "gpus",
    "compute-profile-tool": "compute_profile_tool",
    "profiling-mode": "profiling_mode",
    "execution-profile-tool": "execution_profile_tool",
    "notify": "notify",
}
CHECKED_FIELDS = {
    "prune-startup-oom": "prune_startup_oom",
    "skip-build": "skip_build",
    "resume-run": "resume",
    "idle-debug": "idle_debug",
}


def collect_config(inputs: Mapping[str, str], selects: Mapping[str, str],
                   checks: Mapping[str, bool], *, project_dir: Path,
                   allow_empty_model: bool = False) -> RunConfig:
    values = {**{field: inputs[key] for key, field in INPUT_FIELDS.items()},
              **{field: selects[key] for key, field in SELECT_FIELDS.items()},
              **{field: checks[key] for key, field in CHECKED_FIELDS.items()}}
    config = RunConfig(**values)
    if allow_empty_model and not config.model:
        validated = replace(config, model="settings/default-model").validate(project_dir=project_dir)
        return replace(validated, model="")
    return config.validate(project_dir=project_dir)


def config_values(config: RunConfig) -> tuple[dict[str, str], dict[str, str], dict[str, bool]]:
    inputs = {key: format_input_number(value) if isinstance(value, (int, float)) else value
              for key, field in INPUT_FIELDS.items() for value in [getattr(config, field)]}
    selects = {key: getattr(config, field) for key, field in SELECT_FIELDS.items()}
    checks = {key: getattr(config, field) for key, field in CHECKED_FIELDS.items()}
    return inputs, selects, checks


def infer_preset(config: RunConfig) -> str:
    model = config.model
    if config == RunConfig.smoke(model):
        return "smoke"
    if config == RunConfig.main_matrix(model):
        return "main"
    if config == RunConfig(model=model):
        return "default"
    return "custom"


def matches_preset(config: RunConfig, preset: str) -> bool:
    if preset == "smoke":
        return config == RunConfig.smoke(config.model)
    if preset == "main":
        return config == RunConfig.main_matrix(config.model)
    if preset == "default":
        return config == RunConfig(model=config.model)
    return preset == "custom"
