"""Typed options shared by the CLI, saved configuration and command builders.

Declare a form option once here. Its default and CLI metadata drive both parsers
and command serialization; presentation and validation remain separate.
"""
from __future__ import annotations

import argparse
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Literal, get_type_hints

from acprof.config import (
    DEFAULT_COMPUTE_PROFILE_TOOL,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from acprof.hf_endpoints import HF_DOWNLOAD_MODES


@dataclass(frozen=True)
class CliOption:
    help: str
    probe: bool = False
    probe_help: str = ""
    group: str = "run"
    choices: tuple[str, ...] = ()
    required: bool = False
    negative: str = ""
    negative_help: str = ""
    cli_default: Any = field(default_factory=lambda: MISSING)


def option(default: Any, **metadata: Any) -> Any:
    """Attach transport metadata without changing the declared Python type."""
    return field(default=default, metadata={"cli": CliOption(**metadata)})


@dataclass(frozen=True)
class RunOptions:
    """Explicitly typed configuration fields shared by CLI and presentation."""

    model: str = option(
        '', help='HuggingFace model ID', probe=True, required=True, cli_default=None,
    )
    model_source: str = option(
        'huggingface', help='Model artifact source (default huggingface); never switched automatically', probe=True, group="download", choices=('huggingface', 'modelscope'), cli_default=None,
    )
    revision: str = option(
        '', help='Model branch, tag or full commit SHA', probe=True, cli_default=None,
    )
    task: str = option(
        '', help='Override pipeline_tag (e.g., text-generation)', probe=True, cli_default=None,
    )
    task_family: str = option(
        '', help='Override task family (nlp/cv/audio/timeseries/diffusion/multimodal/structured)', probe=True, cli_default=None,
    )
    backend: str = option(
        '', help='Override runtime backend (transformers_pipeline/chronos/...)', probe=True, cli_default=None,
    )
    cpus: str = option(
        '1,2,4,8', help='CPU core counts (comma-separated)', probe=True,
    )
    cpuset_cpus: str = option(
        '', help='Optional fixed CPU IDs for formal/startup containers, e.g. 0-3,8',
    )
    mems: str = option(
        '2,4,8,16', help='Memory caps in GB (comma-separated)', probe=True,
        probe_help='Memory GB candidates scanned from low to high',
    )
    gpus: str = option(
        'off,on', help='GPU modes (comma-separated: off,on)', probe=True,
    )
    input_scales: str = option(
        '', help='Override input scale values (comma-separated)', probe=True, cli_default=None,
        probe_help='Optional input scales; the largest value is probed',
    )
    input_scale_policy: str = option(
        'auto', help='Without explicit scales, plan a full range or one smallest declared workload scale', choices=('auto', 'minimal'),
    )
    workload_spec: str = option(
        '', help='Path to an NLP, CV, audio, multimodal, diffusion or structured workload manifest. ASR defaults to the bundled LibriSpeech short-form manifest.', probe=True, cli_default=None,
    )
    model_spec: str = option(
        '', help='Local acprof_model.json interface declaration; baked into the immutable service image', probe=True, cli_default=None,
    )
    download_mode: str = option(
        'auto', help='Advanced/compatibility HF Hub entry policy (default auto); TUI always uses auto', probe=True, group="download", choices=HF_DOWNLOAD_MODES, cli_default=None,
    )
    max_download: str = option(
        '', help='Bulk download budget, e.g. 5GB; unknown size stops before download', probe=True, group="download", cli_default=None,
    )
    model_store: str = option(
        '', help='Host Model Store path shared across CPU/GPU runs', probe=True, group="download", cli_default=None,
    )
    model_store_max: str = option(
        '', help='Model Store capacity, e.g. 100GB', probe=True, group="download", cli_default=None,
    )
    output_dir: str = option(
        'results', help='Output directory', probe=True,
    )
    batch_size: int = option(
        1, help='Batch size', probe=True,
    )
    warmup: int = option(
        2, help='Warmup iterations',
    )
    repeat: int = option(
        5, help='Measurement repeat count',
    )
    repeat_in_window: int = option(
        DEFAULT_REPEAT_IN_WINDOW, help='Requests per energy window; 0 enables auto calibration',
    )
    repeat_window_seconds: float = option(
        DEFAULT_REPEAT_WINDOW_SECONDS, help='Target workload window duration for auto repeat-in-window',
    )
    request_timeout_seconds: float = option(
        DEFAULT_REQUEST_TIMEOUT_SECONDS, help='Connect and read-inactivity timeout per /predict (not a total deadline; no retries)',
    )
    sample_hz: float = option(
        20.0, help='GPU energy sampling rate',
    )
    sampling_policy: str = option(
        'burst', help='Late sampling policy: preserve absolute-phase catch-up or skip missed ticks', choices=('burst', 'skip'),
    )
    raw_sample_limit: int = option(
        100000, help='Maximum retained raw samples per collector (integer of at least 2)',
    )
    idle_seconds: float = option(
        DEFAULT_IDLE_SECONDS, help='Idle baseline measurement duration before each workload window',
    )
    idle_cooldown_seconds: float = option(
        DEFAULT_IDLE_COOLDOWN_SECONDS, help='Cooldown duration before collecting idle baselines for each workload window',
    )
    compute_profile_tool: str = option(
        DEFAULT_COMPUTE_PROFILE_TOOL, help='Compute FLOP profiler (default: none): none skips all compute probes; both independently collects torch_profiler_eager logical FLOP and ncu GPU executed FLOP', choices=('none', 'both', 'torch', 'ncu', 'vendor'),
    )
    profiling_mode: str = option(
        'full', help='full requires packet latency, RAPL and perf (default); basic measures application latency, throughput, CPU and memory', choices=('full', 'basic'),
    )
    execution_profile_tool: str = option(
        'none', help='Optional execution profiler: Massif for CPU heap peaks, Nsight Systems for CUDA/GPU timelines, both, or none (default)', choices=('none', 'both', 'massif', 'nsys'),
    )
    sniff_iface: str = option(
        'docker0', help='Network interface for tcpdump',
    )
    notify: str = option(
        'auto', help='Notification mode: auto (default) enables WeCom when ACPROF_WECOM_WEBHOOK_URL is configured; none disables notifications', choices=('auto', 'none', 'wecom'),
    )
    prune_startup_oom: bool = option(
        True, help='Probe startup-only OOM evidence before freezing the formal matrix (default)', negative="--no-prune-startup-oom", negative_help='Disable startup-OOM pruning and independently attempt every selected CPU/memory/GPU case',
    )
    rebuild_image: bool = option(
        False, help='Rebuild the model image (Docker cache allowed); by default reuse a verified image or build if missing', probe=True,
    )
    resume: bool = option(
        False, help='Resume the same experiment using its recorded image, input plan and completed cases',
    )
    idle_debug: bool = option(
        False, help='Write CPU idle baseline timestamps and per-row diagnostic JSONL sidecars',
    )


def add_option_arguments(
    parser: argparse.ArgumentParser,
    *,
    group: str | None = "run",
    purpose: Literal["run", "probe"] = "run",
    default_notify_provider: str = "auto",
    automatic: bool = False,
) -> None:
    """Register the selected schema group, preserving public CLI defaults."""
    hints = get_type_hints(RunOptions)
    for item in fields(RunOptions):
        spec: CliOption = item.metadata["cli"]
        if (group is not None and spec.group != group) or (purpose == "probe" and not spec.probe):
            continue
        help_text = spec.probe_help if purpose == "probe" and spec.probe_help else spec.help
        flag = "--" + item.name.replace("_", "-")
        if item.name == "model" and automatic:
            parser.add_argument("model", help="Exact Hugging Face model ID; ambiguous names are not searched")
            continue
        default = item.default if spec.cli_default is MISSING else spec.cli_default
        if item.name == "notify":
            default = default_notify_provider
        if isinstance(item.default, bool):
            if spec.negative:
                exclusive = parser.add_mutually_exclusive_group()
                exclusive.add_argument(flag, dest=item.name, action="store_true", help=help_text)
                exclusive.add_argument(spec.negative, dest=item.name, action="store_false", help=spec.negative_help)
                parser.set_defaults(**{item.name: default})
            else:
                parser.add_argument(flag, action="store_true", default=default, help=help_text)
            continue
        kwargs: dict[str, Any] = {"default": default, "help": help_text}
        if hints[item.name] in (int, float):
            kwargs["type"] = hints[item.name]
        if spec.choices:
            kwargs["choices"] = spec.choices + (("auto",) if item.name == "profiling_mode" and automatic else ())
        if spec.required:
            kwargs["required"] = True
        parser.add_argument(flag, **kwargs)


def configuration_arguments(config: RunOptions, *, purpose: Literal["run", "probe"]) -> list[str]:
    """Serialize configuration values using their declared command applicability."""
    arguments: list[str] = []
    for item in fields(RunOptions):
        spec: CliOption = item.metadata["cli"]
        if purpose == "probe" and not spec.probe:
            continue
        value = getattr(config, item.name)
        flag = "--" + item.name.replace("_", "-")
        if isinstance(value, bool):
            if value and not item.default:
                arguments.append(flag)
            elif not value and spec.negative:
                arguments.append(spec.negative)
        elif value is not None and value != "":
            # Preserve the payload of string subclasses, including str-based enums.
            text = str.__str__(value) if isinstance(value, str) else str(value)
            arguments.extend((flag, text))
    return arguments
