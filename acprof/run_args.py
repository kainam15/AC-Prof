"""Shared profiling argument declarations and frozen-option serialization."""
import argparse

from acprof.config import (
    DEFAULT_COMPUTE_PROFILE_TOOL,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from acprof.hf_endpoints import HF_DOWNLOAD_MODES


def add_download_arguments(parser):
    parser.add_argument("--download-mode", choices=HF_DOWNLOAD_MODES, default=None,
                        help="HF source mode (default mirror-only; official is explicit)")
    parser.add_argument("--max-download", default=None, help="Bulk download budget, e.g. 5GB; unknown size stops before download")
    parser.add_argument("--model-store", default=None, help="Host Model Store path shared across CPU/GPU runs")
    parser.add_argument("--model-store-max", default=None, help="Model Store capacity, e.g. 100GB")


def build_parser(*, default_notify_provider: str = "auto", automatic: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AC-Prof: Hugging Face model profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  acprof run --model bert-base-uncased
  acprof run --model google/vit-base-patch16-224 --cpus 1,2 --mems 4,8 --gpus off
  acprof run --model amazon/chronos-bolt-base --task-family timeseries --backend chronos
  acprof run --model stable-diffusion-v1-5/stable-diffusion-v1-5 --gpus on
        """,
    )

    # Required
    if automatic:
        parser.description = "AC-Prof: resolve, preflight and collect an exact Hugging Face model ID"
        parser.epilog = "Example: acprof auto openai-community/gpt2 --profiling-mode auto --gpus off"
        parser.add_argument("model", help="Exact Hugging Face model ID; ambiguous names are not searched")
    else:
        parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--revision", help="Model branch, tag or full commit SHA")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the same experiment using its recorded image, input plan and completed cases")

    # Detection overrides
    parser.add_argument("--task", default=None, help="Override pipeline_tag (e.g., text-generation)")
    parser.add_argument(
        "--task-family",
        default=None,
        help="Override task family (nlp/cv/audio/timeseries/diffusion/multimodal/structured)",
    )
    parser.add_argument("--backend", default=None, help="Override runtime backend (transformers_pipeline/chronos/...)")
    parser.add_argument("--model-spec", default=None,
                        help="Local acprof_model.json interface declaration; baked into the immutable service image")

    # Resource matrix
    parser.add_argument("--cpus", default="1,2,4,8", help="CPU core counts (comma-separated)")
    parser.add_argument("--cpuset-cpus", default="", help="Optional fixed CPU IDs for formal/startup containers, e.g. 0-3,8")
    parser.add_argument("--mems", default="2,4,8,16", help="Memory caps in GB (comma-separated)")
    parser.add_argument("--gpus", default="off,on", help="GPU modes (comma-separated: off,on)")
    parser.add_argument("--matrix-order", choices=("seeded", "declared"), default="seeded",
                        help="Frozen case/scale execution order (default: seeded)")
    parser.add_argument("--matrix-seed", type=int, default=0,
                        help="Seed for domain-separated deterministic case and input-scale ordering")
    parser.add_argument("--dram-energy", choices=("auto", "off", "required"), default="auto",
                        help="Optional host DRAM RAPL measurement; only required makes absence fatal")
    parser.add_argument("--gpu-device", default=None, help="One physical GPU index or UUID (default: ACPROF_GPU_DEVICE, DEVICE_INDEX, then 0)")
    startup_oom_pruning_group = parser.add_mutually_exclusive_group()
    startup_oom_pruning_group.add_argument(
        "--prune-startup-oom",
        dest="prune_startup_oom",
        action="store_true",
        help=(
            "Probe startup-only OOM evidence before freezing the formal matrix (default)"
        ),
    )
    startup_oom_pruning_group.add_argument(
        "--no-prune-startup-oom",
        dest="prune_startup_oom",
        action="store_false",
        help=(
            "Disable startup-OOM pruning and independently attempt every "
            "selected CPU/memory/GPU case"
        ),
    )
    parser.set_defaults(prune_startup_oom=True)

    # Experiment parameters
    parser.add_argument(
        "--profiling-mode", choices=("full", "basic", "auto") if automatic else ("full", "basic"), default="full",
        help="full requires packet latency, RAPL and perf (default); basic measures application latency, throughput, CPU and memory",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=5, help="Measurement repeat count")
    parser.add_argument(
        "--repeat-in-window",
        type=int,
        default=DEFAULT_REPEAT_IN_WINDOW,
        help="Requests per energy window; 0 enables auto calibration",
    )
    parser.add_argument(
        "--repeat-window-seconds",
        type=float,
        default=DEFAULT_REPEAT_WINDOW_SECONDS,
        help="Target workload window duration for auto repeat-in-window",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Connect and read-inactivity timeout per /predict (not a total deadline; no retries)",
    )
    parser.add_argument(
        "--latency-slo", action="append", default=[], metavar="SELECTOR=SECONDS",
        help="Latency SLO (repeatable): task:TAG=SECONDS, profile:ID=SECONDS or default=SECONDS; no implicit threshold",
    )
    parser.add_argument("--sample-hz", type=float, default=20.0, help="GPU energy sampling rate")
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=DEFAULT_IDLE_SECONDS,
        help="Idle baseline measurement duration before each workload window",
    )
    parser.add_argument(
        "--idle-cooldown-seconds",
        type=float,
        default=DEFAULT_IDLE_COOLDOWN_SECONDS,
        help="Cooldown duration before collecting idle baselines for each workload window",
    )
    parser.add_argument(
        "--idle-debug",
        action="store_true",
        help="Write CPU idle baseline timestamps and per-row diagnostic JSONL sidecars",
    )
    parser.add_argument("--input-scales", default=None, help="Override input scale values (comma-separated)")
    parser.add_argument("--input-scale-policy", choices=("auto", "minimal"), default="auto",
                        help="Without explicit scales, plan a full range or one smallest declared workload scale")
    parser.add_argument(
        "--workload-spec",
        default=None,
        help=(
            "Path to an NLP, CV, audio, multimodal, diffusion or structured workload manifest. ASR defaults to the "
            "bundled LibriSpeech short-form manifest."
        ),
    )

    # Compute profiling
    parser.add_argument(
        "--compute-profile-tool",
        choices=("none", "both", "torch", "ncu", "vendor"),
        default=DEFAULT_COMPUTE_PROFILE_TOOL,
        help=(
            "Compute FLOP profiler (default: none): none skips all compute "
            "probes; both independently collects torch_profiler_eager logical "
            "FLOP and ncu GPU executed FLOP"
        ),
    )
    parser.add_argument("--advisor-root", default=None, help="Host Intel Advisor install root or advisor executable")
    parser.add_argument("--ncu-root", default=None, help="Host Nsight Compute install root or ncu executable")
    parser.add_argument("--advisor-repeat", type=int, default=20, help="Intel Advisor profiled inference repetitions")
    parser.add_argument(
        "--torch-profiler-repeat",
        type=int,
        default=1,
        help="torch_profiler_eager profiled inference repetitions",
    )
    parser.add_argument("--ncu-repeat", type=int, default=1, help="ncu profiled inference repetitions")
    parser.add_argument("--compute-profile-cpus", type=int, default=None, help="CPU cores for temporary compute profiler containers (default: host logical CPUs)")
    parser.add_argument("--compute-profile-mem", type=int, default=None, help="Memory GB for temporary compute profiler containers (default: 75%% of host memory)")
    profile_artifact_group = parser.add_mutually_exclusive_group()
    profile_artifact_group.add_argument(
        "--keep-compute-profiles",
        dest="keep_compute_profiles",
        action="store_true",
        help="Keep raw Advisor/ncu profiler artifacts (default)",
    )
    profile_artifact_group.add_argument(
        "--discard-compute-profiles",
        dest="keep_compute_profiles",
        action="store_false",
        help="Discard raw profiler artifacts after summaries are recorded",
    )
    parser.set_defaults(keep_compute_profiles=True)

    # High-overhead execution profiling. These probes are intentionally
    # opt-in and use reduced resource sampling by default.
    parser.add_argument(
        "--execution-profile-tool",
        choices=("none", "both", "massif", "nsys"),
        default="none",
        help=(
            "Optional execution profiler: Massif for CPU heap peaks, Nsight "
            "Systems for CUDA/GPU timelines, both, or none (default)"
        ),
    )
    parser.add_argument(
        "--massif-sampling",
        choices=("per-scale", "full"),
        default="per-scale",
        help=(
            "Massif resource sampling: per-scale profiles the largest selected "
            "CPU/memory case and reuses it across CPU-only rows (default); "
            "full profiles every selected CPU/memory case"
        ),
    )
    parser.add_argument(
        "--massif-reference-cpu",
        type=int,
        default=None,
        help="Representative CPU for Massif per-scale sampling (default: largest selected)",
    )
    parser.add_argument(
        "--massif-reference-mem",
        type=int,
        default=None,
        help="Representative memory GB for Massif per-scale sampling (default: largest selected)",
    )
    parser.add_argument(
        "--massif-repeat",
        type=int,
        default=1,
        help="Inference repetitions inside each Valgrind Massif probe",
    )
    parser.add_argument(
        "--nsys-sampling",
        choices=("per-cpu-scale", "per-scale", "full"),
        default="per-cpu-scale",
        help=(
            "Nsight Systems resource sampling: per-cpu-scale profiles every "
            "selected CPU at the largest selected memory (default); per-scale "
            "uses one representative CPU/memory case; full profiles every case"
        ),
    )
    parser.add_argument(
        "--nsys-reference-cpu",
        type=int,
        default=None,
        help="Representative CPU for Nsys per-scale sampling (default: largest selected)",
    )
    parser.add_argument(
        "--nsys-reference-mem",
        type=int,
        default=None,
        help=(
            "Representative memory GB for Nsys reduced sampling "
            "(default: largest selected)"
        ),
    )
    parser.add_argument(
        "--nsys-repeat",
        type=int,
        default=1,
        help="Inference repetitions inside each Nsight Systems capture range",
    )
    parser.add_argument(
        "--nsys-root",
        default=None,
        help="Host Nsight Systems install root or nsys executable",
    )
    execution_artifact_group = parser.add_mutually_exclusive_group()
    execution_artifact_group.add_argument(
        "--keep-execution-profiles",
        dest="keep_execution_profiles",
        action="store_true",
        help=(
            "Keep raw Massif .out and Nsight Systems .nsys-rep artifacts "
            "(default); derived Nsight SQLite caches are always discarded"
        ),
    )
    execution_artifact_group.add_argument(
        "--discard-execution-profiles",
        dest="keep_execution_profiles",
        action="store_false",
        help="Discard raw execution-profiler artifacts after summaries are recorded",
    )
    parser.set_defaults(keep_execution_profiles=True)

    # Infrastructure
    parser.add_argument(
        "--model-download-policy", choices=("auto", "full"), default="auto",
        help="Model files: auto selects verified loader formats; full keeps the complete repository",
    )
    parser.add_argument("--sniff-iface", default="docker0", help="Network interface for tcpdump")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the local model image if present; automatically build it if missing",
    )
    parser.add_argument(
        "--notify",
        choices=("auto", "none", "wecom"),
        default=default_notify_provider,
        help=(
            "Notification mode: auto (default) enables WeCom when "
            "ACPROF_WECOM_WEBHOOK_URL is configured; none disables notifications"
        ),
    )

    add_download_arguments(parser)
    return parser


def arguments_from_options(options: dict) -> list[str]:
    """Serialize recorded public options through the authoritative parser declarations."""
    actions: dict[str, list] = {}
    for action in build_parser()._actions:
        if action.dest != 'help':
            actions.setdefault(action.dest, []).append(action)
    arguments = []
    for name, value in options.items():
        if name not in actions:
            raise ValueError(f'unsupported frozen option: {name}')
        candidates = actions[name]
        action = candidates[0]
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if type(value) is not bool:
                raise ValueError(f'invalid frozen boolean: {name}')
            matched = next((item for item in candidates if item.const is value), None)
            if matched is not None:
                arguments.append(matched.option_strings[0])
            elif action.default is not value:
                raise ValueError(f'cannot restore frozen boolean: {name}')
            continue
        if value is None:
            continue
        values = value if isinstance(action, argparse._AppendAction) else [value]
        if not isinstance(values, list):
            raise ValueError(f'invalid frozen list: {name}')
        for item in values:
            if not isinstance(item, (str, int, float)) or isinstance(item, bool):
                raise ValueError(f'invalid frozen option: {name}')
            try:
                parsed = action.type(item) if action.type else item
            except (ValueError, TypeError) as exc:
                raise ValueError(f'invalid frozen option: {name}') from exc
            if action.choices is not None and parsed not in action.choices:
                raise ValueError(f'invalid frozen choice: {name}')
            arguments.extend((action.option_strings[0], str(item)))
    return arguments
