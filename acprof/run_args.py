"""Shared profiling argument declarations and frozen-option serialization."""
import argparse

from acprof.config_constraints import finite_number, integer
from acprof.run_options import add_option_arguments


def add_download_arguments(parser):
    add_option_arguments(parser, group="download")


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

    add_option_arguments(parser, default_notify_provider=default_notify_provider, automatic=automatic)

    if automatic:
        parser.description = "AC-Prof: resolve, preflight and collect an exact Hugging Face model ID"
        parser.epilog = "Example: acprof auto openai-community/gpt2 --profiling-mode auto --gpus off"

    # Resource matrix
    parser.add_argument("--matrix-order", choices=("seeded", "declared"), default="seeded",
                        help="Frozen case/scale execution order (default: seeded)")
    parser.add_argument("--matrix-seed", type=int, default=0,
                        help="Seed for domain-separated deterministic case and input-scale ordering")
    parser.add_argument("--dram-energy", choices=("auto", "off", "required"), default="auto",
                        help="Optional host DRAM RAPL measurement; only required makes absence fatal")
    parser.add_argument("--gpu-device", default=None, help="One physical GPU index or UUID (default: ACPROF_GPU_DEVICE, DEVICE_INDEX, then 0)")

    # Additional recorded options
    parser.add_argument(
        "--latency-slo", action="append", default=[], metavar="SELECTOR=SECONDS",
        help="Latency SLO (repeatable): task:TAG=SECONDS, profile:ID=SECONDS or default=SECONDS; no implicit threshold",
    )

    # Compute profiling
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

    add_download_arguments(parser)
    return parser


def flatten_run_options(options: dict) -> dict:
    """Recover the public fields from saved RunConfig containers without losing values."""
    flattened = dict(options)
    # Retired execution preference was never part of measurement identity.
    if "skip_build" in flattened:
        if type(flattened.pop("skip_build")) is not bool:
            raise ValueError("invalid frozen boolean: skip_build")
    extra = flattened.pop("extra_options", {})
    if not isinstance(extra, dict):
        raise ValueError("invalid frozen extra_options: expected an object")
    # Old experiment states include this internal form container. Validate its
    # public contents before flattening; never silently resolve contradictory data.
    arguments_from_options(extra)
    for name, value in extra.items():
        if name in flattened and flattened[name] != value:
            raise ValueError(f"conflicting frozen option: {name}")
        flattened[name] = value
    return flattened


def arguments_from_options(options: dict, *, parser: argparse.ArgumentParser | None = None) -> list[str]:
    """Serialize recorded public options through the authoritative parser declarations."""
    actions: dict[str, list] = {}
    for action in (parser or build_parser())._actions:
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
            parsed: object
            try:
                if action.type is int:
                    parsed = integer(item, allow_text=True)
                elif action.type is float:
                    parsed = finite_number(item, allow_text=True)
                else:
                    parsed = action.type(item) if action.type else item
            except (ValueError, TypeError) as exc:
                raise ValueError(f'invalid frozen option: {name}') from exc
            if action.choices is not None and parsed not in action.choices:
                raise ValueError(f'invalid frozen choice: {name}')
            arguments.extend((action.option_strings[0], str(parsed)))
    return arguments
