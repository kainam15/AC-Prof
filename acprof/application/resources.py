"""Validate profiling resource axes independently of runtime preparation."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from typing import TYPE_CHECKING

from acprof.gpu_mode import normalize_gpu_mode

if TYPE_CHECKING:
    from acprof.experiments.configuration import RunConfig

DEFAULT_DOCKER_CPUS = (1, 2, 4, 8)
DEFAULT_DOCKER_MEMS = (2, 4, 8, 16)


def resolve_runtime_options(config: RunConfig, requested: Namespace, *, recorded_runtime: str = "docker") -> Namespace:
    """Resolve one execution stage without rewriting its validated request.

    A resume with auto runtime follows recorded evidence. Only scalar execution
    options are replaced; unrelated validated options retain their existing values.
    """
    from acprof.host.runtimes.selection import require_runtime_requests, select_runtime

    runtime = recorded_runtime if requested.resume and requested.runtime == "auto" else requested.runtime
    selection = select_runtime(runtime, profiling_mode=requested.profiling_mode)
    resolved = Namespace(**vars(requested))
    resolved.runtime = selection.selected
    resolved.runtime_selection = selection.to_dict()
    require_runtime_requests(resolved)
    if resolved.runtime == "docker":
        if resolved.cpus == "auto":
            resolved.cpus = ",".join(map(str, DEFAULT_DOCKER_CPUS))
        if resolved.mems == "auto":
            resolved.mems = ",".join(map(str, DEFAULT_DOCKER_MEMS))
        # Validate automatic resources before resolution, downloads or state writes.
        replace(config, runtime="docker", cpus=resolved.cpus, mems=resolved.mems).validate()
    elif resolved.prune_startup_oom:
        resolved.prune_startup_oom = False
    return resolved


def _parse_int_list(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"





def resource_matrix(args):
    """Validate resource selection before building the runtime."""
    process = getattr(args, 'runtime', 'docker') == 'process'
    cpu_list = ([None] if process else list(DEFAULT_DOCKER_CPUS)) if args.cpus == 'auto' else _parse_int_list(args.cpus)
    mem_list = ([None] if process else list(DEFAULT_DOCKER_MEMS)) if args.mems == 'auto' else _parse_int_list(args.mems)
    gpu_list = [normalize_gpu_mode(mode) for mode in _parse_str_list(args.gpus)]
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")
    if (not cpu_list or not mem_list or not gpu_list
            or any(value is not None and value <= 0 for value in cpu_list + mem_list)):
        raise ValueError("resource lists must be non-empty, CPUs/memory positive, and GPUs off/on")
    if any(len(values) != len(set(values)) for values in (cpu_list, mem_list, gpu_list)):
        raise ValueError("resource lists must not contain duplicate cases")
    massif_selected = args.execution_profile_tool in {"massif", "both"}
    nsys_selected = args.execution_profile_tool in {"nsys", "both"}
    reference_checks = []
    if massif_selected and args.massif_sampling == "per-scale":
        reference_checks.extend(
            [
                ("--massif-reference-cpu", args.massif_reference_cpu, cpu_list),
                ("--massif-reference-mem", args.massif_reference_mem, mem_list),
            ]
        )
    if nsys_selected and args.nsys_sampling != "full":
        reference_checks.append(
            ("--nsys-reference-mem", args.nsys_reference_mem, mem_list)
        )
    if nsys_selected and args.nsys_sampling == "per-scale":
        reference_checks.append(
            ("--nsys-reference-cpu", args.nsys_reference_cpu, cpu_list)
        )
    for option, value, resources in reference_checks:
        if value is not None and value not in resources:
            raise ValueError(
                f"{option}={value} must be present in the selected resource "
                f"matrix {resources}"
            )

    return cpu_list, mem_list, gpu_list
