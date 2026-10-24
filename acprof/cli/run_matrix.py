"""Validate profiling resource axes independently of runtime preparation."""
from __future__ import annotations


def _parse_int_list(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"





def _resource_matrix(args, parser):
    """Validate resource selection before building the runtime."""
    cpu_list = _parse_int_list(args.cpus)
    mem_list = _parse_int_list(args.mems)
    gpu_list = [mode.lower() for mode in _parse_str_list(args.gpus)]
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0")
    if (not cpu_list or not mem_list or not gpu_list
            or any(value <= 0 for value in cpu_list + mem_list)
            or any(mode not in ("off", "on") for mode in gpu_list)):
        parser.error("resource lists must be non-empty, CPUs/memory positive, and GPUs off/on")
    if any(len(values) != len(set(values)) for values in (cpu_list, mem_list, gpu_list)):
        parser.error("resource lists must not contain duplicate cases")
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
            parser.error(
                f"{option}={value} must be present in the selected resource "
                f"matrix {resources}"
            )

    return cpu_list, mem_list, gpu_list
