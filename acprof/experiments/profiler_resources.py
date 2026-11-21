"""Resource axes used by the independent execution profilers."""
from collections.abc import Iterator, Sequence

DEFAULT_MASSIF_SAMPLING = "per-scale"
DEFAULT_NSYS_SAMPLING = "per-cpu-scale"


def enabled_execution_profilers(selection: str, gpu_modes: Sequence[str]) -> tuple[str, ...]:
    """Match the tool selection to the normalized devices in the experiment."""
    return tuple(
        tool for tool, device in (("massif", "off"), ("nsys", "on"))
        if selection in (tool, "both") and device in gpu_modes
    )


def profiler_reference_axes(
    selection: str, gpu_modes: Sequence[str], *, massif_sampling: object, nsys_sampling: object,
) -> Iterator[tuple[str, str]]:
    """Yield only reference options that determine a sampled resource case."""
    tools = enabled_execution_profilers(selection, gpu_modes)
    if massif_sampling is None:
        massif_sampling = DEFAULT_MASSIF_SAMPLING
    if nsys_sampling is None:
        nsys_sampling = DEFAULT_NSYS_SAMPLING
    if "massif" in tools and massif_sampling == "per-scale":
        yield "massif_reference_cpu", "cpus"
        yield "massif_reference_mem", "mems"
    if "nsys" in tools and nsys_sampling in ("per-cpu-scale", "per-scale"):
        yield "nsys_reference_mem", "mems"
        if nsys_sampling == "per-scale":
            yield "nsys_reference_cpu", "cpus"
