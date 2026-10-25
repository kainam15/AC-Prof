"""Versioned raw window evidence and pure replay of existing metric reductions.

Capture/serialization are called after collector shutdown. The retention limit
bounds the persisted evidence, not the collectors' existing reduction buffers.
Old aggregate-only rows never acquire a synthetic raw-sample capability.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TextIO

from acprof.artifacts import atomic_write, loads_finite_json

TELEMETRY_SCHEMA_VERSION = 1
REDUCTION_VERSION = "window-reduction-v1"


def finite_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_dataclass(value) and not isinstance(value, type):
        return finite_json(asdict(value))
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(finite_json(value), sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def capture_samples[T](samples: Sequence[T], limit: int,
                    encode: Callable[[T], Any] = finite_json) -> dict[str, Any]:
    """Select at most limit samples without copying the full collector buffer."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2:
        raise ValueError("raw_sample_limit must be an integer of at least 2")
    count = len(samples)
    indices: Iterable[int]
    if count <= limit:
        indices = range(count)
    else:
        # Preserve boundaries and deterministic coverage; omitted observations
        # cannot be recovered and this subset is not an exact reduction input.
        indices = (index * (count - 1) // (limit - 1) for index in range(limit))
    selected = [encode(samples[index]) for index in indices]
    return {
        "samples": selected,
        "retention": {
            "limit": limit, "observed_samples": count, "retained_samples": len(selected),
            "omitted_samples": count - len(selected), "complete": count <= limit,
            "selection": "all" if count <= limit else "uniform_index_with_boundaries",
            "scope": "persisted_evidence_only; in_memory_reduction_buffer_is_not_bounded",
        },
    }


def window_evidence(*, identity: dict[str, Any], context: dict[str, Any],
                    collectors: dict[str, Any]) -> dict[str, Any]:
    return finite_json({
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "reduction_version": REDUCTION_VERSION,
        "identity": identity, "context": context,
        "clock": {
            "name": "time.perf_counter", "unit": "seconds", "monotonic": True,
            "scope": "one_host_process", "wall_clock_conversion": "unavailable",
            "sample_timestamp_semantics": "logical_read_start; not exact hardware sample time",
        },
        "collectors": collectors,
    })


def write_window_evidence(path: str | Path, evidence: dict[str, Any]) -> dict[str, Any]:
    """Publish bytes atomically; repeated identical publication is idempotent."""
    if type(evidence.get("schema_version")) is not int or evidence.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
        raise ValueError("unsupported telemetry schema")
    destination = Path(path)
    payload = canonical_bytes(evidence) + b"\n"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f"refusing to replace different window evidence: {destination}")
    else:
        def write(stream: TextIO) -> None:
            stream.write(payload.decode("utf-8"))
        atomic_write(destination, write)
    return {
        "path": str(destination), "sha256": hashlib.sha256(payload).hexdigest(),
        "schema_version": TELEMETRY_SCHEMA_VERSION, "size_bytes": len(payload),
    }


def read_window_evidence(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("window telemetry hash mismatch")
    evidence = loads_finite_json(payload.decode("utf-8"))
    if (not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int
            or evidence.get("schema_version") != TELEMETRY_SCHEMA_VERSION):
        raise ValueError("unsupported telemetry schema")
    return evidence


def raw_capability(evidence: dict[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {"status": "unavailable", "reason": "legacy_or_missing_raw_evidence", "can_recompute": False}
    if type(evidence.get("schema_version")) is not int or evidence.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
        raise ValueError("unsupported telemetry schema")
    collectors = evidence.get("collectors", {})
    available = {name: item for name, item in collectors.items()
                 if isinstance(item, dict) and item.get("status") == "recorded"}
    complete = bool(available) and all(item.get("retention", {}).get("complete") is True
                                       for item in available.values())
    return {"status": "complete" if complete else "incomplete" if available else "unavailable",
            "can_recompute": complete, "collectors": sorted(available),
            "scope": "recorded_workload_metrics_using_recorded_control_baseline",
            "control_samples_complete": all(
                not item.get("control") or item["control"].get("retention", {}).get("complete") is True
                for item in available.values())}


def _number(value: Any) -> float:
    return float("nan") if value is None else float(value)


def recompute_window(evidence: dict[str, Any]) -> dict[str, Any]:
    """Replay recorded reductions without hardware, environment reads or I/O.

Missing collectors stay unavailable. A truncated collector is rejected rather
than producing plausible metrics from an incomplete series. GPU source choice
is the recorded, versioned choice, not a fresh hardware capability decision.
"""
    raw_capability(evidence)
    if evidence.get("reduction_version") != REDUCTION_VERSION:
        raise ValueError("unsupported telemetry reduction version")
    output: dict[str, Any] = {}
    for name, item in evidence.get("collectors", {}).items():
        if not isinstance(item, dict) or item.get("status") != "recorded":
            output[name] = {"status": "unavailable"}
            continue
        if item.get("retention", {}).get("complete") is not True:
            raise ValueError(f"cannot exactly recompute truncated {name} evidence")
        if name == "cpu":
            output[name] = _recompute_cpu(item)
        elif name == "gpu":
            output[name] = _recompute_gpu(item)
        elif name == "resource":
            output[name] = _recompute_resource(item)
        else:
            output[name] = {"status": "unavailable", "reason": "no_raw_reducer"}
    return finite_json(output)


def _recompute_cpu(item: dict[str, Any]) -> dict[str, Any]:
    from acprof.monitors.energy_cpu import (
        CPUSample,
        RaplDomain,
        reduce_cpu_samples,
        reduce_dram_samples,
    )
    context = item["context"]
    samples = [CPUSample(**{**sample, "host_active_s": _number(sample["host_active_s"])})
               for sample in item["samples"]]
    domains = [RaplDomain(**domain) for domain in context["domains"]]
    result = reduce_cpu_samples(samples, _number(context["idle_power_w"]), domains,
                                  min_power_interval_s=context["min_power_interval_s"])
    result.dram = reduce_dram_samples(
        samples, [RaplDomain(**domain) for domain in context["dram_domains"]],
        _number(context["dram_idle_power_w"]), context["dram_status"],
        context["dram_error"], min_power_interval_s=context["min_power_interval_s"],
    )
    return asdict(result)


def _recompute_gpu(item: dict[str, Any]) -> dict[str, Any]:
    from acprof.monitors.gpu_metrics import apply_gpu_energy_counter, reduce_gpu_samples
    context = item["context"]
    samples = [(float(sample["timestamp"]), _number(sample["power_w"])) for sample in item["samples"]]
    result = reduce_gpu_samples(samples, _number(context["idle_power_w"]))
    result.counter_start_mj = context["counter_start_mj"]
    result.counter_end_mj = context["counter_end_mj"]
    result.measurement_duration_s = _number(context["measurement_duration_s"])
    result.energy_fallback_reason = context["energy_fallback_reason"]
    source = context["energy_source"]
    if source == "nvml_total_energy":
        apply_gpu_energy_counter(result)
    elif source not in {"power_integration", "unavailable"}:
        raise ValueError("unknown recorded GPU energy source")
    if context.get("control_source_reselected"):
        result.energy_eff_j = result.energy_total_j - result.idle_power_w * result.measurement_duration_s
        result.avg_power_eff_w = result.avg_power_total_w - result.idle_power_w
        result.peak_power_eff_w = result.peak_power_total_w - result.idle_power_w
    return asdict(result)


def _recompute_resource(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("source") == "psutil":
        from acprof.monitors.process_usage import reduce_process_samples
        values = reduce_process_samples(item["samples"])
        if item.get("lost_members"):
            values.update(process_cpu_time_s=None, process_cpu_util_avg_pct=None)
        return values
    from acprof.monitors import resource_metrics
    context = item["context"]
    samples = [resource_metrics.ResourceUsageSample(**sample) for sample in item["samples"]]
    result = resource_metrics.reduce_resource_samples(
        samples, _number(context["cpu_cores"]), _number(context["mem_limit_bytes"]),
        min_cpu_interval_s=context["min_cpu_interval_s"],
    )
    if context["swap_limit_bytes"] is not None:
        result.container_swap_limit_bytes = float(context["swap_limit_bytes"])
    before, after = context["io_start"], context["io_end"]
    if before is not None and after is not None:
        for direction, index in (("read", 0), ("write", 1)):
            setattr(result, f"container_io_{direction}_bytes", resource_metrics.counter_delta(
                {direction: before[index]}, {direction: after[index]}, direction))
    def counters(value: dict[str, Any]) -> resource_metrics.WindowCounterSnapshots:
        return resource_metrics.WindowCounterSnapshots(**{
            key: None if values is None else {name: _number(number) for name, number in values.items()}
            for key, values in value.items()
        })
    resource_metrics.apply_window_counter_metrics(result, counters(context["counters_start"]),
                                                   counters(context["counters_end"]),
                                                   context["window_duration_s"])
    return asdict(result)
