"""Describe the field sources that actually contributed to a completed request window."""
from __future__ import annotations

import math

from acprof.capabilities import capability_from_error
from acprof.host.measurement.numbers import to_float_or_nan
from acprof.monitors.resource_readers import CGROUP_READER_FIELDS
from acprof.results.coverage import GPU_QUERIES

_FAILED = {"permission_denied", "unsupported", "tool_missing", "error", "unavailable"}


def gpu_status(config, observations, use_gpu, query, fields):
    if not use_gpu:
        return "not_requested"
    planned = config.collection_plan.get("probes", {}).get("gpu_" + query, {})
    if config.profiling_mode == "best_effort" and planned.get("status") in _FAILED:
        return planned["status"]
    resource = observations.resource_result
    quality = getattr(resource, "metric_evidence", {})
    if all(math.isfinite(to_float_or_nan(getattr(resource, field, None)))
           and quality.get(field, {}).get("quality", "complete") == "complete" for field in fields):
        return "verified"
    raw = observations.collector_evidence.get("resource", {}).get("gpu", {})
    status = raw.get(query + "_status", raw.get("status", "unavailable"))
    return "unavailable" if status in {"available", "verified"} else status


def _collector_state(name, probe, config, observations, use_gpu):
    state = {"status": probe["status"], "reason": probe["detail"]}
    if name.startswith("gpu_") and not use_gpu:
        return {"status": "not_applicable", "reason": "GPU disabled"}
    if state["status"] != "available" or name == "packet_latency":
        return state
    owner = {"cpu_energy": "cpu", "dram_energy": "cpu", "gpu_power": "gpu",
             "cpu_instructions": "mips"}.get(name, "resource")
    raw = observations.collector_evidence.get(owner) or {}
    if raw.get("reason") != "collector_has_no_raw_evidence_protocol" and raw.get("status") in _FAILED | {"partial"}:
        state.update(status=raw["status"], reason=raw.get("reason") or raw.get("error", ""))
    if name in {"cpu_energy", "dram_energy", "gpu_power", "cgroup"}:
        state["sampling"] = {"sample_hz": config.sample_hz, "policy": config.sampling_policy}
    if name in {"gpu_power", "cpu_energy", "dram_energy"}:
        state["sampling"]["idle_seconds"] = config.idle_seconds
        if raw.get("control_error"):
            state["control_error"] = raw["control_error"]
    if name == "cgroup":
        if raw.get("cgroup_error"):
            failure = capability_from_error(raw["cgroup_error"]).to_dict()
            state.update(status=failure["status"], reason=failure["detail"])
        if raw.get("reader_errors"):
            state["reader_errors"] = {field: capability_from_error(reason).to_dict() for reader, reason in raw["reader_errors"].items()
                for field in config.collection_plan.get("fields", {}).get("cgroup", ())
                if any(pattern in field for pattern in CGROUP_READER_FIELDS.get(reader, ()))}
    if name == "dram_energy":
        dram = getattr(observations.cpu_result, "dram", None)
        if dram and dram.status not in {"verified", "available"}:
            state.update(status=dram.status, reason=dram.error)
    return state


def window_quality(config, progress, observations, *, use_gpu, status, error, latency_slo):
    quality = dict(getattr(observations.resource_result, "metric_evidence", {}))
    for field, sizes in (("request_payload_bytes", progress.payload_sizes),
                         ("response_payload_bytes", progress.response_sizes)):
        observed = sum(math.isfinite(size) for size in sizes)
        complete = bool(observed and observed == progress.count)
        quality[field] = {"quality": "complete" if complete else "partial" if observed else "unavailable",
                          "reason": "" if complete else "missing_http_body_length",
                          "observed_requests": observed, "completed_requests": progress.count}
    evidence = {"schema_version": 1, "metrics": quality}
    if config.profiling_mode != "best_effort":
        return evidence
    states = {name: _collector_state(name, probe, config, observations, use_gpu)
              for name, probe in config.collection_plan["probes"].items()}
    states["client_http"] = {"status": "available" if progress.count and status != "error" else "error",
                             "reason": error, "latency_slo": latency_slo}
    if "packet_latency" in states:
        states["packet_latency"]["latency_slo"] = latency_slo
    for name in ("gpu_power", "cpu_energy"):
        if name in states:
            states[name]["idle_debug"] = config.idle_debug
    for name in ("protocol", "input_plan", "derived", "host_cpufreq", "managed_process_to_ready", "docker_and_server"):
        states[name] = {"status": "available", "reason": ""}
    states["input_plan"]["task_family"] = config.task_family
    for tool in ("torch", "ncu", "massif", "nsys"):
        selected = config.collection_plan.get("independent_tools", {}).get(tool, False)
        states[tool] = {"status": "available" if selected else "not_requested",
                        "reason": "" if selected else "independent_profiler_not_selected"}
    for query, names in GPU_QUERIES.items():
        state = states.get("gpu_" + query, {})
        if state.get("status") in _FAILED | {"not_applicable"}:
            for field in names:
                quality[field] = {"quality": state["status"], "reason": state["reason"]}
    attribution_error = (observations.collector_evidence.get("cpu") or {}).get("attribution_error")
    if attribution_error:
        from acprof.results.metrics import METRICS
        failure = capability_from_error(attribution_error).to_dict()
        for name, metric in METRICS.items():
            if metric.source == "rapl_cgroup_attribution" or name.startswith("container_attributed_"):
                owner = "cpu_energy" if metric.source == "rapl_cgroup_attribution" else "derived"
                states[owner].setdefault("reader_errors", {})[name] = failure
    for field, probe in config.collection_plan.get("field_probes", {}).items():
        if probe.get("scope") == "preflight_namespace":
            states["cgroup"].setdefault("preflight_missing", {})[field] = probe
        else:
            quality.setdefault(field, {"quality": probe["status"], "reason": probe["detail"]})
    evidence["collectors"] = states
    return evidence
