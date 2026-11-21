"""Read-only metric provenance shared by audit, statistics and presentation."""
from __future__ import annotations

import json
import math
from functools import lru_cache

from acprof.results.coverage import collector_contract, field_collectors
from acprof.results.metrics import METRICS

QUALITIES = {'complete', 'partial', 'unsupported', 'permission_denied', 'not_requested', 'error', 'unavailable', 'unknown', 'not_applicable', 'tool_missing'}
USABLE_QUALITIES = frozenset({'complete', 'unknown'})
RESOURCE_SOURCES = {'psutil_observed_process_tree', 'procfs_observed_process_tree', 'service_cgroup', 'nvml_selected_device'}


def observation_signature(observation: dict) -> str:
    return json.dumps({key: observation[key] for key in ('source', 'scope', 'window', 'derivation', 'sampling')
                       if key in observation}, sort_keys=True, allow_nan=False)


def observations_comparable(left: list[dict], right: list[dict]) -> bool:
    return (all(item.get('quality') in USABLE_QUALITIES for item in [*left, *right])
            and len({observation_signature(item) for item in [*left, *right]}) <= 1
            and bool(left) == bool(right))


@lru_cache(maxsize=128)
def _decode(text: str) -> dict:
    if text.strip().lower() in {'', 'nan', 'null', 'none'}:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('resource_metric_quality must be an object')
    if 'schema_version' in value:
        if type(value['schema_version']) is not int or value['schema_version'] != 1:
            raise ValueError('unsupported resource metric quality version')
        value = value.get('metrics')
    if not isinstance(value, dict) or any(not isinstance(item, dict) for item in value.values()):
        raise ValueError('invalid resource metric evidence')
    return value


@lru_cache(maxsize=128)
def _collector_states(text: str) -> dict:
    if not text.startswith("{"):
        return {}
    obj = json.loads(text)
    states = obj.get("collectors", {})
    if not isinstance(states, dict):
        raise ValueError("invalid window collector outcomes")
    return states


def _from_collector(row, metric, group, states):
    item = states.get(group)
    if not isinstance(item, dict):
        return None
    status, reason = item.get("status"), item.get("reason", "")
    if group == "packet_latency" and status == "available":
        status = {"ok": "available", "partial": "partial"}.get(row.get("packet_status"), "unavailable")
        if status != "available":
            reason = "incomplete_packet_request_correlation"
    if status != "available":
        return {"quality": status, "reason": reason}
    needs_baseline = metric.window == "control_window" or any(part in metric.name for part in ("_eff_", "_effective_", "_idle_"))
    if group in {"gpu_power", "cpu_energy", "dram_energy"} and needs_baseline:
        if item.get("sampling", {}).get("idle_seconds") == 0:
            return {"quality": "not_requested", "reason": "idle_window_disabled"}
        if item.get("control_error"):
            return {"quality": "error", "reason": item["control_error"]}
    if metric.applicability in {"gpu", "gpu_idle_debug"} and row.get("gpu_mode") == "off" or (
            metric.applicability == "cpu" and row.get("gpu_mode") == "on"):
        return {"quality": "not_applicable", "reason": "device_not_selected"}
    if metric.name in {"cpu_cores", "mem_cap_gb"} and row.get("resource_limit_source") == "inherited":
        return {"quality": "not_applicable", "reason": "inherited_limit_not_controlled"}
    if metric.applicability == "audio" and item.get("task_family") not in {"audio", "multimodal"} and str(row.get(metric.name, "")).lower() in {"", "nan", "none"}:
        return {"quality": "not_applicable", "reason": "no_audio_samples"}
    if metric.applicability == "pixel_geometry":
        direction = "output" if "output_megapixel" in metric.name or metric.name.startswith("output_pixels") else "input"
        if str(row.get(direction + "_pixels_per_request", "")).lower() in {"", "none", "nan"}:
            return {"quality": "not_applicable", "reason": "no_pixel_geometry"}
    if metric.name.endswith("_slow_ratio") and not item.get("latency_slo"):
        return {"quality": "not_requested", "reason": "no_latency_slo"}
    if metric.name.endswith("_per_cpu_core") and str(row.get("cpu_cores", "")).lower() in {"", "none", "nan"}:
        return {"quality": "not_applicable", "reason": "no_controlled_cpu_quota"}
    if metric.applicability in {"idle_debug", "gpu_idle_debug"} and not item.get("idle_debug"):
        return {"quality": "not_requested", "reason": "idle_debug_disabled"}
    value = row.get(metric.name)
    try:
        present = math.isfinite(float(value)) if metric.kind == "number" else str(value or "").lower() not in {"", "unknown", "nan", "unavailable"}
    except (ValueError, TypeError):
        present = False
    result: dict = {"quality": "complete" if present else "unavailable",
              "reason": "" if present else reason or "missing_window_value"}
    if reader_error := item.get("reader_errors", {}).get(metric.name):
        result.update(quality="partial" if present else reader_error["status"], reason=reader_error["detail"])
    elif not present and (hint := item.get("preflight_missing", {}).get(metric.name)):
        result.update(quality=hint["status"], reason=hint["detail"])
    if metric.name.startswith("throughput_samples_per_s"):
        packet = row.get("packet_status") in {"ok", "partial"}
        result["sampling"] = {"latency_denominator": "packet" if packet else "application"}
        if row.get("packet_status") == "partial":
            result.update(quality="partial", reason="incomplete_packet_request_correlation")
    if group in {"gpu_power", "cpu_energy", "dram_energy", "cpu_instructions", "cgroup"}:
        result["sampling"] = dict(item.get("sampling", {}))
        if group == "gpu_power":
            result["sampling"]["energy_method"] = row.get("gpu_energy_source")
        if group == "cpu_instructions" and present:
            try:
                running = float(row.get("cpu_perf_running_pct", "nan"))
            except (ValueError, TypeError):
                running = float("nan")
            if not (0 < running <= 100):
                result.update(quality="partial", reason="unknown_pmu_running_fraction")
    return result


def metric_observation(row: dict, name: str) -> dict:
    metric = METRICS[name]
    runtime = "process" if row.get("runtime_kind") == "process" else "docker"
    contract = collector_contract(metric, runtime)
    result = {'source': metric.source, 'scope': contract.scope, 'window': metric.window,
              'derivation': contract.derivation, 'quality': 'unknown', 'reason': 'no_metric_quality_evidence'}
    try:
        text = str(row.get("resource_metric_quality", ""))
        evidence = _decode(text)
        item = evidence.get(name)
        states = _collector_states(text)
        if states and runtime not in contract.runtimes:
            return {**result, "quality": "not_applicable", "reason": "runtime_not_implemented"}
        if states and name in field_collectors(runtime):
            state = states.get(field_collectors(runtime)[name], {})
            if state.get("status") == "not_applicable":
                return {**result, "quality": "not_applicable", "reason": state.get("reason", "")}
        if item is None:
            group = field_collectors(runtime).get(name)
            item = _from_collector(row, metric, group, states) if group else None
            if (metric.applicability == "text_tokenizer" and item and item["quality"] == "complete"
                    and "completed_requests" not in evidence.get("output_token_count_avg", {})):
                return {**result, "reason": "output_token_coverage_unrecorded"}
        if item is None:
            return result
        if item.get('quality') not in QUALITIES:
            raise ValueError('invalid metric quality')
        if any(key in item and item[key] != result[key] for key in ('source', 'scope', 'window', 'derivation')):
            raise ValueError('metric source contract conflicts with registry')
        observation_signature(item)  # Reject non-finite sampling metadata before downstream hashing.
        if not isinstance(item.get('reason', ''), str):
            raise ValueError('metric reason must be text')
        return {**result, 'reason': '', **item}
    except (TypeError, ValueError) as exc:
        return {**result, 'quality': 'error', 'reason': str(exc)}


def metric_usable(row: dict, name: str) -> bool:
    """Preserve legacy analysis; explicit incomplete evidence never becomes a valid value."""
    return name not in METRICS or metric_observation(row, name)['quality'] in USABLE_QUALITIES


def collection_field_outcomes(rows) -> dict:
    """Count formal windows by field and quality. A supported source alone proves no value."""
    outcomes: dict = {}
    for row in rows:
        runtime = "process" if row.get("runtime_kind") == "process" else "docker"
        for name, collector in field_collectors(runtime).items():
            metric = METRICS[name]
            if metric.kind != "number":
                continue
            obs = metric_observation(row, name)
            if collector.startswith("gpu_") and row.get("gpu_mode") == "off":
                obs = {**obs, "quality": "not_applicable", "reason": "GPU disabled"}
            if row.get("status") == "error" and obs["quality"] not in {"not_requested", "not_applicable"}:
                obs = {**obs, "quality": "error", "reason": row.get("error") or "measurement_failed"}
            entry = outcomes.setdefault(name, {"collector": collector, "source": obs["source"],
                "scope": obs["scope"], "window": obs["window"], "unit": metric.unit, "derivation": obs["derivation"],
                "windows": {}, "reasons": {}})
            counts = entry["windows"]
            counts[obs["quality"]] = counts.get(obs["quality"], 0) + 1
            if reason := obs.get("reason"):
                entry["reasons"][reason] = entry["reasons"].get(reason, 0) + 1
    return outcomes


def common_runtime_metric(name: str) -> bool:
    metric = METRICS.get(name)
    return bool(metric and (metric.source in {"psutil_observed_process_tree", "procfs_observed_process_tree",
                                              "rapl_package", "rapl_dram", "nvml_selected_device"}
                or metric.source == "client_http" and (name.startswith(("latency_app_", "throughput_app_"))
                                                       or name in {"request_payload_bytes", "response_payload_bytes"})))
