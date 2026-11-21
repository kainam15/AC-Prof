"""Server histogram snapshots: explicit populations, reset checks and offline derivation."""
from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

from acprof.artifacts import reject_isolated_surrogates, require_schema_version

if TYPE_CHECKING:
    from prometheus_client.samples import Sample

METRICS_BY_EXPORTER = {
    "vllm": {
        "queue": "vllm:request_queue_time_seconds",
        "prefill": "vllm:request_prefill_time_seconds",
        "decode": "vllm:request_decode_time_seconds",
        "inference": "vllm:request_inference_time_seconds",
        "e2e": "vllm:e2e_request_latency_seconds",
    },
    "sglang": {
        "queue": "sglang:queue_time_seconds",
        "ttft": "sglang:time_to_first_token_seconds",
        "itl": "sglang:inter_token_latency_seconds",
        "e2e": "sglang:e2e_request_latency_seconds",
        "stage": "sglang:per_stage_req_latency_seconds",
    },
}
type _Series = dict[tuple[str, str], dict[str, Any]]
MAX_METRICS_BYTES = 4 * 1024 * 1024
MAX_METRICS_SERIES = 4096
SERVER_METRICS_SCOPE = (
    "Server model histograms between two scrapes; other traffic and delayed exporter updates "
    "may contribute. Not client-request attribution, GPU execution time or energy."
)


def _phase_metrics(kind: str) -> dict[str, str]:
    if not isinstance(kind, str) or kind not in METRICS_BY_EXPORTER:
        raise ValueError("unsupported server metrics exporter")
    return METRICS_BY_EXPORTER[kind]


def metrics_declaration(kind: str, model: str, timeout_s: float) -> dict[str, Any]:
    _phase_metrics(kind)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("server metrics require a nonempty model")
    if _number(timeout_s) == 0:
        raise ValueError("server metrics timeout must be positive")
    return {
        "schema_version": 1, "kind": kind, "path": "/metrics", "model": model,
        "selection": "exact_model_name", "boundary": "after_warmup_before_load_and_after_load",
        "timeout_seconds": min(timeout_s, 10.0), "max_response_bytes": MAX_METRICS_BYTES,
    }


def _number(value: Any) -> float:
    try:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError
        return float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("server metrics require finite nonnegative numbers") from error


def _series_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["phase"], json.dumps(row["labels"], sort_keys=True, ensure_ascii=True)


def _series_by_phase(rows: Any, model: str, metrics: dict[str, str]) -> dict[str, _Series]:
    if not isinstance(rows, list) or len(rows) > MAX_METRICS_SERIES:
        raise ValueError("invalid server metrics series list")
    groups: dict[str, _Series] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"phase", "labels", "count", "sum_s", "created_s"}:
            raise ValueError("invalid server metric series")
        phase, labels = row["phase"], row["labels"]
        if (not isinstance(phase, str) or phase not in metrics or not isinstance(labels, dict)
                or labels.get("model_name") != model
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in labels.items())):
            raise ValueError("invalid server metric phase or model labels")
        if phase == "stage":
            stage = labels.get("stage", "")
            if not stage.strip() or not stage.isprintable():
                raise ValueError("server stage histogram requires a nonempty printable stage label")
            phase = f"stage:{stage}"
        count, total = row["count"], _number(row["sum_s"])
        if type(count) is not int or not 0 <= count <= 2 ** 53 or (count == 0 and total != 0):
            raise ValueError("invalid server histogram count/sum")
        if row["created_s"] is not None:
            _number(row["created_s"])
        key = _series_key(row)
        selected = groups.setdefault(phase, {})
        if key in selected:
            raise ValueError("duplicate server metric series")
        selected[key] = row
    return groups


def _collect_sample(
    sample: Sample, names: dict[str, tuple[str, str]], rows: _Series,
) -> None:
    phase, suffix = names[sample.name]
    value = _number(sample.value)
    if sample.timestamp is not None or (suffix == "count" and (
        sample.value > 2 ** 53 or not value.is_integer()
    )):
        raise ValueError("server histograms require exact counts and no explicit sample timestamps")
    key = (phase, json.dumps(sample.labels, sort_keys=True))
    row = rows.setdefault(key, {"phase": phase, "labels": dict(sample.labels)})
    if suffix in row or len(rows) > MAX_METRICS_SERIES:
        raise ValueError("duplicate or excessive server metric series")
    row[suffix] = value


def _complete_series(rows: _Series, types: dict[str, str], metrics: dict[str, str]) -> list[dict[str, Any]]:
    series = []
    for row in rows.values():
        if types.get(metrics[row["phase"]]) != "histogram" or "count" not in row or "sum" not in row:
            raise ValueError("server timing requires complete histogram count/sum pairs")
        series.append({"phase": row["phase"], "labels": row["labels"], "count": int(row["count"]),
                       "sum_s": row["sum"], "created_s": row.get("created")})
    return sorted(series, key=_series_key)


def parse_server_metrics(text: str, model: str, *, kind: str = "vllm") -> dict[str, Any]:
    """Use the upstream exposition grammar; retain only named model histogram evidence."""
    from prometheus_client.parser import text_string_to_metric_families

    metrics = _phase_metrics(kind)
    names = {f"{metric}_{suffix}": (phase, suffix) for phase, metric in metrics.items()
             for suffix in ("count", "sum", "created")}
    rows: _Series = {}
    types: dict[str, str] = {}
    process_start = None
    try:
        for family in text_string_to_metric_families(text):
            if family.name in metrics.values():
                types[family.name] = family.type
            for sample in family.samples:
                if sample.name == "process_start_time_seconds" and not sample.labels:
                    if process_start is not None or sample.timestamp is not None:
                        raise ValueError("duplicate or timestamped process start time")
                    process_start = _number(sample.value)
                if sample.name not in names or sample.labels.get("model_name") != model:
                    continue
                _collect_sample(sample, names, rows)
    except (ValueError, OverflowError) as error:
        raise ValueError("invalid Prometheus server metrics exposition") from error
    series = _complete_series(rows, types, metrics)
    reject_isolated_surrogates(series)
    _series_by_phase(series, model, metrics)
    if not series:
        raise ValueError(f"no supported {kind} server histograms match the exact model_name")
    return {"series": series, "process_start_time_s": process_start}


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, "count": None, "sum_s": None, "mean_s": None}


def _phase_delta(before: _Series, after: _Series) -> dict[str, Any]:
    if not before and not after:
        return _unavailable("metric_not_exposed")
    if before.keys() != after.keys():
        return _unavailable("series_changed")
    count, totals = 0, []
    for key, left in before.items():
        right = after[key]
        if (left["created_s"] != right["created_s"] or right["count"] < left["count"]
                or right["sum_s"] < left["sum_s"]):
            return _unavailable("counter_reset_or_creation_changed")
        delta_count = right["count"] - left["count"]
        delta_sum = right["sum_s"] - left["sum_s"]
        if delta_count == 0 and delta_sum != 0:
            return _unavailable("inconsistent_count_sum_delta")
        count += delta_count
        totals.append(delta_sum)
    try:
        total = math.fsum(totals)
        _number(total)
    except (OverflowError, ValueError):
        return _unavailable("delta_overflow")
    return {
        "status": "ok" if count else "unavailable", "reason": None if count else "no_new_observations",
        "count": count, "sum_s": total, "mean_s": total / count if count else None,
    }


def summarize_server_metrics(
    before: dict[str, Any], after: dict[str, Any], model: str, *, kind: str = "vllm",
) -> dict[str, Any]:
    """Differences are service observations, never divided by the client's request count."""
    metrics = _phase_metrics(kind)
    left, right = (
        _series_by_phase(snapshot["series"], model, metrics) if snapshot["status"] == "ok" else {}
        for snapshot in (before, after)
    )
    names = [phase for phase in metrics if phase != "stage"]
    if "stage" in metrics:
        names.extend(sorted((left.keys() | right.keys()) - set(names)) or ["stage"])
    if before["status"] != "ok" or after["status"] != "ok":
        reason = "snapshot_error" if "error" in (before["status"], after["status"]) else "not_collected"
        return {"status": "error" if reason == "snapshot_error" else "unavailable",
                "phases": {phase: _unavailable(reason) for phase in names}}
    process_changed = before["process_start_time_s"] != after["process_start_time_s"]
    phases = {
        phase: _unavailable("process_start_changed") if process_changed else _phase_delta(
            left.get(phase, {}), right.get(phase, {}),
        ) for phase in names
    }
    available = sum(row["status"] == "ok" for row in phases.values())
    return {"status": "ok" if available == len(phases) else "partial" if available else "unavailable",
            "phases": phases}


def _validate_snapshot_source(snapshot: dict[str, Any]) -> None:
    if snapshot["status"] == "ok" or snapshot["sha256"] is not None:
        if (not isinstance(snapshot["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", snapshot["sha256"])
                or type(snapshot["bytes"]) is not int or not 0 <= snapshot["bytes"] <= MAX_METRICS_BYTES):
            raise ValueError("invalid server metrics source digest or byte count")
    elif snapshot["bytes"] is not None:
        raise ValueError("server metrics byte count requires its digest")
    if snapshot["process_start_time_s"] is not None:
        _number(snapshot["process_start_time_s"])


def _validate_snapshot(snapshot: Any, model: str, metrics: dict[str, str]) -> None:
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "status", "error", "started_s", "finished_s", "sha256", "bytes", "series", "process_start_time_s",
    } or snapshot["status"] not in ("ok", "error", "not_collected"):
        raise ValueError("invalid server metrics snapshot")
    status = snapshot["status"]
    if status == "not_collected":
        if any(value is not None for key, value in snapshot.items() if key != "status"):
            raise ValueError("uncollected server metrics cannot contain evidence")
        return
    if _number(snapshot["finished_s"]) < _number(snapshot["started_s"]):
        raise ValueError("server metrics snapshot clock went backwards")
    if status == "error":
        if not isinstance(snapshot["error"], str) or not snapshot["error"]:
            raise ValueError("failed server metrics require an error")
        if any(snapshot[key] is not None for key in ("series", "process_start_time_s")):
            raise ValueError("failed server metrics cannot contain parsed evidence")
    elif snapshot["error"] is not None or not _series_by_phase(snapshot["series"], model, metrics):
        raise ValueError("successful server metrics require parsed evidence and no error")
    _validate_snapshot_source(snapshot)


def _validate_boundaries(before: dict[str, Any], after: dict[str, Any], result: Any) -> None:
    if after["status"] == "not_collected":
        return
    if before["status"] != "ok" or after["started_s"] < before["finished_s"]:
        raise ValueError("server metrics snapshots are out of order")
    if isinstance(result, dict) and after["started_s"] - before["finished_s"] < _number(result.get("duration_s")):
        raise ValueError("server metrics scrape interval does not enclose the load duration")


def validate_server_metrics(experiment: dict[str, Any]) -> dict[str, Any] | None:
    identity = experiment.get("identity", {})
    declaration = identity.get("server_metrics")
    observation = experiment.get("server_metrics")
    if "server_metrics" not in identity and "server_metrics" not in experiment:
        return None
    if not isinstance(declaration, dict) or not isinstance(observation, dict) or experiment["schema_version"] < 2:
        raise ValueError("server metrics require declaration and observation in schema v2/v3")
    require_schema_version(declaration, 1, "server metrics declaration")
    kind = declaration.get("kind", "")
    expected = metrics_declaration(kind, identity.get("model"), identity["load_config"]["timeout_seconds"])
    if json.dumps(declaration, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("server metrics declaration differs from the load inputs")
    require_schema_version(observation, 1, "server metrics observation")
    if (set(observation) != {"schema_version", "kind", "scope", "before", "after", "summary"}
            or observation["kind"] != kind or observation["scope"] != SERVER_METRICS_SCOPE):
        raise ValueError("invalid server metrics observation contract")
    reject_isolated_surrogates(observation)
    before, after = observation["before"], observation["after"]
    for snapshot in (before, after):
        _validate_snapshot(snapshot, declaration["model"], METRICS_BY_EXPORTER[kind])
    _validate_boundaries(before, after, experiment.get("result"))
    summary = summarize_server_metrics(before, after, declaration["model"], kind=kind)
    if json.dumps(summary, sort_keys=True, allow_nan=False) != json.dumps(observation["summary"], sort_keys=True, allow_nan=False):
        raise ValueError("server metrics summary differs from the retained snapshots")
    if experiment.get("status") == "complete" and any(s["status"] != "ok" for s in (before, after)):
        raise ValueError("complete experiment requires both requested server metrics snapshots")
    return summary
