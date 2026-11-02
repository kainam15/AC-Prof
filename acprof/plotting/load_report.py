"""Offline single-point capacity report for stream_load_experiment v1/v2."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any, TextIO

from acprof.analysis.streaming import STREAMING_LATENCY_METRICS, streaming_summary
from acprof.analysis.streaming_capacity import SLO_METRICS, build_capacity
from acprof.artifacts import atomic_write, read_json_object


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _validate_v2_service_declaration(
    experiment: Mapping[str, Any], identity: Mapping[str, Any],
) -> None:
    declaration = identity.get("service")
    service = experiment.get("service_identity")
    if not isinstance(declaration, dict) or not isinstance(service, dict):
        raise ValueError("stream load v2 requires a service declaration and service_identity")
    required = {"schema_version", "backend", "version", "probe"}
    if not required <= set(declaration) <= required | {"model", "revision"}:
        raise ValueError("stream load identity has an invalid service declaration")
    if declaration.get("schema_version") != 1:
        raise ValueError("stream load identity has an invalid service declaration")
    model = declaration.get("model", identity.get("model"))
    revision = declaration.get("revision", identity.get("model_revision"))
    expected = {
        "backend": declaration["backend"], "version": declaration["version"],
        "model": model, "revision": revision,
    }
    if service.get("declared") != expected:
        raise ValueError("stream load service declaration differs from service_identity.declared")
    if identity.get("model") != model or identity.get("model_revision") != revision:
        raise ValueError("stream load model declarations differ")
    probe = service.get("probe")
    if not isinstance(probe, dict) or probe.get("kind") != declaration["probe"]:
        raise ValueError("stream load service probe declaration differs")


def _v2_capacity(experiment: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    identity = experiment.get("identity")
    if not isinstance(identity, dict) or "slo_thresholds_s" not in identity:
        raise ValueError("stream load v2 requires declared identity and SLO thresholds")
    expected_identity = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if experiment.get("identity_sha256") != expected_identity:
        raise ValueError("stream load identity_sha256 does not match declared inputs")
    _validate_v2_service_declaration(experiment, identity)
    protocol = result.get("protocol")
    requests = result.get("requests")
    if not isinstance(protocol, dict) or not isinstance(requests, list):
        raise ValueError("stream load v2 requires protocol and raw requests")
    load_config = identity.get("load_config")
    concurrency = load_config.get("concurrency") if isinstance(load_config, dict) else None
    if not isinstance(concurrency, int) or protocol.get("concurrency") != concurrency:
        raise ValueError("stream load declared and observed client concurrency differ")
    expected = build_capacity(
        requests, result.get("duration_s"), concurrency, identity["slo_thresholds_s"],
    )
    persisted = result.get("capacity")
    if not isinstance(persisted, dict) or _canonical(persisted) != _canonical(expected):
        raise ValueError("load capacity is inconsistent with raw requests, duration or SLO")
    return persisted


def _v2_streaming(result: Mapping[str, Any]) -> dict[str, Any]:
    requests = result.get("requests")
    if not isinstance(requests, list):
        raise ValueError("stream load v2 requires raw requests")
    expected = streaming_summary(requests)
    persisted = result.get("streaming")
    if not isinstance(persisted, dict) or _canonical(persisted) != _canonical(expected):
        raise ValueError("streaming latency is inconsistent with raw requests")
    return expected


def read_load_report(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    path = path / "load.json" if path.is_dir() else path
    if path.name != "load.json":
        raise ValueError("capacity report input must be load.json or its containing directory")
    experiment = read_json_object(path, label="stream load experiment", bounded=False)
    version = experiment.get("schema_version")
    if experiment.get("kind") != "stream_load_experiment" or version not in {1, 2}:
        raise ValueError("capacity report supports only stream_load_experiment schema v1/v2")
    result = experiment.get("result")
    if not isinstance(result, dict) or result.get("kind") != "stream_load_result":
        raise ValueError("stream load experiment has no stream_load_result")
    if result.get("schema_version") != version:
        raise ValueError("stream load experiment and result schema versions differ")
    capacity = _v2_capacity(experiment, result) if version == 2 else None
    streaming = _v2_streaming(result) if version == 2 else result.get("streaming")
    return {
        "experiment": experiment, "result": result, "capacity": capacity,
        "streaming": streaming, "version": version,
    }


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _value(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _cell(value: Any) -> str:
    return escape(_value(value))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th scope=\"col\">{escape(item)}</th>" for item in headers)
    body = []
    for row in rows:
        body.append("<tr><th scope=\"row\">" + _cell(row[0]) + "</th>" + "".join(
            f"<td>{_cell(value)}</td>" for value in row[1:]
        ) + "</tr>")
    return f"<table><thead><tr><th scope=\"col\">Metric</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _service_section(experiment: Mapping[str, Any], version: int) -> str:
    identity = _object(experiment.get("identity"))
    service = _object(experiment.get("service_identity"))
    declared = _object(service.get("declared"))
    if not declared:
        declared = {
            "backend": None, "version": None,
            "model": identity.get("model"), "revision": identity.get("model_revision"),
        }
    observed = _object(service.get("observed"))
    probe = _object(service.get("probe"))
    rows = []
    for name in ("backend", "version", "model", "revision", "root", "max_model_len"):
        rows.append([name, declared.get(name), observed.get(name)])
    mismatches = service.get("mismatches") if version == 2 else None
    mismatch_text = ", ".join(mismatches) if isinstance(mismatches, list) and mismatches else (
        "none" if mismatches == [] else None
    )
    return (
        "<section><h2>Service identity</h2>"
        + _table(["Declared", "Observed"], rows)
        + _table(["Value"], [["probe", probe.get("status")], ["mismatches", mismatch_text]])
        + "</section>"
    )


def _capacity_section(result: Mapping[str, Any], capacity: Mapping[str, Any] | None) -> str:
    counts = _object(capacity.get("counts") if capacity is not None else result.get("counts"))
    protocol = _object(result.get("protocol"))
    concurrency = capacity.get("client_concurrency") if capacity is not None else protocol.get("concurrency")
    duration = capacity.get("duration_s") if capacity is not None else result.get("duration_s")
    successful_rate = (
        capacity.get("successful_requests_per_s") if capacity is not None
        else result.get("successful_requests_per_s")
    )
    rows = [[name, counts.get(name)] for name in ("offered", "sent", "succeeded", "failed", "dropped", "cancelled")]
    rows.extend([
        ["duration_s", duration], ["configured_client_concurrency", concurrency],
        ["successful_requests_per_s", successful_rate],
        ["goodput_requests_per_s", capacity.get("goodput_requests_per_s") if capacity else None],
    ])
    return "<section><h2>Single load point</h2>" + _table(["Value"], rows) + "</section>"


def _slo_section(capacity: Mapping[str, Any] | None) -> str:
    slo = capacity.get("slo") if capacity is not None else None
    if not isinstance(slo, dict):
        return "<section><h2>SLO goodput</h2><p>unavailable</p></section>"
    thresholds = _object(slo.get("thresholds_s"))
    counts = _object(slo.get("counts"))
    per_metric = _object(slo.get("per_metric"))
    rows = []
    for name in SLO_METRICS:
        metric = _object(per_metric.get(name))
        rows.append([
            name, thresholds.get(name), metric.get("satisfied"),
            metric.get("violated"), metric.get("unevaluable"),
        ])
    overall = [[name, counts.get(name)] for name in ("satisfied", "violated", "unevaluable")]
    return (
        "<section><h2>SLO goodput</h2>"
        + _table(["Threshold s", "Satisfied", "Violated", "Unevaluable"], rows)
        + _table(["Requests"], overall) + "</section>"
    )


def _token_section(capacity: Mapping[str, Any] | None) -> str:
    throughput = _object(capacity.get("token_throughput") if capacity is not None else None)
    rows = []
    for name in ("input", "output", "total"):
        component = _object(throughput.get(name))
        rows.append([
            name, component.get("tokens"), component.get("tokens_per_s"),
            component.get("observed_tokens"), component.get("observed_tokens_per_s"),
            component.get("covered_requests"), component.get("successful_requests"),
            component.get("coverage_ratio"),
        ])
    scope = throughput.get("scope_request_counts")
    sources = throughput.get("source_request_counts")
    provenance = _table(["Request counts"], [
        ["scope", json.dumps(scope, sort_keys=True) if isinstance(scope, dict) else None],
        ["source", json.dumps(sources, sort_keys=True) if isinstance(sources, dict) else None],
    ])
    return (
        "<section><h2>Token throughput over result.duration_s</h2>"
        + _table(["Tokens", "Tokens/s", "Observed tokens", "Observed tokens/s",
                  "Covered requests", "Successful requests", "Coverage ratio"], rows)
        + provenance + "</section>"
    )


def _latency_section(streaming: Any) -> str:
    streaming = _object(streaming)
    rows = []
    for name in STREAMING_LATENCY_METRICS:
        distribution = _object(streaming.get(name))
        rows.append([name, *[distribution.get(key) for key in ("mean", "p50", "p90", "p95", "p99", "sample_count")]])
    return (
        "<section><h2>Streaming latency distribution</h2>"
        + _table(["Mean", "p50", "p90", "p95", "p99", "Samples"], rows) + "</section>"
    )


def render_load_report(data: Mapping[str, Any]) -> str:
    experiment = data["experiment"]
    result = data["result"]
    version = data["version"]
    capacity = data["capacity"]
    content = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>AC-Prof single-point load capacity</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;"
        "color:#172033}table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}"
        "th,td{border:1px solid #ccd3df;padding:.45rem;text-align:left}th{background:#f2f5f9}"
        ".boundary{padding:1rem;background:#fff4ce;border:1px solid #d6b656}</style></head><body>"
        "<h1>AC-Prof single-point load capacity</h1>"
        "<p class=\"boundary\">One offered load point only. This is not a capacity curve or a maximum-capacity estimate.</p>"
        + _table(["Value"], [
            ["run_id", experiment.get("run_id")], ["schema_version", version],
            ["experiment_status", experiment.get("status")], ["request_successful", result.get("successful")],
        ])
        + _service_section(experiment, version)
        + _capacity_section(result, capacity)
        + _slo_section(capacity)
        + _token_section(capacity)
        + _latency_section(data.get("streaming"))
        + "</body></html>\n"
    )
    return content


def write_load_report(source: str | Path, output: str | Path) -> Path:
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"output must be a new file: {destination}")
    document = render_load_report(read_load_report(source))

    def write(stream: TextIO) -> None:
        stream.write(document)

    atomic_write(destination, write, overwrite=False)
    return destination
