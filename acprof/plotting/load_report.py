"""Offline single-point capacity report for stream_load_experiment v1/v2/v3."""
from __future__ import annotations

import json
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any, TextIO

from acprof.analysis.load_report import read_load_report as read_load_report
from acprof.analysis.streaming import STREAMING_LATENCY_METRICS
from acprof.analysis.streaming_capacity import SLO_METRICS
from acprof.artifacts import atomic_write


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
    return (
        "<div class=\"table-scroll\"><table><thead><tr><th scope=\"col\">Metric</th>"
        + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def _service_configuration(observed: Mapping[str, Any]) -> str:
    configuration = _object(observed.get("configuration"))
    if not configuration:
        return ""
    values = _object(configuration.get("values"))
    return (
        "<h3>Reported server startup configuration</h3>"
        + _table(["Value"], [["source", configuration.get("source")], ["scope", configuration.get("scope")]])
        + _table(["Value"], [[name, value] for name, value in values.items()])
    )


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
    mismatches = service.get("mismatches") if version >= 2 else None
    mismatch_text = ", ".join(mismatches) if isinstance(mismatches, list) and mismatches else (
        "none" if mismatches == [] else None
    )
    errors = _object(probe.get("errors"))
    diagnostics = (
        "<h3>Service probe errors</h3>" + _table(["Error"], [[name, value] for name, value in errors.items()])
        if errors else ""
    )
    return (
        "<section><h2>Service identity</h2>"
        + _table(["Declared", "Observed"], rows)
        + _table(["Value"], [["probe", probe.get("status")], ["mismatches", mismatch_text]])
        + _service_configuration(observed)
        + diagnostics
        + "</section>"
    )


def _workload_section(experiment: Mapping[str, Any]) -> str:
    identity = _object(experiment.get("identity"))
    dataset = _object(identity.get("request_dataset"))
    if not dataset:
        return ""
    source = _object(dataset.get("source"))
    return (
        "<section><h2>Request workload</h2>"
        + _table(["Value"], [["selection", dataset.get("selection")], ["source", source.get("path")],
                            ["source rows", source.get("rows")], ["source SHA256", source.get("sha256")],
                            ["prepared input SHA256", dataset.get("sha256")],
                            ["measured requests", _object(identity.get("load_config")).get("requests")],
                            ["warmup requests", identity.get("warmup_requests")]])
        + "<p>Warmup and measurement each start at input index 0. Inputs follow file order without automatic repetition.</p></section>"
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


def _diagnostics_section(streaming: Any) -> str:
    streaming = _object(streaming)
    errors = streaming.get("errors")
    if not isinstance(errors, dict):
        error_content = "<p>unavailable</p>"
    elif not errors:
        error_content = "<p>No request errors recorded.</p>"
    else:
        error_content = _table(["Requests"], [[code, count] for code, count in sorted(errors.items())])
    return (
        "<section><h2>Request diagnostics</h2>"
        + _table(["Requests"], [
            ["excluded_requests", streaming.get("excluded_requests")],
            ["token_count_errors", streaming.get("token_count_errors")],
        ])
        + "<p>Latency distributions include successful requests only. "
        "Token count errors are counted among successful requests.</p>"
        + "<h3>Request error codes</h3>" + error_content + "</section>"
    )


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
    coverage = []
    for name in (*STREAMING_LATENCY_METRICS, "output_event_interval_s"):
        distribution = _object(streaming.get(name))
        rows.append([name, *[distribution.get(key) for key in ("mean", "p50", "p90", "p95", "p99", "sample_count")]])
        sample_unit = distribution.get("sample_unit")
        requests_with_samples = distribution.get(
            "sample_count" if sample_unit == "request" else "requests_with_samples"
        )
        coverage.append([
            name, distribution.get("unit"), sample_unit,
            requests_with_samples, distribution.get("missing_requests"),
        ])
    return (
        "<section><h2>Streaming latency distribution</h2>"
        + _table(["Mean", "p50", "p90", "p95", "p99", "Samples"], rows)
        + "<h3>Sample units and coverage</h3>"
        + _table(["Unit", "Sample unit", "Requests with samples", "Missing requests"], coverage)
        + "</section>"
    )


def _server_metrics_section(data: Mapping[str, Any]) -> str:
    metrics = data.get("server_metrics")
    if metrics is None:
        return ""
    observation = data["experiment"]["server_metrics"]
    rows = [[phase, row["status"], row["count"], row["sum_s"], row["mean_s"], row["reason"]]
            for phase, row in metrics["phases"].items()]
    snapshots = [[name, observation[name]["status"], observation[name]["started_s"],
                  observation[name]["finished_s"], observation[name]["error"]]
                 for name in ("before", "after")]
    return (
        "<section><h2>Server phase timing</h2><p class=\"boundary\">"
        + _cell(observation["scope"]) + "</p>"
        + _table(["Status", "Observations", "Total (s)", "Mean (s)", "Unavailable reason"], rows)
        + _table(["Status", "Scrape start (s)", "Scrape end (s)", "Error"], snapshots)
        + "<p>Scrape times are relative to the observation start, outside the client load duration. "
        "Means use server histogram counts. Warmup or earlier requests may appear in delayed updates; "
        "current requests may not yet be exported. Two scrapes cannot detect every hidden counter reset.</p></section>"
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
        "color:#172033}.table-scroll{max-width:100%;overflow-x:auto}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}"
        "th,td{border:1px solid #ccd3df;padding:.45rem;text-align:left}th{background:#f2f5f9}"
        ".boundary{padding:1rem;background:#fff4ce;border:1px solid #d6b656}</style></head><body>"
        "<h1>AC-Prof single-point load capacity</h1>"
        "<p class=\"boundary\">One offered load point only. This is not a capacity curve or a maximum-capacity estimate.</p>"
        + _table(["Value"], [
            ["run_id", experiment.get("run_id")], ["schema_version", version],
            ["experiment_status", experiment.get("status")], ["request_successful", result.get("successful")],
        ])
        + _service_section(experiment, version)
        + _workload_section(experiment)
        + _server_metrics_section(data)
        + _capacity_section(result, capacity)
        + _diagnostics_section(data.get("streaming"))
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
