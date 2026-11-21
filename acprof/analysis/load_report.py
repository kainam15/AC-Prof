"""Shared offline validation for stream_load_experiment v1/v2/v3."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from acprof.analysis.load_inputs import validate_dataset_inputs
from acprof.analysis.server_metrics import validate_server_metrics
from acprof.analysis.streaming import streaming_summary
from acprof.analysis.streaming_capacity import build_capacity
from acprof.artifacts import read_json_object, require_schema_version
from acprof.experiments.streaming import nontext_chat_output, prepare_streaming_payload


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
    require_schema_version(declaration, 1, "stream load service declaration")
    require_schema_version(service, 1, "stream load service_identity")
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


def _validate_chat_output(experiment: Mapping[str, Any], identity: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    declared = identity.get("adapter")
    text_only = isinstance(declared, dict) and "output_modalities" in declared
    if text_only and experiment.get("schema_version") == 2:
        body = identity.get("request_body")
        if not isinstance(body, dict):
            raise ValueError("stream load requires the declared Chat request body")
        prepare_streaming_payload(body, endpoint=identity.get("endpoint", ""), recorded_adapter=declared)
    for phase in (result, experiment.get("warmup")):
        if not isinstance(phase, Mapping):
            continue
        protocol = phase.get("protocol")
        observed = protocol.get("adapter") if isinstance(protocol, Mapping) else None
        if not text_only and not (isinstance(observed, dict) and "output_modalities" in observed):
            continue
        if _canonical(observed) != _canonical(declared):
            raise ValueError("recorded Chat output adapter differs from the declared adapter")
        rows = phase.get("requests")
        if not isinstance(rows, list):
            raise ValueError("recorded Chat output requires request records")
        for row in rows:
            if isinstance(row, dict) and row.get("status") == "ok":
                reason = nontext_chat_output(row.get("finish_reason"), row.get("server_usage"))
                if reason:
                    raise ValueError(f"successful Chat request contradicts recorded output: {reason}")


def _v2_capacity(experiment: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    identity = experiment.get("identity")
    if not isinstance(identity, dict) or "slo_thresholds_s" not in identity:
        raise ValueError("stream load v2 requires declared identity and SLO thresholds")
    expected_identity = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if experiment.get("identity_sha256") != expected_identity:
        raise ValueError("stream load identity_sha256 does not match declared inputs")
    _validate_chat_output(experiment, identity, result)
    _validate_v2_service_declaration(experiment, identity)
    protocol = result.get("protocol")
    requests = result.get("requests")
    if not isinstance(protocol, dict) or not isinstance(requests, list):
        raise ValueError("stream load v2 requires protocol and raw requests")
    load_config = identity.get("load_config")
    if not isinstance(load_config, dict):
        raise ValueError("stream load v2 requires a declared load configuration")
    concurrency = load_config.get("concurrency")
    if not isinstance(concurrency, int) or protocol.get("concurrency") != concurrency:
        raise ValueError("stream load declared and observed client concurrency differ")
    # Compare only declared fields, preserving JSON types and explicit null values.
    protocol_config = {key: protocol[key] for key in load_config if key in protocol}
    if _canonical(protocol_config) != _canonical(load_config):
        raise ValueError("stream load declared configuration differs from result protocol")
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


def _validate_completion(
    experiment: Mapping[str, Any], result: Mapping[str, Any], all_requests_succeeded: bool,
) -> None:
    state = experiment.get("status")
    if not isinstance(state, str) or state not in {"running", "complete", "failed", "cancelled"}:
        raise ValueError("invalid stream load experiment status")
    successful = result.get("successful")
    if type(successful) is not bool or (successful and not all_requests_succeeded):
        raise ValueError("stream load result success contradicts its requests")
    if state == "complete" and successful != all_requests_succeeded:
        raise ValueError("completed stream load success contradicts its requests")
    # Cleanup or cancellation may fail after all requests have succeeded.
    expected = state == "complete" and successful
    if type(experiment.get("successful")) is not bool or experiment["successful"] != expected:
        raise ValueError("stream load experiment success contradicts its result")


def _validate_summary(
    experiment: Mapping[str, Any], result: Mapping[str, Any], capacity: Mapping[str, Any],
) -> None:
    counts = capacity["counts"]
    if counts["offered"] < 1 or _canonical(result.get("counts")) != _canonical(counts):
        raise ValueError("stream load counts are inconsistent with raw requests")
    config = experiment["identity"]["load_config"]
    if "requests" in config and (type(config["requests"]) is not int or config["requests"] != counts["offered"]):
        raise ValueError("stream load request budget differs from retained records")
    expected_rates: dict[str, float] = {
        "success_rate": counts["succeeded"] / counts["offered"],
        "successful_requests_per_s": capacity["successful_requests_per_s"],
    }
    if "sent_requests_per_s" in result:
        expected_rates["sent_requests_per_s"] = counts["sent"] / capacity["duration_s"]
    if "completed_requests_per_s" in result:
        expected_rates["completed_requests_per_s"] = sum(
            row.get("stream_completed_s") is not None for row in result["requests"]
        ) / capacity["duration_s"]
    for name, expected in expected_rates.items():
        value = result.get(name)
        message = f"stream load {name} is inconsistent with raw requests or duration"
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(message)
        # Compare against a finite derived rate without converting untrusted integers to float.
        if not math.isfinite(expected) or value != expected:
            raise ValueError(message)
    _validate_completion(experiment, result, counts["succeeded"] == counts["offered"])


def _first_warmup_failure(warmup: Any) -> str | None:
    records = warmup.get("requests") if isinstance(warmup, dict) else None
    if not isinstance(records, list):
        return None
    for index, row in enumerate(records):
        if not isinstance(row, dict) or row.get("status") not in ("error", "cancelled", "dropped"):
            continue
        code = row.get("error_code")
        code = code if isinstance(code, str) and code else row["status"]
        status = row.get("http_status")
        http = f" (HTTP {status})" if type(status) is int and 100 <= status <= 599 else ""
        return f"warmup.requests[{index}]: {code}{http}"
    return None


def load_failure_diagnostics(experiment: Mapping[str, Any]) -> list[str]:
    """Describe recorded terminal failures without using warmup to derive load metrics."""
    if experiment.get("status") not in ("failed", "cancelled"):
        return []
    error = experiment.get("error")
    messages: list[str] = []
    phases = experiment.get("phase_errors")
    for phase in ("warmup", "result"):
        entry = phases.get(phase) if isinstance(phases, dict) else None
        if not isinstance(entry, dict):
            continue
        kind, detail = entry.get("exception_type"), entry.get("detail")
        if not isinstance(kind, str) or not kind or not isinstance(detail, str):
            continue
        cause = f"{kind}: {detail}"
        # Outer cleanup may have appended notes after the phase recorded the cause.
        if isinstance(error, str) and (error == cause or error.startswith(cause + "\n")):
            cause, error = error, None
        messages.append(f"{phase}: {cause}")
    if isinstance(error, str) and error.strip():
        messages.insert(0, error)
    warmup = _first_warmup_failure(experiment.get("warmup"))
    if warmup:
        messages.append(warmup)
    return messages


def read_load_report(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    path = path / "load.json" if path.is_dir() else path
    if path.name != "load.json":
        raise ValueError("capacity report input must be load.json or its containing directory")
    experiment = read_json_object(path, label="stream load experiment", bounded=False)
    return analyze_load_report(experiment)


def analyze_load_report(experiment: dict[str, Any]) -> dict[str, Any]:
    """Validate an already-read point without loading its raw requests again."""
    version = experiment.get("schema_version")
    if (
        experiment.get("kind") != "stream_load_experiment"
        or type(version) is not int
        or version not in {1, 2, 3}
    ):
        raise ValueError("capacity report supports only stream_load_experiment schema v1/v2/v3")
    result = experiment.get("result")
    if not isinstance(result, dict) or result.get("kind") != "stream_load_result":
        message = "stream load experiment has no stream_load_result"
        if "result" not in experiment:
            message = "\n".join([message, *load_failure_diagnostics(experiment)])
        raise ValueError(message)
    require_schema_version(result, version, "stream load result")
    if version == 3:
        validate_dataset_inputs(experiment)
    elif ((isinstance(experiment.get("identity"), dict) and "request_dataset" in experiment["identity"])
          or (isinstance(result.get("protocol"), dict) and "request_selection" in result["protocol"])):
        raise ValueError("request datasets require stream_load_experiment schema v3")
    capacity = _v2_capacity(experiment, result) if version >= 2 else None
    streaming = _v2_streaming(result) if version >= 2 else result.get("streaming")
    if capacity is not None:
        _validate_summary(experiment, result, capacity)
    server_metrics = validate_server_metrics(experiment)
    return {
        "experiment": experiment, "result": result, "capacity": capacity,
        "streaming": streaming, "version": version, "server_metrics": server_metrics,
    }
