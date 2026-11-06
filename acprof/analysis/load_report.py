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
        raise ValueError("stream load experiment has no stream_load_result")
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
