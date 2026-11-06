"""Validate ordered request inputs without accessing the original dataset file."""
from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from acprof.artifacts import require_schema_version
from acprof.experiments.load import LoadConfig
from acprof.experiments.streaming import (
    prepare_streaming_payload,
    request_digest,
    validate_request_credentials,
)


def _source(source: Any, count: int) -> None:
    if not isinstance(source, dict):
        raise ValueError("request dataset requires source provenance")
    if not isinstance(source.get("path"), str) or not source["path"]:
        raise ValueError("request dataset source path must be nonempty text")
    digest = source.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("request dataset source SHA256 is invalid")
    if type(source.get("rows")) is not int or source["rows"] < count:
        raise ValueError("request dataset source row count is invalid")
    lines = source.get("lines")
    if (not isinstance(lines, list) or len(lines) != count
            or any(type(line) is not int or line < 1 for line in lines)
            or lines != sorted(set(lines))):
        raise ValueError("request dataset source lines must be increasing positive integers")


def _bodies(identity: dict[str, Any], needed: int) -> list[dict[str, Any]]:
    dataset = identity.get("request_dataset")
    if not isinstance(dataset, dict) or dataset.get("selection") != "listed_prefix":
        raise ValueError("stream load v3 requires a listed_prefix request dataset")
    if "request_body" in identity or "request_body_sha256" in identity:
        raise ValueError("request dataset cannot also declare a repeated request body")
    bodies = dataset.get("bodies")
    if not isinstance(bodies, list) or len(bodies) != needed:
        raise ValueError("request dataset must retain exactly max(requests, warmup) bodies")
    if dataset.get("sha256") != request_digest(bodies):
        raise ValueError("request dataset digest differs from ordered bodies")
    _source(dataset.get("source"), len(bodies))
    endpoint = identity.get("endpoint")
    if not isinstance(endpoint, str):
        raise ValueError("request dataset endpoint must be text")
    for body in bodies:
        normalized = prepare_streaming_payload(body, endpoint=endpoint)
        validate_request_credentials(body)
        if request_digest(normalized) != request_digest(body) or body["model"] != identity.get("model"):
            raise ValueError("request dataset body is not normalized or differs from the served model")
    return bodies


def _phase(phase: Any, bodies: list[dict[str, Any]], config: LoadConfig, token: str, adapter: dict[str, Any]) -> None:
    if not isinstance(phase, dict) or phase.get("kind") != "stream_load_result":
        raise ValueError("request dataset phase requires a streaming result")
    require_schema_version(phase, 3, "request dataset phase")
    protocol = phase.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("request_selection") != "listed_prefix":
        raise ValueError("request dataset phase selection differs")
    if request_digest({key: protocol.get(key) for key in asdict(config)}) != request_digest(asdict(config)):
        raise ValueError("request dataset phase configuration differs")
    if protocol.get("kind") != "openai_stream_http_load" or request_digest(protocol.get("adapter")) != request_digest(adapter):
        raise ValueError("request dataset phase adapter differs from declared input")
    if (protocol.get("request_inputs_sha256") != request_digest(bodies[:config.requests])
            or phase.get("protocol_sha256") != request_digest(protocol)):
        raise ValueError("request dataset phase input or protocol digest differs")
    records = phase.get("requests")
    if not isinstance(records, list) or len(records) != config.requests:
        raise ValueError("request dataset phase records differ from request count")
    for index, record in enumerate(records):
        if (not isinstance(record, dict) or type(record.get("input_index")) is not int
                or record["input_index"] != index or record.get("request_id") != f"{token}:{index}"):
            raise ValueError("request dataset record input_index or request_id differs from scheduled order")


def validate_dataset_inputs(experiment: dict[str, Any]) -> None:
    identity = experiment.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("load_config"), dict):
        raise ValueError("stream load v3 requires load configuration")
    try:
        config = LoadConfig(**identity["load_config"])
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid request dataset configuration: {error}") from error
    warmup = identity.get("warmup_requests")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("request dataset warmup count must be a nonnegative integer")
    if identity.get("warmup_scenario") != "serial" or identity.get("warmup_connections") != "close":
        raise ValueError("request dataset warmup must declare serial closed connections")
    adapter = identity.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("endpoint") != identity.get("endpoint"):
        raise ValueError("request dataset adapter endpoint differs")
    run_id = experiment.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("request dataset experiment requires a run ID")
    bodies = _bodies(identity, max(config.requests, warmup))
    _phase(experiment.get("result"), bodies, config, run_id, adapter)
    if warmup:
        _phase(experiment.get("warmup"), bodies, LoadConfig(requests=warmup, timeout_seconds=config.timeout_seconds), "warmup", adapter)
    elif "warmup" in experiment:
        raise ValueError("request dataset declares zero warmup but contains warmup records")
