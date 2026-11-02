"""Declared and observed service identity, kept outside the load window."""
from __future__ import annotations

import http.client
import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from acprof.artifacts import loads_finite_json

MAX_PROBE_RESPONSE_BYTES = 1024 * 1024
_SERVICE_FIELDS = ("backend", "version", "model", "revision")


def initial_service_identity(declared: Mapping[str, Any], probe: str) -> dict[str, Any]:
    if probe not in {"none", "vllm"}:
        raise ValueError("unsupported service probe")
    return {
        "schema_version": 1,
        "declared": {name: declared.get(name) for name in _SERVICE_FIELDS},
        "observed": None,
        "probe": {
            "kind": probe,
            "status": "not_requested" if probe == "none" else "pending",
            "errors": {},
        },
        "matches": dict.fromkeys(_SERVICE_FIELDS),
        "mismatches": [],
    }


def _read_json(origin: str, path: str, headers: Mapping[str, str], timeout_s: float) -> dict[str, Any]:
    target = urlsplit(origin)
    if target.scheme not in {"http", "https"} or not target.hostname:
        raise ValueError("service probe requires an HTTP(S) origin")
    factory = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    connection = factory(target.hostname, target.port, timeout=timeout_s)
    response: http.client.HTTPResponse | None = None
    try:
        connection.request("GET", path, headers={**headers, "Connection": "close"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"GET {path} returned HTTP {response.status}")
        raw = response.read(MAX_PROBE_RESPONSE_BYTES + 1)
        if len(raw) > MAX_PROBE_RESPONSE_BYTES:
            raise ValueError(f"GET {path} exceeded the 1 MiB response limit")
        payload = loads_finite_json(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"GET {path} must return a JSON object")
        return payload
    finally:
        if response is not None:
            response.close()
        connection.close()


def _models(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("GET /v1/models must contain a data list")
    models = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("GET /v1/models entries must be objects")
        model = {name: item.get(name) for name in ("id", "root", "max_model_len")}
        if any(model[name] is not None and not isinstance(model[name], str) for name in ("id", "root")):
            raise ValueError("model id/root must be text or null")
        if model["max_model_len"] is not None and (
            type(model["max_model_len"]) is not int or model["max_model_len"] < 1
        ):
            raise ValueError("max_model_len must be a positive integer or null")
        models.append(model)
    return models


def _matches(declared: Mapping[str, Any], observed: Mapping[str, Any] | None, model_list_known: bool) -> dict[str, bool | None]:
    matches: dict[str, bool | None] = {}
    for name in _SERVICE_FIELDS:
        left = declared.get(name)
        right = observed.get(name) if observed is not None else None
        matches[name] = left == right if left is not None and right is not None else None
    if observed is not None and model_list_known:
        matches["model"] = observed.get("model") is not None
    return matches


def probe_vllm(
    origin: str, declared: Mapping[str, Any], *, headers: Mapping[str, str] | None = None,
    timeout_s: float = 10,
) -> dict[str, Any]:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("service probe timeout must be finite and positive")
    result = initial_service_identity(declared, "vllm")
    errors: dict[str, str] = {}
    version_payload: dict[str, Any] | None = None
    models: list[dict[str, Any]] | None = None
    for name, path in (("version", "/version"), ("models", "/v1/models")):
        try:
            payload = _read_json(origin, path, headers or {}, timeout_s)
            if name == "version":
                version_payload = payload
            else:
                models = _models(payload)
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    observed: dict[str, Any] | None = None
    if version_payload is not None or models is not None:
        version = version_payload.get("version") if version_payload is not None else None
        if version_payload is not None and (not isinstance(version, str) or not version):
            errors["version"] = "ValueError: version must be nonempty text"
            version = None
        matched_models = [model for model in models or [] if model["id"] == declared.get("model")]
        matched = matched_models[0] if len(matched_models) == 1 else None
        observed = {
            "backend": "vllm",
            "version": version,
            "model": matched["id"] if matched is not None else None,
            "revision": None,
            "root": matched["root"] if matched is not None else None,
            "max_model_len": matched["max_model_len"] if matched is not None else None,
            "models": models,
        }
    if not errors:
        status = "available"
    elif version_payload is None and models is None:
        status = "unavailable"
    else:
        status = "partial"
    result.update(observed=observed)
    result["probe"] = {"kind": "vllm", "status": status, "errors": errors}
    result["matches"] = _matches(declared, observed, models is not None)
    result["mismatches"] = [name for name, matched in result["matches"].items() if matched is False]
    return result


def probe_service(
    origin: str, declared: Mapping[str, Any], probe: str, *, headers: Mapping[str, str] | None = None,
    timeout_s: float = 10,
) -> dict[str, Any]:
    if probe == "none":
        return initial_service_identity(declared, probe)
    if probe == "vllm":
        return probe_vllm(origin, declared, headers=headers, timeout_s=timeout_s)
    raise ValueError("unsupported service probe")
