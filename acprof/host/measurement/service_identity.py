"""Declared and observed service identity, kept outside the load window."""
from __future__ import annotations

import http.client
import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from acprof.artifacts import loads_finite_json, reject_isolated_surrogates

MAX_PROBE_RESPONSE_BYTES = 1024 * 1024
SERVICE_PROBES = ("none", "vllm", "sglang")
_SERVICE_FIELDS = ("backend", "version", "model", "revision")
_SGLANG_CONFIGURATION_TYPES = {
    **dict.fromkeys(("model_path", "served_model_name", "revision", "dtype", "kv_cache_dtype",
                     "quantization", "schedule_policy"), str),
    **dict.fromkeys(("tp_size", "pp_size", "dp_size", "attn_dp_size", "context_length",
                     "max_running_requests", "max_total_tokens", "max_prefill_tokens"), int),
    "disable_radix_cache": bool,
}


class _ProbeHTTPError(ValueError):
    def __init__(self, path: str, status: int) -> None:
        super().__init__(f"GET {path} returned HTTP {status}")
        self.status = status


def initial_service_identity(declared: Mapping[str, Any], probe: str) -> dict[str, Any]:
    if probe not in SERVICE_PROBES:
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


def _read_json(
    origin: str, path: str, headers: Mapping[str, str], timeout_s: float, *,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = urlsplit(origin)
    hostname = target.hostname
    if target.scheme not in {"http", "https"} or not hostname:
        raise ValueError("service probe requires an HTTP(S) origin")
    factory = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    connection = factory(hostname, target.port, timeout=timeout_s)
    response: http.client.HTTPResponse | None = None
    try:
        connection.request("GET", path, headers={**headers, "Connection": "close"})
        received = connection.getresponse()
        response = received
        if observation is not None:
            observation["http_status"] = received.status
        if received.status != 200:
            raise _ProbeHTTPError(path, received.status)
        raw = received.read(MAX_PROBE_RESPONSE_BYTES + 1)
        if len(raw) > MAX_PROBE_RESPONSE_BYTES:
            raise ValueError(f"GET {path} exceeded the 1 MiB response limit")
        payload = loads_finite_json(raw.decode("utf-8"))
        reject_isolated_surrogates(payload)
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


def _observed_identity(
    backend: str, version: str | None, models: list[dict[str, Any]] | None,
    declared: Mapping[str, Any],
) -> dict[str, Any]:
    matched_models = [model for model in models or [] if model["id"] == declared.get("model")]
    matched = matched_models[0] if len(matched_models) == 1 else None
    return {
        "backend": backend if version is not None else None, "version": version,
        "model": matched["id"] if matched is not None else None, "revision": None,
        "root": matched["root"] if matched is not None else None,
        "max_model_len": matched["max_model_len"] if matched is not None else None,
        "models": models,
    }


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
        observed = _observed_identity("vllm", version, models, declared)
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


def _read_recorded(
    origin: str, path: str, headers: Mapping[str, str], timeout_s: float, attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    attempt: dict[str, Any] = {"path": path, "http_status": None}
    attempts.append(attempt)
    try:
        payload = _read_json(origin, path, headers, timeout_s, observation=attempt)
    except Exception as error:
        attempt["error"] = f"{type(error).__name__}: {error}"
        raise
    return payload


def _sglang_configuration(payload: Mapping[str, Any], errors: dict[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, expected in _SGLANG_CONFIGURATION_TYPES.items():
        value = payload.get(name)
        if value is None:
            valid = True
        elif expected is str:
            valid = isinstance(value, str) and bool(value.strip())
        elif expected is bool:
            valid = type(value) is bool
        else:
            valid = type(value) is int and value > 0
        values[name] = value if valid else None
        if not valid:
            kind = "nonempty text" if expected is str else "boolean" if expected is bool else "positive integer"
            errors[f"configuration.{name}"] = f"ValueError: {name} must be {kind} or null"
    return values


def probe_sglang(
    origin: str, declared: Mapping[str, Any], *, headers: Mapping[str, str] | None = None,
    timeout_s: float = 10,
) -> dict[str, Any]:
    """Observe SGLang identity and whitelisted startup values, with explicit legacy routing."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("service probe timeout must be finite and positive")
    result = initial_service_identity(declared, "sglang")
    errors: dict[str, str] = {}
    attempts: list[dict[str, Any]] = []
    info: dict[str, Any] | None = None
    models: list[dict[str, Any]] | None = None
    info_path = "/server_info"
    try:
        try:
            info = _read_recorded(origin, info_path, headers or {}, timeout_s, attempts)
        except _ProbeHTTPError as error:
            if error.status != 404:
                raise
            info_path = "/get_server_info"
            info = _read_recorded(origin, info_path, headers or {}, timeout_s, attempts)
    except Exception as error:
        errors["server_info"] = f"{type(error).__name__}: {error}"
    try:
        models = _models(_read_recorded(origin, "/v1/models", headers or {}, timeout_s, attempts))
    except Exception as error:
        errors["models"] = f"{type(error).__name__}: {error}"
    observed: dict[str, Any] | None = None
    if info is not None or models is not None:
        version = info.get("version") if info is not None else None
        if info is not None and (not isinstance(version, str) or not version.strip()):
            errors["version"] = "ValueError: version must be nonempty text"
            version = None
        observed = _observed_identity("sglang", version, models, declared)
        observed["configuration"] = None if info is None else {
            "source": info_path, "scope": "reported_startup_configuration",
            "values": _sglang_configuration(info, errors),
        }
    status = "available" if not errors else "unavailable" if info is None and models is None else "partial"
    result["observed"] = observed
    result["probe"] = {"kind": "sglang", "status": status, "errors": errors, "requests": attempts}
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
    if probe == "sglang":
        return probe_sglang(origin, declared, headers=headers, timeout_s=timeout_s)
    raise ValueError("unsupported service probe")
