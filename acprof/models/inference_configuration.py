"""Immutable inference configuration identities shared by host and runtimes.

Resolved values are confirmed by content digest. Runtime observations are stored
separately and never rewrite this object or authorize another automatic choice.
"""
from __future__ import annotations

import base64
import binascii
import copy
import json
import re
from typing import Any

from acprof.artifacts import loads_finite_json
from acprof.extensions.parameters import ParameterRequest
from acprof.models.evidence import content_digest, pinned_revision

CONFIGURATION_SCHEMA_VERSION = 1
CONFIGURATION_RULE_VERSION = "acprof-inference-v1"
MAX_CONFIGURATION_BYTES = 65536
_CONFIGURATION_FIELDS = {
    "schema_version", "state", "identity_sha256", "rule_version", "model", "backend",
    "loading", "tasks", "requests", "decisions", "workload", "validation", "derivation",
}


def _object(value: Any, path: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{path} must be an object")
    return value


def _text(value: Any, path: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _digest(value: Any, path: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{path} must be a SHA256 digest")
    return value


def _json_bytes(value: Any) -> bytes:
    ParameterRequest("explicit", value)  # Shared finite JSON and depth validation.
    data = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_CONFIGURATION_BYTES:
        raise ValueError(f"inference configuration exceeds {MAX_CONFIGURATION_BYTES} bytes")
    return data


def configuration_identity(configuration: dict) -> str:
    """Bind values, user intent, sources and rules; exclude only lifecycle state."""
    _object(configuration, "inference_configuration")
    body = {key: value for key, value in configuration.items()
            if key not in {"identity_sha256", "state"}}
    _json_bytes(body)
    return content_digest(body)


def _validate_configuration(configuration: dict, *, states: set[str]) -> None:
    _object(configuration, "inference_configuration")
    _json_bytes(configuration)
    unknown = set(configuration) - _CONFIGURATION_FIELDS
    required = _CONFIGURATION_FIELDS - {"derivation"}
    if unknown or required - configuration.keys():
        raise ValueError("inference_configuration has unknown or missing fields: "
                         f"{sorted(unknown | (required - configuration.keys()))}")
    if type(configuration["schema_version"]) is not int or configuration["schema_version"] != CONFIGURATION_SCHEMA_VERSION:
        raise ValueError("unsupported inference_configuration.schema_version")
    if type(configuration["state"]) is not str or configuration["state"] not in states:
        raise ValueError(f"inference_configuration.state must be one of {sorted(states)}")
    if configuration["rule_version"] != CONFIGURATION_RULE_VERSION:
        raise ValueError("unsupported inference_configuration.rule_version")
    model = _object(configuration["model"], "model")
    for key in ("source", "model_id"):
        _text(model.get(key), f"model.{key}")
    if not pinned_revision(model.get("revision")):
        raise ValueError("model.revision must be a fixed 40-character commit")
    backend = _object(configuration["backend"], "backend")
    for key in ("id", "adapter", "extension_id", "runtime_profile"):
        _text(backend.get(key), f"backend.{key}")
    loading = _object(configuration["loading"], "loading")
    if not loading or set(loading) - {"off", "on"}:
        raise ValueError("loading must name the selected off/on devices")
    for device, values in loading.items():
        _object(values, f"loading.{device}")
        if set(values) - {"dtype", "attention_implementation"}:
            raise ValueError(f"loading.{device} contains undeclared loading options")
        if "dtype" in values and (type(values["dtype"]) is not str or values["dtype"] not in {"FP32", "FP16", "BF16", "FP64"}):
            raise ValueError(f"loading.{device}.dtype is not a registered precision")
        if "attention_implementation" in values and (type(values["attention_implementation"]) is not str
                or values["attention_implementation"] not in {"eager", "sdpa"}):
            raise ValueError(f"loading.{device}.attention_implementation must be a concrete registered implementation")
    tasks = _object(configuration["tasks"], "tasks")
    if not tasks:
        raise ValueError("inference_configuration.tasks must not be empty")
    for task_id, task in tasks.items():
        _text(task_id, "tasks key")
        _object(task, f"tasks.{task_id}")
        _text(task.get("interface"), f"tasks.{task_id}.interface")
        _digest(task.get("fingerprint"), f"tasks.{task_id}.fingerprint")
        _object(task.get("generation"), f"tasks.{task_id}.generation")
        if "snapshot" in task:
            from acprof.tasks.workloads import task_fingerprint
            snapshot = _object(task["snapshot"], f"tasks.{task_id}.snapshot")
            if not {"task_id", "interface", "kind", "config"} <= snapshot.keys() or type(snapshot["config"]) is not dict:
                raise ValueError(f"tasks.{task_id}.snapshot is incomplete")
            if (snapshot.get("task_id") != task_id or snapshot.get("interface") != task["interface"]
                    or task_fingerprint(snapshot) != task["fingerprint"]
                    or snapshot.get("config", {}).get("generation", {}) != task["generation"]):
                raise ValueError(f"tasks.{task_id} does not match its frozen task snapshot")
    _object(configuration["requests"], "requests")
    _object(configuration["workload"], "workload")
    if type(configuration["decisions"]) is not list:
        raise ValueError("decisions must be an array")
    if configuration["validation"] != {"static": "validated", "runtime": "not_run"}:
        raise ValueError("frozen configuration must keep runtime observations in a separate report")
    if "derivation" in configuration:
        derivation = _object(configuration["derivation"], "derivation")
        if set(derivation) != {"parent_identity_sha256", "tool", "reason", "loading_overrides"}:
            raise ValueError("derivation must describe its parent, tool, reason and loading overrides")
        _digest(derivation["parent_identity_sha256"], "derivation.parent_identity_sha256")
        _text(derivation["tool"], "derivation.tool")
        _text(derivation["reason"], "derivation.reason")
        _object(derivation["loading_overrides"], "derivation.loading_overrides")
    recorded = _digest(configuration["identity_sha256"], "identity_sha256")
    if recorded != configuration_identity(configuration):
        raise ValueError("inference_configuration identity_sha256 does not match its contents")


def validate_frozen_configuration(configuration: dict) -> None:
    """Reject malformed, mutated or unconfirmed execution configuration."""
    _validate_configuration(configuration, states={"frozen"})


def freeze_inference_configuration(configuration: dict, confirmation_digest: str) -> dict:
    """Confirm exactly the presented candidate; do not resolve or change values."""
    _validate_configuration(configuration, states={"resolved"})
    if confirmation_digest != configuration["identity_sha256"]:
        raise ValueError("inference configuration confirmation is stale; review the current values")
    frozen = copy.deepcopy(configuration)
    frozen["state"] = "frozen"
    validate_frozen_configuration(frozen)
    return frozen


def validate_configuration_binding(configuration: dict, task_info: Any, *,
                                   allow_task_subset: bool = False) -> None:
    validate_frozen_configuration(configuration)
    model, backend = configuration["model"], configuration["backend"]
    for key, attribute in (("source", "model_source"), ("model_id", "model_id"), ("revision", "model_revision")):
        if model[key] != getattr(task_info, attribute, None):
            raise ValueError(f"inference_configuration.model.{key} differs from the active model")
    for key, attribute in (("id", "runtime_backend"), ("adapter", "model_adapter"),
                           ("runtime_profile", "runtime_profile_id")):
        if backend[key] != getattr(task_info, attribute, None):
            raise ValueError(f"inference_configuration.backend.{key} differs from the active backend")
    selected = getattr(task_info, "selected_tasks", ())
    identifiers = {item["task_id"] for item in selected}
    frozen_ids = set(configuration["tasks"])
    if not identifiers or (not allow_task_subset and identifiers != frozen_ids) or identifiers - frozen_ids:
        raise ValueError("inference_configuration tasks differ from the selected tasks")
    from acprof.tasks.workloads import task_fingerprint
    for task in selected:
        expected = configuration["tasks"][task["task_id"]]
        if (task.get("fingerprint") != expected["fingerprint"]
                or task_fingerprint(task) != expected["fingerprint"]
                or task["interface"] != expected["interface"]
                or task["config"].get("generation", {}) != expected["generation"]):
            raise ValueError(f"inference_configuration.tasks.{task['task_id']} differs from the active task")


def declared_generation_consumer(*, family: str, backend: str, adapter: str, task: str) -> bool:
    """Whether the registered Handler has an observable generate boundary."""
    from acprof.extensions import get_extension

    extension = get_extension(family, backend, adapter=adapter, task=task)
    return (family == "nlp" and backend in {"transformers_model", "transformers_pipeline"}
            and extension.handler_entrypoint == "acprof.container.handlers.nlp:NLPHandler"
            and any(spec.scope == "inference" and any(consumer.stage == "predict" for consumer in spec.consumers)
                    for spec in extension.parameters))


def configuration_device(device: str) -> str:
    if type(device) is not str:
        raise ValueError("inference configuration device must be a string")
    if device in {"off", "cpu"}:
        return "off"
    if device in {"on", "gpu", "cuda"} or re.fullmatch(r"cuda:[0-9]+", device):
        return "on"
    raise ValueError(f"unsupported inference configuration device: {device!r}")


def configuration_load_options(configuration: dict, device: str) -> dict:
    if not configuration:
        return {}
    validate_frozen_configuration(configuration)
    key = configuration_device(device)
    if key not in configuration["loading"]:
        raise ValueError(f"device {key} is absent from the frozen inference configuration")
    return copy.deepcopy(configuration["loading"][key])


def configuration_task(configuration: dict, task_id: str) -> dict:
    validate_frozen_configuration(configuration)
    if task_id not in configuration["tasks"]:
        raise ValueError(f"task {task_id!r} is absent from the frozen inference configuration")
    return copy.deepcopy(configuration["tasks"][task_id])


def encode_inference_configuration(configuration: dict) -> str:
    if not configuration:
        return ""
    validate_frozen_configuration(configuration)
    return base64.urlsafe_b64encode(_json_bytes(configuration)).decode("ascii")


def decode_inference_configuration(encoded: str) -> dict:
    if encoded == "":
        return {}
    if type(encoded) is not str or len(encoded) > ((MAX_CONFIGURATION_BYTES + 2) // 3) * 4:
        raise ValueError("encoded inference configuration is invalid or too large")
    try:
        raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if len(raw) > MAX_CONFIGURATION_BYTES:
            raise ValueError("decoded inference configuration is too large")
        configuration = loads_finite_json(raw.decode("utf-8"))
    except (ValueError, UnicodeError, binascii.Error) as exc:
        raise ValueError("invalid encoded inference configuration") from exc
    validate_frozen_configuration(configuration)
    return configuration


def derive_inference_configuration(configuration: dict, *, loading_overrides: dict[str, dict],
                                   reason: str, tool: str) -> dict:
    """Create a separately identified profiler variant without changing its parent."""
    validate_frozen_configuration(configuration)
    if "derivation" in configuration:
        raise ValueError("nested inference configuration derivations are not supported")
    _text(reason, "derivation.reason")
    _text(tool, "derivation.tool")
    _object(loading_overrides, "derivation.loading_overrides")
    derived = copy.deepcopy(configuration)
    for device, options in loading_overrides.items():
        if device not in derived["loading"]:
            raise ValueError(f"derived device {device!r} is absent from the parent configuration")
        _object(options, f"derivation.loading_overrides.{device}")
        if set(options) - set(derived["loading"][device]):
            raise ValueError("profiler derivation cannot introduce undeclared loading options")
        derived["loading"][device].update(copy.deepcopy(options))
    derived["derivation"] = {
        "parent_identity_sha256": configuration["identity_sha256"], "tool": tool, "reason": reason,
        "loading_overrides": copy.deepcopy(loading_overrides),
    }
    derived["identity_sha256"] = configuration_identity(derived)
    validate_frozen_configuration(derived)
    return derived
