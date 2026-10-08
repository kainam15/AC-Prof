"""Resolve the registered runtime policy before any model loader is invoked."""

from __future__ import annotations

import json
import os
from pathlib import Path

from acprof.extensions import CATALOG
from acprof.failures import Failure, RuntimeFailure
from acprof.precision import remote_code_allowed, resolve_precision
from acprof.runtime_profiles import DEFAULT_PROFILES, PROFILES, locked_transformers_version

# The environment image installs this exact lock; it does not ship project dockerfiles/.
CONTAINER_REQUIREMENTS_LOCK = Path("/opt/acprof/requirements.lock")


def registered_policy(task, backend, device, load_options=None):
    family = CATALOG.task_families[task]
    adapter = os.getenv("ACPROF_MODEL_ADAPTER", "family-default")
    extension = CATALOG.get_extension(family, backend, adapter, task)
    profile_id = os.getenv("ACPROF_RUNTIME_PROFILE") or extension.profile or DEFAULT_PROFILES[(family, "cpu" if device == "cpu" else "cu128")]
    profile = PROFILES[profile_id]
    if profile.family != family or profile.adapter != adapter:
        raise ValueError(f"runtime profile {profile_id} does not match {family}/{adapter}")
    options = dict(load_options or {})
    trust = remote_code_allowed(profile, extension)
    requested_trust = options.get("trust_remote_code", trust)
    if type(requested_trust) is not bool or requested_trust and not trust:
        raise RuntimeFailure(Failure("load", "remote_code_disallowed", "load options cannot enable remote code forbidden by the registered policy",
                                     device, profile_id))
    return profile, extension, requested_trust


def load_policy(model_source, task, backend, device, load_options=None):
    import torch

    profile, extension, trust = registered_policy(task, backend, device, load_options)
    config_file = Path(model_source) / "config.json"
    config = json.loads(config_file.read_text()) if config_file.is_file() else {}
    options = dict(load_options or {})
    version = ""
    if backend in {"transformers_model", "transformers_pipeline", "sentence_transformers", "cross_encoder"}:
        import transformers
        version = transformers.__version__
        image_lock = CONTAINER_REQUIREMENTS_LOCK if os.getenv("ACPROF_DEPENDENCY_LOCK_SHA256") else None
        expected = locked_transformers_version(profile.environment, lock_path=image_lock)
        if version != expected:
            raise RuntimeFailure(Failure("load", "runtime_dependency_incompatible",
                f"Runtime profile requires transformers=={expected}; installed version is {version}",
                device, profile.profile_id, "after_configuration",
                {"distribution": "transformers", "required": expected, "installed": version}))
    precision = resolve_precision(profile, extension, device=device, task=task,
                                  model_type=config.get("model_type", ""), version=version, requested=options.get("dtype"))
    return {"dtype": getattr(torch, precision["torch_dtype"]), "trust_remote_code": trust,
            "precision": precision, "runtime_profile": profile.profile_id}


def load_processor(loader, *args, **kwargs):
    from acprof.failures import failure_from_exception
    try:
        return loader(*args, **kwargs)
    except Exception as exc:
        failure = failure_from_exception(exc, stage="processor", runtime_profile=os.getenv("ACPROF_RUNTIME_PROFILE", ""))
        if failure.reason_code == "runtime_initialization_failed":
            from dataclasses import replace
            failure = replace(failure, reason_code="processor_incompatible")
        raise RuntimeFailure(failure) from exc


def actual_dtype(context):
    model = context.get("model")
    if model is None:
        pipeline = context.get("pipeline")
        model = getattr(pipeline, "model", pipeline)
    if callable(getattr(model, "parameters", None)):
        observed = sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()})
        if observed:
            return observed[0] if len(observed) == 1 else "mixed[" + ", ".join(observed) + "]"
    dtype = getattr(model, "dtype", None)
    return str(dtype) if dtype is not None else "unknown"
