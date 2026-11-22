"""Frozen configuration binding and observations outside measured inference.

Loading observations describe floating parameters and model configuration. They
do not establish storage precision, compute precision or executed attention kernels.
Consumer receipts are collected only by the independent validation entrypoint.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from acprof.failures import Failure, RuntimeFailure
from acprof.models.inference_configuration import (
    configuration_load_options,
    configuration_task,
    declared_generation_consumer,
    validate_frozen_configuration,
)
from acprof.tasks.workloads import generation_length_resets

_DTYPE_NAMES = {
    "torch.float32": "FP32", "torch.float16": "FP16",
    "torch.bfloat16": "BF16", "torch.float64": "FP64",
}
_MISSING = object()


def _frozen(config) -> dict:
    return getattr(config, "inference_configuration", {}) or {}


def _failure(config, device: str, detail: str, *, unverified: bool = False,
             stage: str = "configuration", evidence: dict | None = None) -> RuntimeFailure:
    frozen = _frozen(config)
    return RuntimeFailure(Failure(
        stage, "inference_configuration_unverified" if unverified else "inference_configuration_mismatch",
        detail, device, frozen.get("backend", {}).get("runtime_profile", ""),
        "after_configuration", {
            "inference_configuration_sha256": frozen.get("identity_sha256", ""),
            **(evidence or {}),
        },
    ))


def bound_load_options(config, device: str, load_options: dict | None) -> dict | None:
    """Validate the active binding and reject an override of confirmed values."""
    frozen = _frozen(config)
    variant = getattr(config, "model_variant", "")
    if not frozen:
        if variant and not variant.endswith("/"):
            return {**(load_options or {}), "variant": variant}
        return load_options
    from acprof.container.load_policy import registered_policy

    try:
        validate_frozen_configuration(frozen)
        task = configuration_task(frozen, config.inference_task_id)
        expected = configuration_load_options(frozen, device)
        for name, actual in (("model_id", config.model_id), ("revision", config.model_revision)):
            if frozen["model"][name] != actual:
                raise ValueError(f"frozen model.{name} differs from the active model")
        if variant != frozen["model"].get("variant", ""):
            raise ValueError("frozen checkpoint variant differs from the active model")
        for name, actual in (("id", config.backend), ("adapter", config.adapter)):
            if frozen["backend"][name] != actual:
                raise ValueError(f"frozen backend.{name} differs from the active backend")
        if task["interface"] != config.task_type:
            raise ValueError("frozen task interface differs from the active task")
        profile, extension, _trust = registered_policy(
            config.task_type, config.backend, device, expected, adapter=config.adapter,
        )
        if frozen["backend"]["runtime_profile"] != profile.profile_id:
            raise ValueError("frozen runtime profile differs from the registered active profile")
        if frozen["backend"]["extension_id"] != extension.extension_id:
            raise ValueError("frozen extension differs from the registered active extension")
        if declared_generation_consumer(
            family=config.task_family, backend=config.backend, adapter=config.adapter, task=config.task_type,
        ):
            if set(expected) != {"dtype", "attention_implementation"}:
                raise ValueError("native NLP generation requires frozen dtype and attention_implementation")
            if ("do_sample" not in task["generation"]
                    or not task["generation"].keys() & {"max_length", "max_new_tokens"}):
                raise ValueError("native NLP generation requires frozen length and sampling configuration")
        if load_options is not None:
            if type(load_options) is not dict or any(
                key not in expected or type(value) is not type(expected[key]) or value != expected[key]
                for key, value in load_options.items()
            ):
                raise ValueError("load options differ from the frozen configuration; derive a separate profiler configuration")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, RuntimeFailure):
            raise
        raise _failure(config, device, str(exc)) from exc
    if variant and not variant.endswith("/"):
        expected["variant"] = variant
    return expected


def attach_configuration_context(config, device: str, context: dict) -> None:
    """Copy receipt expectations into the session without mutating the frozen object."""
    frozen = _frozen(config)
    if not frozen:
        return
    context["_inference_configuration_sha256"] = frozen["identity_sha256"]
    if declared_generation_consumer(
        family=config.task_family, backend=config.backend, adapter=config.adapter, task=config.task_type,
    ):
        context["_frozen_generation"] = configuration_task(frozen, config.inference_task_id)["generation"]
    context["_inference_task_id"] = config.inference_task_id
    context["_configuration_device"] = device
    inference = configuration_task(frozen, config.inference_task_id).get("inference", {})
    if inference:
        from acprof.models.task_configuration import task_consumer_specs
        task_consumer_specs(family=config.task_family, backend=config.backend, adapter=config.adapter,
                            task=config.task_type, values=inference)
        context["_frozen_inference"] = inference


def _model(context: dict):
    model = context.get("model")
    if model is None:
        pipeline = context.get("pipeline")
        model = getattr(pipeline, "model", pipeline)
    return model


def _parameter_dtypes(model) -> list[str]:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return []
    return sorted({
        _DTYPE_NAMES.get(str(parameter.dtype), str(parameter.dtype))
        for parameter in parameters() if parameter.is_floating_point()
    })


def _attention_implementations(model) -> list[str]:
    values = set()
    model_config = getattr(model, "config", None)
    for name in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
        value = getattr(model_config, name, None)
        if isinstance(value, str) and value:
            values.add(value)
        elif isinstance(value, dict):
            values.update(str(item) for item in value.values() if item is not None)
    return sorted(values)


def verify_model_session_configuration(session) -> dict:
    """Observe all floating parameters and configured attention after load timing."""
    config, context, device = session.config, session.context, session.device
    frozen = _frozen(config)
    if not frozen:
        return {}
    context.pop("configuration_validation", None)
    try:
        expected = configuration_load_options(frozen, device)
    except (TypeError, ValueError) as exc:
        raise _failure(config, device, str(exc)) from exc
    if context.get("_inference_configuration_sha256") != frozen["identity_sha256"]:
        raise _failure(config, device, "loaded session is not bound to this frozen configuration")
    loading: dict[str, dict[str, Any]] = {}
    model = _model(context)
    observations: dict[str, tuple[Callable[[Any], list[str]], str, str]] = {
        "dtype": (_parameter_dtypes, "floating_parameter_dtypes", "model_parameters"),
        "attention_implementation": (_attention_implementations, "model_config", "configured_attention"),
    }
    if "scheduler" in expected:
        observations["scheduler"] = (
            lambda _: [type(context["pipeline"].scheduler).__name__],
            "scheduler.class", "pipeline_scheduler",
        )
    for field, (observe, method, scope) in observations.items():
        if field not in expected:
            loading[field] = {"status": "not_applicable", "reason": "not_declared"}
            continue
        try:
            actual = observe(model)
        except Exception as exc:
            raise _failure(config, device, f"could not observe {field}: {type(exc).__name__}",
                           unverified=True, evidence={"field": field, "expected": expected[field]}) from exc
        status = "unverified" if not actual else "verified" if actual == [expected[field]] else "mismatch"
        loading[field] = {
            "status": status, "expected": expected[field], "observed": actual,
            "method": method, "scope": scope,
        }
        if status != "verified":
            raise _failure(
                config, device, f"runtime {field} does not verify the frozen value",
                unverified=status == "unverified", evidence={"loading": loading},
            )
    report = {
        "status": "verified", "inference_configuration_sha256": frozen["identity_sha256"],
        "task_id": config.inference_task_id, "device": device, "loading": loading,
    }
    context["configuration_validation"] = report
    return copy.deepcopy(report)


def configuration_metadata(session) -> dict:
    """Return cached observations; never scan a model in an HTTP request."""
    frozen = _frozen(session.config)
    if not frozen:
        return {}
    report = session.context.get("configuration_validation", {})
    if (report.get("status") != "verified"
            or report.get("inference_configuration_sha256") != frozen["identity_sha256"]):
        raise _failure(session.config, session.device,
                       "configuration must be verified before the service becomes ready", unverified=True)
    return {
        "inference_configuration_sha256": frozen["identity_sha256"],
        "configuration_validation": copy.deepcopy(report),
    }


def _same_value(expected: Any, actual: Any) -> bool:
    try:
        return json.dumps(expected, sort_keys=True, allow_nan=False) == json.dumps(
            actual, sort_keys=True, allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        return False


def validate_generation_parameters(context: dict, params: Any) -> None:
    """Reject request drift instead of adding or replacing frozen generation values."""
    if "_frozen_generation" not in context:
        return
    expected = context["_frozen_generation"]
    if not _same_value(expected, params):
        raise RuntimeFailure(Failure(
            "configuration_input", "inference_configuration_mismatch",
            "request generation parameters differ from the frozen task",
            context.get("_configuration_device", "unknown"), retryability="after_configuration",
            evidence={
                "inference_configuration_sha256": context["_inference_configuration_sha256"],
                "task_id": context["_inference_task_id"], "expected": copy.deepcopy(expected),
                "actual_parameter_names": sorted(params) if isinstance(params, dict) else None,
            },
        ))


def validate_inference_payload(session, payload: dict) -> None:
    if _frozen(session.config):
        validate_generation_parameters(session.context, payload.get("params", {}))
        expected = session.context.get("_frozen_inference")
        if expected is not None:
            actual = payload if session.config.task_family == "timeseries" else payload.get("params", {})
            if not isinstance(actual, dict) or any(
                name not in actual or not _same_value(value, actual[name]) for name, value in expected.items()
            ):
                raise _failure(session.config, session.device, "request parameters differ from the frozen task",
                               stage="configuration_input")


def _observed_generation_value(model, kwargs: dict, name: str):
    if name in kwargs:
        return kwargs[name], "generate.kwargs"
    generation_config = kwargs.get("generation_config")
    if generation_config is not None and hasattr(generation_config, name):
        return getattr(generation_config, name), "generate.generation_config"
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, name):
        return getattr(generation_config, name), "model.generation_config"
    return _MISSING, "unobserved"


def _task_consumer(session, expected):
    from acprof.models.task_configuration import task_consumer_specs
    config = session.config
    specs = task_consumer_specs(family=config.task_family, backend=config.backend, adapter=config.adapter,
                                task=config.task_type, values=expected)
    model = _model(session.context)
    pipeline = session.context.get("pipeline")
    for spec in specs.values():
        if spec.model_types and getattr(getattr(model, "config", None), "model_type", None) not in spec.model_types:
            raise _failure(config, session.device, "loaded model does not support the declared task parameters")
        if spec.pipeline_classes and type(pipeline).__name__ not in spec.pipeline_classes:
            raise _failure(config, session.device, "loaded pipeline does not support the declared task parameters")
    method = next(iter(specs.values())).validation_method
    mapping = {name: spec.consumers[0].argument.rsplit(".", 1)[-1] for name, spec in specs.items()}
    if method == "handler.synthesize.kwargs":
        return session.handler, "synthesize", mapping, method
    target, operation = {
        "generate.kwargs": (model, "generate"),
        "pipeline.predict.kwargs": (pipeline, "predict"),
        "pipeline.call.kwargs": (pipeline, "__call__"),
    }[method]
    return target, operation, mapping, method


class _ObservedPipeline:
    """Intercept only this session's call; Python resolves __call__ on the type."""
    def __init__(self, pipeline, call):
        self._pipeline, self._call = pipeline, call

    def __getattr__(self, name):
        return getattr(self._pipeline, name)

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)


@contextmanager
def capture_generation_consumption(session):
    """Temporarily observe the declared consumer in independent validation."""
    frozen = _frozen(session.config)
    receipt: dict[str, Any] = {}
    if not frozen:
        yield receipt
        return
    expected = session.context.get("_frozen_inference", session.context.get("_frozen_generation"))
    receipt.update({
        "status": "pending", "inference_configuration_sha256": frozen["identity_sha256"],
        "task_id": session.config.inference_task_id, "calls": [],
    })
    if expected is None:
        receipt.update(status="not_applicable", generation={"status": "not_applicable", "reason": "consumer_not_declared"})
        yield receipt
        return
    resets: dict[str, None] = {}
    if "_frozen_inference" in session.context:
        model, operation, mapping, method = _task_consumer(session, expected)
    else:
        resets = generation_length_resets(expected)
        expected = {**expected, **resets}
        model, operation, method = _model(session.context), "generate", "generate.kwargs"
        mapping = {name: name for name in expected}
    generate = getattr(model, operation, None)
    consumer = "model.generate" if operation == "generate" else method.removesuffix(".kwargs")
    if not callable(generate):
        raise _failure(session.config, session.device, f"no observable {consumer} consumer",
                       unverified=True, stage="configuration_consumption")
    previous = getattr(model, "__dict__", {}).get(operation, _MISSING)

    def observed_generate(*args, **kwargs):
        parameters = {}
        for name, value in expected.items():
            argument = mapping[name]
            actual, source = (_observed_generation_value(model, kwargs, argument) if operation == "generate"
                              else (kwargs.get(argument, _MISSING), method))
            status = "unverified" if actual is _MISSING else "verified" if _same_value(value, actual) else "mismatch"
            if name in resets and actual is None and source != "generate.kwargs":
                # Transformers can still fill a config-level None from model defaults.
                status = "unverified"
            observation = {"status": status, "expected": copy.deepcopy(value), "source": source}
            if actual is not _MISSING:
                try:
                    observation["observed"] = json.loads(json.dumps(actual, allow_nan=False))
                except (TypeError, ValueError, RecursionError):
                    observation["observed_type"] = type(actual).__name__
            parameters[name] = observation
        call = {"parameters": parameters}
        receipt["calls"].append(call)
        failed = [value for value in parameters.values() if value["status"] != "verified"]
        if failed:
            raise _failure(
                session.config, session.device, f"{consumer} parameters do not match the frozen task",
                unverified=all(value["status"] == "unverified" for value in failed),
                stage="configuration_consumption", evidence={"consumer_receipt": copy.deepcopy(receipt)},
            )
        return generate(*args, **kwargs)

    try:
        if operation == "__call__":
            session.context["pipeline"] = _ObservedPipeline(model, observed_generate)
        else:
            setattr(model, operation, observed_generate)
    except (AttributeError, TypeError) as exc:
        raise _failure(session.config, session.device, f"{consumer} cannot be observed",
                       unverified=True, stage="configuration_consumption") from exc
    try:
        yield receipt
    finally:
        if operation == "__call__":
            session.context["pipeline"] = model
        elif previous is _MISSING:
            delattr(model, operation)
        else:
            setattr(model, operation, previous)


def verify_generation_consumption(session, receipt: dict) -> dict:
    if not _frozen(session.config):
        return {}
    if (receipt.get("status") == "not_applicable" and "_frozen_generation" not in session.context
            and "_frozen_inference" not in session.context):
        return copy.deepcopy(receipt)
    if not receipt.get("calls"):
        consumer = "declared task consumer" if "_frozen_inference" in session.context else "model.generate"
        raise _failure(session.config, session.device, f"inference did not reach {consumer}",
                       unverified=True, stage="configuration_consumption")
    if any(observation["status"] != "verified" for call in receipt["calls"]
           for observation in call["parameters"].values()):
        raise _failure(session.config, session.device, "failed generation consumption cannot be retried implicitly",
                       stage="configuration_consumption", evidence={"consumer_receipt": copy.deepcopy(receipt)})
    receipt["status"] = "verified"
    return copy.deepcopy(receipt)
