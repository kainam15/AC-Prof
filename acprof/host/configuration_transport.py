"""Bind one frozen inference configuration to each isolated runtime invocation."""
from __future__ import annotations

import json
from typing import Any

from acprof.failures import Failure, RuntimeFailure
from acprof.models.inference_configuration import (
    configuration_device,
    declared_generation_consumer,
    encode_inference_configuration,
    validate_configuration_binding,
)
from acprof.tasks.workloads import EMBEDDING_INTERFACES, generation_length_resets

CONFIGURATION_ENV = "ACPROF_INFERENCE_CONFIGURATION"
TASK_ENV = "ACPROF_INFERENCE_TASK_ID"
_GENERATION_OBSERVATION_SOURCES = {"generate.kwargs", "generate.generation_config", "model.generation_config"}


def _same_json_value(expected: Any, actual: Any) -> bool:
    try:
        return json.dumps(expected, sort_keys=True, allow_nan=False) == json.dumps(
            actual, sort_keys=True, allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        return False


def bound_inference_configuration(task_info: Any, *, configuration: dict | None = None) -> dict:
    """Read the frozen object without deriving new defaults or changing its identity."""
    frozen = getattr(task_info, "inference_configuration", {}) if configuration is None else configuration
    if type(frozen) is not dict:
        raise ValueError("inference_configuration must be an object")
    if frozen:
        validate_configuration_binding(frozen, task_info, allow_task_subset=True)
    return frozen


def inference_environment(task_info: Any, *, configuration: dict | None = None) -> dict[str, str]:
    """Reuse frozen values; historical tasks without a configuration remain readable."""
    frozen = bound_inference_configuration(task_info, configuration=configuration)
    if not frozen:
        return {}
    identifiers = {
        item["task_id"] for item in task_info.selected_tasks
        if item["interface"] == task_info.pipeline_tag
    }
    if len(identifiers) != 1:
        raise ValueError("select exactly one frozen inference task before starting its runtime")
    identifier = next(iter(identifiers))
    if identifier not in frozen["tasks"]:
        raise ValueError("selected inference task is absent from the frozen configuration")
    return {CONFIGURATION_ENV: encode_inference_configuration(frozen), TASK_ENV: identifier}


def inference_docker_env_args(task_info: Any, *, configuration: dict | None = None) -> list[str]:
    return [part for name, value in inference_environment(task_info, configuration=configuration).items()
            for part in ("-e", f"{name}={value}")]


def _mismatched_embedding_parameters(plan: dict, expected: dict) -> str | None:
    workload = plan.get("workload")
    if not isinstance(workload, dict) or not _same_json_value(expected, workload.get("params", {})):
        return "workload.params"
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        return "entries"
    for index, entry in enumerate(entries):
        payload = entry.get("payload") if isinstance(entry, dict) else None
        if not isinstance(payload, dict) or not _same_json_value(expected, payload.get("params", {})):
            return f"entries[{index}].payload.params"
    return None


def verify_embedding_plan_parameters(task_info: Any, plan: dict, frozen: dict) -> None:
    """Check each request against the encoder options in the approved workload."""
    workload = frozen.get("workload", {})
    if "parameter_defaults" not in workload or task_info.pipeline_tag not in EMBEDDING_INTERFACES:
        return
    field = _mismatched_embedding_parameters(plan, workload["parameter_defaults"])
    if field:
        raise RuntimeFailure(Failure(
            "input_plan", "inference_configuration_mismatch",
            f"Input plan {field} differs from the confirmed sentence embedding parameters",
            runtime_profile=task_info.runtime_profile_id,
            evidence={"identity_sha256": frozen["identity_sha256"], "field": field},
        ))


def verify_plan_configuration(task_info: Any, plan: dict) -> dict:
    """Reject a workload plan prepared for a different experiment configuration."""
    frozen = bound_inference_configuration(task_info)
    expected = frozen.get("identity_sha256")
    observed = plan.get("inference_configuration_sha256")
    if (frozen or observed is not None) and observed != expected:
        raise RuntimeFailure(Failure(
            "input_plan", "inference_configuration_mismatch",
            "Input plan does not belong to the frozen inference configuration",
            runtime_profile=task_info.runtime_profile_id,
            evidence={"expected": expected, "observed": observed},
        ))
    verify_embedding_plan_parameters(task_info, plan, frozen)
    return frozen


def verify_runtime_configuration(task_info: Any, report: dict, *, expected_device: str | None = None,
                                 require_consumers: bool = False, configuration: dict | None = None) -> None:
    """Require independent observations to refer to exactly the frozen configuration."""
    frozen = bound_inference_configuration(task_info, configuration=configuration)
    if not frozen:
        return
    expected = frozen["identity_sha256"]
    if expected_device is None:
        raise RuntimeFailure(Failure(
            "runtime_validation", "inference_configuration_unverified",
            "Expected device is required to verify a frozen inference configuration",
            runtime_profile=task_info.runtime_profile_id,
        ))
    expected_mode = configuration_device(expected_device)
    active_task_id = inference_environment(task_info, configuration=frozen)[TASK_ENV]

    def mismatch(field: str, actual: Any, required: Any) -> RuntimeFailure:
        return RuntimeFailure(Failure(
            "runtime_validation", "inference_configuration_mismatch",
            f"Runtime {field} differs from the active frozen configuration binding",
            device=expected_mode, runtime_profile=task_info.runtime_profile_id,
            evidence={"identity_sha256": expected, "field": field, "expected": required, "observed": actual},
        ))

    def unverified(field: str, actual: Any, *, status: str = "verified") -> RuntimeFailure:
        return RuntimeFailure(Failure(
            "runtime_validation", "inference_configuration_unverified",
            f"Runtime did not provide {status} {field} for the frozen inference configuration",
            device=expected_mode, runtime_profile=task_info.runtime_profile_id,
            evidence={"identity_sha256": expected, "report_field": field, "observed": actual},
        ))

    def verify_parameter(field: str, observation: Any, value: Any, *, loading: bool = False,
                         sources: set[str] | None = None) -> None:
        if (not isinstance(observation, dict) or observation.get("status") != "verified"
                or "expected" not in observation or "observed" not in observation):
            raise unverified(field, observation)
        if not _same_json_value(value, observation["expected"]):
            raise mismatch(f"{field}.expected", observation["expected"], value)
        required = [value] if loading else value
        if not _same_json_value(required, observation["observed"]):
            raise mismatch(f"{field}.observed", observation["observed"], required)
        if not loading:
            source = observation.get("source")
            if not isinstance(source, str) or source not in (sources or _GENERATION_OBSERVATION_SOURCES):
                raise unverified(f"{field}.source", source)

    def verify_device(field: str, actual: Any) -> None:
        try:
            actual_mode = configuration_device(actual)
        except ValueError as exc:
            raise mismatch(field, actual, expected_mode) from exc
        if actual_mode != expected_mode:
            raise mismatch(field, actual, expected_mode)

    if expected_mode not in frozen["loading"]:
        raise mismatch("device", expected_mode, sorted(frozen["loading"]))
    observed = report.get("inference_configuration_sha256") if isinstance(report, dict) else None
    if observed != expected:
        raise RuntimeFailure(Failure(
            "runtime_validation", "inference_configuration_mismatch",
            "Runtime did not report the frozen inference configuration identity",
            runtime_profile=task_info.runtime_profile_id,
            evidence={"expected": expected, "observed": observed},
        ))
    if "device" in report:
        verify_device("device", report["device"])
    requirements = {"configuration_validation": "verified"}
    consumer_declared = False
    inference = frozen["tasks"][active_task_id].get("inference", {})
    consumer_specs = {}
    if inference:
        from acprof.models.task_configuration import task_consumer_specs
        consumer_specs = task_consumer_specs(
            family=task_info.task_family, backend=task_info.runtime_backend,
            adapter=task_info.model_adapter, task=task_info.pipeline_tag, values=inference,
        )
    if require_consumers:
        consumer_declared = declared_generation_consumer(
            family=task_info.task_family, backend=task_info.runtime_backend,
            adapter=task_info.model_adapter, task=task_info.pipeline_tag,
        )
        consumer_declared = consumer_declared or bool(consumer_specs)
        requirements["consumer_receipt"] = "verified" if consumer_declared else "not_applicable"
    for field, status in requirements.items():
        evidence = report.get(field)
        if not isinstance(evidence, dict) or evidence.get("status") != status:
            raise unverified(field, evidence, status=status)
        valid = True
        if field == "configuration_validation":
            for key, required in (("inference_configuration_sha256", expected), ("task_id", active_task_id)):
                if evidence.get(key) != required:
                    raise mismatch(f"{field}.{key}", evidence.get(key), required)
            verify_device(f"{field}.device", evidence.get("device"))
            loading = evidence.get("loading")
            if not isinstance(loading, dict):
                raise unverified(f"{field}.loading", loading)
            expected_loading = frozen["loading"][expected_mode]
            for name, value in expected_loading.items():
                verify_parameter(f"{field}.loading.{name}", loading.get(name), value, loading=True)
            for name in {"dtype", "attention_implementation"} - expected_loading.keys():
                observation = loading.get(name)
                if (not isinstance(observation, dict) or observation.get("status") != "not_applicable"
                        or observation.get("reason") != "not_declared"):
                    raise unverified(f"{field}.loading.{name}", observation, status="not_applicable")
        if field == "consumer_receipt":
            valid = (evidence.get("inference_configuration_sha256") == expected
                     and evidence.get("task_id") == active_task_id)
            if valid and consumer_declared:
                calls = evidence.get("calls")
                valid = isinstance(calls, list) and bool(calls)
                if isinstance(calls, list) and calls:
                    generation = inference or frozen["tasks"][active_task_id]["generation"]
                    resets = {} if inference else generation_length_resets(generation)
                    for index, call in enumerate(calls):
                        path = f"{field}.calls[{index}].parameters"
                        parameters = call.get("parameters") if isinstance(call, dict) else None
                        if not isinstance(parameters, dict):
                            raise unverified(path, parameters)
                        for name, value in generation.items():
                            verify_parameter(f"{path}.{name}", parameters.get(name), value,
                                sources={consumer_specs[name].validation_method} if consumer_specs else None)
                        for name in resets:
                            verify_parameter(f"{path}.{name}", parameters.get(name), None, sources={"generate.kwargs"})
            elif valid:
                generation = evidence.get("generation")
                valid = (isinstance(generation, dict) and generation.get("status") == "not_applicable"
                         and generation.get("reason") == "consumer_not_declared" and evidence.get("calls") == [])
        if not valid:
            raise unverified(field, evidence, status=status)
