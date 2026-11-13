"""Runtime loading evidence must bind its actual device, active task and configuration."""
from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.failures import RuntimeFailure
from acprof.host.configuration_transport import verify_runtime_configuration
from acprof.host.input_plan.materialize import write_scale_plan_file
from acprof.host.runtime_validation import RuntimeValidationError
from acprof.host.runtimes import validation
from acprof.models.inference_configuration import configuration_identity
from tests.support.inference_configuration_fixture import frozen_task, verified_runtime_report


def test_frozen_runtime_observation_requires_an_expected_device():
    task = frozen_task()

    with pytest.raises(RuntimeFailure, match="Expected device is required") as error:
        verify_runtime_configuration(task, verified_runtime_report(task))

    assert error.value.failure.reason_code == "inference_configuration_unverified"


@pytest.mark.parametrize("source", ["generate.kwargs", "generate.generation_config", "model.generation_config"])
def test_each_runtime_generate_observation_source_is_accepted(source):
    task = frozen_task()
    report = verified_runtime_report(task)
    for observation in report["consumer_receipt"]["calls"][0]["parameters"].values():
        observation["source"] = source

    verify_runtime_configuration(task, report, expected_device="off", require_consumers=True)


@pytest.mark.parametrize("change", [None, "loading", "consumer", "calls"])
def test_undeclared_route_records_not_applicable_without_claiming_observations(change):
    from acprof.tasks.workloads import parse_task

    task = frozen_task()
    selected = parse_task("fill-mask", {}, encoder_decoder=False, source="fixture")
    task.pipeline_tag = "fill-mask"
    task.selected_tasks = [selected]
    configuration = task.inference_configuration
    configuration["loading"] = {"off": {}}
    configuration["requests"] = {}
    configuration["tasks"] = {selected["task_id"]: {
        "interface": selected["interface"], "fingerprint": selected["fingerprint"], "generation": {},
    }}
    configuration["identity_sha256"] = configuration_identity(configuration)
    report = verified_runtime_report(task)
    report["configuration_validation"]["loading"] = {
        name: {"status": "not_applicable", "reason": "not_declared"}
        for name in ("dtype", "attention_implementation")
    }
    report["consumer_receipt"].update(
        status="not_applicable", generation={"status": "not_applicable", "reason": "consumer_not_declared"}, calls=[],
    )
    if change is None:
        verify_runtime_configuration(task, report, expected_device="off", require_consumers=True)
        return
    if change == "loading":
        report["configuration_validation"]["loading"]["dtype"]["status"] = "verified"
    elif change == "consumer":
        report["consumer_receipt"]["generation"]["status"] = "verified"
    else:
        report["consumer_receipt"]["calls"] = [{"parameters": {}}]

    with pytest.raises(RuntimeFailure) as error:
        verify_runtime_configuration(task, report, expected_device="off", require_consumers=True)

    assert error.value.failure.reason_code == "inference_configuration_unverified"


@pytest.mark.parametrize(("expected", "observed"), [
    ("off", "cpu"), ("cpu", "off"), ("on", "cuda:0"), ("cuda:2", "gpu"),
])
def test_device_aliases_are_normalized_before_binding(expected, observed):
    task = frozen_task()
    task.inference_configuration["loading"]["on"] = copy.deepcopy(task.inference_configuration["loading"]["off"])
    task.inference_configuration["identity_sha256"] = configuration_identity(task.inference_configuration)
    report = verified_runtime_report(task)
    report["device"] = observed
    report["configuration_validation"]["device"] = observed

    verify_runtime_configuration(task, report, expected_device=expected, require_consumers=True)


@pytest.mark.parametrize("change", [
    "loading_missing", "loading_status", "loading_expected", "loading_observed",
    "consumer_empty_parameters", "consumer_missing_parameter", "consumer_status", "consumer_expected",
    "consumer_observed", "consumer_source", "consumer_second_call",
])
def test_process_rejects_incomplete_or_contradictory_parameter_evidence(tmp_path, monkeypatch, change):
    task = frozen_task()
    path = tmp_path / "input_scale_plan.json"
    write_scale_plan_file(str(path), task, [{"input_scale": 1, "payload": {"text": "fixture"}}])
    report = verified_runtime_report(task)
    loading = report["configuration_validation"]["loading"]
    calls = report["consumer_receipt"]["calls"]
    parameters = calls[0]["parameters"]
    if change == "loading_missing":
        loading.pop("attention_implementation")
    elif change.startswith("loading_"):
        key = change.removeprefix("loading_")
        loading["dtype"][key] = {"status": "unverified", "expected": "BF16", "observed": ["FP16"]}[key]
    elif change == "consumer_empty_parameters":
        calls[0]["parameters"] = {}
    elif change == "consumer_missing_parameter":
        parameters.pop("max_new_tokens")
    elif change == "consumer_second_call":
        calls.append({"parameters": {}})
    else:
        key = change.removeprefix("consumer_")
        parameters["max_new_tokens"][key] = {
            "status": "unverified", "expected": 11, "observed": 7.0, "source": "model_card",
        }[key]
    report["stages"] = [{"stage": stage, "status": "verified"} for stage in
                        ("execution", "load", "preprocess", "predict", "completion", "postprocess", "validate_output")]
    report["validation"] = {"protocol": {"status": "verified"}, "task": {"status": "verified"}}
    monkeypatch.setattr(validation, "run_validation", Mock(return_value=report))

    with pytest.raises(RuntimeValidationError) as error:
        validation.validate_process(
            task_info=task, image_info=SimpleNamespace(environment_id="fixture"),
            planned=SimpleNamespace(plan_file=str(path)), cpu_list=[1], mem_list=[4],
            gpu_list=["off"], output_dir=str(tmp_path),
        )

    assert error.value.failure.reason_code in {"inference_configuration_mismatch", "inference_configuration_unverified"}


@pytest.mark.parametrize(("field", "value", "device"), [
    ("device", "cpu", "on"),
    ("task_id", "another-task", "off"),
    ("inference_configuration_sha256", "b" * 64, "off"),
    ("top_level_device", "cuda", "off"),
])
def test_process_rejects_loading_evidence_from_another_binding(tmp_path, monkeypatch, field, value, device):
    task = frozen_task()
    task.inference_configuration["loading"]["on"] = copy.deepcopy(task.inference_configuration["loading"]["off"])
    task.inference_configuration["identity_sha256"] = configuration_identity(task.inference_configuration)
    path = tmp_path / "input_scale_plan.json"
    write_scale_plan_file(str(path), task, [{"input_scale": 1, "payload": {"text": "fixture"}}])
    report = verified_runtime_report(task)
    report["device"] = "cuda" if device == "on" else "cpu"
    report["configuration_validation"].update(
        inference_configuration_sha256=task.inference_configuration["identity_sha256"],
        task_id=task.selected_tasks[0]["task_id"], device=report["device"],
    )
    if field == "top_level_device":
        report["device"] = value
    else:
        report["configuration_validation"][field] = value
    report["stages"] = [{"stage": stage, "status": "verified"} for stage in
                        ("execution", "load", "preprocess", "predict", "completion", "postprocess", "validate_output")]
    report["validation"] = {"protocol": {"status": "verified"}, "task": {"status": "verified"}}
    monkeypatch.setattr(validation, "run_validation", Mock(return_value=report))

    with pytest.raises(RuntimeValidationError) as error:
        validation.validate_process(
            task_info=task, image_info=SimpleNamespace(environment_id="fixture"),
            planned=SimpleNamespace(plan_file=str(path)), cpu_list=[1], mem_list=[4],
            gpu_list=[device], output_dir=str(tmp_path),
        )

    assert error.value.failure.reason_code == "inference_configuration_mismatch"
