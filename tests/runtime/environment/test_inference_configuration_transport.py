"""Frozen values survive host boundaries and require matching runtime observations."""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.failures import RuntimeFailure
from acprof.host.configuration_transport import (
    CONFIGURATION_ENV,
    TASK_ENV,
    inference_environment,
    verify_plan_configuration,
    verify_runtime_configuration,
)
from acprof.host.images.runtime import ImageInfo, configure_runtime_profile, request_fingerprint
from acprof.host.profiling.support import profiler_inference_configuration
from acprof.host.runtimes.process import service_environment
from acprof.models.inference_configuration import decode_inference_configuration
from acprof.runtime import profiles
from tests.support.inference_configuration_fixture import frozen_task, verified_runtime_report


def test_environment_transports_exact_frozen_values_and_one_selected_task():
    task = frozen_task()
    original = copy.deepcopy(task.inference_configuration)
    environment = inference_environment(task)

    assert environment[TASK_ENV] == "text-generation"
    assert decode_inference_configuration(environment[CONFIGURATION_ENV]) == original
    assert task.inference_configuration == original
    task.inference_configuration = {}
    assert inference_environment(task) == {}


def test_transport_never_selects_an_inactive_task_from_the_full_frozen_collection():
    from acprof.models.inference_configuration import configuration_identity
    from acprof.tasks.workloads import parse_task

    task = frozen_task()
    other = parse_task("fill-mask", {}, encoder_decoder=False, source="fixture")
    task.inference_configuration["tasks"]["fill-mask"] = {
        "interface": other["interface"], "fingerprint": other["fingerprint"], "generation": {},
    }
    task.inference_configuration["identity_sha256"] = configuration_identity(task.inference_configuration)
    task.pipeline_tag = "fill-mask"

    with pytest.raises(ValueError, match="select exactly one frozen inference task"):
        inference_environment(task)


@pytest.mark.parametrize("bad", [None, [], False])
def test_false_nonobjects_do_not_enable_legacy_transport(bad):
    task = frozen_task()
    task.inference_configuration = bad

    with pytest.raises(ValueError, match="must be an object"):
        inference_environment(task)


def test_process_environment_uses_the_same_configuration_and_never_inherits_another(tmp_path, monkeypatch):
    task = frozen_task()
    monkeypatch.setenv(CONFIGURATION_ENV, "unrelated-parent-configuration")
    info = SimpleNamespace(python=str(tmp_path / "venv" / "bin" / "python"),
                           code_root=str(tmp_path), model_path=str(tmp_path / "model"), runtime_environment={})

    environment = service_environment(task, info)

    assert environment[TASK_ENV] == "text-generation"
    assert decode_inference_configuration(environment[CONFIGURATION_ENV]) == task.inference_configuration


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("inference_configuration_sha256", "b" * 64, "inference_configuration_mismatch"),
    ("configuration_validation", {"status": "not_run"}, "inference_configuration_unverified"),
    ("consumer_receipt", {"status": "not_applicable"}, "inference_configuration_unverified"),
])
def test_validation_requires_identity_loading_and_consumer_receipts(field, value, reason):
    task = frozen_task()
    report = verified_runtime_report(task)
    report[field] = value

    with pytest.raises(RuntimeFailure) as error:
        verify_runtime_configuration(task, report, expected_device="off", require_consumers=True)
    assert error.value.failure.reason_code == reason


def test_readiness_accepts_loading_observation_without_claiming_consumer_validation():
    task = frozen_task()
    report = verified_runtime_report(task)
    report.pop("consumer_receipt")

    verify_runtime_configuration(task, report, expected_device="off")
    with pytest.raises(RuntimeFailure, match="consumer_receipt"):
        verify_runtime_configuration(task, report, expected_device="off", require_consumers=True)
    verify_runtime_configuration(replace(task, inference_configuration={}), {})


@pytest.mark.parametrize(("total", "new"), [("max_length", "max_new_tokens"), ("min_length", "min_new_tokens")])
def test_host_requires_each_total_length_call_to_rule_out_an_inherited_override(total, new):
    task = frozen_task(generation={total: 10})
    report = verified_runtime_report(task)
    calls = report["consumer_receipt"]["calls"]
    calls.append(copy.deepcopy(calls[0]))
    del calls[1]["parameters"][new]
    observed = {"status": "verified", "expected": None, "observed": None, "source": "generate.kwargs"}
    calls[0]["parameters"][new] = observed
    for old_observation in (None, {**observed, "observed": 256}, {**observed, "source": "model.generation_config"}):
        if old_observation is not None:
            calls[1]["parameters"][new] = old_observation
        with pytest.raises(RuntimeFailure, match=rf"calls\[1\].*{new}"):
            verify_runtime_configuration(task, report, expected_device="off", require_consumers=True)
    calls[1]["parameters"][new] = observed
    verify_runtime_configuration(task, report, expected_device="off", require_consumers=True)


def test_foreign_input_plan_is_rejected_before_any_runtime_dispatch():
    task = frozen_task()
    with pytest.raises(RuntimeFailure) as error:
        verify_plan_configuration(task, {"inference_configuration_sha256": "b" * 64})
    assert error.value.failure.reason_code == "inference_configuration_mismatch"
    assert verify_plan_configuration(task, {
        "inference_configuration_sha256": task.inference_configuration["identity_sha256"],
    }) is task.inference_configuration


def test_mutated_configuration_stops_docker_before_resource_or_cleanup_actions(monkeypatch):
    from acprof.host import docker_runtime

    task = frozen_task()
    task.inference_configuration["loading"]["off"]["dtype"] = "FP16"
    recover = Mock(side_effect=AssertionError("cleanup must not run"))
    monkeypatch.setattr(docker_runtime, "recover_abandoned_containers", recover)

    with pytest.raises(ValueError, match="identity_sha256"):
        docker_runtime.start_container_session(
            task, 1, 4, "off", ImageInfo("sha256:" + "b" * 64), "fixture", "[fixture]",
        )
    recover.assert_not_called()


def test_frozen_profile_cannot_change_after_hardware_selection(monkeypatch):
    from acprof.host.images import runtime

    task = frozen_task()
    support = Mock(side_effect=AssertionError("revalidation must not run"))
    monkeypatch.setattr("acprof.host.preparation.task_support.require_task_support", support)
    monkeypatch.setattr(runtime, "runtime_build_overrides", lambda: {"ACPROF_NLP_TORCH_SPEC": ""})

    with pytest.raises(RuntimeFailure) as error:
        configure_runtime_profile(task, torch_index_url="https://download.pytorch.org/whl/cu128")
    assert error.value.failure.reason_code == "inference_configuration_mismatch"
    assert task.runtime_profile_id == "nlp-cpu"
    assert task.inference_configuration["backend"]["runtime_profile"] == "nlp-cpu"
    support.assert_not_called()


def test_generation_changes_experiment_but_not_image_request_identity():
    first, second = frozen_task(tokens=7), frozen_task(tokens=11)

    assert first.inference_configuration["identity_sha256"] != second.inference_configuration["identity_sha256"]
    assert request_fingerprint(first) == request_fingerprint(second)


def test_eager_profiler_gets_a_separate_frozen_identity_without_mutating_parent():
    task = frozen_task()
    original = copy.deepcopy(task.inference_configuration)

    derived = profiler_inference_configuration(task, use_gpu=False, profile_mode="torch_eager_cpu")

    assert derived["identity_sha256"] != original["identity_sha256"]
    assert derived["derivation"]["parent_identity_sha256"] == original["identity_sha256"]
    assert derived["loading"]["off"] == {"dtype": "FP32", "attention_implementation": "eager"}
    assert task.inference_configuration == original
    assert profiler_inference_configuration(task, use_gpu=False, profile_mode="cpu") == original


def test_profiler_command_transports_derived_configuration(tmp_path, monkeypatch):
    from acprof.host.profiling.support import profiler_container_command

    task = frozen_task()
    mount = SimpleNamespace(args=[], close=Mock())
    monkeypatch.setattr("acprof.host.model_store.store.acquire_mount", lambda _: mount)

    with profiler_container_command(
        task_info=task, image_tag="sha256:" + "b" * 64, cpu=1, mem=4, use_gpu=False,
        payload_file=str(tmp_path / "input_scale_plan.json"), profile_root=str(tmp_path),
        tool_mount_roots=(), profile_mode="torch_eager_cpu",
    ) as command:
        environment = dict(item.split("=", 1) for index, item in enumerate(command)
                           if index and command[index - 1] == "-e")
        transported = decode_inference_configuration(environment[CONFIGURATION_ENV])
        assert environment[TASK_ENV] == "text-generation"
        assert transported["derivation"]["parent_identity_sha256"] == task.inference_configuration["identity_sha256"]
        assert transported["loading"]["off"]["attention_implementation"] == "eager"

    mount.close.assert_called_once()


def test_eager_profiler_does_not_invent_an_undeclared_loading_option():
    from acprof.models.inference_configuration import configuration_identity

    task = frozen_task()
    task.inference_configuration["loading"]["off"] = {}
    task.inference_configuration["identity_sha256"] = configuration_identity(task.inference_configuration)

    with pytest.raises(ValueError, match="no verified eager profiler configuration"):
        profiler_inference_configuration(task, use_gpu=False, profile_mode="torch_eager_cpu")


def test_process_validation_persists_missing_consumer_receipt_as_typed_failure(tmp_path, monkeypatch):
    import json

    from acprof.host.input_plan.materialize import write_scale_plan_file
    from acprof.host.runtime_validation import RuntimeValidationError
    from acprof.host.runtimes import validation
    from acprof.results.layout import ArtifactLayout

    task = frozen_task()
    path = tmp_path / "input_scale_plan.json"
    write_scale_plan_file(str(path), task, [{"input_scale": 1, "payload": {"text": "fixture"}}])
    response = verified_runtime_report(task)
    response.pop("consumer_receipt")
    response["stages"] = [{"stage": stage, "status": "verified"} for stage in
                          ("execution", "load", "preprocess", "predict", "completion", "postprocess", "validate_output")]
    response["validation"] = {"protocol": {"status": "verified"}, "task": {"status": "verified"}}
    runner = Mock(return_value=response)
    monkeypatch.setattr(validation, "run_validation", runner)

    with pytest.raises(RuntimeValidationError) as error:
        validation.validate_process(
            task_info=task, image_info=SimpleNamespace(environment_id="fixture"),
            planned=SimpleNamespace(plan_file=str(path)), cpu_list=[1], mem_list=[4],
            gpu_list=["off"], output_dir=str(tmp_path),
        )

    assert error.value.failure.reason_code == "inference_configuration_unverified"
    report = json.loads(ArtifactLayout.discover(tmp_path).path("runtime_validation.json").read_text())
    assert report["inference_configuration_sha256"] == task.inference_configuration["identity_sha256"]
    assert report["devices"]["off"]["failure"]["reason_code"] == "inference_configuration_unverified"
    assert report["status"] == "error"
    runner.assert_called_once()


@pytest.mark.parametrize("interface", ["fill-mask", "feature-extraction", "text-classification"])
def test_nongenerative_catalog_routes_do_not_claim_generate_consumers(interface):
    from acprof.models.inference_configuration import declared_generation_consumer

    assert not declared_generation_consumer(
        family="nlp", backend="transformers_model", adapter="family-default", task=interface,
    )


@pytest.mark.parametrize(("field", "value"), [
    ("calls", []), ("task_id", "another-task"), ("inference_configuration_sha256", "b" * 64),
])
def test_consumer_evidence_must_cover_the_selected_task_and_configuration(field, value):
    task = frozen_task()
    report = verified_runtime_report(task)
    report["consumer_receipt"][field] = value

    with pytest.raises(RuntimeFailure, match="consumer_receipt"):
        verify_runtime_configuration(task, report, expected_device="off", require_consumers=True)


@pytest.mark.parametrize(("overrides", "diagnostic"), [
    ({"backends": ("transformers_pipeline",)}, "backend"),
    ({"task_types": ("fill-mask",)}, "task"),
    ({"model_types": ("bert",)}, "model type"),
    ({"adapter": "different-adapter"}, "adapter"),
])
def test_generic_profile_enforces_its_full_declared_route(monkeypatch, overrides, diagnostic):
    task = frozen_task()
    restricted = replace(profiles.PROFILES["nlp-cpu"], **overrides)
    monkeypatch.setitem(profiles.PROFILES, "nlp-cpu", restricted)

    with pytest.raises(RuntimeFailure, match=diagnostic):
        profiles.select_runtime_profile(task)


@pytest.mark.parametrize(("family", "interface", "backend", "profile"), [
    ("nlp", "sentence-similarity", "sentence_transformers", "nlp-cpu"),
    ("nlp", "text-ranking", "cross_encoder", "nlp-cpu"),
    ("audio", "voice-activity-detection", "torchscript", "audio-cpu"),
    ("diffusion", "text-to-image", "diffusers", "diffusion-cpu"),
    ("structured", "tabular-regression", "skops", "structured-cpu"),
    ("timeseries", "time-series-forecasting", "chronos", "timeseries-cpu"),
])
def test_registered_generic_backend_routes_remain_available(family, interface, backend, profile):
    task = replace(frozen_task(), task_family=family, pipeline_tag=interface, runtime_backend=backend,
                   runtime_profile_id=profile, model_config={}, inference_configuration={}, selected_tasks=[])

    assert profiles.select_runtime_profile(task).profile_id == profile
