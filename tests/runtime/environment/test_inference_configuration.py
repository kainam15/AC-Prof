"""Offline contracts for frozen runtime values and independent observations."""
from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.container import inference_configuration as verification
from acprof.container.model_session import ModelLoadConfig, ModelSession, load_model_session
from acprof.failures import RuntimeFailure
from acprof.models.inference_configuration import (
    configuration_identity,
    encode_inference_configuration,
)


def frozen_configuration(*, generation=None, loading=None):
    result = {
        "schema_version": 1, "state": "frozen", "rule_version": "acprof-inference-v1",
        "model": {"source": "huggingface", "model_id": "fixture/model", "revision": "a" * 40},
        "backend": {"id": "transformers_pipeline", "adapter": "family-default",
                    "extension_id": "torch-nlp", "runtime_profile": "fixture-profile"},
        "loading": {"off": {"dtype": "FP32", "attention_implementation": "eager"} if loading is None else loading},
        "tasks": {"generate": {"interface": "text-generation", "fingerprint": "b" * 64,
                              "generation": {"max_new_tokens": 7, "do_sample": False} if generation is None else generation}},
        "requests": {}, "decisions": [], "workload": {},
        "validation": {"static": "validated", "runtime": "not_run"},
    }
    result["identity_sha256"] = configuration_identity(result)
    return result


class Model:
    def __init__(self, dtypes=("torch.float32",), attention="eager"):
        self.dtypes = dtypes
        self.config = SimpleNamespace(_attn_implementation=attention)
        self.generation_config = SimpleNamespace(max_new_tokens=7, do_sample=False)
        self.calls = []

    def parameters(self):
        return (SimpleNamespace(dtype=dtype, is_floating_point=lambda: True) for dtype in self.dtypes)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [1, 2]


def session(*, frozen=None, model=None):
    frozen = frozen_configuration() if frozen is None else frozen
    config = ModelLoadConfig(
        "fixture/model", "nlp", "text-generation", "transformers_pipeline",
        model_revision="a" * 40, inference_configuration=frozen, inference_task_id="generate",
    )
    context = {"model": Model() if model is None else model}
    verification.attach_configuration_context(config, "cpu", context)
    return ModelSession(config, Mock(), context, SimpleNamespace(inference_context=nullcontext), "cpu")


@pytest.fixture
def registered(monkeypatch):
    from acprof.container import load_policy

    extension = SimpleNamespace(
        extension_id="torch-nlp", handler_entrypoint="acprof.container.handlers.nlp:NLPHandler",
        parameters=(SimpleNamespace(scope="inference", consumers=(SimpleNamespace(stage="predict"),)),),
    )
    profile = SimpleNamespace(profile_id="fixture-profile")
    monkeypatch.setattr(load_policy, "registered_policy", lambda *args, **kwargs: (profile, extension, False))
    return profile, extension


def test_environment_decodes_frozen_identity_and_task(monkeypatch):
    frozen = frozen_configuration()
    monkeypatch.setenv("ACPROF_INFERENCE_CONFIGURATION", encode_inference_configuration(frozen))
    monkeypatch.setenv("ACPROF_INFERENCE_TASK_ID", "generate")
    config = ModelLoadConfig.from_environment()
    assert config.inference_configuration == frozen
    assert config.inference_task_id == "generate"


def test_shared_loading_forwards_exact_options_without_observing_model(monkeypatch, registered):
    from acprof.container import handlers

    instance = session()
    original = copy.deepcopy(instance.config.inference_configuration)
    instance.context["model"].parameters = Mock(side_effect=AssertionError("observations belong after load timing"))
    loader = Mock(return_value=instance.context)
    monkeypatch.setattr(handlers, "load_handler", loader)
    loaded = load_model_session(instance.config, instance.execution, "cpu", handler=instance.handler)

    assert loader.call_args.kwargs["load_options"] == {"dtype": "FP32", "attention_implementation": "eager"}
    assert loaded.context["_frozen_generation"] == {"max_new_tokens": 7, "do_sample": False}
    assert instance.config.inference_configuration == original
    assert "configuration_validation" not in loaded.context


@pytest.mark.parametrize("change", [
    {"model_id": "different"}, {"model_revision": "c" * 40}, {"backend": "transformers_model"},
    {"adapter": "different"}, {"task_type": "summarization"}, {"inference_task_id": "missing"},
])
def test_active_binding_mismatch_is_rejected_before_loading(registered, change):
    instance = session()
    with pytest.raises(RuntimeFailure) as caught:
        verification.bound_load_options(replace(instance.config, **change), "cpu", None)
    assert caught.value.failure.reason_code == "inference_configuration_mismatch"


@pytest.mark.parametrize("options", [
    {"dtype": "FP16"}, {"attention_implementation": "sdpa"}, {"trust_remote_code": True},
])
def test_unconfirmed_loading_override_is_rejected(registered, options):
    instance = session()
    with pytest.raises(RuntimeFailure, match="derive a separate profiler"):
        verification.bound_load_options(instance.config, "cpu", options)


def test_identical_profiler_subset_does_not_discard_other_frozen_options(registered):
    instance = session()
    assert verification.bound_load_options(
        instance.config, "cpu", {"attention_implementation": "eager"},
    ) == {"dtype": "FP32", "attention_implementation": "eager"}


@pytest.mark.parametrize(("attribute", "value"), [("profile_id", "wrong-profile"), ("extension_id", "wrong-extension")])
def test_registered_runtime_identity_must_match_frozen_backend(registered, attribute, value):
    profile, extension = registered
    setattr(profile if attribute == "profile_id" else extension, attribute, value)
    with pytest.raises(RuntimeFailure, match="registered active"):
        verification.bound_load_options(session().config, "cpu", None)


def test_observations_are_independent_and_use_all_parameter_dtypes():
    instance = session()
    frozen_before = copy.deepcopy(instance.config.inference_configuration)
    report = verification.verify_model_session_configuration(instance)
    assert report["loading"]["dtype"] == {
        "status": "verified", "expected": "FP32", "observed": ["FP32"],
        "method": "floating_parameter_dtypes", "scope": "model_parameters",
    }
    assert report["loading"]["attention_implementation"]["observed"] == ["eager"]
    assert instance.config.inference_configuration == frozen_before
    assert instance.config.inference_configuration["validation"]["runtime"] == "not_run"
    report["loading"].clear()
    assert verification.configuration_metadata(instance)["configuration_validation"]["loading"]


def test_mixed_parameter_dtypes_fail_even_when_first_parameter_matches():
    instance = session(model=Model(("torch.float32", "torch.float16")))
    with pytest.raises(RuntimeFailure) as caught:
        verification.verify_model_session_configuration(instance)
    assert caught.value.failure.reason_code == "inference_configuration_mismatch"
    assert caught.value.failure.evidence["loading"]["dtype"]["observed"] == ["FP16", "FP32"]


@pytest.mark.parametrize(("model", "field"), [(Model((), "eager"), "dtype"), (Model(attention=None), "attention_implementation")])
def test_missing_observation_does_not_accept_requested_value_echo(model, field):
    instance = session(model=model)
    instance.context.update(dtype="FP32", attention_implementation="eager")
    with pytest.raises(RuntimeFailure) as caught:
        verification.verify_model_session_configuration(instance)
    assert caught.value.failure.reason_code == "inference_configuration_unverified"
    assert caught.value.failure.evidence["loading"][field]["status"] == "unverified"


def test_attention_mismatch_blocks_readiness():
    instance = session(model=Model(attention="sdpa"))
    with pytest.raises(RuntimeFailure, match="runtime attention_implementation"):
        verification.verify_model_session_configuration(instance)
    with pytest.raises(RuntimeFailure, match="before the service becomes ready"):
        verification.configuration_metadata(instance)


def test_undeclared_configuration_fields_are_reported_as_not_applicable():
    frozen = frozen_configuration(generation={}, loading={})
    frozen["tasks"]["generate"]["interface"] = "text-to-image"
    frozen["backend"].update(id="diffusers", extension_id="torch-diffusion")
    frozen["identity_sha256"] = configuration_identity(frozen)
    config = ModelLoadConfig("fixture/model", "diffusion", "text-to-image", "diffusers",
                             model_revision="a" * 40, inference_configuration=frozen, inference_task_id="generate")
    context = {"model": object()}
    verification.attach_configuration_context(config, "cpu", context)
    instance = ModelSession(config, Mock(), context, SimpleNamespace(), "cpu")
    verification.validate_inference_payload(instance, {"params": {"num_inference_steps": 25, "seed": 0}})
    report = verification.verify_model_session_configuration(instance)
    assert report["status"] == "verified"
    assert all(field["status"] == "not_applicable" for field in report["loading"].values())
    with verification.capture_generation_consumption(instance) as receipt:
        pass
    assert verification.verify_generation_consumption(instance, receipt)["status"] == "not_applicable"
    assert receipt["generation"] == {"status": "not_applicable", "reason": "consumer_not_declared"}
    assert "_frozen_generation" not in context


def test_generate_boundary_records_actual_kwargs_and_restores_method():
    instance = session()
    model = instance.context["model"]
    with verification.capture_generation_consumption(instance) as receipt:
        assert model.generate(max_new_tokens=7, do_sample=False) == [1, 2]
    report = verification.verify_generation_consumption(instance, receipt)
    assert report["status"] == "verified"
    assert report["calls"][0]["parameters"]["max_new_tokens"]["observed"] == 7
    assert report["calls"][0]["parameters"]["do_sample"]["source"] == "generate.kwargs"
    assert "generate" not in vars(model)
    assert model.calls == [{"max_new_tokens": 7, "do_sample": False}]


def test_generate_receipt_reads_supplied_generation_config_before_model_defaults():
    instance = session()
    model = instance.context["model"]
    model.generation_config.max_new_tokens = 999
    with verification.capture_generation_consumption(instance) as receipt:
        model.generate(generation_config=SimpleNamespace(max_new_tokens=7, do_sample=False))
    report = verification.verify_generation_consumption(instance, receipt)
    assert report["calls"][0]["parameters"]["max_new_tokens"]["source"] == "generate.generation_config"


def test_parameter_replacement_fails_at_generate_boundary_before_inference():
    instance = session()
    model = instance.context["model"]
    with pytest.raises(RuntimeFailure) as caught:
        with verification.capture_generation_consumption(instance):
            model.generate(max_new_tokens=99, do_sample=False)
    assert caught.value.failure.reason_code == "inference_configuration_mismatch"
    assert model.calls == []
    assert "generate" not in vars(model)


def test_handler_success_without_generate_is_not_consumer_verification():
    instance = session()
    with verification.capture_generation_consumption(instance) as receipt:
        pass
    with pytest.raises(RuntimeFailure, match="did not reach model.generate"):
        verification.verify_generation_consumption(instance, receipt)


def test_frozen_request_parameters_are_not_silently_replaced():
    instance = session()
    with pytest.raises(RuntimeFailure, match="request generation"):
        verification.validate_inference_payload(instance, {"params": {"max_new_tokens": 12, "do_sample": False}})
    verification.validate_inference_payload(instance, {"params": {"max_new_tokens": 7, "do_sample": False}})
    assert instance.context["_frozen_generation"] == {"max_new_tokens": 7, "do_sample": False}


def test_nlp_handler_keeps_frozen_kwargs_and_does_not_add_legacy_defaults(monkeypatch):
    from acprof.container.handlers import nlp

    params = {"max_length": 9}
    instance = session(frozen=frozen_configuration(generation=params))
    pipe = Mock(return_value=[1])
    instance.context.update(pipeline=pipe, task_type="text-generation", backend="transformers_pipeline")
    monkeypatch.setattr(nlp, "check_nlp_input", lambda *args: None)
    output = nlp.NLPHandler().predict(instance.context, {"text": "hello", "params": params})
    assert output == [1]
    pipe.assert_called_once_with("hello", max_length=9, truncation=True)
    assert params == {"max_length": 9}


@pytest.mark.parametrize("mismatch", [False, True])
def test_independent_validation_keeps_configuration_and_real_consumption_stages(monkeypatch, mismatch):
    from acprof.container import execution, model_session, runtime_validate

    instance = session(model=Model(attention="sdpa" if mismatch else "eager"))
    instance.execution.metadata = lambda: {}
    instance.handler.preprocess.side_effect = lambda context, payload: payload
    instance.handler.predict.side_effect = lambda context, processed: context["model"].generate(**processed["params"])
    instance.handler.postprocess.return_value = {"task": "text-generation"}
    instance.handler.validate_output.return_value = {
        "protocol": {"status": "verified"}, "task": {"status": "verified"}, "workload_contract": {},
    }
    for name, value in {
        "TASK_FAMILY": "nlp", "RUNTIME_BACKEND": "transformers_pipeline",
        "TASK_TYPE": "text-generation", "MODEL_ID": "fixture/model", "MODEL_REVISION": "a" * 40,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(model_session.ModelLoadConfig, "from_environment", lambda **kwargs: instance.config)
    monkeypatch.setattr(model_session, "prepare_execution", lambda config: (instance.execution, "cpu"))
    monkeypatch.setattr(model_session, "load_model_session", lambda *args: instance)
    monkeypatch.setattr(execution, "complete_prediction", lambda execution, context, output: output)
    stages = []
    payload = {"text": "hello", "params": {"max_new_tokens": 7, "do_sample": False}}
    if mismatch:
        with pytest.raises(RuntimeFailure):
            runtime_validate.validate(payload, stages=stages)
        assert stages[-1] == {
            "stage": "configuration", "status": "error",
            "error": stages[-1]["error"],
        }
        instance.handler.preprocess.assert_not_called()
        instance.handler.predict.assert_not_called()
    else:
        result = runtime_validate.validate(payload, stages=stages)
        assert result["status"] == "ok"
        assert result["configuration_validation"]["status"] == "verified"
        assert result["consumer_receipt"]["status"] == "verified"
        assert result["consumer_receipt"]["calls"][0]["parameters"]["max_new_tokens"]["observed"] == 7
        names = [stage["stage"] for stage in stages]
        assert names.index("load") < names.index("configuration") < names.index("predict")
        assert names.index("predict") < names.index("configuration_consumption")
        assert instance.config.inference_configuration["validation"]["runtime"] == "not_run"


@pytest.fixture
def fake_flask(monkeypatch):
    import sys
    from types import ModuleType

    module = ModuleType("flask")
    apps = []

    class FakeFlask:
        def __init__(self, name):
            self.config, self.routes = {}, {}
            apps.append(self)

        def route(self, path, **options):
            def register(function):
                self.routes[path] = function
                return function
            return register

        def errorhandler(self, code):
            return lambda function: function

    module.Flask = FakeFlask
    module.jsonify = lambda value: value
    module.request = SimpleNamespace(headers={}, get_json=lambda **kwargs: {})
    monkeypatch.setitem(sys.modules, "flask", module)
    return module, apps


def test_server_exposes_cached_validation_and_rejects_unverified_session(fake_flask):
    from acprof.container import server

    module, apps = fake_flask
    instance = session()
    with pytest.raises(RuntimeFailure, match="before the service becomes ready"):
        server.create_app(instance)
    assert apps[-1].routes == {}
    verification.verify_model_session_configuration(instance)
    instance.context["model"].parameters = Mock(side_effect=AssertionError("HTTP must not rescan"))
    instance.execution.metadata = lambda: {}
    app = server.create_app(instance)
    for route in ("/ready", "/meta"):
        result = app.routes[route]()
        assert result["inference_configuration_sha256"] == instance.config.inference_configuration["identity_sha256"]
        assert result["configuration_validation"]["status"] == "verified"
    module.request.get_json = lambda **kwargs: {"text": "hello", "params": {"max_new_tokens": 99}}
    result, status = app.routes["/predict"]()
    assert status == 500
    assert result["failure"]["reason_code"] == "inference_configuration_mismatch"
    instance.handler.preprocess.assert_not_called()


@pytest.mark.parametrize("mismatch", [False, True])
def test_startup_verifies_after_load_timer_and_before_readiness(monkeypatch, fake_flask, mismatch):
    from acprof.container import model_session, server

    _module, apps = fake_flask
    instance = session(model=Model(attention="sdpa" if mismatch else "eager"))
    elapsed, wall_times, events = [0.0], [], []
    monkeypatch.setattr(server, "SERVER_PROCESS_STARTED_AT", 1000.0)
    monkeypatch.setattr(server, "SERVER_PROCESS_STARTED_PERF", 100.0)

    def wall_time():
        wall_times.append(1000.0 + elapsed[0])
        return wall_times[-1]

    monkeypatch.setattr(server, "time", SimpleNamespace(time=wall_time, perf_counter=lambda: 100.0 + elapsed[0]))
    monkeypatch.setattr(model_session.ModelLoadConfig, "from_environment", lambda: instance.config)
    monkeypatch.setattr(model_session, "select_handler", lambda config: instance.handler)

    def prepare(config):
        elapsed[0] += 2.0
        return instance.execution, "cpu"

    def load(*args, **kwargs):
        events.append("load")
        elapsed[0] += 11.0
        return instance

    observe = verification.verify_model_session_configuration

    def verify(loaded):
        events.append("verify")
        elapsed[0] += 41.0
        return observe(loaded)

    monkeypatch.setattr(model_session, "prepare_execution", prepare)
    monkeypatch.setattr(model_session, "load_model_session", load)
    monkeypatch.setattr(verification, "verify_model_session_configuration", verify)
    if mismatch:
        with pytest.raises(RuntimeFailure, match="runtime attention_implementation"):
            server.create_runtime_app()
        assert apps[-1].routes == {}
    else:
        app = server.create_runtime_app()
        ready = app.routes["/ready"]()
        assert ready["load_time_s"] == 11.0
        assert ready["startup_timing"]["model_load_completed_at_epoch_s"] == 1013.0
        assert ready["configuration_validation"]["status"] == "verified"
    assert events == ["load", "verify"]
    assert wall_times == [1002.0, 1013.0]
    assert elapsed[0] == 54.0


def test_caught_consumer_mismatch_cannot_become_verified_by_later_retry():
    instance = session()
    with verification.capture_generation_consumption(instance) as receipt:
        with pytest.raises(RuntimeFailure):
            instance.context["model"].generate(max_new_tokens=99)
        instance.context["model"].generate(max_new_tokens=7, do_sample=False)
    with pytest.raises(RuntimeFailure, match="failed generation consumption"):
        verification.verify_generation_consumption(instance, receipt)


def test_failed_reverification_does_not_leave_a_stale_ready_result():
    instance = session()
    verification.verify_model_session_configuration(instance)
    instance.context["model"].dtypes = ("torch.float16",)
    with pytest.raises(RuntimeFailure):
        verification.verify_model_session_configuration(instance)
    with pytest.raises(RuntimeFailure, match="before the service becomes ready"):
        verification.configuration_metadata(instance)


def test_profiler_reports_configuration_before_its_measurement_window(monkeypatch, capsys):
    import importlib
    import json
    import os
    import sys
    from unittest.mock import patch

    with patch.dict(os.environ):
        runner = importlib.import_module("acprof.container.compute_profile_runner")
    instance = session()
    events = []
    instance.execution.synchronize = lambda: None
    instance.handler.preprocess.side_effect = lambda context, payload: payload
    instance.handler.predict.side_effect = lambda context, processed: events.append("predict") or [1]
    instance.handler.postprocess.return_value = {}

    def load(*args, **kwargs):
        events.append("load")
        return instance

    observe = verification.verify_model_session_configuration

    def verify(loaded):
        events.append("verify")
        result = observe(loaded)
        loaded.context["model"].parameters = Mock(side_effect=AssertionError("profiling must not rescan"))
        return result

    monkeypatch.setattr(runner, "ModelLoadConfig", SimpleNamespace(from_environment=lambda **kwargs: instance.config))
    monkeypatch.setattr(runner, "prepare_execution", lambda config: (instance.execution, "cpu"))
    monkeypatch.setattr(runner, "select_handler", lambda config: instance.handler)
    monkeypatch.setattr(runner, "load_model_session", load)
    monkeypatch.setattr(runner, "verify_model_session_configuration", verify)
    monkeypatch.setattr(runner, "complete_prediction", lambda execution, context, output: output)
    monkeypatch.setattr(runner, "_find_payload", lambda *args: {
        "text": "hello", "params": {"max_new_tokens": 7, "do_sample": False},
    })
    monkeypatch.setattr(runner, "_gpu_metadata", lambda device: {})
    monkeypatch.setattr(runner, "_ITTControl", lambda: SimpleNamespace(
        resume=lambda: events.append("profile_start"), pause=lambda: events.append("profile_end"),
    ))
    monkeypatch.setattr(sys, "argv", [
        "compute_profile_runner", "--payload-file", "fixture.json", "--input-scale", "1",
        "--repeat", "1", "--profile-mode", "cpu",
    ])
    runner.main()
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["inference_configuration_sha256"] == instance.config.inference_configuration["identity_sha256"]
    assert result["configuration_validation"]["status"] == "verified"
    assert events == ["load", "verify", "predict", "profile_start", "predict", "profile_end"]
