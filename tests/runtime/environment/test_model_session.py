"""Offline coverage of the shared loading boundary and explicit configuration."""
import importlib
import os
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock, patch

import pytest

from acprof.container.model_session import (
    ModelLoadConfig,
    load_model_session,
    prepare_execution,
    select_handler,
)


@pytest.mark.parametrize("load_options", [None, {"attention_implementation": "eager"}])
def test_loading_uses_captured_source_revision_and_profile_options(tmp_path, monkeypatch, load_options):
    with patch.dict(os.environ, {
        "MODEL_ID": "local/model", "MODEL_REVISION": "pinned-revision",
        "MODEL_LOCAL_PATH": str(tmp_path), "TASK_FAMILY": "nlp",
        "TASK_TYPE": "text-generation", "RUNTIME_BACKEND": "transformers_pipeline",
    }, clear=True):
        config = ModelLoadConfig.from_environment()
    monkeypatch.delenv("ACPROF_MODEL_ADAPTER", raising=False)
    monkeypatch.setenv("MODEL_LOCAL_PATH", str(tmp_path / "later-missing-snapshot"))
    monkeypatch.setenv("MODEL_REVISION", "later-revision")
    handlers = importlib.import_module("acprof.container.handlers")
    handler, execution, context = object(), object(), {"loaded": True}
    loader = Mock(return_value=context)
    monkeypatch.setattr(handlers, "load_handler", loader)
    registry = Mock(return_value=handler)
    monkeypatch.setattr(handlers.HandlerRegistry, "get", registry)

    session = load_model_session(config, execution, "cpu", load_options=load_options)

    args = (handler, str(tmp_path), "text-generation", "transformers_pipeline", "cpu", "pinned-revision")
    if load_options is None:
        loader.assert_called_once_with(*args, adapter="family-default")
    else:
        loader.assert_called_once_with(*args, adapter="family-default", load_options=load_options)
        assert loader.call_args.kwargs["load_options"] is load_options
    registry.assert_called_once_with("nlp", "transformers_pipeline", adapter="family-default")
    assert session.config is config
    assert session.handler is handler
    assert session.context is context
    assert session.execution is execution


def test_environment_defaults_and_entrypoint_overrides_are_explicit():
    with patch.dict(os.environ, {}, clear=True):
        config = ModelLoadConfig.from_environment()
        validation = ModelLoadConfig.from_environment(default_threads=1, use_gpu=False)
        profiler = ModelLoadConfig.from_environment(task_type_default="text-classification")
    assert (config.task_family, config.backend, config.task_type) == (
        "nlp", "transformers_pipeline", "text-generation",
    )
    assert (config.model_id, config.model_revision, config.model_path, config.adapter) == (
        "", "main", "", "family-default",
    )
    assert config.threads == 0
    assert validation.threads == 1
    assert profiler.task_type == "text-classification"
    with pytest.raises(FrozenInstanceError):
        config.model_id = "changed"


def test_execution_preparation_uses_captured_adapter_device_and_threads(monkeypatch):
    with patch.dict(os.environ, {
        "TASK_FAMILY": "structured", "RUNTIME_BACKEND": "onnxruntime",
        "ACPROF_MODEL_ADAPTER": "fixture-adapter", "USE_GPU": "1",
        "ACPROF_RUNTIME_THREADS": "4",
    }, clear=True):
        config = ModelLoadConfig.from_environment()
    monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "later-adapter")
    monkeypatch.setenv("USE_GPU", "0")
    monkeypatch.setenv("ACPROF_RUNTIME_THREADS", "9")
    module = importlib.import_module("acprof.container.execution")
    expected = object(), "cuda"
    prepare = Mock(return_value=expected)
    monkeypatch.setattr(module, "configured_execution", prepare)

    assert prepare_execution(config) is expected

    prepare.assert_called_once_with(
        "structured", "onnxruntime", use_gpu=True, threads=4, adapter="fixture-adapter",
    )


@pytest.fixture
def adapter_runtime(monkeypatch):
    """Exercise the real loading chain with isolated declarations and no model framework."""
    import sys
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace

    import acprof.extensions as extensions
    from acprof.container import handlers, load_policy as policies
    from acprof.extensions import ExtensionCatalog
    from acprof.extensions.schema import ExtensionDeclaration
    from acprof.models import spec

    catalog = ExtensionCatalog()
    profiles = {}
    runtimes = {}
    loads = []
    base_profile = next(iter(policies.PROFILES.values()))

    class FixtureHandler(handlers.BaseHandler):
        identity = ""

        def load(self, model_source, task_type, backend, device, model_revision="main",
                 load_options=None, *, adapter="family-default"):
            loads.append((self.identity, model_source))
            policy = policies.load_policy(
                model_source, task_type, backend, device, load_options, adapter=adapter,
            )
            return {
                "handler_adapter": self.identity, "adapter": adapter,
                "runtime_profile": policy["runtime_profile"], "dtype": policy["dtype"],
                "source": model_source, "revision": model_revision, "device": device,
                "model": object(), "load_options": dict(load_options or {}),
            }

        def preprocess(self, model_ctx, raw_input):
            return raw_input

        def predict(self, model_ctx, processed_input):
            return processed_input

        def postprocess(self, model_ctx, raw_output):
            return raw_output

        def get_scale_metadata(self, model_ctx, raw_input):
            return {
                "handler_adapter": self.identity, "runtime_profile": model_ctx["runtime_profile"],
                "declaration_adapter": model_ctx["model_spec"]["format"],
            }

    monkeypatch.setattr(handlers.HandlerRegistry, "_handlers", {})
    monkeypatch.setattr(handlers.HandlerRegistry, "_adapters", {})
    for adapter in ("family-default", "adapter-a", "adapter-b"):
        runtime = ModuleType(f"acprof_fixture_execution_{adapter.replace('-', '_')}")
        runtime.adapter = adapter
        runtime.prepare_device = lambda use_gpu, threads: "cpu"
        runtime.inference_context = nullcontext
        runtime.metadata = lambda expected=adapter: {"execution_adapter": expected}

        def validate(context, payload, processed, output, response, *, expected=adapter):
            return {
                "protocol": {"status": "verified"}, "task": {"status": "verified"},
                "workload_contract": {}, "validator_adapter": expected,
                "runtime_profile": context["runtime_profile"],
            }

        runtime.validate = validate
        runtimes[adapter] = runtime
        monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
        profile_id = f"fixture-{adapter}"
        profiles[profile_id] = replace(
            base_profile, profile_id=profile_id, family="fixture", adapter=adapter,
            trust_remote_code=False,
            precision_policy={"supported_dtypes": ["FP32"], "preferred_dtype": "FP32"},
        )
        catalog.add(ExtensionDeclaration(
            extension_id=profile_id, family="fixture", runtime="fixture",
            backends=("fixture-runtime",), tasks=("fixture-task",),
            handler_entrypoint=f"{__name__}:FixtureHandler",
            validation_entrypoint=f"{runtime.__name__}:validate",
            execution={"cpu": "available", "cuda": "unsupported"},
            execution_entrypoint=runtime.__name__, dtypes=("FP32",),
            input_modalities=("text",), adapter=adapter, profile=profile_id,
            handler_options={"model_spec_format": adapter}, trust_remote_code=False,
        ))
        handler_class = type(f"Handler_{adapter.replace('-', '_')}", (FixtureHandler,), {"identity": adapter})
        if adapter == "family-default":
            handlers.HandlerRegistry.register("fixture", "fixture-runtime", handler_class)
        else:
            handlers.HandlerRegistry.register_adapter(adapter, "fixture", "fixture-runtime", handler_class)

    monkeypatch.setattr(extensions, "CATALOG", catalog)
    monkeypatch.setattr(extensions, "get_extension", catalog.get_extension)
    monkeypatch.setattr(policies, "CATALOG", catalog)
    monkeypatch.setattr(policies, "PROFILES", profiles)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="fixture-float32"))
    monkeypatch.setattr(
        spec, "load_model_spec",
        lambda source, task, *, expected_format: {"format": expected_format, "source": source},
    )
    monkeypatch.delenv("ACPROF_RUNTIME_PROFILE", raising=False)
    return SimpleNamespace(profiles=profiles, runtimes=runtimes, loads=loads)


@pytest.fixture
def adapter_startup_environment(adapter_runtime, monkeypatch):
    for name, value in {
        "MODEL_ID": "fixture/model", "MODEL_REVISION": "frozen-revision",
        "TASK_FAMILY": "fixture", "TASK_TYPE": "fixture-task",
        "RUNTIME_BACKEND": "fixture-runtime", "ACPROF_MODEL_ADAPTER": "adapter-a",
        "USE_GPU": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MODEL_LOCAL_PATH", raising=False)

    def prepare_device(use_gpu, threads):
        monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "adapter-b")
        return "cpu"

    monkeypatch.setattr(adapter_runtime.runtimes["adapter-a"], "prepare_device", prepare_device)
    return adapter_runtime


def test_server_startup_keeps_captured_adapter_after_execution_setup(adapter_startup_environment):
    pytest.importorskip("flask")
    from acprof.container.server import create_runtime_app

    app = create_runtime_app()
    with app.test_client() as client:
        ready = client.get("/ready")
        metadata = client.get("/scale_meta")

    assert ready.status_code == metadata.status_code == 200
    assert ready.get_json()["model_revision"] == "frozen-revision"
    assert metadata.get_json() == {
        "handler_adapter": "adapter-a", "runtime_profile": "fixture-adapter-a",
        "declaration_adapter": "adapter-a",
    }
    assert os.environ["ACPROF_MODEL_ADAPTER"] == "adapter-b"
    assert adapter_startup_environment.loads == [("adapter-a", "fixture/model")]


def test_runtime_validation_keeps_captured_adapter_and_declared_validator(adapter_startup_environment):
    from acprof.container.runtime_validate import validate

    result = validate({"input_scale": 1})

    assert result["status"] == "ok"
    assert result["adapter"] == "Handler_adapter_a"
    assert result["execution_adapter"] == "adapter-a"
    assert result["validation"]["validator_adapter"] == "adapter-a"
    assert result["validation"]["runtime_profile"] == "fixture-adapter-a"
    assert result["model_spec"]["format"] == "adapter-a"
    assert all(stage["status"] == "verified" for stage in result["stages"])
    assert os.environ["ACPROF_MODEL_ADAPTER"] == "adapter-b"
    assert adapter_startup_environment.loads == [("adapter-a", "fixture/model")]


@pytest.mark.parametrize("source", ["default", "explicit", "environment"])
def test_adapter_identity_is_captured_once_through_the_loading_chain(adapter_runtime, monkeypatch, source):
    expected = "family-default" if source == "default" else "adapter-a"
    if source == "environment":
        with patch.dict(os.environ, {
            "MODEL_ID": "fixture/model", "MODEL_REVISION": "frozen-revision",
            "TASK_FAMILY": "fixture", "TASK_TYPE": "fixture-task",
            "RUNTIME_BACKEND": "fixture-runtime", "ACPROF_MODEL_ADAPTER": "adapter-a",
        }, clear=True):
            config = ModelLoadConfig.from_environment()
    else:
        options = {} if source == "default" else {"adapter": "adapter-a"}
        config = ModelLoadConfig(
            "fixture/model", "fixture", "fixture-task", "fixture-runtime",
            model_revision="frozen-revision", **options,
        )
    monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "adapter-b")

    execution, device = prepare_execution(config)
    session = load_model_session(config, execution, device)

    assert config.adapter == expected
    assert execution.adapter == expected
    assert session.handler is select_handler(config)
    assert session.context["handler_adapter"] == expected
    assert session.context["adapter"] == expected
    assert session.context["runtime_profile"] == f"fixture-{expected}"
    assert session.context["model_spec"] == {"format": expected, "source": "fixture/model"}
    assert session.context["revision"] == "frozen-revision"
    assert session.context["dtype"] == "fixture-float32"
    assert adapter_runtime.loads == [(expected, "fixture/model")]


def test_interleaved_adapters_reuse_handlers_without_sharing_model_context(adapter_runtime, monkeypatch):
    monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "unrelated-environment-adapter")
    sessions = []
    for adapter, source in (("adapter-a", "fixture/first"), ("adapter-b", "fixture/second"),
                            ("adapter-a", "fixture/third")):
        config = ModelLoadConfig(source, "fixture", "fixture-task", "fixture-runtime", adapter=adapter)
        execution, device = prepare_execution(config)
        sessions.append(load_model_session(config, execution, device))

    first, second, third = sessions
    assert first.handler is third.handler
    assert first.handler is not second.handler
    assert [session.context["source"] for session in sessions] == [
        "fixture/first", "fixture/second", "fixture/third",
    ]
    assert [session.context["model_spec"]["format"] for session in sessions] == [
        "adapter-a", "adapter-b", "adapter-a",
    ]
    assert len({id(session.context) for session in sessions}) == 3
    assert len({id(session.context["model"]) for session in sessions}) == 3
    assert vars(first.handler) == vars(second.handler) == {}
    first.context["load_options"]["sentinel"] = True
    assert second.context["load_options"] == third.context["load_options"] == {}


def test_unknown_adapter_does_not_use_the_default_handler(adapter_runtime):
    from acprof.container.handlers import HandlerNotRegisteredError

    config = ModelLoadConfig(
        "fixture/model", "fixture", "fixture-task", "fixture-runtime", adapter="unknown-adapter",
    )
    with pytest.raises(HandlerNotRegisteredError, match="adapter=unknown-adapter"):
        load_model_session(config, object(), "cpu")
    assert adapter_runtime.loads == []


@pytest.mark.parametrize("conflict", ["adapter", "family"])
def test_runtime_profile_conflict_is_rejected_before_handler_loading(adapter_runtime, monkeypatch, conflict):
    from acprof.container.handlers import HandlerInitializationError

    config = ModelLoadConfig(
        "fixture/model", "fixture", "fixture-task", "fixture-runtime", adapter="adapter-a",
    )
    if conflict == "adapter":
        monkeypatch.setenv("ACPROF_RUNTIME_PROFILE", "fixture-adapter-b")
    else:
        adapter_runtime.profiles["fixture-adapter-a"] = replace(
            adapter_runtime.profiles["fixture-adapter-a"], family="other-family",
        )
    monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "adapter-b")

    with pytest.raises(HandlerInitializationError, match="runtime profile .* does not match fixture/adapter-a") as caught:
        load_model_session(config, object(), "cpu")

    assert isinstance(caught.value.__cause__, ValueError)
    assert adapter_runtime.loads == []


def test_explicit_adapter_does_not_bypass_remote_code_policy(adapter_runtime):
    from acprof.container.handlers import HandlerInitializationError
    from acprof.failures import RuntimeFailure

    config = ModelLoadConfig(
        "fixture/model", "fixture", "fixture-task", "fixture-runtime", adapter="adapter-a",
    )
    with pytest.raises(HandlerInitializationError, match="remote code forbidden") as caught:
        load_model_session(config, object(), "cpu", load_options={"trust_remote_code": True})
    assert isinstance(caught.value.__cause__, RuntimeFailure)
    assert caught.value.__cause__.failure.reason_code == "remote_code_disallowed"
    assert adapter_runtime.loads == []


def test_loading_preserves_error_identity_and_does_not_prepare_execution(monkeypatch):
    config = ModelLoadConfig("local/model", "nlp", "text-generation", "transformers_pipeline")
    monkeypatch.delenv("ACPROF_MODEL_ADAPTER", raising=False)
    handlers = importlib.import_module("acprof.container.handlers")
    module = importlib.import_module("acprof.container.execution")
    error = TimeoutError("loading timed out")
    monkeypatch.setattr(handlers, "load_handler", Mock(side_effect=error))
    prepare = Mock(side_effect=AssertionError("loading must not initialize execution again"))
    monkeypatch.setattr(module, "configured_execution", prepare)

    with pytest.raises(TimeoutError) as caught:
        load_model_session(config, object(), "cpu", handler=object())

    assert caught.value is error
    prepare.assert_not_called()
