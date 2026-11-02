"""Offline coverage of the shared loading boundary and explicit configuration."""
import importlib
import os
from dataclasses import FrozenInstanceError
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
        loader.assert_called_once_with(*args)
    else:
        loader.assert_called_once_with(*args, load_options=load_options)
        assert loader.call_args.kwargs["load_options"] is load_options
    registry.assert_called_once_with("nlp", "transformers_pipeline")
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
    monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "fixture-adapter")
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


@pytest.mark.parametrize("operation", ["select", "prepare", "load"])
def test_changed_adapter_is_rejected_before_mixing_runtime_identities(monkeypatch, operation):
    config = ModelLoadConfig("local/model", "nlp", "text-generation", "transformers_pipeline")
    monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "different-adapter")
    operations = {
        "select": lambda: select_handler(config),
        "prepare": lambda: prepare_execution(config),
        "load": lambda: load_model_session(config, object(), "cpu", handler=object()),
    }
    with pytest.raises(ValueError, match="adapter changed"):
        operations[operation]()


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
