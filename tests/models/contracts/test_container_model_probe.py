"""Offline interface probes preserve the captured adapter without running a model."""
import json
from unittest.mock import patch

import pytest

from acprof.container.handlers import BaseHandler, HandlerNotRegisteredError, HandlerRegistry
from acprof.container.model_probe import validate_interface
from acprof.container.model_session import ModelLoadConfig
from acprof.failures import RuntimeFailure


class ProbeHandler(BaseHandler):
    def load(self, model_source, task_type, backend, device, model_revision="main", *, adapter):
        raise AssertionError("interface inspection must not load a model")

    def preprocess(self, context, payload):
        raise AssertionError("interface inspection must not preprocess")

    def predict(self, context, inputs):
        raise AssertionError("interface inspection must not predict")

    def postprocess(self, context, output):
        raise AssertionError("interface inspection must not postprocess")


@pytest.fixture
def probe_environment(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    for name, value in {
        "MODEL_ID": "fixture/interface", "MODEL_LOCAL_PATH": str(tmp_path),
        "MODEL_REVISION": "pinned-fixture", "TASK_FAMILY": "structured",
        "TASK_TYPE": "tabular-regression", "RUNTIME_BACKEND": "onnxruntime",
        "ACPROF_MODEL_ADAPTER": "family-default", "ACPROF_RUNTIME_PROFILE": "",
        "USE_GPU": "0", "ACPROF_RUNTIME_THREADS": "1",
    }.items():
        monkeypatch.setenv(name, value)
    with patch.dict(HandlerRegistry._handlers, clear=True), patch.dict(HandlerRegistry._adapters, clear=True):
        yield tmp_path


def change_adapter_after_capture(monkeypatch):
    resolve_source = ModelLoadConfig.resolve_source

    def resolve(config):
        monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "unselected-adapter")
        return resolve_source(config)

    monkeypatch.setattr(ModelLoadConfig, "resolve_source", resolve)


@pytest.mark.parametrize("adapter", ["family-default", "probe-adapter"])
def test_probe_keeps_captured_adapter_and_does_not_load(probe_environment, monkeypatch, adapter):
    class UnselectedHandler(ProbeHandler):
        def __init__(self):
            raise AssertionError("later environment must not select this handler")

    HandlerRegistry.register("structured", "onnxruntime", ProbeHandler)
    HandlerRegistry.register_adapter("probe-adapter", "structured", "onnxruntime", ProbeHandler)
    HandlerRegistry.register_adapter("unselected-adapter", "structured", "onnxruntime", UnselectedHandler)
    if adapter == "family-default":
        monkeypatch.delenv("ACPROF_MODEL_ADAPTER")
    else:
        monkeypatch.setenv("ACPROF_MODEL_ADAPTER", adapter)
    change_adapter_after_capture(monkeypatch)

    result = validate_interface({})

    assert result["status"] == "ok"
    assert "adapter" in result["signatures"]["load"]
    assert set(result["signatures"]) == {"load", "preprocess", "predict", "postprocess", "validate_output"}
    assert result["inference"] == result["preprocess"] == result["output_validation"] == "not_run"


def test_probe_rejects_handler_without_adapter_keyword(probe_environment):
    class MissingAdapterHandler(ProbeHandler):
        def load(self, model_source, task_type, backend, device, model_revision="main"):
            raise AssertionError("signature validation must not call load")

    HandlerRegistry.register("structured", "onnxruntime", MissingAdapterHandler)

    with pytest.raises(TypeError, match="adapter"):
        validate_interface({})


def test_unknown_probe_adapter_cannot_use_default_handler(probe_environment, monkeypatch):
    HandlerRegistry.register("structured", "onnxruntime", ProbeHandler)
    monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "unknown-probe-adapter")

    with pytest.raises(HandlerNotRegisteredError, match="unknown-probe-adapter"):
        validate_interface({})


def custom_pipeline_metadata(probe_environment, monkeypatch):
    (probe_environment / "config.json").write_text(
        json.dumps({"custom_pipelines": {"fixture-pipeline": {}}}), encoding="utf-8",
    )
    monkeypatch.setenv("TASK_FAMILY", "multimodal")
    monkeypatch.setenv("TASK_TYPE", "audio-text-to-text")
    monkeypatch.setenv("RUNTIME_BACKEND", "transformers_model")
    monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "moss-transcribe-diarize")
    monkeypatch.setenv("ACPROF_RUNTIME_PROFILE", "moss-transformers5104")


def test_custom_pipeline_policy_keeps_adapter_and_does_not_forward_it(
    probe_environment, monkeypatch,
):
    class Pipeline:
        def preprocess(self, inputs):
            raise AssertionError("probe must not preprocess")

        def _forward(self, inputs, *, max_new_tokens):
            raise AssertionError("probe must not infer")

        def postprocess(self, output):
            raise AssertionError("probe must not postprocess")

    custom_pipeline_metadata(probe_environment, monkeypatch)
    # This branch has never required the handler-only family/revision fields.
    monkeypatch.delenv("TASK_FAMILY")
    monkeypatch.delenv("MODEL_REVISION")
    monkeypatch.setattr("acprof.container.local_pipeline.load_local_pipeline_class", lambda *_args: Pipeline)
    monkeypatch.setattr("acprof.models.spec.pipeline_task", lambda *_args: "fixture-pipeline")
    monkeypatch.setattr("acprof.models.spec.load_model_spec", lambda *_args, **_kwargs: {
        "multimodal": {"forward_kwargs": {"max_new_tokens": "$max_new_tokens"}},
    })
    change_adapter_after_capture(monkeypatch)

    result = validate_interface({})

    assert result["status"] == "ok"
    assert set(result["signatures"]) == {"preprocess", "_forward", "postprocess"}
    assert result["inference"] == "not_run"


def test_custom_pipeline_probe_rejects_actual_profile_mismatch(probe_environment, monkeypatch):
    custom_pipeline_metadata(probe_environment, monkeypatch)
    monkeypatch.setenv("ACPROF_RUNTIME_PROFILE", "nlp-cpu")

    with pytest.raises(ValueError, match="runtime profile .* does not match"):
        validate_interface({})


def test_probe_preserves_remote_code_permission(probe_environment):
    (probe_environment / "config.json").write_text(
        json.dumps({"custom_pipelines": {"fixture-pipeline": {}}}), encoding="utf-8",
    )

    with pytest.raises(RuntimeFailure, match="Interface inspection requires"):
        validate_interface({})
