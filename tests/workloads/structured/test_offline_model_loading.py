import json
import sys
import tempfile
import types
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from acprof.container.handlers.audio import AudioHandler
from acprof.container.handlers.cv import CVHandler
from acprof.container.handlers.multimodal import MultimodalHandler
from acprof.container.handlers.nlp import NLPHandler
from acprof.container.handlers.structured import StructuredHandler
from acprof.container.handlers.timeseries import (
    ChronosHandler,
)


def test_transformers_handlers_load_local_snapshot_without_hub_revision() -> None:
    calls = []

    def fake_pipeline(**kwargs):
        calls.append(kwargs)
        return object()

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "4.57.6"
    fake_transformers.pipeline = fake_pipeline

    handlers = [
        (NLPHandler(), "fill-mask", "transformers_pipeline"),
        (CVHandler(), "image-classification", "transformers_pipeline"),
        (CVHandler(), "image-to-text", "transformers_pipeline"),
        (AudioHandler(), "audio-classification", "transformers_pipeline"),
    ]

    with tempfile.TemporaryDirectory() as model_source, patch.dict(
        sys.modules,
        {"torch": fake_torch, "transformers": fake_transformers},
    ):
        for handler, task_type, backend in handlers:
            handler.load(
                model_source,
                task_type,
                backend,
                "cpu",
                "0123456789abcdef",
            )

    assert (len(calls)) == (len(handlers))
    for call in calls:
        assert (call["model"]) == (model_source)
        assert ("revision") not in (call)

def test_transformers_handler_keeps_revision_when_loading_by_model_id() -> None:
    calls = []

    def fake_pipeline(**kwargs):
        calls.append(kwargs)
        return object()

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "4.57.6"
    fake_transformers.pipeline = fake_pipeline

    with patch.dict(
        sys.modules,
        {"torch": fake_torch, "transformers": fake_transformers},
    ):
        NLPHandler().load(
            "google-bert/bert-base-uncased",
            "fill-mask",
            "transformers_pipeline",
            "cpu",
            "0123456789abcdef",
        )

    assert (calls[0]["revision"]) == ("0123456789abcdef")

def test_chronos_loads_local_snapshot_without_hub_revision() -> None:
    calls = []

    class FakeChronosBoltPipeline:
        @classmethod
        def from_pretrained(cls, model_source, **kwargs):
            calls.append((model_source, kwargs))
            return object()

    fake_chronos = types.ModuleType("chronos")
    fake_chronos.BaseChronosPipeline = FakeChronosBoltPipeline
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"

    with tempfile.TemporaryDirectory() as model_source, patch.dict(
        sys.modules,
        {"torch": fake_torch, "chronos": fake_chronos},
    ):
        ChronosHandler().load(
            model_source,
            "time-series-forecasting",
            "chronos",
            "cpu",
            "0123456789abcdef",
        )

    assert (calls[0][0]) == (model_source)
    assert ("revision") not in (calls[0][1])
    assert (calls[0][1]["local_files_only"])


@pytest.mark.parametrize(("handler_type", "task", "backend", "model_type"), [
    pytest.param(NLPHandler, "sentence-similarity", "transformers_pipeline", "fixture-encoder",
                 id="sentence-encoder-delegate"),
    pytest.param(AudioHandler, "audio-to-audio", "transformers_model", "encodec", id="audio-codec"),
    pytest.param(CVHandler, "video-classification", "transformers_model", "videomae", id="cv-direct"),
    pytest.param(MultimodalHandler, "image-text-to-text", "transformers_model", "qwen2_5_vl",
                 id="multimodal-declaration"),
    pytest.param(StructuredHandler, "robotics", "torchscript", "fixture-artifact", id="structured-declaration"),
])
def test_explicit_adapter_reaches_nested_loaders_without_entering_sdk_kwargs(
    monkeypatch, tmp_path, handler_type, task, backend, model_type,
):
    import acprof.container.load_policy as policy
    import acprof.extensions as extensions

    monkeypatch.delenv("ACPROF_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("ACPROF_DEPENDENCY_LOCK_SHA256", raising=False)
    profile, declaration, _ = policy.registered_policy(task, backend, "cpu")
    adapter, profile_id = "fixture-selected", "fixture-profile"
    catalog = extensions.ExtensionCatalog()
    catalog.add(replace(
        declaration, extension_id="fixture-extension", adapter=adapter, profile=profile_id,
    ))
    # No default route exists in this fixture: every downstream consumer must
    # keep the explicitly selected adapter, including delegated/helper loads.
    monkeypatch.setattr(extensions, "CATALOG", catalog)
    monkeypatch.setattr(policy, "CATALOG", catalog)
    monkeypatch.setattr(policy, "PROFILES", {
        profile_id: replace(profile, profile_id=profile_id, adapter=adapter),
    })
    monkeypatch.setenv("ACPROF_RUNTIME_PROFILE", profile_id)
    monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "another-adapter")

    config = types.SimpleNamespace(model_type=model_type, audio_channels=1, sampling_rate=24000)
    model = Mock(config=config, num_labels=1)
    model.to.return_value = model
    model.eval.return_value = model
    processor = types.SimpleNamespace(sampling_rate=24000)
    sdk_kwargs, model_calls = [], []

    def load_model(source, **kwargs):
        sdk_kwargs.append(kwargs)
        model_calls.append((source, kwargs))
        return model

    def load_config(source, **kwargs):
        sdk_kwargs.append(kwargs)
        return config

    def load_processor(source, **kwargs):
        sdk_kwargs.append(kwargs)
        return processor

    model_loader = types.SimpleNamespace(from_pretrained=load_model)
    processor_loader = types.SimpleNamespace(from_pretrained=load_processor)
    transformers = types.SimpleNamespace(
        __version__="4.57.6",
        pipeline=Mock(side_effect=AssertionError("selected path must use its direct loader")),
        AutoConfig=types.SimpleNamespace(from_pretrained=load_config),
        AutoProcessor=processor_loader, AutoImageProcessor=processor_loader,
        AutoModelForVideoClassification=model_loader,
        AutoModelForImageTextToText=model_loader, EncodecModel=model_loader,
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        float32="fixture-fp32", float16="fixture-fp16", jit=types.SimpleNamespace(load=load_model),
    ))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(
        SentenceTransformer=load_model, CrossEncoder=load_model,
    ))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    if backend == "torchscript":
        (tmp_path / "model.pt").write_bytes(b"offline-fixture")
        (tmp_path / "acprof_model.json").write_text(json.dumps({
            "schema_version": 1, "task": task, "format": backend,
            "model_file": "model.pt", "feature_dim": 3,
        }))

    context = handler_type().load(str(tmp_path), task, backend, "cpu", "fixed-sha", adapter=adapter)

    assert context.get("pipeline", context.get("model")) is model
    assert context["model_revision"] == "fixed-sha"
    assert context["load_options"] == {}
    assert model_calls
    expected_source = tmp_path / "model.pt" if backend == "torchscript" else tmp_path
    for source, kwargs in model_calls:
        assert source == str(expected_source)
        assert "revision" not in kwargs
        if backend == "torchscript":
            assert kwargs["map_location"] == "cpu"
            assert context["tensor_inputs"] is True
        else:
            assert kwargs.get("torch_dtype", kwargs.get("model_kwargs", {}).get("torch_dtype")) == "fixture-fp32"
    for kwargs in sdk_kwargs:
        assert "adapter" not in kwargs
        assert "adapter" not in kwargs.get("model_kwargs", {})
