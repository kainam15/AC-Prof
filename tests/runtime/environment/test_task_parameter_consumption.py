"""Observe real Mock consumer calls, including dropped and overridden task values."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.container.handlers.audio import AudioHandler
from acprof.container.handlers.multimodal import MultimodalHandler
from acprof.container.handlers.timeseries import ChronosHandler
from acprof.container.inference_configuration import (
    attach_configuration_context,
    capture_generation_consumption,
    validate_inference_payload,
    verify_generation_consumption,
)
from acprof.container.model_session import ModelLoadConfig, ModelSession
from acprof.failures import RuntimeFailure
from acprof.models.inference_configuration import configuration_identity


def session(family, task, backend, parameters, context):
    frozen = {
        "schema_version": 1, "state": "frozen", "rule_version": "acprof-inference-v2",
        "model": {"source": "huggingface", "model_id": "fixture/model", "revision": "a" * 40},
        "backend": {"id": backend, "adapter": "family-default", "extension_id": "torch-" + family,
                    "runtime_profile": "fixture-profile"},
        "loading": {"off": {}}, "tasks": {task: {"interface": task, "fingerprint": "b" * 64,
                                               "generation": {}, "inference": parameters}},
        "requests": {}, "decisions": [], "workload": {},
        "validation": {"static": "validated", "runtime": "not_run"},
    }
    frozen["identity_sha256"] = configuration_identity(frozen)
    config = ModelLoadConfig("fixture/model", family, task, backend, model_revision="a" * 40,
                             inference_configuration=frozen, inference_task_id=task)
    attach_configuration_context(config, "cpu", context)
    return ModelSession(config, Mock(), context, SimpleNamespace(inference_context=nullcontext), "cpu")


def test_chronos_handler_delivers_the_frozen_horizon_and_restores_consumer(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    pipeline = SimpleNamespace(predict=Mock(return_value=object()))
    original = pipeline.predict
    instance = session("timeseries", "time-series-forecasting", "chronos",
                       {"prediction_length": 24}, {"pipeline": pipeline})
    validate_inference_payload(instance, {"prediction_length": 24})
    with capture_generation_consumption(instance) as receipt:
        ChronosHandler().predict(instance.context, {"context": object(), "series": [object()], "prediction_length": 24})
    report = verify_generation_consumption(instance, receipt)
    assert report["calls"][0]["parameters"]["prediction_length"]["observed"] == 24
    assert report["calls"][0]["parameters"]["prediction_length"]["source"] == "pipeline.predict.kwargs"
    assert pipeline.predict is original
    with pytest.raises(RuntimeFailure, match="frozen"):
        validate_inference_payload(instance, {"prediction_length": 64})


def test_whisper_language_reaches_generate_after_the_pipeline_boundary():
    model = SimpleNamespace(config=SimpleNamespace(model_type="whisper"), generate=Mock(return_value={"text": "fixture"}))

    class Pipeline:
        def __init__(self):
            self.model = model

        def __call__(self, inputs, **kwargs):
            return self.model.generate(**kwargs["generate_kwargs"])

    context = {"pipeline": Pipeline(), "task_type": "automatic-speech-recognition",
               "audio_metadata": {"model_type": "whisper", "max_short_form_duration_s": 30}}
    instance = session("audio", context["task_type"], "transformers_pipeline",
                       {"language": "zh", "asr_task": "transcribe"}, context)
    with capture_generation_consumption(instance) as receipt:
        AudioHandler().predict(context, {"audio": [0.0] * 16, "sample_rate": 16000,
                                        "params": {"language": "zh", "asr_task": "transcribe"}})
    assert verify_generation_consumption(instance, receipt)["calls"][0]["parameters"]["asr_task"]["observed"] == "transcribe"
    assert model.generate.call_args.kwargs == {"language": "zh", "task": "transcribe"}


@pytest.mark.parametrize("drop", [False, True])
def test_multimodal_handler_cannot_skip_or_replace_the_consumer(monkeypatch, drop):
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    model = SimpleNamespace(generate=Mock(return_value=[1]), config=SimpleNamespace(model_type="qwen2_5_vl"))
    instance = session("multimodal", "image-text-to-text", "transformers_model",
                       {"max_new_tokens": 32, "do_sample": False}, {"model": model, "mode": "generation"})
    def infer():
        with capture_generation_consumption(instance) as receipt:
            if not drop:
                MultimodalHandler().predict(instance.context, {"inputs": {}, "prompt_length": 1,
                                                               "params": {"max_new_tokens": 64, "do_sample": False}})
        verify_generation_consumption(instance, receipt)
    with pytest.raises(RuntimeFailure):
        infer()


def test_diffusers_special_call_method_is_observed_and_restored():
    class StableDiffusionXLPipeline:
        def __call__(self, **kwargs):
            return kwargs["num_inference_steps"]

    pipeline = StableDiffusionXLPipeline()
    instance = session("diffusion", "text-to-image", "diffusers",
                       {"num_inference_steps": 12, "guidance_scale": 4.5}, {"pipeline": pipeline})
    with capture_generation_consumption(instance) as receipt:
        assert instance.context["pipeline"](num_inference_steps=12, guidance_scale=4.5) == 12
    assert instance.context["pipeline"] is pipeline
    assert verify_generation_consumption(instance, receipt)["status"] == "verified"
