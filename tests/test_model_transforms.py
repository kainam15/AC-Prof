"""Bounded declarative input adapters preserve media and reject executable DSLs."""
import copy

import pytest

from acprof.container.handlers.custom_pipeline import pipeline_inputs
from acprof.model_spec import validate_model_spec


def spec(inputs):
    return {"schema_version": 1, "format": "transformers-pipeline", "task": "audio-text-to-text",
            "pipeline_task": "chat-audio", "multimodal": {"inputs": inputs,
            "forward_kwargs": {"max_new_tokens": "$max_new_tokens", "do_sample": False}}}


def test_chat_template_reaches_preprocess_with_original_media():
    declaration = spec({"turns": {"template": [{"role": "user", "content": {"from": "text"}}]},
                        "audio": {"from": "audio"}, "rate": "sampling_rate",
                        "mode": {"literal": "transcribe"}})
    validate_model_spec(declaration)
    class Pipeline:
        def preprocess(self, payload):
            return payload
    audio = object()
    result = pipeline_inputs({"pipeline": Pipeline(), "pipeline_protocol": declaration["multimodal"]},
                             [{"type": "text", "text": "你好"}], {"audio": [audio], "sampling_rate": 16000})
    assert (result["turns"]) == ([{"role": "user", "content": "你好"}])
    assert (result["audio"]) is (audio)
    assert (result["rate"]) == (16000)
    assert (result["mode"]) == ("transcribe")
    result["turns"][0]["role"] = "changed"
    assert (declaration["multimodal"]["inputs"]["turns"]["template"][0]["role"]) == ("user")

@pytest.mark.parametrize('value_case', range(7))
def test_invalid_or_unbounded_transforms_are_rejected(value_case):
    nested = {"from": "text"}
    for _ in range(12):
        nested = [nested]
    bad = [{"python": "import os"}, {"from": "os.environ"}, {"from": "text", "extra": 1},
           {"literal": float("nan")}, {"template": nested}, {"template": ["x"] * 300},
           {"template": {"content": {"from": "missing"}}}]
    value = tuple(bad)[value_case]
    with pytest.raises((ValueError, TypeError)):
        validate_model_spec(spec({"prompt": value, "audio": "audio", "rate": "sampling_rate"}))

def test_literal_cannot_substitute_for_required_text():
    with pytest.raises(ValueError, match="text"):
        validate_model_spec(spec({"turns": {"literal": [{"content": "text"}]},
                                  "audio": "audio", "rate": "sampling_rate"}))

def test_old_rename_mapping_remains_valid():
    original = spec({"prompt": "text", "audio": "audio", "rate": "sampling_rate"})
    assert (validate_model_spec(copy.deepcopy(original))) == (original)
