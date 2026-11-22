"""The registered causal pipeline receives one user turn and measures the templated request."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from acprof.container.handlers import InputLimitError
from acprof.container.handlers.nlp import NLPHandler
from acprof.input_budget import InputBudgetExceeded


class ChatTokenizer:
    model_max_length = 16
    pad_token_id = None
    eos_token_id = 2
    eos_token = "</s>"
    chat_template = "fixture-template"

    def get_chat_template(self):
        if not self.chat_template:
            raise ValueError("chat template is not set")
        return self.chat_template

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize
        assert add_generation_prompt
        message, = messages
        assert message["role"] == "user"
        return f"<bos> <user> {message['content']} <end> <assistant>"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(len(text.split())))

    def num_special_tokens_to_add(self, pair):
        return 2


class Pipeline:
    def __init__(self):
        self.generation_config = SimpleNamespace(pad_token_id=None)
        self.tokenizer = ChatTokenizer()
        self.model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))
        self.call = Mock()

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


@pytest.fixture
def chat():
    pipe = Pipeline()
    factory = Mock(return_value=pipe)
    with patch.dict("sys.modules", {
        "transformers": SimpleNamespace(__version__="4.57.6", pipeline=factory, set_seed=Mock()),
        "torch": SimpleNamespace(float32="fp32", float16="fp16"),
    }):
        yield NLPHandler(), pipe, factory


@pytest.mark.parametrize("batch", [1, 2])
def test_registered_chat_load_predict_and_output(chat, batch):
    handler, pipe, factory = chat
    ctx = handler.load("fixture/model", "conversational", "transformers_pipeline", "cpu", "fixed")
    assert factory.call_args.kwargs["task"] == "text-generation"
    assert factory.call_args.kwargs["revision"] == "fixed"
    assert pipe.tokenizer.padding_side == "left"
    assert pipe.generation_config.pad_token_id == 2
    params = {"num_beams": 3, "num_return_sequences": 2, "do_sample": False, "max_new_tokens": 3}
    raw = {"text": "one two three", "batch_size": batch, "params": params}
    processed = handler.preprocess(ctx, raw)
    assert processed["_effective_input_scale"] == 7
    assert processed["_input_token_scope"] == "templated_prompt_per_sample_including_special_tokens"
    assert processed["_truncated_by_limit"] is False
    assert handler.get_scale_metadata(ctx, raw)["max_effective_input_scale"] == 13
    sequences = [{"generated_text": "yes"}, {"generated_text": "no"}]
    pipe.call.return_value = sequences if batch == 1 else [sequences, sequences]
    output = handler.predict(ctx, processed)
    messages = [{"role": "user", "content": "one two three"}]
    pipe.call.assert_called_once_with(messages if batch == 1 else [messages, messages], **params,
                                      return_full_text=False, truncation=False,
                                      **({"batch_size": batch} if batch > 1 else {}))
    result = handler.postprocess(ctx, output)
    assert result["n_results"] == 2 * batch
    assert (result["task"], result["output_length"], result["output_token_count"]) == (
        "conversational", 5 * batch, 2 * batch)


def test_chat_rejects_missing_template_at_load(chat):
    handler, pipe, _ = chat
    pipe.tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat.template"):
        handler.load("fixture/model", "conversational", "transformers_pipeline", "cpu")


@pytest.mark.parametrize(("limit", "available"), [
    ({"max_new_tokens": 7}, 9), ({"max_length": 13, "min_new_tokens": 4}, 9),
])
def test_chat_limit_uses_complete_untruncated_template(chat, limit, available):
    handler, pipe, _ = chat
    ctx = handler.load("fixture/model", "conversational", "transformers_pipeline", "cpu")
    raw = {"text": "a b c d e f", "params": limit}
    assert handler.get_scale_metadata(ctx, raw)["max_effective_input_scale"] == available
    with pytest.raises(InputLimitError) as caught:
        handler.preprocess(ctx, raw)
    assert caught.value.probe_metadata["limit_exceeded"]
    assert caught.value.probe_metadata["effective_input_scale"] == 10
    pipe.call.assert_not_called()


def test_chat_text_budget_includes_rendered_template_before_batch_expansion(chat):
    handler, pipe, _ = chat
    ctx = handler.load("fixture/model", "conversational", "transformers_pipeline", "cpu")
    raw = {"text": "hi", "batch_size": 2, "params": {"max_new_tokens": 1},
           "input_budget": {"max_text_characters": 40}}
    with pytest.raises(InputBudgetExceeded):
        handler.preprocess(ctx, raw)
    pipe.call.assert_not_called()


@pytest.mark.parametrize("output", [
    [{"generated_text": [{"role": "user", "content": "input"}]}],
    {"generated_text": "bare result"},
    [],
    [[{"generated_text": "unexpected batch"}]],
])
def test_chat_refuses_outputs_that_do_not_describe_new_text(chat, output):
    handler, pipe, _ = chat
    ctx = handler.load("fixture/model", "conversational", "transformers_pipeline", "cpu")
    processed = handler.preprocess(ctx, {"text": "hi", "params": {"max_new_tokens": 1}})
    pipe.call.return_value = output
    with pytest.raises(ValueError, match="chat.*output"):
        handler.predict(ctx, processed)


def test_custom_pipeline_named_text_generation_keeps_declared_text_protocol(chat, tmp_path):
    handler, pipe, factory = chat
    spec = {"schema_version": 1, "format": "transformers-pipeline", "task": "conversational",
            "pipeline_task": "text-generation"}
    (tmp_path / "acprof_model.json").write_text(json.dumps(spec))
    (tmp_path / "config.json").write_text(json.dumps({"custom_pipelines": {
        "text-generation": {"impl": "custom.Pipeline"}}}))
    (tmp_path / "custom.py").write_text("# fixture")
    pipe.tokenizer.chat_template = None
    ctx = handler.load(str(tmp_path), "conversational", "transformers_pipeline", "cpu")
    assert factory.call_args.kwargs["task"] == "text-generation"
    assert not ctx.get("native_chat")
    sample = handler.preprocess(ctx, {"text": "hello", "params": {"max_new_tokens": 1}})
    assert sample["_effective_input_scale"] == 1
    handler.predict(ctx, sample)
    pipe.call.assert_called_once_with("hello", max_new_tokens=1, do_sample=False, truncation=True)


def test_chat_respects_configured_default_number_of_returns(chat):
    handler, pipe, _ = chat
    pipe.generation_config.num_return_sequences = 2
    ctx = handler.load("fixture/model", "conversational", "transformers_pipeline", "cpu")
    sample = handler.preprocess(ctx, {"text": "hello", "params": {"max_new_tokens": 1}})
    pipe.call.return_value = [{"generated_text": "first"}, {"generated_text": "second"}]
    assert handler.predict(ctx, sample) == pipe.call.return_value


@pytest.mark.parametrize("observed_rows", [1, 2])
def test_chat_counts_generated_ids_for_every_returned_candidate(chat, observed_rows):
    handler, pipe, _ = chat
    prompt = [10, 11, 12, 13, 14]
    pipe.generation_config.eos_token_id = 2
    pipe.preprocess = lambda messages: {"input_ids": np.array([prompt])}
    pipe.postprocess = lambda model_output: [{"generated_text": "yes"}, {"generated_text": "no"}]
    ctx = handler.load("fixture/model", "conversational", "transformers_pipeline", "cpu")

    def generate(chats, **kwargs):
        records = []
        for messages in chats:
            inputs = pipe.preprocess(messages)
            sequences = [prompt + [2, 0], prompt + [99, 2]][:observed_rows]
            records.append(pipe.postprocess({**inputs, "generated_sequence": np.array([sequences])}))
        return records

    pipe.call.side_effect = generate
    sample = handler.preprocess(ctx, {"text": "input", "batch_size": 2,
                                      "params": {"max_new_tokens": 2, "num_return_sequences": 2}})
    response = handler.postprocess(ctx, handler.predict(ctx, sample))
    assert response["actual_input_tokens"] == 10
    assert response["n_results"] == 4
    assert response["actual_output_tokens_per_sequence"] == ([1, 2, 1, 2] if observed_rows == 2 else None)
    assert response["actual_output_tokens"] == (6 if observed_rows == 2 else None)


def test_explicit_total_bounds_chat_when_model_context_is_unknown(chat):
    handler, pipe, _ = chat
    pipe.tokenizer.model_max_length = 10**30
    ctx = handler.load("fixture/model", "conversational", "transformers_pipeline", "cpu")
    meta = handler.get_scale_metadata(ctx, {"params": {"max_length": 8}})
    assert meta["max_effective_input_scale"] == 7
    assert "model context limit unknown" in meta["reason"]
    assert handler.get_scale_metadata(ctx, {"params": {"max_new_tokens": 3}})["max_effective_input_scale"] is None
