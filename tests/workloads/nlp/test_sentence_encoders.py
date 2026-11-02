"""Encoder contracts run without model downloads or inference frameworks."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from acprof.container.handlers import HandlerInitializationError, load_handler
from acprof.container.handlers.nlp import NLPHandler
from acprof.container.handlers.sentence_encoders import SentenceEncoderHandler
from tests.workloads.nlp.test_nlp_handler import FakeTokenizer


@pytest.fixture
def encoder_context():
    def context(task, *, limit=16, **attributes):
        pipe = Mock()
        pipe.tokenizer = FakeTokenizer("[MASK]", model_max_length=limit)
        pipe.max_seq_length = None
        pipe.max_length = None
        pipe.prompts = {}
        pipe.default_prompt_name = None
        pipe.model = SimpleNamespace(config=SimpleNamespace())
        for name, value in attributes.items():
            setattr(pipe, name, value)
        return {"task_type": task, "pipeline": pipe}
    return context


@pytest.fixture
def encoder_sdk(monkeypatch):
    sdk = SimpleNamespace(SentenceTransformer=Mock(), CrossEncoder=Mock())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="fp32", float16="fp16"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", sdk)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        __version__="4.57.6", pipeline=Mock(side_effect=AssertionError("unexpected pipeline load")),
    ))
    return sdk


def test_similarity_uses_encoder_limit_and_keeps_batch_candidates_separate(encoder_context):
    handler = SentenceEncoderHandler("sentence")
    ctx = encoder_context("sentence-similarity", max_seq_length=6)
    embeddings = np.arange(18).reshape(6, 3)
    ctx["pipeline"].encode.return_value = embeddings
    ctx["pipeline"].similarity.side_effect = [np.array([[0.2, 0.9]]), np.array([[0.7, 0.1]])]
    prepared = handler.preprocess(ctx, {
        "query": "one two", "documents": ["one two three four five"] * 2, "batch_size": 2,
    })
    assert prepared["documents"] == ["one two three four"] * 2
    assert prepared["_effective_input_scale"] == 4
    assert prepared["_truncated_by_limit"]
    output = handler.predict(ctx, prepared)
    np.testing.assert_equal(output, [[0.2, 0.9], [0.7, 0.1]])
    args, kwargs = ctx["pipeline"].encode.call_args
    assert args[0] == ["one two", "one two three four", "one two three four"] * 2
    assert kwargs["batch_size"] == 6
    assert kwargs["convert_to_tensor"] is True
    assert kwargs["show_progress_bar"] is False
    first, second = ctx["pipeline"].similarity.call_args_list
    np.testing.assert_equal(first.args[0], embeddings[:1])
    np.testing.assert_equal(first.args[1], embeddings[1:3])
    np.testing.assert_equal(second.args[0], embeddings[3:4])
    np.testing.assert_equal(second.args[1], embeddings[4:6])


def test_ranking_preserves_query_budget_raw_scores_and_batch_order(encoder_context):
    handler = SentenceEncoderHandler("cross")
    ctx = encoder_context("text-ranking", limit=10, max_length=8)
    ctx["pipeline"].predict.return_value = np.array([-2.0, 3.5, 0.6, 0.2])
    prepared = handler.preprocess(ctx, {
        "query": "one two", "documents": ["one two three four five"] * 2, "batch_size": 2,
    })
    assert prepared["_effective_input_scale"] == 4
    assert handler.predict(ctx, prepared) == [
        [{"corpus_id": 1, "score": 3.5}, {"corpus_id": 0, "score": -2.0}],
        [{"corpus_id": 0, "score": 0.6}, {"corpus_id": 1, "score": 0.2}],
    ]
    args, kwargs = ctx["pipeline"].predict.call_args
    assert args[0] == [["one two", "one two three four"]] * 4
    assert kwargs["batch_size"] == 4
    assert kwargs["convert_to_numpy"] is True


def test_ranking_metadata_reserves_query_and_honors_encoder_limit(encoder_context):
    ctx = encoder_context("text-ranking", limit=20, max_length=10)
    metadata = SentenceEncoderHandler("cross").get_scale_metadata(ctx, {"query": "one two three"})
    assert metadata["max_effective_input_scale"] == 5


@pytest.mark.parametrize(("task", "encoder"), [
    ("sentence-similarity", "sentence"), ("text-ranking", "cross"),
])
def test_pair_tasks_reject_absent_candidates(encoder_context, task, encoder):
    with pytest.raises(ValueError, match="documents"):
        SentenceEncoderHandler(encoder).preprocess(encoder_context(task), {"query": "a"})


def test_sentence_load_preserves_revision_dtype_and_profiler_options(encoder_sdk):
    SentenceEncoderHandler("sentence").load(
        "org/model", "sentence-similarity", "sentence_transformers", "cpu", "fixed-sha",
        load_options={"attention_implementation": "eager"},
    )
    args, kwargs = encoder_sdk.SentenceTransformer.call_args
    assert args == ("org/model",)
    assert kwargs["revision"] == "fixed-sha"
    assert kwargs["device"] == "cpu"
    assert kwargs["model_kwargs"] == {"torch_dtype": "fp32", "attn_implementation": "eager"}
    assert kwargs["local_files_only"] is False
    encoder_sdk.CrossEncoder.assert_not_called()


def test_cross_local_load_is_offline_and_rejects_multiclass_ranker(encoder_sdk, tmp_path):
    encoder_sdk.CrossEncoder.return_value = SimpleNamespace(num_labels=3)
    with pytest.raises(ValueError, match="single.*score|num_labels"):
        SentenceEncoderHandler("cross").load(str(tmp_path), "text-ranking", "cross_encoder", "cpu", "fixed-sha")
    assert encoder_sdk.CrossEncoder.call_args.kwargs["local_files_only"] is True
    assert "revision" not in encoder_sdk.CrossEncoder.call_args.kwargs
    encoder_sdk.SentenceTransformer.assert_not_called()


def test_default_prompt_is_preserved_and_included_in_token_budget(encoder_context):
    handler = SentenceEncoderHandler("sentence")
    ctx = encoder_context(
        "sentence-similarity", limit=8, max_seq_length=8,
        prompts={"search": "find passage "}, default_prompt_name="search",
    )
    ctx["pipeline"].encode.return_value = np.eye(3)
    ctx["pipeline"].similarity.return_value = np.array([[0.2, 0.9]])
    prepared = handler.preprocess(ctx, {
        "query": "query", "documents": ["one two three four five"] * 2,
    })
    assert prepared["documents"] == ["one two three four"] * 2
    assert prepared["_truncated_by_limit"]
    handler.predict(ctx, prepared)
    kwargs = ctx["pipeline"].encode.call_args.kwargs
    assert "prompt" not in kwargs
    assert "prompt_name" not in kwargs
    assert handler.get_scale_metadata(ctx, {})["max_effective_input_scale"] == 4


def test_prompt_name_and_normalization_reach_complete_sentence_encoder(encoder_context):
    handler = SentenceEncoderHandler("sentence")
    ctx = encoder_context("feature-extraction", prompts={"search": "query "})
    ctx["pipeline"].encode.return_value = np.ones((2, 3))
    prepared = handler.preprocess(ctx, {
        "text": "one two", "batch_size": 2,
        "params": {"prompt_name": "search", "normalize_embeddings": True},
    })
    output = handler.predict(ctx, prepared)
    assert handler.postprocess(ctx, output) == {
        "task": "feature-extraction", "output_type": "embedding", "output_shape": [2, 3], "n_results": 2,
    }
    kwargs = ctx["pipeline"].encode.call_args.kwargs
    assert kwargs["prompt_name"] == "search"
    assert kwargs["normalize_embeddings"] is True
    assert "prompt" not in kwargs


@pytest.mark.parametrize(("params", "message"), [
    ({"normalize_embeddings": 1}, "must be a boolean"),
    ({"prompt": "query ", "prompt_name": "search"}, "either prompt or prompt_name"),
    ({"prompt_name": "missing"}, "unknown.*prompt_name"),
    ({"prompt": 1}, "prompt must be a string"),
])
def test_embedding_options_fail_before_encoding(encoder_context, params, message):
    ctx = encoder_context("feature-extraction")
    with pytest.raises(ValueError, match=message):
        SentenceEncoderHandler("sentence").preprocess(ctx, {"text": "one", "params": params})
    ctx["pipeline"].encode.assert_not_called()


def test_prompt_and_text_concatenation_is_checked_after_individual_token_budget(encoder_context):
    class BoundaryTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            tokens = super().encode(text, add_special_tokens=add_special_tokens)
            return tokens + [999] if text == "prefix word" else tokens

    ctx = encoder_context("feature-extraction", limit=4)
    ctx["pipeline"].tokenizer = BoundaryTokenizer("[MASK]", model_max_length=4)
    with pytest.raises(ValueError, match="prompt and text exceed model token limit"):
        SentenceEncoderHandler("sentence").preprocess(ctx, {"text": "word", "params": {"prompt": "prefix "}})
    ctx["pipeline"].encode.assert_not_called()


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3), (0, 3), (2, 0)])
def test_sentence_embeddings_must_be_a_nonempty_matrix(encoder_context, shape):
    with pytest.raises(ValueError, match="nonempty batch of sentence embeddings"):
        SentenceEncoderHandler("sentence").postprocess(
            encoder_context("feature-extraction"), SimpleNamespace(shape=shape),
        )


@pytest.mark.parametrize(("backend", "task", "constructor_name", "output_type"), [
    ("sentence_transformers", "feature-extraction", "SentenceTransformer", "embedding"),
    ("sentence_transformers", "sentence-similarity", "SentenceTransformer", "similarity"),
    ("cross_encoder", "text-ranking", "CrossEncoder", "ranking"),
    ("transformers_pipeline", "sentence-similarity", "SentenceTransformer", "similarity"),
    ("transformers_pipeline", "text-ranking", "CrossEncoder", "ranking"),
])
def test_existing_routes_delegate_all_stages_and_keep_original_identity(
    encoder_context, encoder_sdk, monkeypatch, backend, task, constructor_name, output_type,
):
    pipe = encoder_context(task, max_seq_length=8, num_labels=1)["pipeline"]
    pipe.encode.return_value = np.ones((1 if task == "feature-extraction" else 3, 2))
    pipe.similarity.return_value = np.array([[0.2, 0.9]])
    pipe.predict.return_value = np.array([0.2, 0.9])
    constructor = getattr(encoder_sdk, constructor_name)
    constructor.return_value = pipe
    checks = [{"fixture": "single loading quality capture"}]
    capture = Mock(side_effect=lambda _: nullcontext(checks))
    monkeypatch.setattr("acprof.container.loading_quality.capture_loading_quality", capture)
    handler = NLPHandler()
    options = {"attention_implementation": "eager"}
    ctx = load_handler(handler, "org/model", task, backend, "cpu", "fixed-sha", load_options=options)
    assert (ctx["backend"], ctx["task_type"], ctx["model_revision"], ctx["load_options"]) == (
        backend, task, "fixed-sha", options,
    )
    assert ctx["pipeline"] is pipe
    assert ctx["quality_checks"] == checks
    capture.assert_called_once_with(backend)
    payload = {"text": "one two"} if task == "feature-extraction" else {
        "query": "one", "documents": ["two", "three"],
    }
    prepared = handler.preprocess(ctx, payload)
    assert handler.get_scale_metadata(ctx, payload)["max_effective_input_scale"] == (
        5 if task == "text-ranking" else 6
    )
    result = handler.postprocess(ctx, handler.predict(ctx, prepared))
    assert result["output_type"] == output_type
    constructor.assert_called_once()


def test_loading_error_is_wrapped_once_at_the_existing_boundary(encoder_sdk, monkeypatch):
    original = ValueError("fixture encoder initialization failure")
    encoder_sdk.SentenceTransformer.side_effect = original
    capture = Mock(side_effect=lambda _: nullcontext([]))
    monkeypatch.setattr("acprof.container.loading_quality.capture_loading_quality", capture)
    with pytest.raises(HandlerInitializationError) as caught:
        load_handler(NLPHandler(), "org/model", "sentence-similarity", "transformers_pipeline", "cpu")
    assert caught.value.__cause__ is original
    capture.assert_called_once_with("transformers_pipeline")
    encoder_sdk.SentenceTransformer.assert_called_once()


def test_one_cached_handler_keeps_generic_sentence_and_cross_contexts_independent(encoder_context, encoder_sdk):
    generic = encoder_context("text-classification")["pipeline"]
    generic.return_value = [{"label": "fixture"}]
    sys.modules["transformers"].pipeline.side_effect = None
    sys.modules["transformers"].pipeline.return_value = generic
    sentence = encoder_context("sentence-similarity")["pipeline"]
    sentence.encode.return_value = np.eye(3)
    sentence.similarity.return_value = np.array([[0.2, 0.9]])
    encoder_sdk.SentenceTransformer.return_value = sentence
    cross = encoder_context("text-ranking", num_labels=1)["pipeline"]
    cross.predict.return_value = np.array([0.2, 0.9])
    encoder_sdk.CrossEncoder.return_value = cross
    handler = NLPHandler()
    contexts = [handler.load("org/model", task, "transformers_pipeline", "cpu") for task in (
        "text-classification", "sentence-similarity", "text-ranking",
    )]
    cases = [
        (contexts[0], {"text": "one"}, "label"),
        (contexts[1], {"query": "one", "documents": ["two", "three"]}, "similarity"),
        (contexts[2], {"query": "one", "documents": ["two", "three"]}, "ranking"),
    ]
    for ctx, payload, expected in cases * 2:
        prepared = handler.preprocess(ctx, payload)
        result = handler.postprocess(ctx, handler.predict(ctx, prepared))
        assert result["output_type"] == expected
    assert generic.call_count == 2
    assert sentence.encode.call_count == 2
    assert cross.predict.call_count == 2
