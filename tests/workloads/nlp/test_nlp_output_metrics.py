"""Decoded response text metrics are distinct from generation token evidence."""
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from acprof.container.handlers.nlp import NLPHandler


def _context(task, tokenizer):
    return {
        "task_type": task,
        "pipeline": SimpleNamespace(tokenizer=tokenizer),
        "generation_evidence": {"actual_output_tokens": 99},
    }


@pytest.mark.parametrize(("task", "key"), [
    ("text-generation", "generated_text"),
    ("text2text-generation", "generated_text"),
    ("summarization", "summary_text"),
    ("translation", "translation_text"),
])
def test_generated_response_counts_each_decoded_text_without_special_tokens(task, key):
    tokens = {"你好 世界": [10, 11, 12], "a🙂": [13, 14]}
    tokenizer = SimpleNamespace(encode=Mock(side_effect=lambda text, **_: tokens[text]))

    result = NLPHandler().postprocess(
        _context(task, tokenizer), [{key: "你好 世界"}, {key: "a🙂"}],
    )

    assert result["output_length"] == 7
    assert result["output_token_count"] == 5
    assert result["actual_output_tokens"] == 99
    assert result["n_results"] == 2
    assert tokenizer.encode.call_args_list == [
        call("你好 世界", add_special_tokens=False), call("a🙂", add_special_tokens=False),
    ]


def test_nested_batch_counts_all_returned_sequences_including_prompt_text():
    tokens = {"prompt answer": [1, 2, 3], "second": [4, 5], "third": [6]}
    tokenizer = SimpleNamespace(encode=Mock(side_effect=lambda text, **_: tokens[text]))
    raw = [[{"generated_text": "prompt answer"}, {"generated_text": "second"}],
           [{"generated_text": "third"}]]

    result = NLPHandler().postprocess(_context("text-generation", tokenizer), raw)

    assert result["output_length"] == 24
    assert result["output_token_count"] == 6
    assert result["n_results"] == 2
    assert tokenizer.encode.call_count == 3


def test_valid_empty_text_has_zero_length_and_tokens():
    tokenizer = SimpleNamespace(encode=Mock(return_value=[]))

    result = NLPHandler().postprocess(
        _context("text2text-generation", tokenizer), [{"generated_text": ""}],
    )

    assert result["output_length"] == 0
    assert result["output_token_count"] == 0


@pytest.mark.parametrize("tokenizer", [None, SimpleNamespace(), SimpleNamespace(encode=None)])
def test_missing_tokenizer_preserves_known_character_count(tokenizer):
    result = NLPHandler().postprocess(
        _context("summarization", tokenizer), [{"generated_text": "你好"}],
    )

    assert result["output_length"] == 2
    assert result["output_token_count"] is None


@pytest.mark.parametrize("error", [TypeError, ValueError, RuntimeError])
def test_tokenization_failure_does_not_publish_partial_token_count(error):
    tokenizer = SimpleNamespace(encode=Mock(side_effect=[[1, 2], error("cannot tokenize")]))

    result = NLPHandler().postprocess(
        _context("translation", tokenizer), [{"generated_text": "one"}, {"generated_text": "two"}],
    )

    assert result["output_length"] == 6
    assert result["output_token_count"] is None


@pytest.mark.parametrize("raw", [
    [], [{"generated_token_ids": [1, 2]}], [{"generated_text": None}],
    [{"generated_text": "valid"}, {"score": 0.5}],
])
def test_unavailable_text_does_not_become_zero_or_partial_metrics(raw):
    tokenizer = SimpleNamespace(encode=Mock(return_value=[1]))

    result = NLPHandler().postprocess(_context("text-generation", tokenizer), raw)

    assert result["output_length"] is None
    assert result["output_token_count"] is None


def test_non_generative_output_keeps_text_metrics_inapplicable():
    tokenizer = SimpleNamespace(encode=Mock())

    result = NLPHandler().postprocess(
        _context("text-classification", tokenizer), [{"label": "positive", "score": 0.9}],
    )

    assert "output_length" not in result
    assert "output_token_count" not in result
    tokenizer.encode.assert_not_called()
