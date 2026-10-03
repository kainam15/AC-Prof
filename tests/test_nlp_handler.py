import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from acprof.container.handlers.nlp import NLPHandler


class FakeTokenizer:
    def __init__(
        self,
        mask_token,
        *,
        mask_token_id=99,
        model_max_length=16,
        special_tokens=2,
    ):
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id
        self._encoded_mask_token_id = (
            99 if mask_token_id is None else mask_token_id
        )
        self.model_max_length = model_max_length
        self._special_tokens = special_tokens
        self._next_token_id = 100
        self._token_to_id = {}
        self._id_to_token = {}

    def convert_tokens_to_ids(self, token):
        if token == self.mask_token:
            return self._encoded_mask_token_id
        return self._id_for_token(token)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [
            self._encoded_mask_token_id
            if token == self.mask_token
            else self._id_for_token(token)
            for token in text.split()
        ]

    def decode(self, token_ids, skip_special_tokens):
        tokens = []
        for token_id in token_ids:
            if token_id == self._encoded_mask_token_id:
                if not skip_special_tokens:
                    tokens.append(self.mask_token)
            else:
                tokens.append(self._id_to_token[token_id])
        return " ".join(tokens)

    def num_special_tokens_to_add(self, pair):
        del pair
        return self._special_tokens

    def _id_for_token(self, token):
        if token not in self._token_to_id:
            token_id = self._next_token_id
            self._next_token_id += 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]


class TestNLPHandlerFillMask:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.handler = NLPHandler()

    def _preprocess(self, tokenizer, text, task_type="fill-mask"):
        model_ctx = {
            "task_type": task_type,
            "pipeline": SimpleNamespace(tokenizer=tokenizer),
        }
        return self.handler.preprocess(
            model_ctx,
            {"text": text, "params": {}},
        )

    def test_keeps_bert_native_mask_token(self):
        processed = self._preprocess(
            FakeTokenizer("[MASK]"),
            "the [MASK] token",
        )

        assert (processed["text"]) == ("the [MASK] token")
        assert (processed["_effective_input_scale"]) == (3)
        assert not (processed["_truncated_by_limit"])

    def test_translates_portable_placeholder_to_roberta_mask_token(self):
        processed = self._preprocess(
            FakeTokenizer("<mask>"),
            "the [MASK] token",
        )

        assert (processed["text"]) == ("the <mask> token")
        assert (processed["_effective_input_scale"]) == (3)
        assert not (processed["_truncated_by_limit"])

    def test_uses_arbitrary_tokenizer_mask_token_without_model_name_rules(self):
        tokenizer = FakeTokenizer("<custom-mask>", mask_token_id=None)

        processed = self._preprocess(
            tokenizer,
            "the [MASK] token",
        )

        assert (processed["text"]) == ("the <custom-mask> token")
        assert (99) in (tokenizer.encode(processed["text"]))

    def test_appends_native_mask_when_input_has_no_placeholder(self):
        processed = self._preprocess(
            FakeTokenizer("<mask>"),
            "the token",
        )

        assert (processed["text"]) == ("the token <mask>")
        assert (processed["_effective_input_scale"]) == (3)

    def test_preserves_native_mask_when_truncating_past_its_position(self):
        tokenizer = FakeTokenizer("<mask>", model_max_length=6)

        processed = self._preprocess(
            tokenizer,
            "one two three four five [MASK]",
        )

        assert (processed["text"]) == ("one two three <mask>")
        assert (processed["_effective_input_scale"]) == (4)
        assert (processed["_truncated_by_limit"])
        assert (tokenizer.mask_token_id) in (tokenizer.encode(processed["text"]))

    def test_does_not_normalize_mask_placeholder_for_other_tasks(self):
        processed = self._preprocess(
            FakeTokenizer("<mask>"),
            "the [MASK] token",
            task_type="text-classification",
        )

        assert (processed["text"]) == ("the [MASK] token")

    def test_rejects_fill_mask_tokenizer_without_mask_token(self):
        tokenizer = FakeTokenizer(None, mask_token_id=None)

        with pytest.raises(ValueError, match="tokenizer with a configured mask_token"):
            self._preprocess(tokenizer, "the [MASK] token")


class TestNLPTaskCompatibility:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.handler = NLPHandler()

    def context(self, task, *, limit=16, **kwargs):
        pipe = Mock()
        pipe.tokenizer = FakeTokenizer("[MASK]", model_max_length=limit)
        pipe.max_seq_length = None
        pipe.max_length = None
        for name, value in kwargs.items():
            setattr(pipe, name, value)
        return {"task_type": task, "pipeline": pipe}

    def test_zero_shot_reserves_tokens_for_longest_candidate_hypothesis(self):
        ctx = self.context("zero-shot-classification", limit=12)
        processed = self.handler.preprocess(ctx, {
            "text": "one two three four five six seven eight nine",
            "candidate_labels": ["news", "world politics"],
            "hypothesis_template": "This is {}.",
        })
        assert (processed["_effective_input_scale"]) == (6)
        assert (processed["_truncated_by_limit"])
        self.handler.predict(ctx, processed)
        assert (ctx["pipeline"].call_args.kwargs["candidate_labels"]) == (["news", "world politics"])
        assert ("truncation") not in (ctx["pipeline"].call_args.kwargs)

    def test_zero_shot_requires_candidate_labels(self):
        with pytest.raises(ValueError, match="candidate_labels"):
            self.handler.preprocess(self.context("zero-shot-classification"),
                                    {"text": "one two"})

    def test_sentence_similarity_uses_encoder_limit_and_real_candidates(self):
        ctx = self.context("sentence-similarity", max_seq_length=6)
        ctx["pipeline"].encode.return_value = np.eye(3)
        ctx["pipeline"].similarity.return_value = np.array([[0.2, 0.9]])
        processed = self.handler.preprocess(ctx, {
            "query": "one two", "documents": ["one two three four five"] * 2,
        })
        assert (processed["documents"]) == (["one two three four"] * 2)
        assert (processed["_effective_input_scale"]) == (4)
        assert (processed["_truncated_by_limit"])
        output = self.handler.predict(ctx, processed)
        np.testing.assert_equal(output[0], [0.2, 0.9])
        args, kwargs = ctx["pipeline"].encode.call_args
        assert (args[0]) == (["one two", "one two three four", "one two three four"])
        assert not (kwargs["show_progress_bar"])

    def test_ranking_preserves_query_budget_and_orders_document_scores(self):
        ctx = self.context("text-ranking", limit=10, max_length=8)
        ctx["pipeline"].predict.return_value = np.array([0.1, 0.8, 0.6, 0.2])
        processed = self.handler.preprocess(ctx, {
            "query": "one two", "documents": ["one two three four five"] * 2,
            "batch_size": 2,
        })
        assert (processed["_effective_input_scale"]) == (4)
        output = self.handler.predict(ctx, processed)
        assert ([item[0]["corpus_id"] for item in output]) == ([1, 0])
        assert (len(ctx["pipeline"].predict.call_args.args[0])) == (4)
        assert (ctx["pipeline"].predict.call_args.kwargs["batch_size"]) == (4)

    def test_pair_task_metadata_reserves_query_and_honors_encoder_limit(self):
        ctx = self.context("text-ranking", limit=20, max_length=10)
        metadata = self.handler.get_scale_metadata(ctx, {"query": "one two three"})
        assert (metadata["max_effective_input_scale"]) == (5)

    @pytest.mark.parametrize('task', ('sentence-similarity', 'text-ranking'))
    def test_pair_tasks_reject_absent_candidates(self, task):
        with pytest.raises(ValueError, match="documents"):
            self.handler.preprocess(self.context(task), {"query": "a"})

    def test_token_classification_calls_supported_pipeline_signature_and_batches(self):
        ctx = self.context("token-classification")
        processed = self.handler.preprocess(ctx, {"text": "one two", "batch_size": 3})
        self.handler.predict(ctx, processed)
        args, kwargs = ctx["pipeline"].call_args
        assert (args[0]) == (["one two"] * 3)
        assert (kwargs["batch_size"]) == (3)
        assert ("truncation") not in (kwargs)

    def test_question_answering_batches_complete_question_context_pairs(self):
        ctx = self.context("question-answering")
        processed = self.handler.preprocess(ctx, {
            "question": "who", "context": "one two", "batch_size": 2,
        })
        self.handler.predict(ctx, processed)
        assert (ctx["pipeline"].call_args.args[0]) == ([{"question": "who", "context": "one two"}] * 2)

    def test_table_qa_passes_rectangular_table_and_preserves_row_scale(self):
        ctx = self.context("table-question-answering", limit=128)
        ctx["pipeline"].tokenizer = Mock(model_max_length=128)
        ctx["pipeline"].tokenizer.return_value = {"input_ids": [1] * 14}
        table = {"name": ["A", "B"], "value": ["1", "2"]}
        processed = self.handler.preprocess(ctx, {"table": table, "query": "which name", "batch_size": 2})
        assert (processed["_effective_input_scale"]) == (2)
        assert not (processed["_truncated_by_limit"])
        self.handler.predict(ctx, processed)
        _, kwargs = ctx["pipeline"].call_args
        assert (kwargs["table"].to_dict(orient="list")) == (table)
        assert (kwargs["query"]) == (["which name", "which name"])
        assert (kwargs["batch_size"]) == (1)
        assert (kwargs["truncation"]) is (False)

    def test_table_qa_rejects_ragged_columns_and_never_silently_drops_rows(self):
        ctx = self.context("table-question-answering", limit=10)
        with pytest.raises(ValueError, match="same number of rows"):
            self.handler.preprocess(ctx, {"table": {"a": ["1"], "b": []}, "query": "q"})
        ctx["pipeline"].tokenizer = Mock(model_max_length=10)
        ctx["pipeline"].tokenizer.return_value = {"input_ids": [1] * 11}
        with pytest.raises(ValueError, match="table.*model.*limit"):
            self.handler.preprocess(ctx, {"table": {"a": ["1"]}, "query": "q"})

    def test_sentence_transformer_load_preserves_revision_and_profiler_options(self):
        constructor = Mock(return_value=SimpleNamespace())
        with patch.dict("sys.modules", {
            "torch": SimpleNamespace(float32="fp32", float16="fp16"),
            "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor),
            "transformers": SimpleNamespace(__version__="4.57.6", pipeline=Mock()),
        }):
            self.handler.load("org/model", "sentence-similarity", "sentence_transformers",
                              "cpu", model_revision="fixed-sha",
                              load_options={"attention_implementation": "eager"})
        assert (constructor.call_args.args) == (("org/model",))
        assert (constructor.call_args.kwargs["revision"]) == ("fixed-sha")
        assert (constructor.call_args.kwargs["model_kwargs"]["attn_implementation"]) == ("eager")

    def test_cross_encoder_local_load_is_offline_and_rejects_multiclass_ranker(self):
        constructor = Mock(return_value=SimpleNamespace(num_labels=3))
        with tempfile.TemporaryDirectory() as local, patch.dict("sys.modules", {
            "torch": SimpleNamespace(float32="fp32", float16="fp16"),
            "sentence_transformers": SimpleNamespace(CrossEncoder=constructor),
            "transformers": SimpleNamespace(__version__="4.57.6", pipeline=Mock()),
        }):
            with pytest.raises(ValueError, match="single.*score|num_labels"):
                self.handler.load(local, "text-ranking", "cross_encoder", "cpu", "fixed-sha")
        assert (constructor.call_args.kwargs["local_files_only"])
        assert ("revision") not in (constructor.call_args.kwargs)

    def test_causal_generation_load_configures_padding_for_real_batches(self):
        tokenizer = SimpleNamespace(pad_token_id=None, eos_token="</s>", eos_token_id=2,
                                    pad_token=None, padding_side="right")
        pipe = SimpleNamespace(tokenizer=tokenizer, generation_config=SimpleNamespace(pad_token_id=None))
        with patch.dict("sys.modules", {
            "torch": SimpleNamespace(float32="fp32", float16="fp16"),
            "transformers": SimpleNamespace(__version__="4.57.6", pipeline=Mock(return_value=pipe)),
        }):
            self.handler.load("org/model", "text-generation", "transformers_pipeline", "cpu")
        assert (tokenizer.pad_token) == ("</s>")
        assert (tokenizer.padding_side) == ("left")
        assert (pipe.generation_config.pad_token_id) == (2)

    def test_decoder_generation_reserves_output_with_architecture_context_limit(self):
        ctx = self.context("text-generation", limit=100)
        ctx["pipeline"].model = SimpleNamespace(config=SimpleNamespace(
            is_encoder_decoder=False, max_position_embeddings=12,
        ))
        raw_input = {"text": "one two three four five six seven eight nine",
                     "params": {"max_new_tokens": 4}}
        metadata = self.handler.get_scale_metadata(ctx, raw_input)
        processed = self.handler.preprocess(ctx, raw_input)
        assert (metadata["max_effective_input_scale"]) == (6)
        assert (processed["_effective_input_scale"]) == (6)
        assert (processed["_truncated_by_limit"])
        assert ("reserved_output_tokens=4") in (processed["_probe_reason"])

    def test_encoder_decoder_input_limit_is_independent_of_output_tokens(self):
        ctx = self.context("summarization", limit=12)
        ctx["pipeline"].model = SimpleNamespace(config=SimpleNamespace(
            is_encoder_decoder=True, max_position_embeddings=12,
        ))
        raw_input = {"text": "one two three four five six seven eight nine",
                     "params": {"max_new_tokens": 4}}
        assert (self.handler.get_scale_metadata(ctx, raw_input)["max_effective_input_scale"]) == (10)
        processed = self.handler.preprocess(ctx, raw_input)
        assert (processed["_effective_input_scale"]) == (9)
        assert not (processed["_truncated_by_limit"])

    def test_generation_rejects_output_budget_that_fills_entire_context(self):
        ctx = self.context("text-generation", limit=8)
        with pytest.raises(ValueError, match="max_new_tokens.*input token budget"):
            self.handler.get_scale_metadata(ctx, {"params": {"max_new_tokens": 8}})
