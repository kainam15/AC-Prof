"""Sentence embedding, similarity and ranking rules behind the existing NLP route."""
from __future__ import annotations

import os
from typing import Any

from acprof.container.handlers import (
    BaseHandler,
    model_revision_kwargs,
    transformers_pipeline_load_kwargs,
)
from acprof.container.handlers.text_inputs import (
    get_model_max_length,
    max_effective_qa_context_length,
    max_effective_single_text_length,
    string_list,
    truncate_qa_context,
    truncate_single_text,
)
from acprof.container.load_policy import load_policy
from acprof.input_budget import check_nlp_input, positive_integer


class SentenceEncoderHandler(BaseHandler):
    """One loaded encoder selection, independent of generic Pipeline behavior."""

    def __init__(self, encoder: str):
        if encoder not in {"sentence", "cross"}:
            raise ValueError(f"unsupported sentence encoder: {encoder}")
        self._encoder = encoder

    def load(
        self, model_source: str, task_type: str, backend: str, device: str,
        model_revision: str = "main", load_options: dict[str, Any] | None = None,
        *, adapter: str = "family-default",
    ) -> dict[str, Any]:
        policy = load_policy(model_source, task_type, backend, device, load_options, adapter=adapter)
        options = transformers_pipeline_load_kwargs(load_options)
        if self._encoder == "sentence":
            from sentence_transformers import SentenceTransformer as Encoder
        else:
            from sentence_transformers import CrossEncoder as Encoder
        pipe = Encoder(
            model_source, **model_revision_kwargs(model_source, model_revision),
            device=device, local_files_only=os.path.isdir(model_source),
            trust_remote_code=policy["trust_remote_code"],
            model_kwargs={"torch_dtype": policy["dtype"], **options.get("model_kwargs", {})},
        )
        if task_type == "text-ranking" and pipe.num_labels != 1:
            raise ValueError("text-ranking requires a single relevance score (num_labels=1)")
        return {
            "pipeline": pipe, "task_type": task_type, "backend": backend,
            "device": device, "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
        }

    def preprocess(self, model_ctx: dict[str, Any], raw_input: dict[str, Any]) -> Any:
        task_type, pipe = model_ctx["task_type"], model_ctx["pipeline"]
        params = raw_input.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        prompt = self._embedding_options(pipe, params)[1] if task_type != "text-ranking" else ""
        budget = check_nlp_input(raw_input, task_type, embedding_prompt=prompt)
        common = {"params": params,
                  "batch_size": positive_integer(raw_input.get("batch_size", 1), "batch_size"),
                  "input_budget": budget.as_dict()}
        if task_type == "feature-extraction":
            text, scale, truncated, reason = self._prepare_embedding_text(
                pipe, raw_input.get("text", ""), task_type, params=params,
            )
            return {**common, "text": text, "_effective_input_scale": scale,
                    "_truncated_by_limit": truncated, "_probe_reason": reason}

        documents = string_list(raw_input, "documents")
        query = raw_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{task_type} requires a nonempty query")
        prepare = self._prepare_embedding_text if task_type == "sentence-similarity" else truncate_single_text
        options: dict[str, Any] = {"params": params} if task_type == "sentence-similarity" else {}
        checked_query, _, query_truncated, _ = prepare(pipe, query, task_type, **options)
        if query_truncated:
            raise ValueError("query exceeds model token limit")
        results = [truncate_qa_context(pipe, checked_query, doc)
                   if task_type == "text-ranking" else
                   prepare(pipe, doc, task_type, **options) for doc in documents]
        if any(item[1] == 0 for item in results):
            raise ValueError("query leaves no model token budget for documents")
        scales = [item[1] for item in results if item[1] is not None]
        return {**common, "query": checked_query, "documents": [item[0] for item in results],
                "_effective_input_scale": max(scales) if scales else None,
                "_truncated_by_limit": any(item[2] for item in results),
                "_probe_reason": "per_candidate_tokens; " + results[0][3]}

    def get_scale_metadata(self, model_ctx: dict[str, Any], raw_input: dict[str, Any]) -> dict[str, Any]:
        pipe = model_ctx["pipeline"]
        if model_ctx["task_type"] == "text-ranking":
            available, reason = max_effective_qa_context_length(pipe, raw_input.get("query", ""))
        else:
            _, prompt = self._embedding_options(pipe, raw_input.get("params", {}))
            tokenizer = getattr(pipe, "tokenizer", None)
            reserve = len(tokenizer.encode(prompt, add_special_tokens=False)) if prompt and tokenizer else 0
            available, reason = max_effective_single_text_length(pipe, reserved_tokens=reserve)
        return {"input_scale_type": "seq_length", "max_effective_input_scale": available, "reason": reason}

    @staticmethod
    def _embedding_options(pipe: Any, params: dict[str, Any]) -> tuple[dict, str]:
        options = {key: params[key] for key in ("prompt", "prompt_name", "normalize_embeddings") if key in params}
        if "normalize_embeddings" in options and not isinstance(options["normalize_embeddings"], bool):
            raise ValueError("normalize_embeddings must be a boolean")
        if "prompt" in options and "prompt_name" in options:
            raise ValueError("set either prompt or prompt_name, not both")
        prompts = getattr(pipe, "prompts", {})
        prompts = prompts if isinstance(prompts, dict) else {}
        name = options.get("prompt_name", getattr(pipe, "default_prompt_name", None))
        if "prompt_name" in options and name not in prompts:
            raise ValueError(f"unknown sentence embedding prompt_name: {name!r}")
        prompt = options.get("prompt", prompts.get(name, "") if isinstance(name, str) else "")
        if not isinstance(prompt, str):
            raise ValueError("sentence embedding prompt must be a string")
        return options, prompt

    def _prepare_embedding_text(self, pipe: Any, text: str, task_type: str, *, params: dict) -> tuple:
        _, prompt = self._embedding_options(pipe, params)
        tokenizer = getattr(pipe, "tokenizer", None)
        reserve = len(tokenizer.encode(prompt, add_special_tokens=False)) if prompt and tokenizer else 0
        result = truncate_single_text(pipe, text, task_type, reserved_tokens=reserve)
        limit = get_model_max_length(pipe)
        if tokenizer and limit is not None:
            # Check the concatenation as well: BPE boundaries need not be additive.
            length = len(tokenizer.encode(prompt + result[0], add_special_tokens=False))
            if length + tokenizer.num_special_tokens_to_add(pair=False) > limit:
                raise ValueError("sentence embedding prompt and text exceed model token limit")
        return result

    def predict(self, model_ctx: dict[str, Any], processed_input: Any) -> Any:
        pipe, task_type = model_ctx["pipeline"], model_ctx["task_type"]
        params = processed_input.get("params", {})
        batch_size = processed_input.get("batch_size", 1)
        check_nlp_input(processed_input, task_type)
        if task_type == "feature-extraction":
            return pipe.encode([processed_input["text"]] * batch_size, batch_size=batch_size,
                               show_progress_bar=False, convert_to_tensor=True,
                               **self._embedding_options(pipe, params)[0])
        if task_type == "sentence-similarity":
            documents = processed_input["documents"]
            sample = [processed_input["query"], *documents]
            embeddings = pipe.encode(sample * batch_size, batch_size=len(sample) * batch_size,
                                     show_progress_bar=False, convert_to_tensor=True,
                                     **self._embedding_options(pipe, params)[0])
            return [pipe.similarity(embeddings[offset:offset + 1],
                                    embeddings[offset + 1:offset + len(sample)])[0]
                    for offset in range(0, len(sample) * batch_size, len(sample))]

        documents = processed_input["documents"]
        pairs = [[processed_input["query"], document] for document in documents] * batch_size
        scores = pipe.predict(pairs, batch_size=len(pairs), show_progress_bar=False,
                              convert_to_numpy=True)
        count = len(documents)
        return [sorted([{"corpus_id": index, "score": float(scores[offset + index])}
                        for index in range(count)], key=lambda item: item["score"], reverse=True)
                for offset in range(0, len(pairs), count)]

    def postprocess(self, model_ctx: dict[str, Any], raw_output: Any) -> dict[str, Any]:
        task_type = model_ctx["task_type"]
        if task_type == "feature-extraction":
            shape = list(raw_output.shape)
            if len(shape) != 2 or any(size <= 0 for size in shape):
                raise ValueError("sentence encoder must return a nonempty batch of sentence embeddings")
            return {"task": task_type, "output_type": "embedding", "output_shape": shape,
                    "n_results": shape[0]}
        return {"task": task_type,
                "output_type": "similarity" if task_type == "sentence-similarity" else "ranking",
                "n_results": len(raw_output) if isinstance(raw_output, list) else 1}
