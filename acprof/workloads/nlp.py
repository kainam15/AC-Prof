"""NLP workload generator - synthetic text at target token lengths."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from acprof.input_budget import InputBudget, check_nlp_input, positive_integer
from acprof.workloads import WorkloadGenerator, register_generator

# Fixed corpus for deterministic text generation
_CORPUS_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "a journey of a thousand miles begins with a single step "
    "to be or not to be that is the question "
    "all that glitters is not gold "
    "the early bird catches the worm "
    "knowledge is power and power is knowledge "
    "actions speak louder than words in every situation "
    "time and tide wait for no man in this world "
).split()

DEFAULT_QA_QUESTION = "What is the main topic?"
DEFAULT_PAIR_QUERY = "What is the main topic?"
DEFAULT_CANDIDATE_LABELS = ["science", "sports", "politics"]
DEFAULT_HYPOTHESIS_TEMPLATE = "This example is {}."


class NLPWorkloadGenerator(WorkloadGenerator):
    """Generate synthetic text of target sequence length (in approximate tokens)."""

    def __init__(self, model_id: str, task_type: str, batch_size: int,
                 workload_spec_path: str | None = None):
        super().__init__(model_id, task_type, batch_size)
        self.batch_size = positive_integer(batch_size, "batch_size")
        self._budget = InputBudget()
        self._params: dict = {}
        self._spec_metadata: dict = {}
        self._task_workload: dict = {}
        if workload_spec_path:
            raw = Path(workload_spec_path).read_bytes()
            spec = json.loads(raw)
            if not isinstance(spec, dict) or spec.get("schema_version") != 1:
                raise ValueError("NLP workload requires schema_version=1")
            unknown = set(spec) - {"schema_version", "task", "params", "input_budget", "task_workload"}
            if unknown or spec.get("task", task_type) != task_type:
                raise ValueError(f"invalid NLP workload task or fields: {sorted(unknown)}")
            self._budget = InputBudget.from_value(spec.get("input_budget"))
            params = spec.get("params", {})
            from acprof.tasks.workloads import (
                GENERATION_KEYS,
                task_fingerprint,
                validate_generation,
            )
            self._task_workload = spec.get("task_workload", {})
            if self._task_workload:
                if (self._task_workload.get("interface") != task_type
                        or self._task_workload.get("fingerprint") != task_fingerprint(self._task_workload)):
                    raise ValueError("task workload fingerprint or interface mismatch")
            allowed = ({"prompt", "prompt_name", "normalize_embeddings"} if task_type in
                       {"feature-extraction", "sentence-similarity"} else
                       GENERATION_KEYS if task_type in {"text-generation", "text2text-generation", "summarization",
                                                          "translation", "conversational"} else set())
            if not isinstance(params, dict) or set(params) - allowed:
                raise ValueError(f"unsupported params for NLP workload task {task_type}")
            if allowed == GENERATION_KEYS:
                validate_generation(params)
            if "max_new_tokens" in params and (type(params["max_new_tokens"]) is not int or params["max_new_tokens"] < 1):
                raise ValueError("max_new_tokens must be a positive integer")
            if "normalize_embeddings" in params and not isinstance(params["normalize_embeddings"], bool):
                raise ValueError("normalize_embeddings must be a boolean")
            if any(key in params and not isinstance(params[key], str) for key in ("prompt", "prompt_name")):
                raise ValueError("embedding prompt and prompt_name must be strings")
            if "prompt" in params and "prompt_name" in params:
                raise ValueError("set either prompt or prompt_name, not both")
            self._params = params
            self._spec_metadata = {"workload_spec_sha256": hashlib.sha256(raw).hexdigest(),
                                   "params": copy.deepcopy(params)}
        self._budget.batch(self.batch_size)

    def _generate_text_from_word_count(self, word_count: int) -> str:
        n_words = max(1, int(word_count))
        cycles, rest = divmod(n_words, len(_CORPUS_WORDS))
        characters = cycles * sum(map(len, _CORPUS_WORDS)) + sum(map(len, _CORPUS_WORDS[:rest])) + n_words - 1
        candidates = 3 if self.task_type in {"sentence-similarity", "zero-shot-classification"} else 2 if self.task_type == "text-ranking" else 1
        self._budget.text(characters * self.batch_size * candidates,
                          batch_items=self.batch_size * candidates)
        words = []
        for i in range(n_words):
            words.append(_CORPUS_WORDS[i % len(_CORPUS_WORDS)])
        return " ".join(words)

    def generate_for_word_count(self, word_count: int) -> dict[str, Any]:
        text = self._generate_text_from_word_count(word_count)
        if self._task_workload:
            from acprof.tasks.workloads import CORPORA, render_task_text
            if self._task_workload["kind"] in CORPORA or self._task_workload["config"].get("input_text"):
                text = render_task_text(self._task_workload, word_count)

        payload: dict[str, Any] = {
            "text": text, "params": {}, "batch_size": self.batch_size,
            "input_budget": self._budget.as_dict(),
        }

        if self.task_type == "fill-mask":
            # Portable placeholder: the container replaces it with the loaded
            # tokenizer's native mask token (for example, RoBERTa uses <mask>).
            words = text.split()
            if len(words) > 1:
                mask_pos = len(words) // 2
                words[mask_pos] = "[MASK]"
            else:
                words.append("[MASK]")
            payload["text"] = " ".join(words)
        elif self.task_type in ("text-generation", "text2text-generation",
                              "summarization", "translation", "conversational"):
            if not (self._params.keys() & {"max_length", "max_new_tokens"}):
                payload["params"]["max_new_tokens"] = 64
        elif self.task_type == "question-answering":
            payload["question"] = DEFAULT_QA_QUESTION
            payload["context"] = text
        elif self.task_type == "zero-shot-classification":
            payload["candidate_labels"] = list(DEFAULT_CANDIDATE_LABELS)
            payload["hypothesis_template"] = DEFAULT_HYPOTHESIS_TEMPLATE
        elif self.task_type in {"sentence-similarity", "text-ranking"}:
            payload["query"] = DEFAULT_PAIR_QUERY
            # Two candidates are part of one request sample. Length is varied,
            # while candidate count and the query stay fixed across scales.
            payload["documents"] = [text, " ".join(reversed(text.split()))]

        payload["params"].update(copy.deepcopy(self._params))
        if self._task_workload:
            payload["task_workload"] = {key: self._task_workload[key] for key in ("task_id", "fingerprint", "interface")}
        check_nlp_input(payload, self.task_type, embedding_prompt=self._params.get("prompt", ""))
        return payload

    def generate(self, scale_value: float) -> dict[str, Any]:
        if self.task_type == "table-question-answering":
            if not math.isfinite(scale_value) or scale_value < 1 or int(scale_value) != scale_value:
                raise ValueError("table_rows must be a positive integer")
            rows = int(scale_value)
            self._budget.text(rows * (24 + 2 * len(str(rows))) * self.batch_size,
                              batch_items=self.batch_size)
            return {
                "table": {
                    "name": [f"item {index + 1}" for index in range(rows)],
                    "value": [str(index + 1) for index in range(rows)],
                },
                "query": "What is the value of item 1?",
                "params": {},
                "batch_size": self.batch_size,
                "input_budget": self._budget.as_dict(),
                "input_scale_type": "table_rows",
            }
        return self.generate_for_word_count(int(scale_value))

    def scale_label(self, scale_value: float) -> str:
        if self.task_type == "table-question-answering":
            return f"rows{int(scale_value)}"
        return f"seq{int(scale_value)}"

    def default_input_scales(self) -> list[float] | None:
        if self.task_type == "table-question-answering":
            return [1, 2, 4, 8, 16, 32]
        return None

    def plan_metadata(self) -> dict[str, Any]:
        metadata = copy.deepcopy(self._spec_metadata)
        if self._task_workload:
            from acprof.models.evidence import content_digest
            from acprof.tasks.workloads import CORPORA
            metadata.update(task_workload=copy.deepcopy(self._task_workload),
                            data_source="acprof_builtin_text", seed=0,
                            template_sha256=content_digest({"config": self._task_workload["config"],
                                                           "corpus": CORPORA.get(self._task_workload["kind"], _CORPUS_WORDS)}))
        metadata["input_budget"] = self._budget.as_dict()
        if self.task_type == "table-question-answering":
            return {**metadata, "input_scale_type": "table_rows", "columns": ["name", "value"]}
        if self.task_type in {"sentence-similarity", "text-ranking"}:
            return {**metadata, "input_scale_type": "seq_length", "scale_scope": "per_candidate",
                    "candidate_count": 2, "query": DEFAULT_PAIR_QUERY}
        if self.task_type == "zero-shot-classification":
            return {**metadata, "candidate_labels": list(DEFAULT_CANDIDATE_LABELS),
                    "hypothesis_template": DEFAULT_HYPOTHESIS_TEMPLATE}
        return metadata


register_generator("nlp", NLPWorkloadGenerator)
