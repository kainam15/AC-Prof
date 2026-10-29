"""Serializable task semantics shared by discovery, input planning and runtimes.

Registered interfaces and explicit usage declarations enter this protocol.
An executable interface does not prove author intent or task quality.
"""
from __future__ import annotations

import copy
import re

from acprof.models.evidence import content_digest

LANGUAGES = {
    "en": ("English", "英语"), "de": ("German", "德语"), "fr": ("French", "法语"),
    "ro": ("Romanian", "罗马尼亚语"), "es": ("Spanish", "西班牙语"),
    "zh": ("Chinese", "中文"), "ja": ("Japanese", "日语"), "it": ("Italian", "意大利语"),
    "pt": ("Portuguese", "葡萄牙语"), "ru": ("Russian", "俄语"),
}
GENERATION_KEYS = frozenset({
    "max_length", "min_length", "max_new_tokens", "min_new_tokens", "num_beams",
    "early_stopping", "length_penalty", "no_repeat_ngram_size", "do_sample",
    "temperature", "top_k", "top_p", "repetition_penalty", "decoder_start_token_id",
    "bos_token_id", "eos_token_id", "pad_token_id", "forced_bos_token_id", "forced_eos_token_id",
    "num_return_sequences", "use_cache", "encoder_no_repeat_ngram_size",
})
GENERATIVE_INTERFACES = frozenset({"text-generation", "text2text-generation", "summarization",
                                  "translation", "conversational"})
TEMPLATE_VERSION = "acprof-task-text-v1"
CORPORA = {
    "translation": "The library opens every morning. People read books and learn about the world. ",
    "summarization": ("The town opened a new public library on Monday. It offers books, study rooms "
                      "and free classes. Residents welcomed the new space for learning. "),
    "generative-question-answering": ("The public library opens at nine in the morning. "
                                       "It closes at six in the evening. "),
    "sentiment-analysis": "The service was helpful and the experience was pleasant. ",
    "text-classification": "The researchers studied plants and published their findings. ",
    "regression": "The service was helpful and the experience was pleasant. ",
}


def task_label(task: dict, *, language: str = "zh_CN") -> str:
    """Name a semantic declaration in CLI inspection, never an execution group."""
    config = task["config"]
    if task["kind"] == "translation":
        index = 1 if language.startswith("zh") else 0
        return f"{LANGUAGES[config['source_language']][index]} → {LANGUAGES[config['target_language']][index]}"
    labels = {"summarization": ("Summarization", "文本摘要"),
              "generative-question-answering": ("Generative question answering", "生成式问答"),
              "sentiment-analysis": ("Sentiment analysis", "情感分析"),
              "text-classification": ("Text classification", "文本分类"),
              "regression": ("Regression", "回归")}
    return labels.get(task["kind"], (task["task_id"], task["task_id"]))[1 if language.startswith("zh") else 0]


def canonical_task_id(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("task name must be a string")
    if name.startswith("translation-"):
        name = "translation_" + name[len("translation-"):].replace("-to-", "_to_")
    return {"sentiment_analysis": "sentiment-analysis", "question_answering": "question-answering"}.get(name, name)


def parse_task(name: str, params: dict, *, encoder_decoder: bool, source: str, generative: bool = False,
               default_generation: dict | None = None) -> dict:
    from acprof.extensions import CATALOG

    if not isinstance(params, dict):
        raise ValueError("task configuration must be an object")
    name = canonical_task_id(name)
    interface, kind = name, name
    config: dict = {}
    pair = re.fullmatch(r"translation_([a-z]{2,3})_to_([a-z]{2,3})", name)
    if pair:
        src, target = pair.groups()
        if src not in LANGUAGES or target not in LANGUAGES:
            raise ValueError("translation language requires an explicit supported text template")
        interface = "text2text-generation" if encoder_decoder else "text-generation"
        kind = "translation"
        config.update(source_language=src, target_language=target)
        config["prefix"] = f"translate {LANGUAGES[src][0]} to {LANGUAGES[target][0]}: "
        # Built-in input corpora are English; do not feed English to another source language.
        if src != "en" and not params.get("input_text"):
            raise ValueError("non-English translation requires input_text in the task declaration")
    elif name == "translation":
        raise ValueError("translation requires a declared source and target language")
    elif name == "summarization":
        interface = "text2text-generation" if encoder_decoder else "text-generation"
        config["prefix"] = "summarize: " if encoder_decoder else "Summarize the following text: "
    elif name == "generative-question-answering" or name == "question-answering" and (encoder_decoder or generative):
        name = kind = "generative-question-answering"
        interface = "text2text-generation" if encoder_decoder else "text-generation"
        config.update(prefix="question: When does the library open? context: ")
    elif name in {"sentiment-analysis", "text-classification", "regression"}:
        if encoder_decoder:
            interface = "text2text-generation"
            if not params.get("prefix"):
                raise ValueError("generative classification/regression requires a declared prefix")
        else:
            interface = "text-classification"
    if interface not in CATALOG.task_families:
        raise ValueError(f"unsupported inference interface: {interface}")
    allowed = (GENERATION_KEYS if interface in GENERATIVE_INTERFACES else frozenset()) | {"prefix", "input_text"}
    unknown = params.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported task configuration: {sorted(unknown)}")
    for key in ("prefix", "input_text"):
        if key in params:
            if not isinstance(params[key], str):
                raise ValueError(f"{key} must be a string")
            config[key] = params[key]
    if interface in GENERATIVE_INTERFACES:
        generation = {key: copy.deepcopy(value) for key, value in (default_generation or {}).items()
                      if key in GENERATION_KEYS and value is not None}
        generation = merge_generation_parameters(generation, {key: value for key, value in params.items()
                                                               if key in GENERATION_KEYS})
        if not (generation.keys() & {"max_new_tokens", "max_length"}):
            generation["max_new_tokens"] = 64
        generation.setdefault("do_sample", False)
        validate_generation(generation)
        config["generation"] = generation
        config["length_semantics"] = "decoder_sequence_including_start" if encoder_decoder else "prompt_plus_completion"
    config.update(template_version=TEMPLATE_VERSION, seed=0)
    task = {"task_id": name, "interface": interface, "kind": kind, "config": config, "evidence": [source]}
    task["fingerprint"] = task_fingerprint(task)
    return task


def merge_generation_parameters(defaults: dict, overrides: dict) -> dict:
    """A higher-priority length spelling replaces the inherited alternate spelling."""
    merged = copy.deepcopy(defaults)
    for total, new in (("max_length", "max_new_tokens"), ("min_length", "min_new_tokens")):
        if total in overrides or new in overrides:
            merged.pop(total, None)
            merged.pop(new, None)
    merged.update(copy.deepcopy(overrides))
    for total, new in (("max_length", "max_new_tokens"), ("min_length", "min_new_tokens")):
        if new in merged:
            merged.pop(total, None)
    return merged


def validate_generation(params: dict) -> None:
    if set(params) - GENERATION_KEYS:
        raise ValueError(f"unsupported generation parameters: {sorted(set(params) - GENERATION_KEYS)}")
    for key in ("max_length", "max_new_tokens", "num_beams", "num_return_sequences"):
        if key in params and (type(params[key]) is not int or params[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("min_length", "min_new_tokens", "no_repeat_ngram_size", "top_k"):
        if key in params and (type(params[key]) is not int or params[key] < 0):
            raise ValueError(f"{key} must be a nonnegative integer")
    for low, high in (("min_length", "max_length"), ("min_new_tokens", "max_new_tokens")):
        if low in params and high in params and params[low] > params[high]:
            raise ValueError(f"{low} exceeds {high}")
    content_digest(params)  # Reject nonfinite/non-JSON configuration before any runtime work.


def task_fingerprint(task: dict) -> str:
    identity = {key: task[key] for key in ("task_id", "interface", "kind", "config")}
    # A grouped benchmark must also bind the complete semantic-task snapshot:
    # the exemplar alone is insufficient to prove resume/input provenance.
    for key in ("execution_path", "execution_profile", "semantic_tasks"):
        if key in task:
            identity[key] = task[key]
    return content_digest(identity)


def workload_manifest(task: dict) -> dict:
    return {"schema_version": 1, "task": task["interface"], "task_workload": copy.deepcopy(task),
            "params": copy.deepcopy(task["config"].get("generation", {}))}


def render_task_text(task: dict, word_count: int) -> str:
    config = task["config"]
    corpus = config.get("input_text") or CORPORA.get(task["kind"])
    if not corpus:
        raise ValueError(f"no semantic text template for {task['task_id']}")
    words = corpus.split()
    text = " ".join(words[index % len(words)] for index in range(max(1, word_count)))
    return config.get("prefix", "") + text
