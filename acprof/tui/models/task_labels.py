"""Localize selection data; parameter vocabulary never controls comparison."""
from __future__ import annotations

import json
from collections.abc import Callable

from acprof.tasks.presentation import ConfigField, ConfigPath, SelectionGroup, TaskPurpose
from acprof.tui.i18n import message_id

type Translator = Callable[[str], str]

_PURPOSES = {
    "translation": "models.task.translation", "summarization": "models.task.summarization",
    "generative-question-answering": "models.task.generative_question_answering", "sentiment-analysis": "models.task.sentiment_analysis",
    "text-classification": "models.task.text_classification", "regression": "models.task.regression",
    "text-generation": "models.task.text_generation", "text2text-generation": "models.task.text_generation",
    "conversational": "models.task.conversation", "question-answering": "models.task.question_answering",
    "fill-mask": "models.task.fill_mask", "image-classification": "models.task.image_classification",
    "automatic-speech-recognition": "models.task.speech_recognition",
}
_PARAMETERS: dict[ConfigPath, str] = {
    ("backend",): "models.parameter.backend",
    ("generation", "num_return_sequences"): "models.parameter.return_sequences",
    ("generation", "use_cache"): "models.parameter.kv_cache",
    ("generation", "no_repeat_ngram_size"): "models.parameter.no_repeat_ngram",
    ("generation", "encoder_no_repeat_ngram_size"): "models.parameter.encoder_no_repeat_ngram",
    ("generation", "early_stopping"): "models.parameter.early_stopping",
    ("generation", "temperature"): "models.parameter.temperature",
    ("generation", "top_k"): "models.parameter.top_k",
    ("generation", "top_p"): "models.parameter.top_p",
    ("generation", "repetition_penalty"): "models.parameter.repetition_penalty",
    ("generation", "length_penalty"): "models.parameter.beam_length_penalty",
    ("operation",): "models.parameter.operation",
}


def purpose_label(purpose: TaskPurpose, tr: Translator) -> str:
    key = _PURPOSES.get(purpose.kind)
    if key is None:
        return " / ".join(purpose.task_ids)
    label = tr(message_id(key))
    if purpose.kind == "translation" and purpose.directions:
        if len(purpose.directions) > 1:
            return tr(message_id("models.tasks.language_direction_count", label, len(purpose.directions)))
        source, target = purpose.directions[0]
        return f"{label} · {source} → {target}"
    if len(purpose.task_ids) > 1:
        return tr(message_id("models.tasks.input_declaration_count", label, len(purpose.task_ids)))
    return label


def group_label(group: SelectionGroup, tr: Translator) -> str:
    return " / ".join(purpose_label(purpose, tr) for purpose in group.purposes)


def format_path(path: ConfigPath) -> str:
    result = ""
    for part in path:
        if isinstance(part, int):
            result += f"[{part}]"
        elif part.isidentifier():
            result += ("." if result else "") + part
        else:
            result += f"[{json.dumps(part, ensure_ascii=False)}]"
    return result or "$"


def format_parameter(field: ConfigField, tr: Translator) -> str:
    path, value = field.path, field.value
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if not field.present:
        return f"{format_path(path)} = {tr(message_id('models.parameter.not_set'))}"
    if path == ("interface",):
        if value == "text2text-generation":
            return "Seq2Seq"
        if value == "text-generation":
            return tr(message_id("models.interface.autoregressive"))
    if path == ("generation", "num_beams"):
        return f"Beam ×{raw}"
    if path == ("generation", "do_sample") and type(value) is bool:
        return tr(message_id("models.parameter.sampling_enabled" if value else "models.parameter.sampling_disabled"))
    if key := _PARAMETERS.get(path):
        label = tr(message_id(key))
        if type(value) is bool:
            raw = tr(message_id("common.option.on" if value else "models.parameter.disabled"))
        return f"{label} = {raw}"
    return f"{format_path(path)} = {raw}"


def format_parameters(fields: tuple[ConfigField, ...], tr: Translator) -> str:
    # Keep the comparison's deterministic order, with the interface first for orientation.
    ordered = sorted(fields, key=lambda field: field.path != ("interface",))
    return " · ".join(format_parameter(field, tr) for field in ordered)
