"""Localize selection data; parameter vocabulary never controls comparison."""
from __future__ import annotations

import json
from collections.abc import Callable

from acprof.tasks.presentation import ConfigField, ConfigPath, SelectionGroup, TaskPurpose
from acprof.tui.i18n import message_id
from acprof.tui.ui.view_models import FactRow

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
    ("generation", "do_sample"): "models.parameter.random_sampling",
    ("generation", "num_beams"): "models.parameter.beam_count",
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
    ("interface",): "models.parameter.interface",
    ("operation",): "models.parameter.operation",
}


_BOOLEAN_PARAMETERS = {
    ("generation", "do_sample"), ("generation", "early_stopping"), ("generation", "use_cache"),
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


def _parameter_value(field: ConfigField) -> str:
    if not field.present:
        return message_id("models.parameter.not_set")
    if field.path == ("interface",):
        if field.value == "text2text-generation":
            return "Seq2Seq"
        if field.value == "text-generation":
            return message_id("models.interface.autoregressive")
    if field.path in _BOOLEAN_PARAMETERS and type(field.value) is bool:
        return message_id("common.option.on" if field.value else "models.parameter.disabled")
    return json.dumps(field.value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def parameter_fact(field: ConfigField) -> FactRow:
    """Separate translated labels from raw paths and complete typed JSON evidence."""
    key = _PARAMETERS.get(field.path)
    label = message_id(key) if key else message_id("models.parameter.raw_path", format_path(field.path))
    detail = json.dumps({"path": list(field.path), "present": field.present, "value": field.value},
                        ensure_ascii=False, sort_keys=True, allow_nan=False)
    return FactRow(label, _parameter_value(field), detail=detail, copyable=True)
