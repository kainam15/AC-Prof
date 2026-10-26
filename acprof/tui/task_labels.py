"""Localize selection data; parameter vocabulary never controls comparison."""
from __future__ import annotations

import json
from collections.abc import Callable

from acprof.task_presentation import ConfigField, ConfigPath, SelectionGroup, TaskPurpose
from acprof.tui.i18n import message_id

type Translator = Callable[[str], str]

_PURPOSES = {
    "translation": "ui.2177", "summarization": "ui.2168",
    "generative-question-answering": "ui.2169", "sentiment-analysis": "ui.2170",
    "text-classification": "ui.2171", "regression": "ui.2172",
    "text-generation": "ui.2178", "text2text-generation": "ui.2178",
    "conversational": "ui.2179", "question-answering": "ui.2180",
    "fill-mask": "ui.2181", "image-classification": "ui.2182",
    "automatic-speech-recognition": "ui.2183",
}
_PARAMETERS: dict[ConfigPath, str] = {
    ("backend",): "ui.2184",
    ("generation", "num_return_sequences"): "ui.2185",
    ("generation", "use_cache"): "ui.2186",
    ("generation", "no_repeat_ngram_size"): "ui.2187",
    ("generation", "encoder_no_repeat_ngram_size"): "ui.2188",
    ("generation", "early_stopping"): "ui.2189",
    ("generation", "temperature"): "ui.2190",
    ("generation", "top_k"): "ui.2191",
    ("generation", "top_p"): "ui.2192",
}


def purpose_label(purpose: TaskPurpose, tr: Translator) -> str:
    key = _PURPOSES.get(purpose.kind)
    if key is None:
        return " / ".join(purpose.task_ids)
    label = tr(message_id(key))
    if purpose.kind == "translation" and purpose.directions:
        if len(purpose.directions) > 1:
            return tr(message_id("ui.2193", label, len(purpose.directions)))
        source, target = purpose.directions[0]
        return f"{label} · {source} → {target}"
    if len(purpose.task_ids) > 1:
        return tr(message_id("ui.2194", label, len(purpose.task_ids)))
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
        return f"{format_path(path)} = {tr(message_id('ui.2195'))}"
    if path == ("interface",):
        if value == "text2text-generation":
            return "Seq2Seq"
        if value == "text-generation":
            return tr(message_id("ui.2196"))
    if path == ("generation", "num_beams"):
        return f"Beam ×{raw}"
    if path == ("generation", "do_sample") and type(value) is bool:
        return tr(message_id("ui.2197" if value else "ui.2198"))
    if key := _PARAMETERS.get(path):
        label = tr(message_id(key))
        if type(value) is bool:
            raw = tr(message_id("ui.2199" if value else "ui.2200"))
        return f"{label} = {raw}"
    return f"{format_path(path)} = {raw}"


def format_parameters(fields: tuple[ConfigField, ...], tr: Translator) -> str:
    # Keep the comparison's deterministic order, with the interface first for orientation.
    ordered = sorted(fields, key=lambda field: field.path != ("interface",))
    return " · ".join(format_parameter(field, tr) for field in ordered)
