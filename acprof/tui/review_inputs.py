"""The same declared choices and JSON values for both model-review entry points."""
from __future__ import annotations

import json

from textual.widget import Widget
from textual.widgets import Select

from acprof.tui.input import BarCursorInput as Input


def review_input(question: dict, *, identifier: str, translate, classes: str = "") -> Widget:
    options = question.get("options")
    if options:
        value = question.get("value")
        return Select([(str(item), item) for item in options],
                      value=value if value in options else Select.NULL,
                      prompt=translate("请选择"), id=identifier, classes=classes)
    value = question.get("value")
    return Input(value=json.dumps(value, ensure_ascii=False) if value is not None else "",
                 placeholder=translate("填写此字段的 JSON 值"), id=identifier, classes=classes)


def review_answers(screen, questions: list[dict], *, prefix: str) -> dict:
    answers = {}
    for index, question in enumerate(questions):
        if question.get("read_only"):
            continue
        widget = screen.query_one(f"#{prefix}-{index}")
        try:
            if isinstance(widget, Select):
                if widget.value is Select.NULL:
                    raise ValueError(screen.app.tr("请选择"))
                value = widget.value
            elif isinstance(widget, Input):
                value = json.loads(widget.value)
            else:
                raise TypeError("unsupported model review input")
        except (ValueError, TypeError) as exc:
            widget.focus()
            widget.scroll_visible(animate=False, immediate=True)
            raise ValueError(f"{question['path']}: {exc}") from exc
        answers[question["path"]] = value
    return answers
