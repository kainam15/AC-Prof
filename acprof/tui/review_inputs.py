"""Declared choices, paths and JSON values for model confirmation."""
from __future__ import annotations

import json
import math
from typing import Any

from textual.widget import Widget
from textual.widgets import Checkbox, Select

from acprof.tui.i18n import message_id
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.task_choices import TaskChoices


def review_input(question: dict, *, identifier: str, translate, classes: str = "") -> Widget:
    options = question.get("options")
    if question.get("kind") == "multi":
        return TaskChoices(question["profiles"], translate=translate, selected=question.get("value", []),
                           id=identifier, classes=classes)
    if options:
        value = question.get("value")
        labels = question.get("labels", {})
        return Select([(translate(str(labels.get(str(item).lower(), labels.get(item, item)))), item) for item in options],
                      value=value if value in options else Select.NULL,
                      prompt=translate("请选择"), id=identifier, classes=classes)
    value = question.get("value")
    if question.get("kind") in {"integer", "float"}:
        return Input(value=str(value), type="integer" if question["kind"] == "integer" else "number",
                     id=identifier, classes=classes)
    if question.get("kind") in {"path", "text", "scales"}:
        return Input(value=value or "", placeholder=translate(
            "本地文件路径" if question["kind"] == "path" else
            "留空自动规划" if question["kind"] == "scales" else "请输入"),
                     id=identifier, classes=classes)
    return Input(value=json.dumps(value, ensure_ascii=False) if value is not None else "",
                 placeholder=translate("填写此字段的 JSON 值"), id=identifier, classes=classes)


def review_answers(screen, questions: list[dict], *, prefix: str) -> dict:
    answers = {}
    for index, question in enumerate(questions):
        if question.get("read_only"):
            continue
        widget = screen.query_one(f"#{prefix}-{index}")
        value: Any
        try:
            if isinstance(widget, TaskChoices):
                value = widget.selected
                if not value:
                    raise ValueError(screen.app.tr(message_id('models.tasks.selection_required')))
            elif isinstance(widget, Select):
                if widget.value is Select.NULL:
                    raise ValueError(screen.app.tr(message_id('common.input.select_prompt')))
                value = widget.value
            elif isinstance(widget, Input):
                kind = question.get("kind")
                if kind == "integer":
                    value = int(widget.value.strip())
                elif kind == "float":
                    value = float(widget.value.strip())
                    if not math.isfinite(value):
                        raise ValueError("数字必须有限")
                else:
                    value = widget.value.strip() if kind in {"path", "text", "scales"} else json.loads(widget.value)
                if kind == "path" and not value and not question.get("optional"):
                    raise ValueError(screen.app.tr(message_id('common.input.local_file_path')))
                if question.get("kind") == "path" and value:
                    from pathlib import Path
                    if not Path(value).expanduser().is_file():
                        raise ValueError(screen.app.tr(message_id('validation.file.not_found')))
            else:
                raise TypeError("unsupported model review input")
        except (ValueError, TypeError) as exc:
            focus_target = widget.query_one(Checkbox) if isinstance(widget, TaskChoices) else widget
            focus_target.focus()
            focus_target.scroll_visible(animate=False, immediate=True)
            raise ValueError(f"{question['path']}: {exc}") from exc
        answers[question["path"]] = value
    return answers
