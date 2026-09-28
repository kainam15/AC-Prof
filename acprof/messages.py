"""Structured source messages; translation belongs to the presentation layer."""
from __future__ import annotations

from typing import Iterable, cast


class Message(str):
    """A source string with its unformatted template and values attached."""

    template: str
    values: tuple[object, ...]

    def __new__(cls, template: str, *values: object) -> "Message":
        instance = cast(Message, super().__new__(cls, template.format(*values) if values else template))
        instance.template = template
        instance.values = values
        return instance


def message(template: str, *values: object) -> Message:
    return Message(template, *values)


def join_messages(separator: str, values: Iterable[str]) -> Message:
    parts = tuple(values)
    return message(separator.join("{" + str(i) + "}" for i in range(len(parts))), *parts)
