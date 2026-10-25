"""Small, application-scoped translation catalog for the terminal controller.

Messages retain their Chinese string value for validation and progress tracking.
Only the presentation boundary renders English; interpolated user input, paths,
commands and subprocess output are never searched or replaced. There is no
process-global locale, file loading, polling or translation dependency.
"""
from __future__ import annotations

from typing import cast

from acprof.messages import Message, join_messages
from acprof.tui.translation_catalog import ENGLISH, ENGLISH_BY_ID, SOURCE_BY_ID

LANGUAGE_OPTIONS = (("简体中文", "zh"), ("English", "en"))
UI_LANGUAGES = tuple(value for _, value in LANGUAGE_OPTIONS)



def error_message(error: BaseException) -> str:
    """Retain structured validation messages while leaving external errors raw."""
    issues = getattr(error, "issues", None)
    if issues is not None:
        return join_messages("; ", (item.reason for item in issues))
    if error.args and isinstance(error.args[0], Message):
        return error.args[0]
    return str(error)


class CatalogMessage(Message):
    """A source message with a stable UI identity, independent of copy changes."""

    msgid: str

    def __new__(cls, msgid: str, *values: object) -> CatalogMessage:
        template = SOURCE_BY_ID[msgid]  # Unknown IDs are developer errors.
        instance = cast(CatalogMessage, super().__new__(cls, template, *values))
        instance.msgid = msgid
        return instance


def message_id(msgid: str, *values: object) -> CatalogMessage:
    return CatalogMessage(msgid, *values)


def translate(source: str, language: str) -> str:
    if isinstance(source, Message):
        if language == "en" and isinstance(source, CatalogMessage):
            template = ENGLISH_BY_ID[source.msgid]
        else:
            template = ENGLISH.get(source.template, source.template) if language == "en" else source.template
        values = tuple(
            translate(value, language) if isinstance(value, Message) else value
            for value in source.values
        )
        return template.format(*values) if values else template
    return ENGLISH.get(source, source) if language == "en" else source
