"""Function-scoped catalogs with one bilingual definition per semantic key.

The source-text indexes also serve structured messages from non-TUI modules.
They are derived here so those modules never need to import the presentation layer.
"""
from __future__ import annotations

from collections.abc import Mapping

from .common import MESSAGES as COMMON
from .downloads import MESSAGES as DOWNLOADS
from .entry import Entry as Entry
from .images import MESSAGES as IMAGES
from .logs import MESSAGES as LOGS
from .models import MESSAGES as MODELS
from .profiling import MESSAGES as PROFILING
from .results import MESSAGES as RESULTS
from .run import MESSAGES as RUN
from .settings import MESSAGES as SETTINGS
from .validation import MESSAGES as VALIDATION


def merge_messages(*catalogs: Mapping[str, Entry]) -> dict[str, Entry]:
    """Reject duplicate identities instead of silently overwriting a module."""
    messages: dict[str, Entry] = {}
    for catalog in catalogs:
        for key, entry in catalog.items():
            if key in messages:
                raise ValueError(f"Duplicate message key: {key}")
            messages[key] = entry
    return messages


def source_indexes(messages: Mapping[str, Entry]) -> tuple[dict[str, str], dict[str, str]]:
    """Keep source-based translation unambiguous for structured backend messages."""
    identities: dict[str, str] = {}
    english: dict[str, str] = {}
    for key, entry in messages.items():
        if entry.zh in english and english[entry.zh] != entry.en:
            previous = identities[entry.zh]
            raise ValueError(f"Conflicting source translations: {previous}, {key}")
        identities.setdefault(entry.zh, key)
        english[entry.zh] = entry.en
    return identities, english


MESSAGES = merge_messages(
    COMMON, DOWNLOADS, IMAGES, LOGS, MODELS, PROFILING, RESULTS, RUN, SETTINGS, VALIDATION,
)
SOURCE_BY_ID = {key: entry.zh for key, entry in MESSAGES.items()}
ENGLISH_BY_ID = {key: entry.en for key, entry in MESSAGES.items()}
SOURCE_TO_ID, ENGLISH = source_indexes(MESSAGES)
