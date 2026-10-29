"""Bilingual display text, kept together for review and maintenance."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Entry:
    """One message; field order is never used to select the UI language."""

    zh: str
    en: str
