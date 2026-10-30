"""AC-Prof TUI logging/panel.py components."""

from __future__ import annotations

from textual.containers import Vertical


class LogPanel(Vertical):
    """Keep the log controls available when the log fills the terminal."""

    ALLOW_MAXIMIZE = True
