"""Typed host interfaces for TUI widgets mounted under the AC-Prof app.

Widgets see Textual's generic ``App`` through ``self.app``. These protocols
name the capabilities they use without importing the composed controller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast


class TranslatableApp(Protocol):
    def tr(self, source: str) -> str: ...


class CandidatePickerApp(Protocol):
    def open_model_candidates(self) -> None: ...


if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

    def _check_composed_app_contracts(app: AcprofTui) -> tuple[TranslatableApp, CandidatePickerApp]:
        """Ensure the final controller really provides both widget contracts."""
        return app, app


def translated_app(app: object) -> TranslatableApp:
    """Type the translation capability provided by the enclosing TUI app."""
    return cast(TranslatableApp, app)


def candidate_picker_app(app: object) -> CandidatePickerApp:
    """Type the model-picker action provided by the enclosing TUI app."""
    return cast(CandidatePickerApp, app)
