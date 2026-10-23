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


class ImageDetailApp(Protocol):
    def _localized_widget(self, widget: Widget) -> Widget: ...
    def _set_text(self, widget: Widget, source: str, attribute: str = "content") -> None: ...


if TYPE_CHECKING:
    from textual.widget import Widget

    from acprof.tui.app import AcprofTui

    def _check_composed_app_contracts(app: AcprofTui) -> tuple[TranslatableApp, CandidatePickerApp, ImageDetailApp]:
        """Ensure the final controller provides every widget contract."""
        return app, app, app


def translated_app(app: object) -> TranslatableApp:
    """Type the translation capability provided by the enclosing TUI app."""
    return cast(TranslatableApp, app)


def candidate_picker_app(app: object) -> CandidatePickerApp:
    """Type the model-picker action provided by the enclosing TUI app."""
    return cast(CandidatePickerApp, app)


def image_detail_app(app: object) -> ImageDetailApp:
    """Type localization methods provided by the enclosing TUI app."""
    return cast(ImageDetailApp, app)
