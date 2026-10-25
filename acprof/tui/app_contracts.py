"""Small typed capabilities exposed to Textual widgets through their App."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui
    from acprof.tui.localization_controller import LocalizationController
    from acprof.tui.model_controller import ModelController


class TranslatableApp(Protocol):
    def tr(self, source: str) -> str: ...


class CandidatePickerApp(Protocol):
    models: ModelController


class ImageDetailApp(Protocol):
    localization: LocalizationController


if TYPE_CHECKING:
    def _check_composed_app_contracts(app: AcprofTui) -> tuple[TranslatableApp, CandidatePickerApp, ImageDetailApp]:
        """Check the real controller attributes, including their method types."""
        return app, app, app


def translated_app(app: object) -> TranslatableApp:
    """Type the translation capability provided by the enclosing TUI app."""
    return cast(TranslatableApp, app)


def candidate_picker_app(app: object) -> CandidatePickerApp:
    """Type the model controller owned by the enclosing TUI app."""
    return cast(CandidatePickerApp, app)


def image_detail_app(app: object) -> ImageDetailApp:
    """Type the localization controller owned by the enclosing TUI app."""
    return cast(ImageDetailApp, app)
