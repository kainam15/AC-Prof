"""The ordinary Chinese configuration form at a fixed terminal size."""
from __future__ import annotations

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from tests.visual_png.scenarios.common import freeze_for_screenshot


def initial_config() -> RunConfig:
    return RunConfig.smoke("fixture/model")


def mount(app: AcprofTui) -> None:
    freeze_for_screenshot(app)
