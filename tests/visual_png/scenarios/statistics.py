"""HTML-only statistics entry, with no experiment or browser process launched."""
from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from tests.visual_png.scenarios.common import freeze_for_screenshot


def initial_config() -> RunConfig:
    return RunConfig.smoke("fixture/model")


def mount(app: AcprofTui) -> None:
    app.activate_tab("results-tab")
    app.results.show_result_section("statistics-view")
    freeze_for_screenshot(app)
