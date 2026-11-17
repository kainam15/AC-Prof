"""Resource fields remain editable and discoverable inside the collapsed group."""
from dataclasses import replace

import pytest
from textual.widgets import Collapsible, Input, Static
from textual.widgets._collapsible import CollapsibleTitle

from acprof.experiments.configuration import RunConfig
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30)), ("zh", (150, 45))])
async def test_resource_matrix_toggle_preserves_configuration(tmp_path, language, size):
    config = replace(RunConfig.smoke("fixture/model"), cpus="1,2", mems="2,4",
                     gpus="on,off", input_scales="64,128")
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        section = app.query_one("#resource-matrix", Collapsible)
        assert section.collapsed
        assert section.title == ("资源矩阵" if language == "zh" else "Resource matrix")
        assert not app.query_one("#cpus").is_on_screen
        assert app.configuration.collect_config() == config
        title = section.query_one(CollapsibleTitle)
        title.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(title)
        await pilot.pause()
        assert not section.collapsed
        for identifier in ("cpus", "mems", "gpus", "input-scales"):
            control = section.query_one("#" + identifier)
            control.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert control.is_on_screen
            assert control.region.width > 0
            assert control.region.right <= app.query_one("#run-form").content_region.right
        app.query_one("#cpus", Input).value = "1,3"
        await wait_for_condition(
            pilot, lambda: "--cpus 1,3" in str(app.query_one("#command-preview", Static).content),
            description="resource edit updates the command preview", timeout=3,
        )
        title.scroll_visible(animate=False, immediate=True)
        title.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert section.collapsed
        assert app.configuration.collect_config() == replace(config, cpus="1,3")
        app.configuration.preferences = replace(app.configuration.preferences,
                                                language="en" if language == "zh" else "zh")
        app.configuration.apply_ui_preferences()
        await pilot.pause()
        assert section.collapsed
        assert section.title == ("Resource matrix" if language == "zh" else "资源矩阵")


async def test_resource_error_reveals_collapsed_field(tmp_path):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        section = app.query_one("#resource-matrix", Collapsible)
        assert section.collapsed
        cpu = app.query_one("#cpus", Input)
        cpu.value = "invalid-cpus"
        # The existing preset action validates the draft before applying changes.
        app.experiments.preset_main()
        await wait_for_condition(
            pilot, lambda: app.focused is cpu and cpu.is_on_screen,
            description="invalid CPU field is revealed and focused", timeout=3,
        )
        assert not section.collapsed
        assert cpu.value == "invalid-cpus"
        assert app.query_one("#cpus-error", Static).display
