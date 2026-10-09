"""Independent profiler choices in the advanced TUI configuration."""

from pathlib import Path

import pytest
from textual.widgets import Checkbox, ContentSwitcher, Select, Static, TabbedContent
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, build_run_command
from acprof.tui.run_form import PROFILER_CHECKBOXES, config_values, profiler_tools_from_checks


@pytest.mark.parametrize("mask", range(16))
def test_profiler_checkbox_modes_round_trip(mask):
    selections = {
        name: bool(mask & (1 << index))
        for index, name in enumerate(PROFILER_CHECKBOXES)
    }
    expected_compute = ("none", "torch", "ncu", "both")[mask & 3]
    expected_execution = ("none", "massif", "nsys", "both")[(mask >> 2) & 3]

    assert profiler_tools_from_checks(selections) == (
        expected_compute, expected_execution
    )
    config = RunConfig(
        model="test/model",
        compute_profile_tool=expected_compute,
        execution_profile_tool=expected_execution,
    )
    _, _, restored = config_values(config)
    assert {name: restored[name] for name in PROFILER_CHECKBOXES} == selections


@pytest.mark.parametrize("size", [(80, 24), (120, 30)])
@pytest.mark.parametrize("language", ["zh", "en"])
async def test_profiler_checkboxes_update_preview_preflight_and_restore(tmp_path, size, language):
    config = RunConfig(
        model="test/model",
        gpus="off,on",
        compute_profile_tool="torch",
        execution_profile_tool="nsys",
    )
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    async with app.run_test(size=size) as pilot:
        app.query_one("#ui-language", Select).value = language
        app.query_one("#main-tabs", TabbedContent).active = "run-tab"
        app.query_one("#experiment-pages", ContentSwitcher).current = "advanced-form"
        await pilot.pause()
        assert app.query_one("#torch-profiler", Checkbox).value
        assert not app.query_one("#ncu-profiler", Checkbox).value
        assert not app.query_one("#massif-profiler", Checkbox).value
        assert app.query_one("#nsys-profiler", Checkbox).value
        assert app._collect_config() == config

        ncu = app.query_one("#ncu-profiler", Checkbox)
        ncu.scroll_visible(animate=False, immediate=True)
        ncu.focus()
        await pilot.press("space")
        await pilot.pause(0.15)
        assert ncu.value
        assert app._collect_config().compute_profile_tool == "both"
        assert app._preflight_config().compute_profile_tool == "both"
        preview = str(app.query_one("#command-preview", Static).render())
        assert "--compute-profile-tool both" in preview
        command = build_run_command(app._collect_config(), project_dir=Path.cwd())
        assert command[command.index("--execution-profile-tool") + 1] == "nsys"

        if language == "en":
            assert "GPU kernel" in ncu.label.plain
        else:
            assert "GPU 内核" in ncu.label.plain

        app._apply_config(config)
        await pilot.pause()
        assert not ncu.value
        assert app._collect_config() == config
