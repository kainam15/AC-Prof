import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.widgets import ContentSwitcher, Select, TabbedContent
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, build_run_command
from acprof.tui.diagnostics import quick_preflight


async def test_late_refresh_timer_update_pauses_after_tabs_are_unmounted():
    with tempfile.TemporaryDirectory() as root:
        app = AcprofTui(RunConfig(model="test/model"), settings_path=Path(root) / "settings.json")
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._image_refresh_timer.pause()
            app._image_refresh_timer = Mock()
            try:
                await app.query_one("#main-tabs", TabbedContent).remove()
                app._sync_image_refresh_timer()
            finally:
                app._form_ready = False
            app._image_refresh_timer.pause.assert_called()
            app._image_refresh_timer.reset.assert_not_called()

@pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
@pytest.mark.parametrize('language', ('zh', 'en'))
async def test_mode_selector_updates_command_in_both_languages_and_sizes(size, language):
    with tempfile.TemporaryDirectory() as root:
        app = AcprofTui(RunConfig(model="test/model"), settings_path=Path(root) / "settings.json")
        async with app.run_test(size=size) as pilot:
            app.query_one("#ui-language", Select).value = language
            await pilot.pause()
            control = app.query_one("#profiling-mode", Select)
            app.query_one("#main-tabs", TabbedContent).active = "run-tab"
            app.query_one("#experiment-pages", ContentSwitcher).current = "advanced-form"
            await pilot.pause()
            control.scroll_visible(immediate=True, force=True)
            control.focus()
            await pilot.press("enter", "down", "enter")
            await pilot.pause()
            assert (control.value) == ("basic")
            assert (control.region.height) > (0)
            config = app._collect_config()
            assert (config.profiling_mode) == ("basic")
            command = build_run_command(config, project_dir=Path.cwd())
            assert (command[command.index("--profiling-mode") + 1]) == ("basic")

def test_basic_diagnostics_do_not_probe_unrequested_energy_or_perf():
    with tempfile.TemporaryDirectory() as root, patch("acprof.tui.diagnostics.probe_cpu_energy", side_effect=AssertionError("RAPL not requested")), patch(
        "acprof.tui.diagnostics.probe_perf_instructions", side_effect=AssertionError("perf not requested")
    ), patch("acprof.tui.diagnostics.shutil.which", return_value=None):
        checks = quick_preflight(RunConfig(model="test", profiling_mode="basic", gpus="off"), project_dir=root)
    for label in ("CPU RAPL", "perf instructions"):
        check = next(item for item in checks if item.label == label)
        assert (check.capability_status) == ("not_requested")
        assert ("not_requested") in (check.detail)
