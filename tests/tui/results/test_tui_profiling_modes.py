import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Checkbox, Collapsible, Select, Static, TabbedContent
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition

from acprof.experiments.configuration import RunConfig, build_run_command
from acprof.tui.run.diagnostics import quick_preflight


async def test_late_refresh_timer_update_pauses_after_tabs_are_unmounted():
    with tempfile.TemporaryDirectory() as root:
        app = AcprofTui(RunConfig(model="test/model"), settings_path=Path(root) / "settings.json")
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app.images.refresh_timer.pause()
            app.images.refresh_timer = Mock()
            try:
                await app.query_one("#main-tabs", TabbedContent).remove()
                app.images.sync_image_refresh_timer()
            finally:
                app.form_ready = False
            app.images.refresh_timer.pause.assert_called()
            app.images.refresh_timer.reset.assert_not_called()

@pytest.mark.parametrize('size', [(80, 24), (120, 30), (150, 45)])
@pytest.mark.parametrize('language', ['zh', 'en'])
async def test_optional_profiler_updates_command_in_both_languages_and_sizes(size, language, tmp_path):
    app = AcprofTui(RunConfig(model="test/model", profiling_mode="full"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=size) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        assert not app.query("#profiling-mode")
        options = app.query_one("#profiling-options", Collapsible)
        title = options.query_one("CollapsibleTitle")
        title.scroll_visible(animate=False, immediate=True, force=True)
        title.focus()
        await pilot.press("enter")
        assert not options.collapsed
        profiler = options.query_one("#torch-profiler", Checkbox)
        profiler.scroll_visible(animate=False, immediate=True, force=True)
        profiler.focus()
        await pilot.pause()
        assert profiler.region.area > 0 and not profiler.value
        await pilot.press("space")
        preview = app.query_one("#command-preview", Static)
        await wait_for_condition(pilot, lambda: "--compute-profile-tool torch" in str(preview.content),
                                 description="selected profiler in command preview", timeout=3)
        config = app.configuration.collect_config()
        assert profiler.value and config.compute_profile_tool == "torch"
        assert config.profiling_mode == "best_effort"
        command = build_run_command(config, project_dir=Path.cwd())
        assert command[command.index("--compute-profile-tool") + 1] == "torch"
        assert "--profiling-mode" not in command

def test_basic_diagnostics_do_not_probe_unrequested_energy_or_perf():
    with tempfile.TemporaryDirectory() as root, patch("acprof.tui.run.diagnostics.probe_cpu_energy", side_effect=AssertionError("RAPL not requested")), patch(
        "acprof.tui.run.diagnostics.probe_perf_instructions", side_effect=AssertionError("perf not requested")
    ), patch("acprof.tui.run.diagnostics.shutil.which", return_value=None):
        checks = quick_preflight(RunConfig(model="test", runtime="docker", profiling_mode="basic", gpus="off"), project_dir=root)
    for label in ("CPU RAPL", "perf instructions"):
        check = next(item for item in checks if item.label == label)
        assert (check.capability_status) == ("not_requested")
        assert ("not_requested") in (check.detail)
