"""Review the multi-task decision at the minimum supported terminal size."""
from dataclasses import replace

import pytest
from test_tui_task_selection import task_event
from textual.widgets import Button, SelectionList
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.preparation import PreparationScreen


@pytest.mark.parametrize("language", ["zh", "en"])
def test_task_selection(snap_compare, tmp_path, language):
    app = AcprofTui(RunConfig.smoke("google-t5/t5-small"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language, theme="acprof-graphite")

    async def prepare(pilot):
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.clear_notifications()
        app.set_input_cursor_blink_enabled(False)
        screen = PreparationScreen(task_event())
        await app.push_screen(screen)
        await pilot.pause()
        choices = screen.query_one(SelectionList)
        assert choices.selected == []
        assert choices.region.height >= 4
        for identifier in ("#preparation-apply", "#preparation-cancel"):
            button = screen.query_one(identifier, Button)
            assert button.region.bottom <= 24
            assert button.region.width > 0

    assert snap_compare(app, terminal_size=(80, 24), run_before=prepare)
