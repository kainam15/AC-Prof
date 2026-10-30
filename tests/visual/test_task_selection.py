"""Review the multi-task decision at the minimum supported terminal size."""
from dataclasses import replace

import pytest
from textual.widgets import Button, Checkbox

from acprof.experiments.configuration import RunConfig
from acprof.tui.models.task_choices import TaskChoices
from acprof.tui.run.preparation import PreparationScreen
from tests.support.tui_fixtures import AcprofTui
from tests.tui.catalog.test_tui_task_selection import long_profiles, task_event

pytestmark = pytest.mark.visual


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
        choices = screen.query_one(TaskChoices)
        assert choices.selected == []
        assert choices.region.height >= 4
        for identifier in ("#preparation-apply", "#preparation-cancel"):
            button = screen.query_one(identifier, Button)
            assert button.region.bottom <= 24
            assert button.region.width > 0

    assert snap_compare(app, terminal_size=(80, 24), run_before=prepare)


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("size", [(120, 30), (150, 45)])
def test_task_selection_wide(snap_compare, tmp_path, language, size):
    app = AcprofTui(RunConfig.smoke("google-t5/t5-small"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language, theme="acprof-graphite")

    async def prepare(pilot):
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.clear_notifications()
        app.set_input_cursor_blink_enabled(False)
        await app.push_screen(PreparationScreen(task_event()))
        await pilot.pause()

    assert snap_compare(app, terminal_size=size, run_before=prepare)


@pytest.mark.parametrize("language", ["zh", "en"])
def test_long_task_selection(snap_compare, tmp_path, language):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language, theme="acprof-graphite")

    async def prepare(pilot):
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.clear_notifications()
        app.set_input_cursor_blink_enabled(False)
        screen = PreparationScreen(task_event(long_profiles()))
        await app.push_screen(screen)
        await pilot.pause()
        choices = screen.query_one(TaskChoices)
        choices.select_all()
        list(choices.query(Checkbox))[-1].focus()
        await pilot.pause()

    assert snap_compare(app, terminal_size=(80, 24), run_before=prepare)
