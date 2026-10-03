"""Invalid controls remain visible and actionable without reading transient notices."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Collapsible, ContentSwitcher, Select, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.input import BarCursorInput as Input


@pytest.mark.parametrize('size,language', (((80, 24), 'zh'), ((120, 30), 'en')))
@pytest.mark.parametrize('field,invalid,valid,page', (('cpus', 'zero', '1', 'run-form'), ('repeat', '0', '1', 'advanced-form'), ('workload-spec', 'missing-coverage-input.json', '', 'advanced-form'), ('max-download', 'bad', '5GB', 'advanced-form')))
async def test_submit_reveals_and_focuses_each_error_then_clears_its_reason(size, language, field, invalid, valid, page):
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=Path(directory) / "settings.json")
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app.query_one("#ui-language", Select).value = language
            await pilot.pause()
            control = app.query_one(f"#{field}", Input)
            control.value = invalid
            app.query_one("#experiment-pages", ContentSwitcher).current = "run-form"
            for container in control.ancestors:
                if isinstance(container, Collapsible):
                    container.collapsed = True
            await pilot.pause()
            with patch.object(app, "_launch") as launch:
                assert (await pilot.click("#start-run"))
                await pilot.pause()
            launch.assert_not_called()
            assert (app.focused) is (control)
            assert (app.query_one("#experiment-pages", ContentSwitcher).current) == (page)
            assert (control.region.overlaps(app.screen.region))
            error = app.query_one(f"#{field}-error", Static)
            assert (error.display)
            assert (str(error.render()))
            control.value = valid
            # Form edits intentionally coalesce on a 50 ms timer.
            await pilot.pause(0.1)
            assert not (error.display)

async def test_model_review_requires_an_explicit_select_choice():
    from acprof.tui.preparation import PreparationScreen
    question = {"path": "pipeline_task", "value": "second", "reason": "Choose the declared task",
                "options": ["first", "second"]}
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=Path(directory) / "settings.json")
        async with app.run_test(size=(80, 24)) as pilot:
            event = {"stage": "resolution", "request": {"kind": "review", "questions": [question]}}
            screen = PreparationScreen(event)
            app.push_screen(screen)
            await pilot.pause()
            control = screen.query_one("#preparation-answer-0", Select)
            assert control.value == "second"
            control.clear()
            assert await pilot.click("#preparation-apply")
            await pilot.pause()
            assert app.screen is screen
            assert app.focused is control
            assert "pipeline_task" in str(screen.query_one("#preparation-error", Static).render())


async def test_review_invalid_json_identifies_and_focuses_the_same_field():
    from acprof.tui.preparation import PreparationScreen
    question = {"path": "inputs.messages.template", "value": None, "reason": "Provide the message template"}
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=Path(directory) / "settings.json")
        async with app.run_test(size=(80, 24)) as pilot:
            screen = PreparationScreen({"stage": "resolution", "request": {"kind": "review", "questions": [question]}})
            app.push_screen(screen)
            await pilot.pause()
            control = screen.query_one("#preparation-answer-0", Input)
            control.value = "{invalid"
            assert await pilot.click("#preparation-apply")
            await pilot.pause()
            assert app.screen is screen
            assert app.focused is control
            assert question["path"] in str(screen.query_one("#preparation-error", Static).render())
