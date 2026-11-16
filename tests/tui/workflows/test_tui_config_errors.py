"""Invalid controls remain visible and actionable without reading transient notices."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Collapsible, ContentSwitcher, Select, Static

from acprof.experiments.configuration import RunConfig
from acprof.tui.ui.input import BarCursorInput as Input
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


@pytest.mark.parametrize(('size', 'language'), [((80, 24), 'zh'), ((120, 30), 'en')])
@pytest.mark.parametrize(('field', 'invalid', 'valid', 'page'), [('cpus', 'zero', '1', 'run-form'), ('repeat', '0', '1', 'advanced-form'), ('workload-spec', 'missing-coverage-input.json', '', 'advanced-form'), ('max-download', 'bad', '5GB', 'advanced-form')])
async def test_invalid_edits_preserve_focus_and_clear_errors_after_repair(size, language, field, invalid, valid, page):
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=Path(directory) / "settings.json")
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app.query_one("#ui-language", Select).value = language
            start = app.query_one("#start-run", Button)
            await wait_for_condition(
                pilot, lambda: not start.disabled,
                description="initial configuration ready", timeout=5.0,
            )
            if page == "advanced-form":
                assert await pilot.click("#open-run-settings")
            control = app.query_one(f"#{field}", Input)
            for container in control.ancestors:
                if isinstance(container, Collapsible):
                    container.collapsed = False
            await pilot.pause()
            control.focus(scroll_visible=False)
            control.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            error = app.query_one(f"#{field}-error", Static)
            with patch.object(app, 'launch') as launch:
                control.value = invalid
                await wait_for_condition(
                    pilot, lambda: error.display and start.disabled,
                    description=f"{field} error shown and collection blocked", timeout=5.0,
                )
                # The keyboard entry must honor the same invalid-form gate.
                await pilot.press("f5")
                await pilot.pause()
            launch.assert_not_called()
            assert (app.focused) is (control)
            assert (app.query_one("#experiment-pages", ContentSwitcher).current) == (page)
            assert (control.region.overlaps(app.screen.region))
            assert (error.display)
            assert (str(error.render()))
            control.value = valid
            # Wait for the debounced validation result, including its UI update.
            await wait_for_condition(
                pilot, lambda: not error.display,
                description=f"{field} error cleared after a valid edit", timeout=5.0,
                diagnostics=lambda: str(error.render()),
            )
            assert not (error.display)
            await wait_for_condition(
                pilot, lambda: not start.disabled,
                description="repaired configuration ready", timeout=5.0,
            )
            assert app.focused is control
            assert app.query_one("#experiment-pages", ContentSwitcher).current == page

async def test_model_review_requires_an_explicit_select_choice():
    from acprof.tui.run.preparation import PreparationScreen
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
    from acprof.tui.run.preparation import PreparationScreen
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
