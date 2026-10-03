"""Invalid controls remain visible and actionable without reading transient notices."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Button, Collapsible, ContentSwitcher, Select, Static

from acprof.experiment import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.input import BarCursorInput as Input


class TuiConfigErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_reveals_and_focuses_each_error_then_clears_its_reason(self):
        for size, language in (((80, 24), "zh"), ((120, 30), "en")):
            with self.subTest(size=size, language=language), tempfile.TemporaryDirectory() as directory:
                app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=Path(directory) / "settings.json")
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()
                    app.query_one("#ui-language", Select).value = language
                    await pilot.pause()
                    for field, invalid, valid, page in (("cpus", "zero", "1", "run-form"),
                            ("repeat", "0", "1", "advanced-form"),
                            ("workload-spec", "missing-coverage-input.json", "", "advanced-form"),
                            ("max-download", "bad", "5GB", "advanced-form")):
                        with self.subTest(field=field):
                            control = app.query_one(f"#{field}", Input)
                            control.value = invalid
                            app.query_one("#experiment-pages", ContentSwitcher).current = "run-form"
                            for container in control.ancestors:
                                if isinstance(container, Collapsible):
                                    container.collapsed = True
                            await pilot.pause()
                            with patch.object(app, "_launch") as launch:
                                self.assertTrue(await pilot.click("#start-run"))
                                await pilot.pause()
                            launch.assert_not_called()
                            self.assertIs(app.focused, control)
                            self.assertEqual(app.query_one("#experiment-pages", ContentSwitcher).current, page)
                            self.assertTrue(control.region.overlaps(app.screen.region))
                            error = app.query_one(f"#{field}-error", Static)
                            self.assertTrue(error.display)
                            self.assertTrue(str(error.render()))
                            control.value = valid
                            # Form edits intentionally coalesce on a 50 ms timer.
                            await pilot.pause(0.1)
                            self.assertFalse(error.display)

    async def test_model_review_options_use_the_same_select_as_preparation(self):
        from types import SimpleNamespace

        from acprof.tui.model_resolution import ModelResolutionScreen
        from acprof.tui.preparation import PreparationScreen
        question = {"path": "pipeline_task", "value": "second", "reason": "Choose the declared task",
                    "options": ["first", "second"]}
        task = SimpleNamespace(model_resolution={"contract": {"status": "needs_review"}})
        with tempfile.TemporaryDirectory() as directory:
            app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=Path(directory) / "settings.json")
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                with patch("acprof.tui.model_resolution.review_questions", return_value=[question]), patch(
                        "acprof.tui.model_resolution.explain_resolution", return_value="two declared candidates"):
                    screen = ModelResolutionScreen(app.initial_config, task=task)
                    app.push_screen(screen)
                    await pilot.pause()
                    control = screen.query_one("#resolution-answer-0", Select)
                    self.assertEqual(control.value, "second")
                    control.value = "first"
                    screen.query_one("#resolution-apply", Button).scroll_visible(animate=False, immediate=True)
                    await pilot.pause()
                    with patch.object(screen, "review") as review:
                        self.assertTrue(await pilot.click("#resolution-apply"))
                    review.assert_called_once_with({"pipeline_task": "first"})
                    screen.dismiss(None)
                    await pilot.pause()
                event = {"stage": "interface", "request": {"kind": "review", "questions": [question]}}
                preparation = PreparationScreen(event, "pending")
                app.push_screen(preparation)
                await pilot.pause()
                self.assertEqual(preparation.query_one("#preparation-answer-0", Select).value, "second")
                preparation.query_one("#preparation-answer-0", Select).clear()
                preparation.query_one("#preparation-continue", Button).scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                self.assertTrue(await pilot.click("#preparation-continue"))
                await pilot.pause()
                self.assertIs(app.screen, preparation)
                self.assertIs(app.focused, preparation.query_one("#preparation-answer-0", Select))
                self.assertIn("pipeline_task", str(preparation.query_one("#preparation-error", Static).render()))

    async def test_review_invalid_json_identifies_and_focuses_the_same_field(self):
        from types import SimpleNamespace

        from acprof.tui.model_resolution import ModelResolutionScreen
        from acprof.tui.preparation import PreparationScreen
        question = {"path": "inputs.messages.template", "value": None, "reason": "Provide the message template"}
        task = SimpleNamespace(model_resolution={"contract": {"status": "needs_review"}})
        with tempfile.TemporaryDirectory() as directory:
            app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=Path(directory) / "settings.json")
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                with patch("acprof.tui.model_resolution.review_questions", return_value=[question]), patch(
                        "acprof.tui.model_resolution.explain_resolution", return_value="message template required"):
                    screens = (
                        (ModelResolutionScreen(app.initial_config, task=task), "resolution", "apply"),
                        (PreparationScreen({"stage": "interface", "request": {
                            "kind": "review", "questions": [question]}}, "pending"), "preparation", "continue"),
                    )
                    for screen, prefix, action in screens:
                        with self.subTest(screen=prefix):
                            app.push_screen(screen)
                            await pilot.pause()
                            control = screen.query_one(f"#{prefix}-answer-0", Input)
                            control.value = "{invalid"
                            screen.query_one(f"#{prefix}-{action}", Button).scroll_visible(animate=False, immediate=True)
                            await pilot.pause()
                            self.assertTrue(await pilot.click(f"#{prefix}-{action}"))
                            await pilot.pause()
                            self.assertIs(app.screen, screen)
                            self.assertIs(app.focused, control)
                            self.assertIn(question["path"], str(screen.query_one(f"#{prefix}-error", Static).render()))
                            screen.dismiss(None)
                            await pilot.pause()
