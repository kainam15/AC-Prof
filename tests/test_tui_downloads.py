"""Download controls and explicit pruning stay outside experiment execution."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Button, Collapsible, ContentSwitcher, Input, Select

from acprof.experiment import RunConfig, build_run_command
from acprof.tui.app import AcprofTui
from acprof.tui.commands import build_probe_command
from acprof.tui.model_store import ModelStoreScreen
from acprof.tui.settings import TuiSettings, load_settings, save_settings
from acprof.tui.views import ConfirmActionScreen


class TuiDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_settings_and_prune_confirmation_in_both_languages(self):
        project = Path(__file__).resolve().parents[1]
        for size, language in (((80, 24), "zh"), ((120, 40), "en")):
            with self.subTest(size=size, language=language), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                settings_path = root / "tui.json"
                app = AcprofTui(RunConfig.smoke("example/model"), settings_path=settings_path)
                report = {"path": str(root), "total_bytes": 100, "free_bytes": 1000,
                          "models": [{"entry_id": "old"}, {"entry_id": "new"}]}
                preview = {"entries": ["old"], "reclaimable_bytes": 40}
                with patch("acprof.tui.model_store.disk_report", return_value=report), patch(
                    "acprof.tui.model_store.prune_store", return_value=preview,
                ) as prune:
                    async with app.run_test(size=size) as pilot:
                        app.query_one("#ui-language", Select).value = language
                        app.query_one("#experiment-pages", ContentSwitcher).current = "advanced-form"
                        budget = app.query_one("#max-download", Input)
                        for ancestor in budget.ancestors:
                            if isinstance(ancestor, Collapsible):
                                ancestor.collapsed = False
                        budget.focus()
                        await pilot.pause()
                        self.assertTrue(await pilot.click(budget))
                        await pilot.press(*"5GB")
                        app.query_one("#model-store", Input).value = str(root)
                        app.query_one("#model-store-max", Input).value = "100GB"
                        config = app._collect_config()
                        for builder in (build_run_command, build_probe_command):
                            command = builder(config, project_dir=project)
                            self.assertEqual(command[command.index("--max-download") + 1], "5GB")
                            self.assertEqual(command[command.index("--model-store") + 1], str(root))
                        save_settings(settings_path, TuiSettings(run_defaults=config), project)
                        restored, warning = load_settings(settings_path, project)
                        self.assertFalse(warning)
                        self.assertEqual(restored.run_defaults.max_download, "5GB")
                        self.assertEqual(restored.run_defaults.download_mode, "mirror-only")
                        button = app.query_one("#open-model-store", Button)
                        button.focus()
                        await pilot.pause()
                        self.assertTrue(await pilot.click(button))
                        await app.workers.wait_for_complete()
                        await pilot.pause()
                        self.assertIsInstance(app.screen, ModelStoreScreen)
                        self.assertTrue(app._is_busy())
                        for identifier in ("store-close", "store-refresh", "store-prune"):
                            control = app.screen.query_one(f"#{identifier}", Button)
                            self.assertLessEqual(control.region.right, size[0])
                            self.assertLessEqual(control.region.bottom, size[1])
                            self.assertIs(app.get_widget_at(*control.region.center)[0], control)
                        await pilot.click("#store-prune")
                        await pilot.pause()
                        self.assertIsInstance(app.screen, ConfirmActionScreen)
                        self.assertFalse(any(call.kwargs.get("apply") for call in prune.call_args_list))
                        await pilot.press("escape")
                        await pilot.pause()
                        self.assertFalse(any(call.kwargs.get("apply") for call in prune.call_args_list))
                        await pilot.click("#store-prune")
                        await pilot.pause()
                        await pilot.click("#confirm-yes")
                        await app.workers.wait_for_complete()
                        await pilot.pause()
                        applications = [call for call in prune.call_args_list if call.kwargs.get("apply")]
                        self.assertEqual(len(applications), 1)
                        self.assertEqual(applications[0].kwargs["keep"], {"new"})
                        await pilot.click("#store-close")
                        await pilot.pause()
                        self.assertFalse(app._is_busy())
                        prune.reset_mock()
                        app._latest_snapshot = replace(app._latest_snapshot, measurement_active=True)
                        app.open_model_store()
                        self.assertNotIsInstance(app.screen, ModelStoreScreen)
                        prune.assert_not_called()
                        app._latest_snapshot = replace(app._latest_snapshot, measurement_active=False)
