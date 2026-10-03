"""Download controls and explicit pruning stay outside experiment execution."""
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Collapsible, ContentSwitcher, Input, Select
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, build_run_command
from acprof.tui.commands import build_probe_command
from acprof.tui.model_store import ModelStoreScreen
from acprof.tui.settings import TuiSettings, load_settings, save_settings
from acprof.tui.views import ConfirmActionScreen


@pytest.mark.parametrize('size,language', (((80, 24), 'zh'), ((120, 40), 'en')))
async def test_budget_settings_and_prune_confirmation_in_both_languages(size, language):
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary:
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
                assert (await pilot.click(budget))
                await pilot.press(*"5GB")
                app.query_one("#model-store", Input).value = str(root)
                app.query_one("#model-store-max", Input).value = "100GB"
                config = app._collect_config()
                for builder in (build_run_command, build_probe_command):
                    command = builder(config, project_dir=project)
                    assert (command[command.index("--max-download") + 1]) == ("5GB")
                    assert (command[command.index("--model-store") + 1]) == (str(root))
                save_settings(settings_path, TuiSettings(run_defaults=config), project)
                restored, warning = load_settings(settings_path, project)
                assert not (warning)
                assert (restored.run_defaults.max_download) == ("5GB")
                assert (restored.run_defaults.download_mode) == ("mirror-only")
                button = app.query_one("#open-model-store", Button)
                button.focus()
                await pilot.pause()
                assert (await pilot.click(button))
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert isinstance(app.screen, ModelStoreScreen)
                assert (app._is_busy())
                for identifier in ("store-close", "store-refresh", "store-prune"):
                    control = app.screen.query_one(f"#{identifier}", Button)
                    assert (control.region.right) <= (size[0])
                    assert (control.region.bottom) <= (size[1])
                    assert (app.get_widget_at(*control.region.center)[0]) is (control)
                await pilot.click("#store-prune")
                await pilot.pause()
                assert isinstance(app.screen, ConfirmActionScreen)
                assert not (any(call.kwargs.get("apply") for call in prune.call_args_list))
                await pilot.press("escape")
                await pilot.pause()
                assert not (any(call.kwargs.get("apply") for call in prune.call_args_list))
                await pilot.click("#store-prune")
                await pilot.pause()
                await pilot.click("#confirm-yes")
                await app.workers.wait_for_complete()
                await pilot.pause()
                applications = [call for call in prune.call_args_list if call.kwargs.get("apply")]
                assert (len(applications)) == (1)
                assert (applications[0].kwargs["approved_entries"]) == ({"old"})
                await pilot.click("#store-close")
                await pilot.pause()
                assert not (app._is_busy())
                prune.reset_mock()
                app._latest_snapshot = replace(app._latest_snapshot, measurement_active=True)
                app.open_model_store()
                assert not isinstance(app.screen, ModelStoreScreen)
                prune.assert_not_called()
                app._latest_snapshot = replace(app._latest_snapshot, measurement_active=False)
