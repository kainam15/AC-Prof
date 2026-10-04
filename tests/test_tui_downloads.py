"""Download controls and explicit pruning stay outside experiment execution."""
import tempfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Collapsible, ContentSwitcher, Input, Select, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, build_run_command
from acprof.tui.commands import build_probe_command
from acprof.tui.i18n import translate
from acprof.tui.model_store import ModelStoreScreen
from acprof.tui.settings import TuiSettings, load_settings, save_settings
from acprof.tui.views import ConfirmActionScreen


@pytest.fixture
def download_report():
    return {
        "expected_download_bytes": 708_390_484, "direct_download_bytes": 708_390_484,
        "proxy_download_bytes": 0, "max_download_bytes": 5_000_000_000,
        "model_store_path": "/custom/cache",
        "model": {"total_bytes": 708_391_278, "cached_bytes": 794, "endpoint": "https://hf-mirror.com"},
        "disk": {"total_bytes": 985, "free_bytes": 32_982_540_288,
                 "reclaimable_bytes": 794, "remaining_bytes": 32_274_139_804},
        "runtime": {"platform_local": True, "alternatives": {"local_build_bytes": 1_500_000}},
        "docker_storage": {"docker_storage_total_bytes": 30_358_750_032,
                           "docker_storage_available_bytes_at_start": 32_982_540_288,
                           "docker_storage_filesystem": "ext4", "docker_storage_device": "/dev/nvme1n1p3"},
    }


def test_download_summary_formats_sizes_without_mutating_report(download_report):
    from acprof.tui.downloads import download_fields, download_summary

    original = deepcopy(download_report)
    summary = download_summary(download_report, lambda value: translate(value, "en"))
    for expected in (
        "Expected download: 708 MB", "Effective download budget: 5.00 GB", "Model Store path: /custom/cache",
        "Model total: 708 MB | cached: 794 B", "Model Store: 985 B | free: 33.0 GB",
        "Reclaimable: 794 B | remaining: 32.3 GB", "Docker storage: 30.4 GB",
        "Docker available space: 33.0 GB", "ext4", "/dev/nvme1n1p3", "hf-mirror.com",
    ):
        assert expected in summary
    assert download_fields(download_report) == {
        "预计下载": "708 MB", "可用空间": "33.0 GB", "模型来源": "huggingface",
    }
    assert download_report == original


def test_download_summary_keeps_unknown_distinct_from_zero_and_unlimited():
    from acprof.tui.downloads import download_summary

    report = {"expected_download_bytes": None, "direct_download_bytes": None, "proxy_download_bytes": 0}
    summary = download_summary(report, lambda value: translate(value, "en"))
    assert "Expected download: Unknown" in summary
    assert "DIRECT" not in summary and "PROXY" not in summary
    assert "Using the system network environment" in summary
    assert "Effective download budget: Unknown" in summary
    assert "Docker storage: Unknown" in summary
    report["max_download_bytes"] = None
    assert "Effective download budget: Unlimited" in download_summary(report, lambda value: translate(value, "en"))


async def test_network_reports_display_units_and_preserve_input_json(tmp_path, download_report):
    import json

    app = AcprofTui(RunConfig.smoke("example/model"), settings_path=tmp_path / "settings.json")
    line = "[network-preflight] " + json.dumps(download_report)
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#ui-language", Select).value = "en"
        await pilot.pause()
        app._network_preflight_report(line)
        payload = {"category": "model", "verified_new_payload_bytes": 708_390_484,
                   "cache_savings_bytes": 794, "wire_bytes": None}
        app._network_download_report("[network-download] " + json.dumps(payload))
        content = app.query_one("#network-download-summary", Static).content
        assert "Expected download: 708 MB" in content
        assert "Verified new payload: 708 MB" in content
        assert "Cache savings: 794 B" in content
        assert "Wire bytes: Unknown" in content
        assert json.loads(line.split(" ", 1)[1]) == download_report


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
                await pilot.pause()
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
                assert (restored.run_defaults.download_mode) == ("auto")
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
