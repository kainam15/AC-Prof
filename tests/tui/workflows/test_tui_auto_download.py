"""The ordinary flow exposes source choice, never proxy/VPN controls."""
import asyncio
import sys
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import Button, Collapsible, Input, Select, Static, Tooltip

from acprof.experiment import RunConfig
from acprof.tui.log import SelectableLog
from acprof.tui.preparation import ModelSourceScreen, PreparationScreen
from acprof.tui.process import ProcessLifecycle
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize(("size", "language", "stage"), [
    ((80, 24), "zh", "model"), ((120, 40), "en", "model"), ((160, 48), "zh", "model"),
    ((80, 24), "en", "model"), ((80, 24), "en", "runtime"),
])
async def test_failure_actions_show_details_and_require_source_confirmation(tmp_path, size, language, stage):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    app.TOOLTIP_DELAY = 0.01
    replies = []
    event = {"stage": stage, "status": "failed", "request": {"id": 1, "kind": "error",
        "detail": "Traceback: diagnostic-only", "download_error": {"source": "huggingface", "model_id": "demo/model",
        "attempts": [{"stage": "storage", "reason": "TLS", "host": "us.aws.cdn.hf.co"}]}}}
    async with app.run_test(size=size, tooltips=True) as pilot:
        app.query_one("#ui-language", Select).value = language
        assert not app.query("#download-mode")
        screen = PreparationScreen(event, respond=replies.append)
        app.push_screen(screen)
        await pilot.pause()
        assert not screen.query_one("#preparation-diagnostics", Collapsible).collapsed
        assert not screen.query("#preparation-show-diagnostics")
        log = screen.query_one("#preparation-traceback", SelectableLog)
        assert "Traceback: diagnostic-only" in log.text
        assert '"host": "us.aws.cdn.hf.co"' in log.text
        body = str(screen.query_one("#preparation-detail", Static).content)
        assert "Traceback" not in body
        assert "storage / TLS" in body
        assert "us.aws.cdn.hf.co" in body
        identifiers = ["preparation-cancel", "preparation-continue", "preparation-modelscope"]
        if stage == "runtime":
            identifiers.append("preparation-rebuild")
        for identifier in identifiers:
            button = screen.query_one("#" + identifier, Button)
            assert button.region.width > 0
            assert cell_len(str(button.label)) <= button.content_region.width
            assert button.region.right <= size[0]
            assert button.region.bottom <= size[1]
            assert app.get_widget_at(*button.region.center)[0] is button
        assert await pilot.hover("#preparation-modelscope")
        await pilot.pause(app.TOOLTIP_DELAY + 0.05)
        tooltip = screen.query_one(Tooltip)
        assert tooltip.display
        assert "ModelScope" in str(tooltip.content)
        assert "revision" in str(tooltip.content)
        assert screen.region.contains_region(tooltip.region)
        await pilot.click("#preparation-modelscope")
        await pilot.pause()
        assert isinstance(app.screen, ModelSourceScreen)
        assert app.screen.query_one("#source-model-id", Input).value == ""
        await pilot.click("#source-confirm")
        assert not replies
        await pilot.press("escape")
        await pilot.pause()
        assert not replies
        assert app.screen is screen
        await pilot.click("#preparation-modelscope")
        await pilot.pause()
        app.screen.query_one("#source-model-id", Input).value = "new-owner/new-model"
        await pilot.click("#source-confirm")
        await pilot.pause()
        assert replies == [{"action": "switch-source", "model_id": "new-owner/new-model", "revision": ""}]
        assert app.query_one("#model-source", Select).value == "huggingface"


async def test_confirmed_source_switch_stops_old_child_and_returns_new_experiment_config(tmp_path):
    script = '''
from acprof.host.preparation.collection_workflow import PreparationWorkflow
from acprof.hub.huggingface.download import HfDownloadError
workflow = PreparationWorkflow(interactive=True)
def fail():
    raise HfDownloadError("old/model", [{"stage":"storage", "reason":"timeout", "host":"cdn.hf.co"}])
workflow.run("model", fail)
'''
    app = AcprofTui(RunConfig.smoke("old/model"), settings_path=tmp_path / "settings.json")
    original_start = ProcessLifecycle.start
    def start(owner, _command, **kwargs):
        return original_start(owner, [sys.executable, "-c", script], **kwargs)
    with patch("acprof.tui.process.ProcessLifecycle.start", autospec=True, side_effect=start):
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#start-run")
            async def ready():
                while not app.screen.query("#preparation-modelscope"):
                    await pilot.pause()
            await asyncio.wait_for(ready(), timeout=10)
            # The other dialog test exercises physical hit-testing. Here a
            # semantic press avoids a layout race on heavily loaded CI runners.
            app.screen.query_one("#preparation-modelscope", Button).press()
            await pilot.pause()
            async def source_screen_ready():
                while not isinstance(app.screen, ModelSourceScreen) or not app.screen.query("#source-model-id"):
                    await pilot.pause()
            await asyncio.wait_for(source_screen_ready(), timeout=10)
            app.screen.query_one("#source-model-id", Input).value = "new/model"
            await pilot.click("#source-confirm")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=15)
            await pilot.pause()
            assert app.lifecycle.process is None
            config = app.configuration.collect_config()
            assert (config.model_source, config.model) == ("modelscope", "new/model")
            assert not config.resume
            assert "modelscope-" in config.output_dir
            assert not app.query_one("#start-run", Button).disabled
