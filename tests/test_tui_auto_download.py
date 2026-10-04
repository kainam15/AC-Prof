"""The ordinary flow exposes source choice, never proxy/VPN controls."""
import asyncio
import sys
from unittest.mock import patch

import pytest
from textual.widgets import Button, Collapsible, Input, Select, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.preparation import ModelSourceScreen, PreparationScreen
from acprof.tui.process import ProcessLifecycle


@pytest.mark.parametrize("size,language", [((80, 24), "zh"), ((120, 40), "en"), ((160, 48), "zh")])
async def test_failure_actions_keep_diagnostics_folded_and_require_source_confirmation(tmp_path, size, language):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    replies = []
    event = {"stage": "model", "status": "failed", "request": {"id": 1, "kind": "error",
        "detail": "Traceback: diagnostic-only", "download_error": {"source": "huggingface", "model_id": "demo/model",
        "attempts": [{"stage": "storage", "reason": "TLS", "host": "us.aws.cdn.hf.co"}]}}}
    async with app.run_test(size=size) as pilot:
        app.query_one("#ui-language", Select).value = language
        assert not app.query("#download-mode")
        screen = PreparationScreen(event, respond=replies.append)
        app.push_screen(screen)
        await pilot.pause()
        assert screen.query_one("#preparation-diagnostics", Collapsible).collapsed
        body = str(screen.query_one("#preparation-detail", Static).content)
        assert "Traceback" not in body and "storage / TLS" in body and "us.aws.cdn.hf.co" in body
        for identifier in ("preparation-continue", "preparation-show-diagnostics", "preparation-modelscope"):
            button = screen.query_one("#" + identifier, Button)
            assert button.region.width > 0 and button.region.right <= size[0] and button.region.bottom <= size[1]
            assert app.get_widget_at(*button.region.center)[0] is button
        await pilot.click("#preparation-show-diagnostics")
        assert not screen.query_one("#preparation-diagnostics", Collapsible).collapsed
        await pilot.click("#preparation-modelscope")
        await pilot.pause()
        assert isinstance(app.screen, ModelSourceScreen)
        assert app.screen.query_one("#source-model-id", Input).value == ""
        await pilot.click("#source-confirm")
        assert not replies
        await pilot.press("escape")
        await pilot.pause()
        assert not replies and app.screen is screen
        await pilot.click("#preparation-modelscope")
        await pilot.pause()
        app.screen.query_one("#source-model-id", Input).value = "new-owner/new-model"
        await pilot.click("#source-confirm")
        await pilot.pause()
        assert replies == [{"action": "switch-source", "model_id": "new-owner/new-model", "revision": ""}]
        assert app.query_one("#model-source", Select).value == "huggingface"


async def test_confirmed_source_switch_stops_old_child_and_returns_new_experiment_config(tmp_path):
    script = '''
from acprof.host.collection_workflow import PreparationWorkflow
from acprof.hf_download import HfDownloadError
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
            await pilot.click("#preparation-modelscope")
            await pilot.pause()
            app.screen.query_one("#source-model-id", Input).value = "new/model"
            await pilot.click("#source-confirm")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=15)
            await pilot.pause()
            assert app._lifecycle.process is None
            config = app._collect_config()
            assert (config.model_source, config.model) == ("modelscope", "new/model")
            assert not config.resume
            assert "modelscope-" in config.output_dir
            assert not app.query_one("#start-run", Button).disabled
