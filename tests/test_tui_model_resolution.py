"""Resolution UI asks for gaps only and hands probes to the managed subprocess."""
import json
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx
import pytest
import test_model_contract as fixture
from huggingface_hub.errors import RepositoryNotFoundError
from test_model_lookup import hub_error
from textual.widgets import Button, Collapsible, Input, Select, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig


async def test_probe_uses_managed_process_and_preserves_reviewed_revision():
    task = fixture.TestModelContract().discover()
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task) as detect, patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ):
        app = AcprofTui(RunConfig(model=task.model_id, revision=task.model_revision, output_dir=directory), settings_path=Path(directory, "settings.json"))
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#ui-language", Select).value = "en"
            button = app.query_one("#inspect-model", Button)
            button.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            await pilot.click(button)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (detect.call_args.kwargs["revision"]) == (task.model_revision)
            assert (len(list(app.screen.query(".resolution-answer")))) == (0)
            assert (str(app.screen.query_one("#resolution-use", Button).label)) == ("Use contract")
            await pilot.resize_terminal(120, 30)
            with patch.object(app, "_launch") as launch:
                assert (await pilot.click("#resolution-basic"))
                await pilot.pause()
            command = launch.call_args.args[0].command
            assert (command[command.index("--probe") + 1]) == ("basic")
            assert (command[command.index("--expected-revision") + 1]) == (task.model_revision)
            assert ("--gpus") not in (command)

@pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
async def test_only_unknown_input_is_editable_and_export_reaches_run_form(size):
    source = fixture.SOURCE.replace('audio = inputs.get("audio", None)', 'audio = inputs.get("audio", None)\n        speaker = inputs["speaker"]')
    task = fixture.TestModelContract().discover(source)
    with tempfile.TemporaryDirectory() as directory, patch(
        "acprof.host.detect.detect_task", return_value=task,
    ), patch("acprof.host.env_utils.bootstrap_project_env"):
        app = AcprofTui(RunConfig(model=task.model_id, output_dir=directory), settings_path=Path(directory, "settings.json"))
        async with app.run_test(size=size) as pilot:
            button = app.query_one("#inspect-model", Button)
            button.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert (await pilot.click(button))
            await app.workers.wait_for_complete()
            await pilot.pause()
            from acprof.tui.input import BarCursorInput
            fields = list(app.screen.query(".resolution-answer"))
            assert (len(fields)) == (1)
            field = fields[0]
            assert isinstance(field, BarCursorInput)
            assert ("speaker") in (str(app.screen.query_one("#resolution-body", Static).render()))
            field.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            await pilot.click(field)
            await pilot.press(*'{"literal":"narrator"}')
            assert (await pilot.click("#resolution-apply"))
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (len(list(app.screen.query(".resolution-answer")))) == (0)
            assert (await pilot.click("#resolution-use"))
            await pilot.pause()
            path = app.query_one("#model-spec", BarCursorInput).value
            spec = json.loads(Path(path).read_text())
            assert (spec["multimodal"]["inputs"]["speaker"]) == ({"literal": "narrator"})
            assert not (app._is_busy())

async def test_measurement_disables_resolution():
    from acprof.tui.progress import ProgressSnapshot
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(RunConfig(model="fixture/model"), settings_path=Path(directory, "settings.json"))
        async with app.run_test(size=(80, 24)) as pilot:
            app._latest_snapshot = ProgressSnapshot(measurement_active=True)
            app._set_busy(True)
            await pilot.pause()
            assert (app.query_one("#inspect-model", Button).disabled)
@pytest.mark.parametrize('language,retry,hint', (('zh', '重试', '检查网络'), ('en', 'Retry', 'network')))
async def test_network_failure_can_retry_and_clears_the_previous_error(language, retry, hint):
    task = fixture.TestModelContract().discover()
    with tempfile.TemporaryDirectory() as directory, patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ), patch("acprof.host.detect.detect_task", side_effect=[httpx.ReadTimeout("fixture timeout"), task]) as detect:
        app = AcprofTui(RunConfig(model=task.model_id), settings_path=Path(directory, "settings.json"))
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#ui-language", Select).value = language
            await pilot.pause()
            app.inspect_model()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (hint) in (cast(str, app.screen.query_one("#resolution-error", Static).content))
            assert (str(app.screen.query_one("#resolution-retry", Button).label)) == (retry)
            assert (await pilot.click("#resolution-retry"))
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (detect.call_count) == (2)
            assert (app.screen.query_one("#resolution-error", Static).content) == ("")
            assert not (app.screen.query_one("#resolution-use", Button).disabled)

@pytest.mark.parametrize('language,size,title,hint,back', (('zh', (80, 24), '模型检查失败', '未找到模型', '返回修改'), ('en', (120, 30), 'Model check failed', 'not found', 'Edit model ID')))
async def test_missing_model_replaces_loading_and_returns_to_model_field(language, size, title, hint, back):
    with tempfile.TemporaryDirectory() as directory, patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ), patch("huggingface_hub.HfApi.model_info", side_effect=hub_error(RepositoryNotFoundError, 404)), patch(
        "acprof.host.detect._download_metadata", side_effect=OSError("missing"),
    ):
        app = AcprofTui(RunConfig(model="asdf"), settings_path=Path(directory, "settings.json"))
        async with app.run_test(size=size) as pilot:
            app.query_one("#ui-language", Select).value = language
            await pilot.pause()
            app.inspect_model()
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen
            assert (screen.query_one("#resolution-error", Static).content) != ("1")
            assert (screen.query_one("#resolution-title", Static).content) == (title)
            assert (hint) in (cast(str, screen.query_one("#resolution-body", Static).content))
            text = "\n".join(content for widget in screen.query(Static)
                             if isinstance(content := widget.content, str))
            assert ("正在读取") not in (text)
            assert ("Reading model evidence") not in (text)
            assert ("300") not in (text)
            assert (screen.query_one("#resolution-basic", Button).disabled)
            assert (screen.query_one("#resolution-use", Button).disabled)
            assert (screen.query_one(Collapsible).collapsed)
            assert (str(screen.query_one("#resolution-close", Button).label)) == (back)
            assert (await pilot.click("#resolution-close"))
            await pilot.pause()
            model = app.query_one("#model", Input)
            assert (model.value) == ("asdf")
            assert (app.focused) is (model)
            assert not (app._is_busy())
