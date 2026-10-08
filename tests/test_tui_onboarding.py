"""First-run presets must reach the real command without changing saved defaults."""
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Input, Select, Static
from tui_fixtures import AcprofTui

from acprof.cli.tui import main
from acprof.experiment import RunConfig, build_run_command
from acprof.tui.settings import TuiSettings, save_settings


async def test_start_waiting_state_is_translated_without_covering_monitor():
    with tempfile.TemporaryDirectory() as temporary:
        config = replace(
            RunConfig.smoke("demo/model"),
            output_dir=str(Path(temporary) / "results"),
        )
        app = AcprofTui(config, settings_path=Path(temporary) / "settings.json")
        async with app.run_test(size=(120, 30)) as pilot:
            app.ui_preferences = replace(app.ui_preferences, language="en")
            app._apply_ui_preferences()
            await pilot.pause()
            with patch.object(app, "_execute_command"):
                await pilot.press("f5")
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert app.screen.id == "_default"
                assert app.query_one("#main-tabs").active == "monitor-tab"
                assert app._preparation_screen is None
                assert app.query_one("#status-stage", Static).content == "Starting"
                assert app.query_one("#status-detail", Static).content == "Creating subprocess"
                app._process_kind = ""

def test_first_launch_without_saved_configuration_is_a_small_smoke():
    with tempfile.TemporaryDirectory() as temporary:
        app = AcprofTui(settings_path=Path(temporary) / "settings.json")
        assert (app.initial_config) == (RunConfig.smoke())

@pytest.mark.parametrize("legacy_mode", ("mirror-only", "mirror-preferred", "official"))
async def test_changing_presets_preserves_effective_download_constraints(legacy_mode):
    with tempfile.TemporaryDirectory() as temporary:
        config = replace(RunConfig.smoke("demo/model"), model_source="modelscope",
            max_download="5GB", download_mode=legacy_mode, model_store_max="10GB",
            model_store=str(Path(temporary) / "cache"), output_dir=str(Path(temporary) / "results"))
        app = AcprofTui(config, settings_path=Path(temporary) / "settings.json")
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            for preset in ("main", "smoke"):
                app.query_one("#run-preset", Select).value = preset
                await pilot.pause()
                effective = app._collect_config()
                assert (effective.max_download) == ("5GB")
                assert effective.download_mode == "auto"
                assert effective.model_source == "modelscope"
                assert (app.query_one("#model-store", Input).value) == (config.model_store)
                assert effective.model_store == config.model_store
                assert effective.model_store_max == "10GB"
                assert (effective.output_dir) == (config.output_dir)
                command = build_run_command(effective, project_dir=Path.cwd())
                for option, value in (("--download-mode", "auto"), ("--model-source", "modelscope"),
                                      ("--max-download", "5GB"), ("--model-store-max", "10GB"),
                                      ("--model-store", config.model_store), ("--output-dir", config.output_dir)):
                    assert command[command.index(option) + 1] == value

async def test_selecting_smoke_builds_basic_cpu_command_and_preserves_notifications():
    with tempfile.TemporaryDirectory() as temporary:
        app = AcprofTui(RunConfig(model="google-bert/bert-base-uncased", notify="wecom"),
                        settings_path=Path(temporary) / "settings.json")
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            preset = app.query_one("#run-preset", Select)
            preset.focus()
            await pilot.press("enter", "home", "down", "enter")
            await pilot.pause()
            assert (preset.value) == ("smoke")
            command = build_run_command(app._collect_config(), project_dir=Path.cwd())
            for option, value in (("--profiling-mode", "basic"), ("--gpus", "off"),
                                  ("--notify", "wecom"), ("--repeat-in-window", "1")):
                assert (command[command.index(option) + 1]) == (value)

def test_cli_first_run_overrides_saved_full_matrix_without_rewriting_settings():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "settings.json"
        save_settings(path, TuiSettings(run_defaults=RunConfig(model="saved/model")), Path.cwd())
        original = path.read_bytes()
        with patch("acprof.tui.app.default_settings_path", return_value=path), patch(
            "textual.app.App.run", autospec=True,
        ) as run:
            main(["--model", "google-bert/bert-base-uncased", "--preset", "smoke",
                  "--output-dir", "results/first run"])
        app = run.call_args.args[0]
        command = build_run_command(app.initial_config, project_dir=Path.cwd())
        assert (command[command.index("--profiling-mode") + 1]) == ("basic")
        assert (command[command.index("--output-dir") + 1]) == ("results/first run")
        assert (path.read_bytes()) == (original)

def test_starting_without_overrides_preserves_saved_full_matrix():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "settings.json"
        config = RunConfig(model="saved/model", cpus="2,4", output_dir="results/saved")
        save_settings(path, TuiSettings(run_defaults=config), Path.cwd())
        with patch("acprof.tui.app.default_settings_path", return_value=path), patch(
            "textual.app.App.run", autospec=True,
        ) as run:
            main([])
        assert (run.call_args.args[0].initial_config) == (config)

@pytest.mark.parametrize('size,language', (((80, 24), 'zh'), ((120, 30), 'en'), ((150, 45), 'zh')))
async def test_prepared_scales_update_summary_outside_measurement_in_both_languages(size, language):
    with tempfile.TemporaryDirectory() as temporary:
        config = RunConfig.smoke("openai/whisper-tiny")
        app = AcprofTui(config, settings_path=Path(temporary) / "settings.json")
        async with app.run_test(size=size) as pilot:
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            app._active_run_config = config
            app._preparation_event({"stage": "input", "status": "passed",
                "input_plan": {"scales": [1.0], "scale_type": "duration_s"}})
            await pilot.pause()
            assert ("1 s") in (app.query_one("#input-scales", Input).placeholder)
            summary = app.query_one("#run-plan-summary", Static)
            assert ("(s)") in (str(summary.render()))
            button = app.query_one("#start-run", Button)
            assert (summary.region.bottom) <= (app.query_one("#run-actions").region.y)
            assert (app.get_widget_at(button.region.x + 3, button.region.y + 1)[0]) is (button)
            app._latest_snapshot = replace(app._latest_snapshot, measurement_active=True)
            with pytest.raises(RuntimeError, match="measurement window"):
                app._preparation_event({"input_plan": {"scales": [2.0], "scale_type": "duration_s"}})


async def test_revision_edit_invalidates_prepared_scales_but_repeat_edit_keeps_them(tmp_path):
    config = replace(RunConfig.smoke("demo/model"), revision="v1",
                     input_scale_policy="auto", repeat=3)
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    with patch.object(app, "_execute_command") as execute:
        async with app.run_test(size=(120, 30)) as pilot:
            app._active_run_config = app._collect_config()
            app._preparation_event({"stage": "input", "status": "passed",
                "input_plan": {"scales": [8.0, 16.0], "scale_type": "seq_length"}})
            await pilot.pause()
            summary = app.query_one("#run-plan-summary", Static)
            scales = app.query_one("#input-scales", Input)
            assert "输入 2 档 (tokens)" in str(summary.content)
            assert "正式 6 窗口" in str(summary.content)
            assert scales.placeholder == "8,16 tokens"

            await pilot.click("#open-run-settings")
            app.query_one("#repeat", Input).focus()
            await pilot.press("end", "ctrl+u", "4")
            await pilot.pause()
            assert app._collect_config().repeat == 4
            assert "输入 2 档 (tokens)" in str(summary.content)
            assert "正式 8 窗口" in str(summary.content)
            assert scales.placeholder == "8,16 tokens"

            app.query_one("#revision", Input).focus()
            await pilot.press("end", "ctrl+u", "v", "2")
            await pilot.pause()
            assert app._collect_config().revision == "v2"
            assert "输入 待解析 档" in str(summary.content)
            assert "正式 待解析 窗口" in str(summary.content)
            assert scales.placeholder == "留空：自动范围 · 单位待解析"

            app._active_run_config = app._collect_config()
            app._preparation_event({"stage": "input", "status": "passed",
                "input_plan": {"scales": [4.0, 12.0, 24.0], "scale_type": "seq_length"}})
            await pilot.pause()
            assert "输入 3 档 (tokens)" in str(summary.content)
            assert "正式 12 窗口" in str(summary.content)
            assert scales.placeholder == "4,12,24 tokens"
        execute.assert_not_called()
