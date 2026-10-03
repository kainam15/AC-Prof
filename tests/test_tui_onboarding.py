"""First-run presets must reach the real command without changing saved defaults."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Button, Input, Select, Static
from tui_fixtures import AcprofTui

from acprof.cli.tui import main
from acprof.experiment import RunConfig, build_run_command
from acprof.tui.settings import TuiSettings, save_settings


class TuiOnboardingTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_confirmation_translates_planning_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=Path(temporary) / "settings.json")
            async with app.run_test(size=(120, 30)) as pilot:
                app.ui_preferences = replace(app.ui_preferences, language="en")
                app._apply_ui_preferences()
                await pilot.pause()
                await pilot.press("f5")
                await pilot.pause()
                body = str(app.screen.query_one("#confirm-message", Static).render())
                self.assertIn("Download budget", body)
                self.assertIn("assuming 1s/request", body)
                self.assertNotIn("下载预算", body)

    def test_first_launch_without_saved_configuration_is_a_small_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = AcprofTui(settings_path=Path(temporary) / "settings.json")
            self.assertEqual(app.initial_config, RunConfig.smoke())

    async def test_changing_presets_preserves_effective_download_constraints(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(RunConfig.smoke("demo/model"), max_download="5GB", download_mode="official",
                model_store=str(Path(temporary) / "cache"), output_dir=str(Path(temporary) / "results"))
            app = AcprofTui(config, settings_path=Path(temporary) / "settings.json")
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                for preset in ("main", "smoke"):
                    app.query_one("#run-preset", Select).value = preset
                    await pilot.pause()
                    effective = app._collect_config()
                    self.assertEqual(effective.max_download, "5GB")
                    self.assertEqual(effective.download_mode, "official")
                    self.assertEqual(app.query_one("#model-store", Input).value, config.model_store)
                    self.assertEqual(effective.output_dir, config.output_dir)
                    command = build_run_command(effective, project_dir=Path.cwd())
                    self.assertEqual(command[command.index("--max-download") + 1], "5GB")

    async def test_selecting_smoke_builds_basic_cpu_command_and_preserves_notifications(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = AcprofTui(RunConfig(model="google-bert/bert-base-uncased", notify="wecom"),
                            settings_path=Path(temporary) / "settings.json")
            async with app.run_test(size=(120, 30)) as pilot:
                await pilot.pause()
                preset = app.query_one("#run-preset", Select)
                preset.focus()
                await pilot.press("enter", "home", "down", "enter")
                await pilot.pause()
                self.assertEqual(preset.value, "smoke")
                command = build_run_command(app._collect_config(), project_dir=Path.cwd())
                for option, value in (("--profiling-mode", "basic"), ("--gpus", "off"),
                                      ("--notify", "wecom"), ("--repeat-in-window", "1")):
                    self.assertEqual(command[command.index(option) + 1], value)

    def test_cli_first_run_overrides_saved_full_matrix_without_rewriting_settings(self):
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
            self.assertEqual(command[command.index("--profiling-mode") + 1], "basic")
            self.assertEqual(command[command.index("--output-dir") + 1], "results/first run")
            self.assertEqual(path.read_bytes(), original)

    def test_starting_without_overrides_preserves_saved_full_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            config = RunConfig(model="saved/model", cpus="2,4", output_dir="results/saved")
            save_settings(path, TuiSettings(run_defaults=config), Path.cwd())
            with patch("acprof.tui.app.default_settings_path", return_value=path), patch(
                "textual.app.App.run", autospec=True,
            ) as run:
                main([])
            self.assertEqual(run.call_args.args[0].initial_config, config)

    async def test_prepared_scales_update_summary_outside_measurement_in_both_languages(self):
        for size, language in (((80, 24), "zh"), ((120, 30), "en"), ((150, 45), "zh")):
            with self.subTest(size=size, language=language), tempfile.TemporaryDirectory() as temporary:
                config = RunConfig.smoke("openai/whisper-tiny")
                app = AcprofTui(config, settings_path=Path(temporary) / "settings.json")
                async with app.run_test(size=size) as pilot:
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    app._active_run_config = config
                    app._preparation_event({"stage": "input", "status": "passed",
                        "input_plan": {"scales": [1.0], "scale_type": "duration_s"}})
                    await pilot.pause()
                    self.assertIn("1 s", app.query_one("#input-scales", Input).placeholder)
                    summary = app.query_one("#run-plan-summary", Static)
                    self.assertIn("(s)", str(summary.render()))
                    button = app.query_one("#start-run", Button)
                    self.assertLessEqual(summary.region.bottom, app.query_one("#run-actions").region.y)
                    self.assertIs(app.get_widget_at(button.region.x + 3, button.region.y + 1)[0], button)
                    app._latest_snapshot = replace(app._latest_snapshot, measurement_active=True)
                    with self.assertRaisesRegex(RuntimeError, "measurement window"):
                        app._preparation_event({"input_plan": {"scales": [2.0], "scale_type": "duration_s"}})
