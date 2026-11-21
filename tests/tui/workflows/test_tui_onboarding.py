"""First-run presets must reach the real command without changing saved defaults."""
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Input, Static

from acprof.cli.tui import main
from acprof.experiments.configuration import RunConfig, build_run_command
from acprof.tui.i18n import message_id
from acprof.tui.run.plan_panel import RunPlanPanel
from acprof.tui.settings.preferences import TuiSettings, save_settings
from acprof.tui.ui.environment import EnvironmentIndicator
from acprof.tui.ui.structured import FactGrid
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


async def test_start_waiting_state_is_translated_without_covering_monitor():
    with tempfile.TemporaryDirectory() as temporary:
        config = replace(
            RunConfig.smoke("demo/model"),
            output_dir=str(Path(temporary) / "results"),
        )
        app = AcprofTui(config, settings_path=Path(temporary) / "settings.json")
        async with app.run_test(size=(120, 30)) as pilot:
            app.configuration.preferences = replace(app.configuration.preferences, language="en")
            app.configuration.apply_ui_preferences()
            await pilot.pause()
            with patch.object(app.process, 'execute_command'):
                await pilot.press("f5")
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert app.screen.id == "_default"
                assert app.query_one("#main-tabs").active == "monitor-tab"
                assert app.preflight.preparation_screen is None
                assert str(app.query_one("#status-stage", Static).content) == "… Starting"
                details = app.query_one("#monitor-details", FactGrid)
                assert details.section.collapsed
                assert any(app.tr(row.detail) == "Creating subprocess" for row in details.section.rows)
                app.execution.session.kind = ""

def test_first_launch_without_saved_configuration_is_a_small_smoke():
    with tempfile.TemporaryDirectory() as temporary:
        app = AcprofTui(settings_path=Path(temporary) / "settings.json")
        assert app.configuration.initial_config == replace(RunConfig.smoke(), runtime="docker")

@pytest.mark.parametrize(('language', 'size'), [('zh', (80, 24)), ('en', (120, 30))])
async def test_first_open_blank_model_is_neutral_and_locatable(tmp_path, language, size):
    app = AcprofTui(settings_path=tmp_path / 'fresh-settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        await wait_for_condition(pilot, lambda: app.preflight.status.state == 'unconfigured',
                                 description='initial model draft', timeout=3)
        assert app.preflight.status.counts['error'] == 0
        assert app.preflight.status.counts['warning'] == 0
        assert app.query_one('#environment-error', EnvironmentIndicator).content == '× 0'
        assert not app.query_one('#model-error', Static).display
        assert app.query_one('#start-run', Button).disabled
        plan = app.query_one(RunPlanPanel).view
        assert plan.summary.level == 'neutral'
        assert app.tr(plan.summary.headline) == (
            '待配置 · 请输入模型 ID' if language == 'zh'
            else 'Not configured · Enter a model ID'
        )
        assert not plan.callout
        app.query_one('#model-candidates', Button).focus()
        assert await pilot.click('#preflight-run-reason')
        await wait_for_condition(pilot, lambda: app.focused is app.query_one('#model'),
                                 description='pending reason locates model', timeout=3)


async def test_edited_empty_model_does_not_suppress_other_form_errors(tmp_path):
    app = AcprofTui(RunConfig.smoke('fixture/model'), settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        app.query_one('#model', Input).value = ''
        await wait_for_condition(pilot, lambda: app.preflight.status.state == 'unconfigured',
                                 description='editing cleared model', timeout=3)
        app.query_one('#cpus', Input).value = 'invalid-cpu'
        await wait_for_condition(pilot, lambda: app.preflight.status.state == 'incomplete',
                                 description='unrelated form error', timeout=3)
        assert not app.query_one('#model-error', Static).display
        assert app.query_one('#cpus-error', Static).display
        assert app.preflight.status.counts['error'] >= 1
        assert not app.preflight.status.can_start
        app.query_one('#cpus', Input).value = '1'
        await wait_for_condition(pilot, lambda: app.preflight.status.state == 'unconfigured',
                                 description='resolved other error', timeout=3)
        assert app.preflight.status.counts['error'] == 0
        assert not app.query_one('#cpus-error', Static).display


@pytest.mark.parametrize("legacy_mode", ["mirror-only", "mirror-preferred", "official"])
async def test_changing_presets_preserves_effective_download_constraints(legacy_mode):
    with tempfile.TemporaryDirectory() as temporary:
        config = replace(RunConfig.smoke("demo/model"), model_source="modelscope",
            max_download="5GB", download_mode=legacy_mode, model_store_max="10GB",
            model_store=str(Path(temporary) / "cache"), output_dir=str(Path(temporary) / "results"))
        app = AcprofTui(config, settings_path=Path(temporary) / "settings.json")
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            for preset in ("main", "smoke"):
                app.experiments.apply_preset(preset)
                await pilot.pause()
                effective = app.configuration.collect_config()
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

@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30)), ("zh", (150, 45))])
async def test_smoke_command_without_preset_selector_preserves_notifications(language, size):
    with tempfile.TemporaryDirectory() as temporary:
        app = AcprofTui(RunConfig(model="google-bert/bert-base-uncased", notify="wecom"),
                        settings_path=Path(temporary) / "settings.json")
        app.configuration.preferences = replace(app.configuration.preferences, language=language, show_command_bar=True)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            assert not app.query("#run-preset")
            command_input = app.query_one("#slash-command", Input)
            command_input.value = "/smoke"
            command_input.focus()
            await pilot.press("enter")
            await pilot.pause()
            command = build_run_command(app.configuration.collect_config(), project_dir=Path.cwd())
            assert "--profiling-mode" not in command
            for option, value in (("--gpus", "off"),
                                  ("--notify", "wecom"), ("--repeat-in-window", "1")):
                assert (command[command.index(option) + 1]) == (value)


async def test_preset_command_reports_invalid_form_without_replacing_fields(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, show_command_bar=True)
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#cpus", Input).value = "invalid-cpus"
        command_input = app.query_one("#slash-command", Input)
        command_input.value = "/main"
        command_input.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#cpus", Input).value == "invalid-cpus"
        assert not app.query("#profiling-mode")
        assert app.query_one("#cpus-error", Static).display

def test_cli_first_run_overrides_saved_full_matrix_without_rewriting_settings():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "settings.json"
        save_settings(path, TuiSettings(run_defaults=RunConfig(model="saved/model")), Path.cwd())
        original = path.read_bytes()
        with patch("acprof.tui.settings.controller.default_settings_path", return_value=path), patch(
            "textual.app.App.run", autospec=True,
        ) as run:
            main(["--model", "google-bert/bert-base-uncased", "--preset", "smoke",
                  "--output-dir", "results/first run"])
        app = run.call_args.args[0]
        command = build_run_command(app.configuration.initial_config, project_dir=Path.cwd())
        assert "--profiling-mode" not in command
        assert (command[command.index("--output-dir") + 1]) == ("results/first run")
        assert (path.read_bytes()) == (original)

@pytest.mark.parametrize(("saved_runtime", "form_runtime"), [("auto", "docker"), ("process", "process")])
def test_starting_without_overrides_preserves_saved_full_matrix(saved_runtime, form_runtime):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "settings.json"
        config = RunConfig(model="saved/model", runtime=saved_runtime, cpus="2,4", output_dir="results/saved")
        save_settings(path, TuiSettings(run_defaults=config), Path.cwd())
        original = path.read_bytes()
        with patch("acprof.tui.settings.controller.default_settings_path", return_value=path), patch(
            "textual.app.App.run", autospec=True,
        ) as run:
            main([])
        assert run.call_args.args[0].configuration.initial_config == replace(config, runtime=form_runtime)
        assert path.read_bytes() == original

@pytest.mark.parametrize(("size", "language"), [((80, 24), 'zh'), ((120, 30), 'en'), ((150, 45), 'zh')])
async def test_prepared_scales_update_summary_outside_measurement_in_both_languages(size, language):
    with tempfile.TemporaryDirectory() as temporary:
        config = RunConfig.smoke("openai/whisper-tiny")
        app = AcprofTui(config, settings_path=Path(temporary) / "settings.json")
        async with app.run_test(size=size) as pilot:
            app.configuration.preferences = replace(app.configuration.preferences, language=language)
            app.configuration.apply_ui_preferences()
            app.execution.session.config = config
            app.preflight.preparation_event({"stage": "input", "status": "passed",
                "input_plan": {"scales": [1.0], "scale_type": "duration_s"}})
            await pilot.pause()
            assert ("1 s") in (app.query_one("#input-scales", Input).placeholder)
            summary = app.query_one("#run-plan-panel", RunPlanPanel)
            assert summary.view is not None
            facts = {row.label: row for row in summary.view.section.rows}
            assert facts[message_id("run.plan.scales_label")].value == message_id(
                "run.plan.scale_count_unit", 1, "s",
            )
            button = app.query_one("#start-run", Button)
            assert (summary.region.bottom) <= (app.query_one("#run-actions").region.y)
            assert (app.get_widget_at(button.region.x + 3, button.region.y + 1)[0]) is (button)
            app.execution.session.snapshot = replace(app.execution.session.snapshot, measurement_active=True)
            with pytest.raises(RuntimeError, match="measurement window"):
                app.preflight.preparation_event({"input_plan": {"scales": [2.0], "scale_type": "duration_s"}})


async def test_revision_edit_invalidates_prepared_scales_but_repeat_edit_keeps_them(tmp_path):
    config = replace(RunConfig.smoke("demo/model"), revision="v1",
                     input_scale_policy="auto", repeat=3)
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    with patch.object(app.process, 'execute_command') as execute:
        async with app.run_test(size=(120, 30)) as pilot:
            app.execution.session.config = app.configuration.collect_config()
            app.preflight.preparation_event({"stage": "input", "status": "passed",
                "input_plan": {"scales": [8.0, 16.0], "scale_type": "seq_length"}})
            await pilot.pause()
            summary = app.query_one("#run-plan-panel", RunPlanPanel)
            scales = app.query_one("#input-scales", Input)

            def facts():
                assert summary.view is not None
                return {row.label: row.value for row in summary.view.section.rows}

            scale_label = message_id("run.plan.scales_label")
            window_label = message_id("run.plan.windows_label")
            assert facts()[scale_label] == message_id("run.plan.scale_count_unit", 2, "tokens")
            assert facts()[window_label] == message_id("run.plan.window_totals", 0, 6)
            assert scales.placeholder == "8,16 tokens"

            app.configuration.open_config_field("repeat")
            await wait_for_condition(pilot, lambda: app.focused is app.query_one("#repeat", Input),
                                     description="repeat control revealed and focused", timeout=3)
            await pilot.press("end", "ctrl+u", "4")
            await wait_for_condition(
                pilot,
                lambda: facts().get(window_label) == message_id("run.plan.window_totals", 0, 8),
                description="edited repeat count in the collection plan", timeout=3,
            )
            assert app.configuration.collect_config().repeat == 4
            assert facts()[scale_label] == message_id("run.plan.scale_count_unit", 2, "tokens")
            assert scales.placeholder == "8,16 tokens"

            app.configuration.open_config_field("revision")
            await wait_for_condition(pilot, lambda: app.focused is app.query_one("#revision", Input),
                                     description="revision control revealed and focused", timeout=3)
            await pilot.press("end", "ctrl+u", "v", "2")
            pending = message_id("common.state.pending")
            await wait_for_condition(
                pilot,
                lambda: facts().get(window_label) == message_id("run.plan.window_totals", pending, pending),
                description="stale prepared scales removed after a revision edit", timeout=3,
            )
            assert app.configuration.collect_config().revision == "v2"
            assert facts()[scale_label] == message_id(
                "run.plan.scale_count_unit", pending, message_id("run.inputs.unit_pending"),
            )
            assert scales.placeholder == "留空：自动范围 · 单位待解析"

            app.execution.session.config = app.configuration.collect_config()
            app.preflight.preparation_event({"stage": "input", "status": "passed",
                "input_plan": {"scales": [4.0, 12.0, 24.0], "scale_type": "seq_length"}})
            await pilot.pause()
            assert facts()[scale_label] == message_id("run.plan.scale_count_unit", 3, "tokens")
            assert facts()[window_label] == message_id("run.plan.window_totals", 0, 12)
            assert scales.placeholder == "4,12,24 tokens"
        execute.assert_not_called()
