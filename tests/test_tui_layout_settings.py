import os
import tempfile
from dataclasses import replace
from functools import partial
from itertools import product
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    ContentSwitcher,
    Input,
    Select,
    Static,
    TabbedContent,
)
from tui_fixtures import AcprofTui

from acprof.cli.tui import main
from acprof.experiment import RunConfig
from acprof.tui.commands import PendingLaunch
from acprof.tui.log import SelectableLog
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.run_results import RunResult
from acprof.tui.settings import (
    TuiSettings,
    UiPreferences,
    default_settings_path,
    load_settings,
    save_settings,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]


class TestTuiLayoutSettings:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.temporary = tmp_path
        self.settings_path = Path(str(self.temporary)) / "tui.json"

    def assert_button_reachable(self, app, button_id):
        button = app.query_one(f"#{button_id}", Button)
        region = button.region
        bottom = app.query_one("#bottom-panel").region.y
        assert (region.width) > (0), button_id
        assert (region.height) > (0), button_id
        assert (region.x) >= (0), button_id
        assert (region.y) >= (0), button_id
        assert (region.right) <= (app.size.width), button_id
        assert (region.bottom) <= (bottom), button_id
        center = (region.x + region.width // 2, region.y + region.height // 2)
        widget, _ = app.get_widget_at(*center)
        assert (widget) is (button), f"{button_id} is obscured by {widget}"

    async def remember_experiment(self, app, pilot):
        app._activate_tab("run-tab")
        await pilot.pause()
        if app.query_one("#experiment-pages", ContentSwitcher).current != "advanced-form":
            assert (await pilot.click("#open-run-settings", offset=(3, 1)))
            await pilot.pause()
        app.query_one("#advanced-form").scroll_end(animate=False, immediate=True)
        await pilot.pause()
        self.assert_button_reachable(app, "save-run-default")
        assert (await pilot.click("#save-run-default", offset=(3, 1)))
        await pilot.pause()

    @pytest.mark.parametrize('size', ((120, 30), (80, 24), (150, 45)))
    async def test_action_bar_stays_reachable_and_advanced_settings_click_works(self, size):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            action_region = app.query_one("#run-actions").region
            for button_id in ("open-run-settings", "probe-largest", "start-run"):
                self.assert_button_reachable(app, button_id)
            app.query_one("#command-details", Collapsible).collapsed = False
            app.query_one("#run-form").scroll_end(animate=False, immediate=True)
            await pilot.pause()
            assert (app.query_one("#run-actions").region) == (action_region)
            for button_id in ("open-run-settings", "probe-largest", "start-run"):
                self.assert_button_reachable(app, button_id)
            assert (await pilot.click("#open-run-settings", offset=(3, 1)))
            await pilot.pause()
            assert (app.query_one("#main-tabs", TabbedContent).active) == ("run-tab")
            assert (app.query_one("#experiment-pages", ContentSwitcher).current) == ("advanced-form")
            for button_id in ("open-run-settings", "probe-largest", "start-run"):
                self.assert_button_reachable(app, button_id)
            assert (str(app.query_one("#open-run-settings", Button).label)) == ("返回基本配置")
            app.query_one("#advanced-form").scroll_end(animate=False, immediate=True)
            await pilot.pause()
            assert (app.query_one("#run-actions").region) == (action_region)
            self.assert_button_reachable(app, "save-run-default")
            assert (await pilot.click("#open-run-settings", offset=(3, 1)))
            await pilot.pause()
            assert (app.query_one("#experiment-pages", ContentSwitcher).current) == ("run-form")
            assert (str(app.query_one("#open-run-settings", Button).label)) == ("高级参数")
            await pilot.press("f2")
            await pilot.pause()
            for button_id in ("restore-ui-defaults", "save-ui-settings"):
                self.assert_button_reachable(app, button_id)

    @pytest.mark.parametrize('show_command_bar', (False, True))
    @pytest.mark.parametrize('size', ((120, 40), (80, 24), (150, 45)))
    async def test_action_buttons_survive_resize_and_command_bar_visibility(self, show_command_bar, size):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.pause()
            app.query_one("#ui-theme", Select).value = "acprof-mist"
            await pilot.resize_terminal(*size)
            app.action_show_settings()
            app.query_one("#ui-command-bar", Checkbox).value = show_command_bar
            await pilot.pause()
            app._activate_tab("run-tab")
            await pilot.pause()
            for button_id in ("open-run-settings", "probe-largest", "start-run"):
                self.assert_button_reachable(app, button_id)
            assert (await pilot.click("#open-run-settings", offset=(3, 1)))
            await pilot.pause()
            assert (app.query_one("#experiment-pages", ContentSwitcher).current) == ("advanced-form")
            self.assert_button_reachable(app, "open-run-settings")
            self.assert_button_reachable(app, "start-run")
            with patch.object(app, "_launch") as launch:
                assert await pilot.click("#start-run", offset=(3, 1))
                await pilot.pause()
                launch.assert_called_once()
                assert launch.call_args.args[0].kind == "run"
            assert (await pilot.click("#open-run-settings", offset=(3, 1)))
            await pilot.pause()
            assert (app.query_one("#experiment-pages", ContentSwitcher).current) == ("run-form")

    @pytest.mark.parametrize('size_case', range(6))
    async def test_action_bar_keyboard_order_survives_language_resize_and_navigation(self, size_case):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        async with app.run_test(size=(150, 45)) as pilot:
            (size, language) = tuple(product(((80, 24), (120, 30), (150, 45)), ('zh', 'en')))[size_case]
            await pilot.resize_terminal(*size)
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            navigation = app.query_one("#open-run-settings", Button)
            navigation.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert (app.query_one("#experiment-pages", ContentSwitcher).current) == ("advanced-form")
            navigation.focus()
            await pilot.pause()
            for button_id in ("probe-largest", "start-run"):
                await pilot.press("tab")
                button = app.query_one("#" + button_id, Button)
                assert (app.focused) is (button)
                self.assert_button_reachable(app, button_id)
            with patch.object(app, "_launch") as launch:
                await pilot.press("enter")
                await pilot.pause()
                launch.assert_called_once()
                assert launch.call_args.args[0].kind == "run"
            app.query_one("#start-run", Button).focus()
            await pilot.pause()
            for button_id in ("probe-largest", "open-run-settings"):
                await pilot.press("shift+tab")
                assert (app.focused) is (app.query_one("#" + button_id, Button))
            assert (cell_len(navigation.label.plain)) <= (navigation.content_region.width)
            await pilot.press("enter")
            await pilot.pause()
            assert (app.query_one("#experiment-pages", ContentSwitcher).current) == ("run-form")

    async def test_f2_changes_page_from_focused_input(self):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)) as pilot:
            app.query_one("#model", Input).focus()
            await pilot.pause()
            assert (app.focused) is (app.query_one("#model", Input))
            await pilot.press("f2")
            await pilot.pause()
            assert (app.query_one("#main-tabs", TabbedContent).active) == ("settings-tab")
            settings_tab = app.query_one("#settings-tab")
            assert (len(settings_tab.query(".config-control"))) == (0)
            assert (len(settings_tab.query("#save-run-default"))) == (0)
            assert (len(settings_tab.query(".ui-preference"))) == (5)
            assert (app.query_one("#experiment-pages", ContentSwitcher).current) == ("run-form")

    @pytest.mark.parametrize('size', ((120, 30), (80, 24), (150, 45)))
    async def test_monitor_keeps_log_and_stop_button_reachable(self, size):
        config = RunConfig(model="demo/model")
        app = AcprofTui(config, settings_path=self.settings_path)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app._activate_tab("monitor-tab")
            app._render_snapshot(ProgressSnapshot(
                stage="正式测量", detail="正在执行工作负载", current_case=1,
                total_cases=32, cpu="1", mem="2", gpu="off", measurement_active=True,
            ))
            app.query_one("#stop-run", Button).disabled = False
            log = app.query_one("#run-log", SelectableLog)
            log.write("[case] Running workload...")
            await pilot.pause()
            self.assert_button_reachable(app, "stop-run")
            self.assert_button_reachable(app, "clear-log")
            assert (log.region.height) >= (3)
            assert (log.region.bottom) <= (app.query_one("#bottom-panel").region.y)
            center = (
                log.content_region.x + log.content_region.width // 2,
                log.content_region.y + log.content_region.height // 2,
            )
            assert (app.get_widget_at(*center)[0]) is (log)
            with patch.object(app, "action_request_stop") as stop:
                assert (await pilot.click("#stop-run", offset=(3, 0)))
                await pilot.pause()
                stop.assert_called_once_with()

    async def test_wrapped_log_fits_narrow_terminal_without_horizontal_overflow(self):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._activate_tab("monitor-tab")
            await pilot.pause()
            log = app.query_one("#run-log", SelectableLog)
            log.write("[case] 正在执行确定性输入请求 processing deterministic workload " * 8)
            await pilot.pause()
            assert (log.wrap)
            assert (log.virtual_size.height) > (1)
            assert (log.virtual_size.width) <= (log.scrollable_content_region.width)
            assert (log.max_scroll_x) == (0)

    async def test_empty_model_can_be_remembered_without_allowing_collection(self):
        app = AcprofTui(settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            await self.remember_experiment(app, pilot)
            saved, warning = load_settings(self.settings_path, PROJECT_DIR)
            assert (warning) == ("")
            assert (saved.run_defaults) is not None
            assert (saved.run_defaults.model) == ("")
            assert (saved.run_defaults) == (RunConfig.smoke())
            with patch.object(app, "_launch") as launch:
                app.action_request_run()
                await pilot.pause()
                launch.assert_not_called()
            assert (app._pending_launch) is None

    async def test_custom_gpu_order_survives_loading_and_form_collection(self):
        config = replace(RunConfig.smoke("demo/gpu-order"), gpus="on,off")
        save_settings(self.settings_path, TuiSettings(run_defaults=config), PROJECT_DIR)
        app = AcprofTui(settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert (app.query_one("#gpus", Select).value) == ("on,off")
            assert (app._collect_config()) == (config)
            await self.remember_experiment(app, pilot)
            saved, warning = load_settings(self.settings_path, PROJECT_DIR)
            assert (warning) == ("")
            assert (saved.run_defaults.gpus) == ("on,off")

    async def test_ui_preferences_apply_save_and_restore_after_restart(self):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.press("f2")
            app.query_one("#ui-theme", Select).value = "acprof-light"
            app.query_one("#ui-log-lines", Select).value = 1000
            app.query_one("#ui-log-wrap", Checkbox).value = False
            await pilot.pause()
            assert (await pilot.click("#ui-command-bar", offset=(2, 1)))
            await pilot.pause()
            assert (app.theme) == ("acprof-light")
            assert not (app.query_one("#run-log", SelectableLog).wrap)
            assert (app.query_one("#run-log", SelectableLog).max_lines) == (1000)
            assert not (app.query_one("#slash-command-bar").display)
            assert (app.query_one("#bottom-panel").region.height) == (0)
            assert not (self.settings_path.exists())
            assert (await pilot.click("#save-ui-settings", offset=(3, 1)))
            await pilot.pause()
            assert (self.settings_path.is_file())
        restarted = AcprofTui(settings_path=self.settings_path)
        async with restarted.run_test(size=(150, 45)) as pilot:
            await pilot.pause()
            assert (restarted.theme) == ("acprof-light")
            assert not (restarted.query_one("#slash-command-bar").display)
            assert not (restarted.query_one("#run-log", SelectableLog).wrap)
            assert (restarted.query_one("#run-log", SelectableLog).max_lines) == (1000)
            assert (restarted.initial_config) == (RunConfig.smoke())

    async def test_saving_ui_preserves_remembered_experiment_and_saving_experiment_preserves_ui(self):
        config = RunConfig.smoke("demo/remembered")
        app = AcprofTui(config, settings_path=self.settings_path)
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.press("f2")
            app.query_one("#ui-theme", Select).value = "acprof-light"
            await pilot.pause()
            assert (await pilot.click("#save-ui-settings", offset=(3, 1)))
            await pilot.pause()
            # Remembering the experiment must preserve the saved theme,
            # even when a different theme is currently previewed in memory.
            app.query_one("#ui-theme", Select).value = "acprof-dark"
            await pilot.pause()
            await self.remember_experiment(app, pilot)
            saved, warning = load_settings(self.settings_path, PROJECT_DIR)
            assert (warning) == ("")
            assert (saved.run_defaults) == (config)
            assert (saved.ui.theme) == ("acprof-light")
            assert (app.ui_preferences.theme) == ("acprof-dark")

            # Unsaved form edits must not replace the explicitly remembered run
            # when the user only chooses to save interface preferences.
            app.query_one("#model", Input).value = "demo/unsaved"
            app.query_one("#cpus", Input).value = "1,3"
            app.query_one("#ui-log-lines", Select).value = 500
            await pilot.pause(0.1)
            await pilot.press("f2")
            await pilot.pause()
            assert (await pilot.click("#save-ui-settings", offset=(3, 1)))
            await pilot.pause()
            saved, warning = load_settings(self.settings_path, PROJECT_DIR)
            assert (warning) == ("")
            assert (saved.run_defaults) == (config)
            assert (saved.ui.theme) == ("acprof-dark")
            assert (saved.ui.log_max_lines) == (500)
        restarted = AcprofTui(settings_path=self.settings_path)
        assert (restarted.initial_config) == (config)
        assert (restarted.ui_preferences.log_max_lines) == (500)
        assert (restarted.ui_preferences.theme) == ("acprof-dark")


class TestTuiModelMemory:
    @pytest.fixture(autouse=True)
    def _setup(self):
        scratch = PROJECT_DIR / "internal-testing"
        scratch.mkdir(exist_ok=True)
        # These cases exercise project-relative paths, so keep their root inside
        # the checkout instead of using pytest's system temporary directory.
        with tempfile.TemporaryDirectory(prefix="tui-recent-", dir=scratch) as temporary:
            self.temporary = Path(temporary)
            self.settings_path = self.temporary / "tui.json"
            self.saved = TuiSettings(
                ui=UiPreferences(theme="acprof-light"),
                run_defaults=replace(RunConfig.smoke("demo/saved"), cpus="1,3"),
                last_model="demo/previous",
                last_result_dir=str(self.temporary / "previous"),
                last_result_csv=str(self.temporary / "previous" / "custom.csv"),
            )
            yield

    @pytest.mark.parametrize('kind', ('run', 'probe'))
    async def test_confirmed_run_and_probe_restore_model_without_saving_other_edits(self, kind):
        save_settings(self.settings_path, self.saved, PROJECT_DIR)
        original = self.settings_path.read_bytes()
        app = AcprofTui(settings_path=self.settings_path)
        model = f"demo/latest-{kind}"
        expected = replace(self.saved, last_model=model)
        async with app.run_test(size=(120, 30)) as pilot:
            app.query_one("#model", Input).value = f"  {model}  "
            app.query_one("#cpus", Input).value = "2"
            app.query_one("#ui-theme", Select).value = "acprof-dark"
            await pilot.pause(0.12)
            def check_persisted_before_launch(command, launched_kind):
                assert (launched_kind) == (kind)
                assert (load_settings(self.settings_path, PROJECT_DIR)) == ((expected, ""))

            with patch.object(
                app, "_execute_command", side_effect=check_persisted_before_launch,
            ) as execute:
                getattr(app, f"action_request_{kind}")()
                await pilot.pause()
                if kind == "probe":
                    assert self.settings_path.read_bytes() == original
                    assert await pilot.click("#confirm-yes")
                    await pilot.pause()
                execute.assert_called_once()
        restarted = AcprofTui(settings_path=self.settings_path)
        async with restarted.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.12)
            assert (restarted.query_one("#model", Input).value) == (model)
            assert (restarted.query_one("#cpus", Input).value) == ("1,3")
            assert (restarted.theme) == ("acprof-light")
            assert (restarted.query_one("#result-dir", Input).value) == (expected.last_result_dir)
            assert (restarted.query_one("#result-csv", Input).value) == (expected.last_result_csv)

    async def test_draft_preview_and_cancel_do_not_replace_last_model(self):
        save_settings(self.settings_path, self.saved, PROJECT_DIR)
        original = self.settings_path.read_bytes()
        app = AcprofTui(settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)) as pilot:
            app.query_one("#model", Input).value = "demo/draft"
            app.query_one("#result-dir", Input).value = "results/draft"
            app.query_one("#result-csv", Input).value = "results/draft/custom.csv"
            await pilot.pause(0.12)
            assert (self.settings_path.read_bytes()) == (original)
            with patch.object(app, "_execute_command") as execute:
                for kind in ("probe",):
                    getattr(app, f"action_request_{kind}")()
                    await pilot.pause()
                    assert (await pilot.click("#confirm-no"))
                    await pilot.pause()
                    assert (app._pending_launch) is None
                    assert (self.settings_path.read_bytes()) == (original)
                app.query_one("#model", Input).value = ""
                app.action_request_run()
                await pilot.pause(0.12)
                assert (app._pending_launch) is None
                assert (self.settings_path.read_bytes()) == (original)
                execute.assert_not_called()

    async def test_first_launch_remembers_model_but_not_unproduced_results(self):
        config = RunConfig.smoke("demo/first")
        app = AcprofTui(config, settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)) as pilot:
            with patch("acprof.tui.app.subprocess.Popen", side_effect=OSError("test failure")):
                app._launch(PendingLaunch(("unused",), "run", config))
                await app.workers.wait_for_complete()
            await pilot.pause()
            assert not (app._is_busy())
        saved, warning = load_settings(self.settings_path, PROJECT_DIR)
        assert (warning) == ("")
        assert (saved) == (TuiSettings(last_model="demo/first"))
        assert (AcprofTui(settings_path=self.settings_path).initial_config) == (RunConfig.smoke("demo/first"))

    @pytest.mark.parametrize('corrupt', (False, True))
    async def test_write_failure_or_corrupt_file_does_not_block_launch(self, corrupt):
        save_settings(self.settings_path, self.saved, PROJECT_DIR)
        if corrupt:
            self.settings_path.write_text("{broken", encoding="utf-8")
        original = self.settings_path.read_bytes()
        config = RunConfig.smoke("demo/latest")
        app = AcprofTui(config, settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)):
            with (
                patch.object(app, "_execute_command") as execute,
                patch.object(app, "notify") as notify,
                patch("acprof.tui.app.save_settings", side_effect=OSError("disk error")) as save,
            ):
                app._launch(PendingLaunch(("unused",), "run", config))
                execute.assert_called_once()
                notify.assert_called_once()
                assert (notify.call_args.kwargs["title"]) == ("自动记忆未保存")
                assert (save.call_count) == (0 if corrupt else 1)
            assert (self.settings_path.read_bytes()) == (original)

    @pytest.mark.parametrize('remember_run', (False, True))
    async def test_explicit_settings_saves_preserve_last_model(self, remember_run):
        save_settings(self.settings_path, self.saved, PROJECT_DIR)
        config = RunConfig.smoke("demo/edited")
        app = AcprofTui(config, settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.press("f2")
            await pilot.pause()
            app.query_one("#ui-theme", Select).value = "acprof-dark"
            await pilot.pause()
            assert (app.ui_preferences.theme) == ("acprof-dark")
            app._save_settings(remember_run=remember_run)
        saved, warning = load_settings(self.settings_path, PROJECT_DIR)
        assert (warning) == ("")
        assert (saved.last_model) == ("demo/previous")
        assert (saved.last_result_dir) == (self.saved.last_result_dir)
        assert (saved.last_result_csv) == (self.saved.last_result_csv)
        assert (saved.run_defaults) == (config if remember_run else self.saved.run_defaults)
        assert (saved.ui.theme) == ("acprof-light" if remember_run else "acprof-dark")

    def write_result(self):
        path = Path(str(self.temporary)) / "模型 结果" / "custom.csv"
        path.parent.mkdir(exist_ok=True)
        path.write_text("status,warmup,latency_app_s\nok,0,0.02\n", encoding="utf-8")
        return path

    async def test_restore_keeps_missing_paths_without_reading_results_or_saving(self):
        save_settings(self.settings_path, self.saved, PROJECT_DIR)
        original = self.settings_path.read_bytes()
        app = AcprofTui(RunConfig.smoke("demo/other"), settings_path=self.settings_path)
        with patch("acprof.tui.app.summarize_result_csv") as read_results:
            async with app.run_test(size=(80, 24)) as pilot:
                app._activate_tab("plot-tab")
                await pilot.pause()
                assert (app.query_one("#result-csv", Input).value) == (self.saved.last_result_csv)
                app._activate_tab("profile-tab")
                await pilot.pause()
                assert (app.query_one("#result-dir", Input).value) == (self.saved.last_result_dir)
                read_results.assert_not_called()
        assert (self.settings_path.read_bytes()) == (original)

    async def test_run_updates_same_model_output_then_remembers_actual_final_csv(self):
        csv_path = self.write_result()
        save_settings(self.settings_path, self.saved, PROJECT_DIR)
        config = replace(
            self.saved.run_defaults, model=self.saved.last_model,
            output_dir=str(Path(str(self.temporary)) / "new output"),
        )
        app = AcprofTui(config, settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)):
            with patch.object(app, "_execute_command"):
                app._launch(PendingLaunch(("unused",), "run", config))
            started, warning = load_settings(self.settings_path, PROJECT_DIR)
            assert (warning) == ("")
            assert (started.last_result_csv) == (self.saved.last_result_csv)
            assert (started.last_result_dir) == (self.saved.last_result_dir)
            # Only the audited child attempt can replace selected/history paths,
            # even if the form is subsequently edited.
            app._run_result = RunResult(True, True, result_csv=str(csv_path))
            app.query_one("#output-dir", Input).value = "results/draft"
            with patch("acprof.tui.app.save_settings", wraps=save_settings) as save:
                app._process_finished(
                    "run", 0,
                    ProgressSnapshot(stage="已完成", final_csv=str(csv_path.relative_to(PROJECT_DIR))),
                    "",
                )
                await app.workers.wait_for_complete()
                # The automatic summary reuses the remembered CSV, without a
                # duplicate write or any write during the worker's lifetime.
                save.assert_called_once()
            assert (app.query_one("#result-csv", Input).value) == (str(csv_path))
            assert (app.query_one("#result-dir", Input).value) == (str(csv_path.parent))
        expected = replace(
            self.saved, last_result_csv=str(csv_path), last_result_dir=str(csv_path.parent),
        )
        assert (load_settings(self.settings_path, PROJECT_DIR)) == ((expected, ""))
        restarted = AcprofTui(settings_path=self.settings_path)
        async with restarted.run_test(size=(120, 30)):
            assert (restarted.query_one("#result-csv", Input).value) == (str(csv_path))
            assert (restarted.query_one("#result-dir", Input).value) == (str(csv_path.parent))

    async def test_summary_remembers_only_successfully_read_csv(self):
        csv_path = self.write_result()
        save_settings(self.settings_path, self.saved, PROJECT_DIR)
        app = AcprofTui(settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)):
            app.query_one("#result-dir", Input).value = "results/unsubmitted-draft"
            app.query_one("#result-csv", Input).value = str(csv_path.relative_to(PROJECT_DIR))
            app.summarize_results_button()
            await app.workers.wait_for_complete()
            expected = replace(self.saved, last_result_csv=str(csv_path))
            assert (load_settings(self.settings_path, PROJECT_DIR)) == ((expected, ""))
            assert (app.query_one("#result-csv", Input).value) == (str(csv_path))
            assert ("成功 1") in (app.query_one("#result-summary", Static).content)
            original = self.settings_path.read_bytes()
            with patch("acprof.tui.app.save_settings") as save:
                app.summarize_results_button()
                app.query_one("#result-csv", Input).value = str(csv_path.with_name("missing.csv"))
                app.summarize_results_button()
                save.assert_not_called()
            assert (self.settings_path.read_bytes()) == (original)

    @pytest.mark.parametrize('kind_case', range(6))
    async def test_result_tools_remember_confirmed_paths_before_starting(self, kind_case):
        csv_path = self.write_result()
        (kind, frozen) = tuple(product(('plot', 'profile-dry-run', 'profile'), (False, True)))[kind_case]
        with patch("sys.frozen", frozen, create=True):
            save_settings(self.settings_path, self.saved, PROJECT_DIR)
            original = self.settings_path.read_bytes()
            expected = replace(
                self.saved,
                **({"last_result_csv": str(csv_path)} if kind == "plot"
                   else {"last_result_dir": str(csv_path.parent)}),
            )
            app = AcprofTui(settings_path=self.settings_path)
            async with app.run_test(size=(120, 30)) as pilot:
                def check_persisted_before_launch(command, launched_kind):
                    assert (launched_kind) == (kind)
                    assert (load_settings(self.settings_path, PROJECT_DIR)) == ((expected, ""))
                    subcommand = "plot" if kind == "plot" else "profile"
                    assert (command[command.index(subcommand) + 1]) == (str(csv_path if kind == "plot" else csv_path.parent))

                with patch.object(app, "_execute_command", side_effect=check_persisted_before_launch) as execute:
                    if kind == "plot":
                        app._launch_plot(str(csv_path.with_name("missing.csv")))
                        assert (self.settings_path.read_bytes()) == (original)
                        app._launch_plot(str(csv_path.relative_to(PROJECT_DIR)))
                    elif kind == "profile-dry-run":
                        app._launch_profile(dry_run=True, result_dir=str(csv_path.parent / "missing"))
                        assert (self.settings_path.read_bytes()) == (original)
                        app._launch_profile(dry_run=True, result_dir=str(csv_path.parent.relative_to(PROJECT_DIR)))
                    else:
                        app._request_profile_run(result_dir=str(csv_path.parent))
                        await pilot.pause()
                        assert (await pilot.click("#confirm-no"))
                        await pilot.pause()
                        assert (self.settings_path.read_bytes()) == (original)
                        execute.assert_not_called()
                        app._request_profile_run(result_dir=str(csv_path.parent))
                        await pilot.pause()
                        # Confirmation must remember the frozen command's
                        # directory, not a newer draft in the form.
                        app.query_one("#result-dir", Input).value = "results/draft"
                        assert (await pilot.click("#confirm-yes"))
                        await pilot.pause()
                    execute.assert_called_once()

    @pytest.mark.parametrize('corrupt', (False, True))
    async def test_result_memory_failure_does_not_prevent_summary(self, corrupt):
        csv_path = self.write_result()
        save_settings(self.settings_path, self.saved, PROJECT_DIR)
        if corrupt:
            self.settings_path.write_text("{broken", encoding="utf-8")
        original = self.settings_path.read_bytes()
        app = AcprofTui(settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)):
            with patch("acprof.tui.app.save_settings", side_effect=OSError("disk error")) as save:
                app._update_result_summary(str(csv_path))
                await app.workers.wait_for_complete()
                assert (save.call_count) == (0 if corrupt else 1)
            assert ("成功 1") in (app.query_one("#result-summary", Static).content)
            assert (app.query_one("#result-csv", Input).value) == (str(csv_path))
        assert (self.settings_path.read_bytes()) == (original)

    async def test_manual_summary_is_blocked_while_a_task_is_running(self):
        save_settings(self.settings_path, self.saved, PROJECT_DIR)
        original = self.settings_path.read_bytes()
        app = AcprofTui(settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)):
            app._process_kind = "run"
            app._latest_snapshot = ProgressSnapshot(measurement_active=True)
            app._set_busy(True)
            assert (app.query_one("#summarize-results", Button).disabled)
            with patch("acprof.tui.app.summarize_result_csv") as read_results:
                app.summarize_results_button()
                app.slash_command_submitted(Input.Submitted(
                    app.query_one("#slash-command", Input), "/results unused.csv",
                ))
                read_results.assert_not_called()
            assert (self.settings_path.read_bytes()) == (original)
            app._process_kind = ""
            app._set_busy(False)


class TestTuiMainSettings:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.temporary = tmp_path
        self.xdg_patch = patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.temporary)})
        self.xdg_patch.start()
        self._request.addfinalizer(partial(self.xdg_patch.stop))
        self.settings_path = default_settings_path(PROJECT_DIR)
        self.saved_config = replace(RunConfig.smoke("demo/saved"), cpus="1,3", repeat=7)
        save_settings(
            self.settings_path,
            TuiSettings(ui=UiPreferences(theme="acprof-light"), run_defaults=self.saved_config),
            PROJECT_DIR,
        )

    @pytest.mark.parametrize('model', ('', 'demo/latest'))
    @pytest.mark.parametrize('cli_case', range(6))
    def test_cli_explicit_arguments_override_saved_defaults_without_rewriting_them(self, model, cli_case):
        settings, warning = load_settings(self.settings_path, PROJECT_DIR)
        assert (warning) == ("")
        save_settings(self.settings_path, replace(settings, last_model=model), PROJECT_DIR)
        self.check_cli_overrides(model or self.saved_config.model, cli_case)

    def check_cli_overrides(self, model, cli_case):
        cases = (
            ([], replace(self.saved_config, model=model)),
            (["--model", "demo/explicit"], replace(self.saved_config, model="demo/explicit")),
            (["--model", ""], replace(self.saved_config, model="")),
            (["--preset", "smoke"], replace(self.saved_config.with_preset("smoke"), model=model)),
            (["--preset", "main"], replace(self.saved_config.with_preset("main"), model=model)),
            (["--model", "demo/explicit", "--preset", "smoke"], replace(self.saved_config.with_preset("smoke"), model="demo/explicit")),
        )
        original = self.settings_path.read_bytes()
        argv, expected = cases[cli_case]
        with patch("acprof.tui.app.AcprofTui.run", autospec=True) as run:
            main(argv)
        run.assert_called_once()
        app = run.call_args.args[0]
        assert (app.initial_config) == (expected)
        assert (app.ui_preferences.theme) == ("acprof-light")
        from acprof.tui.run_form import infer_preset
        assert (app._initial_preset) == (infer_preset(expected))
        assert (self.settings_path.read_bytes()) == (original)
