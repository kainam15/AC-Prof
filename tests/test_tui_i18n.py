import re
import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path
from string import Formatter
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import (
    Button,
    ContentSwitcher,
    Input,
    Select,
    Static,
    TabbedContent,
)
from textual.widgets.text_area import Selection
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, RunConfigError, build_run_command
from acprof.messages import message
from acprof.tui.app import PROJECT_DIR
from acprof.tui.diagnostics import PreflightCheck, ResultSummary
from acprof.tui.i18n import ENGLISH, error_message, translate
from acprof.tui.log import SelectableLog
from acprof.tui.progress import ProgressSnapshot, RunProgressTracker
from acprof.tui.settings import TuiSettings, load_settings, save_settings


def test_templates_translate_nested_ui_text_and_preserve_user_values():
    value = message("已记住：{0} · CPU {1} · 内存 {2} GB", "等待/{模型}", "1,3", "4")
    assert (translate(value, "en")) == ("Saved: 等待/{模型} · CPU 1,3 · Memory 4 GB")
    assert (str(value)) == ("已记住：等待/{模型} · CPU 1,3 · 内存 4 GB")
    nested = message("{0}必须是{1}", message("每窗口请求数"), message("整数"))
    assert (translate(nested, "en")) == ("Requests per window must be an integer")
    assert (translate(message("{0}", "等待"), "en")) == ("等待")
    assert (translate("unknown {raw} text", "en")) == ("unknown {raw} text")

@pytest.mark.parametrize('config_case', range(4), ids=["(RunConfig(model=''), 'Model ID must not be empty')", "(RunConfig(model='demo/model', mems='invalid'), 'Memory list must be comma-separ", "(RunConfig(model='demo/model', repeat_in_window='oops'), 'Requests per window mu", "(RunConfig(model='demo/model', workload_spec='等待/{raw}'), 'Workload manifest doe"])
def test_validation_keeps_translatable_errors_without_changing_config_contract(config_case):
    (config, expected) = tuple(((RunConfig(model=''), 'Model ID must not be empty'), (RunConfig(model='demo/model', mems='invalid'), 'Memory list must be comma-separated integers'), (RunConfig(model='demo/model', repeat_in_window='oops'), 'Requests per window must be an integer'), (RunConfig(model='demo/model', workload_spec='等待/{raw}'), 'Workload manifest does not exist: 等待/{raw}')))[config_case]
    with pytest.raises(RunConfigError) as caught:
        config.validate(project_dir=PROJECT_DIR)
    assert (expected) in (translate(error_message(caught.value), "en"))
    assert re.search(r"[\u4e00-\u9fff]", str(caught.value))

def test_probe_progress_is_language_independent_and_keeps_duration_semantics():
    tracker = RunProgressTracker()
    snapshot = tracker.feed("[largest-probe] MEMORY_RESULT mem=4 status=startup_oom")
    assert (snapshot.stage) == ("内存可行性探测")
    assert (translate(snapshot.detail, "en")) == ("4GB · Startup OOM; trying next candidate")
    snapshot = tracker.feed(
        "[largest-probe] RESULT status=ok input_scale=512 cpu=1 mem=8 gpu=off "
        "cold_start_s=nan request_s=1.25 ready_plus_request_s=3.5"
    )
    assert (snapshot.stage) == ("探测完成")
    assert ("Single request 1.250s · Cold start Unknown · Ready + request 3.500s") in (translate(snapshot.detail, "en"))
    assert not (snapshot.measurement_active)

@pytest.mark.parametrize("source,translated", [
    pytest.param(source, translated, id=source) for source, translated in sorted(ENGLISH.items())
])
def test_catalog_preserves_interpolation_fields_and_numeric_formats(source, translated):
    def fields(text):
        return sorted((name, spec, conversion or "") for _, name, spec, conversion in Formatter().parse(text) if name is not None)
    assert (fields(source)) == (fields(translated))
    assert (translated.strip())


class TestTuiLanguage:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        scratch = PROJECT_DIR / "internal-testing"
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="tui-language-", dir=scratch)
        self._request.addfinalizer(partial(temporary.cleanup))
        self.settings_path = Path(temporary.name) / "tui.json"

    def make_app(self, config=None):
        return AcprofTui(config, settings_path=self.settings_path)

    async def switch(self, app, pilot, language):
        app.action_show_settings()
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        assert (app.ui_preferences.language) == (language)

    async def test_language_selector_accepts_keyboard_choices_in_both_directions(self):
        app = self.make_app(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("f2")
            await pilot.pause()
            # Select routes clicks through its child label; Pilot.click's
            # return value only matches the exact target widget.
            await pilot.click("#ui-language", offset=(4, 1))
            assert (app.query_one("#ui-language", Select).expanded)
            await pilot.press("end", "enter")
            await pilot.pause()
            assert (app.ui_preferences.language) == ("en")
            assert (app.query_one("#ui-language SelectCurrent #label", Static).content) == ("English")
            await pilot.click("#ui-language", offset=(4, 1))
            assert (app.query_one("#ui-language", Select).expanded)
            await pilot.press("home", "enter")
            await pilot.pause()
            assert (app.ui_preferences.language) == ("zh")
            assert (app.query_one("#ui-language SelectCurrent #label", Static).content) == ("简体中文")
            assert (app.query_one("#run-preset", Select).value) == ("smoke")
            assert not (self.settings_path.exists())

    async def test_switch_keeps_drafts_presets_log_selection_status_and_loaded_results(self):
        config = replace(RunConfig.smoke("demo/model"), gpus="on,off")
        app = self.make_app(config)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app.open_run_settings()
            model = app.query_one("#model", Input)
            model.value = "等待/{模型}"
            model.cursor_position = 2
            await pilot.pause()
            # This test preserves an already rendered preview; debounce timing
            # is covered by the form interaction tests.
            app._cancel_preview_timer()
            app._refresh_command_preview(notify=False, sync_preset=True)
            command = build_run_command(app._collect_config(), project_dir=PROJECT_DIR)
            preview = app.query_one("#command-preview", Static).content
            preset = app.query_one("#run-preset", Select).value
            app._activate_tab("monitor-tab")
            log = app.query_one("#run-log", SelectableLog)
            log.write("\n".join(f"等待 原始日志 {index}" for index in range(100)))
            await pilot.pause()
            log.selection = Selection((30, 0), (31, 4))
            log.scroll_to(y=25, animate=False, immediate=True)
            await pilot.pause()
            old_text, old_selection, old_scroll = log.text, log.selection, log.scroll_y
            assert not (log.following)
            snapshot = ProgressSnapshot(
                stage="case 完成", current_case=1, completed_cases=1, total_cases=2,
                cpu="1", mem="4", gpu="on", detail=message("正在准备 case {0}/{1}", 1, 2),
            )
            app._latest_snapshot = snapshot
            app._render_snapshot(snapshot)
            summary = ResultSummary(3, 2, 1, 1, 2, groups=(dict(cpu_cores=1, mem_cap_gb=4,
                gpu_mode="off", input_scale=64, environment_class="unknown", metric="latency_app_s",
                mean=0.02, unit="s", n_windows=1, reason="insufficient_windows"),))
            with patch("acprof.tui.app.summarize_result_csv", return_value=summary) as read_results:
                app._update_result_summary("read-once.csv", notify=False)
                await app.workers.wait_for_complete()
                await pilot.pause()
                remembered = self.settings_path.read_bytes()
                assert (load_settings(self.settings_path, PROJECT_DIR)) == ((TuiSettings(last_result_csv=str(PROJECT_DIR / "read-once.csv")), ""))
                await self.switch(app, pilot, "en")
                assert (read_results.call_count) == (1)
                assert ("App latency: 20 ms") in (app.query_one("#result-summary", Static).content)
            assert (model.value) == ("等待/{模型}")
            assert (model.cursor_position) == (2)
            assert (app.query_one("#run-preset", Select).value) == (preset)
            assert (app.query_one("#gpus", Select).value) == ("on,off")
            assert (app.query_one("#ui-theme SelectCurrent #label", Static).content) == ("Graphite · Dark (Default)")
            assert (build_run_command(app._collect_config(), project_dir=PROJECT_DIR)) == (command)
            assert (app.query_one("#command-preview", Static).content) == (preview)
            assert (app.query_one("#experiment-pages", ContentSwitcher).current) == ("advanced-form")
            assert (str(app.query_one("#open-run-settings", Button).label)) == ("Basic settings")
            assert (app.query_one("#status-stage", Static).content) == ("Case completed")
            assert (app.query_one("#status-stage").has_class("stage-success"))
            assert (app._latest_snapshot) is (snapshot)
            assert (app.query_one("#status-case", Static).content) == ("Current 1 · Done 1/2")
            assert (app.query_one("#status-resource", Static).content) == ("CPU=1  MEM=4GB  GPU=on")
            assert ((log.text, log.selection, log.scroll_y)) == ((old_text, old_selection, old_scroll))
            assert not (log.following)
            assert (self.settings_path.read_bytes()) == (remembered)
            assert (app.query_one("#result-csv", Input).value) == (str(PROJECT_DIR / "read-once.csv"))
            await self.switch(app, pilot, "zh")
            assert (str(app.query_one("#open-run-settings", Button).label)) == ("返回基本配置")
            assert (app.query_one("#status-case", Static).content) == ("当前 1 · 已完成 1/2")
            assert ((log.text, log.selection, log.scroll_y)) == ((old_text, old_selection, old_scroll))
            assert (self.settings_path.read_bytes()) == (remembered)

    async def test_explicit_save_restart_reset_and_model_memory_keep_their_boundaries(self):
        defaults = RunConfig.smoke("demo/saved")
        save_settings(self.settings_path, TuiSettings(run_defaults=defaults), PROJECT_DIR)
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            await self.switch(app, pilot, "en")
            app._remember_last_used(model="demo/latest")
            saved, warning = load_settings(self.settings_path, PROJECT_DIR)
            assert not (warning)
            assert (saved.ui.language) == ("zh")
            assert ("Model for next launch: demo/latest") in (app.query_one("#saved-run-summary", Static).content)
            assert (await pilot.click("#save-ui-settings"))
            await pilot.pause()
        restarted = self.make_app()
        async with restarted.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert (restarted.ui_preferences.language) == ("en")
            assert (str(restarted.query_one("#start-run", Button).label)) == ("Start run")
            assert (restarted.query_one("#model", Input).value) == ("demo/latest")
            assert (restarted._saved_settings.run_defaults) == (defaults)
            await pilot.press("f2")
            assert (await pilot.click("#restore-ui-defaults"))
            await pilot.pause()
            assert (restarted.ui_preferences.language) == ("zh")
            assert (restarted.query_one("#ui-language", Select).value) == ("zh")
            assert (load_settings(self.settings_path, PROJECT_DIR)[0].ui.language) == ("en")

    async def test_english_validation_dialog_notifications_and_raw_output(self):
        app = self.make_app(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            await self.switch(app, pilot, "en")
            app.query_one("#mems", Input).value = "oops"
            await pilot.pause()
            app._refresh_command_preview(notify=False)
            assert ("Memory list must be comma-separated integers") in (app.query_one("#config-summary", Static).content)
            app.action_request_run()
            await pilot.pause()
            assert (any(n.title == "Invalid configuration" and "Memory list" in n.message for n in app._notifications))
            app.query_one("#mems", Input).value = "4"
            await pilot.pause()
            app.action_request_probe()
            await pilot.pause()
            assert (str(app.screen.query_one("#confirm-no", Button).label)) == ("Cancel")
            assert (str(app.screen.query_one("#confirm-yes", Button).label)) == ("Start probe")
            assert ("Memory candidates: 4GB") in (app.screen.query_one("#confirm-message", Static).content)
            await pilot.press("escape")
            app._check_request = object()
            app._show_quick_check([PreflightCheck(message("原生 Linux"), "ok", message("统一层级可用"))], "", app._check_request)
            log = app.query_one("#run-log", SelectableLog)
            assert ("Native Linux") not in (log.text)
            assert not (app.query_one("#environment-status").display)
            app._consume_process_line("等待 原始子进程输出 {raw}", None, False)
            assert ("等待 原始子进程输出 {raw}") in (log.text)
            app._process_started(123, "probe")
            assert ("probe process started, PID=123") in (log.text)

    @pytest.mark.parametrize('language', ('en', 'zh'))
    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_both_languages_fit_common_terminal_sizes_and_busy_tasks_lock_language(self, language, size):
        app = self.make_app(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert (app.has_class("narrow")) == (size[0] < 110)
            assert (app.has_class("short")) == (size[1] < 35)
            await self.switch(app, pilot, language)
            for tab, ids in (
                ("settings-tab", ("ui-language", "restore-ui-defaults", "save-ui-settings")),
                ("run-tab", ("open-run-settings", "probe-largest", "start-run")),
                ("monitor-tab", ("copy-log", "follow-log", "expand-log", "clear-log", "stop-run")),
                ("plot-tab", ("result-csv", "summarize-results", "plot-results")),
                ("profile-tab", ("result-dir", "profile-dry-run", "profile-run")),
            ):
                tabs = app.query_one("#main-tabs", TabbedContent)
                tab_button = tabs.get_tab(tab)
                assert (tab_button.region.width) > (0), tab
                assert (tab_button.region.x) >= (0), tab
                assert (tab_button.region.right) <= (size[0]), tab
                assert (cell_len(tab_button.label.plain)) <= (tab_button.content_region.width), tab
                assert (await pilot.click(tab_button))
                await pilot.pause()
                assert (tabs.active) == (tab)
                for widget_id in ids:
                    widget = app.query_one(f"#{widget_id}")
                    region = widget.region
                    assert (region.width) > (0), widget_id
                    assert (region.x) >= (0), widget_id
                    assert (region.right) <= (size[0]), widget_id
                    assert (region.bottom) <= (app.query_one("#bottom-panel").region.y), widget_id
                    if isinstance(widget, Button):
                        assert (cell_len(widget.label.plain)) <= (widget.content_region.width), widget_id
                        assert (region.right) <= (widget.parent.content_region.right), widget_id
                        assert (region.x) >= (widget.parent.content_region.x), widget_id
            if language == "en":
                for (widget, attribute), source in app._localized_text.items():
                    rendered = widget.content if attribute == "content" else getattr(widget, attribute)
                    assert (re.search(r"[\u4e00-\u9fff]", str(rendered))) is None, (widget, attribute)
                for widget in app._localized_selects:
                    displayed = widget.query_one("SelectCurrent #label", Static).content
                    assert (re.search(r"[\u4e00-\u9fff]", str(displayed))) is None, widget
            app._process_kind = "run"
            app._set_busy(True)
            assert (app.query_one("#ui-language", Select).disabled)
            app.query_one("#ui-language", Select).value = "zh" if language == "en" else "en"
            await pilot.pause()
            assert app.ui_preferences.language == language
            app._process_kind = ""
            app._set_busy(False)
            assert not (app.query_one("#ui-language", Select).disabled)
