"""Configuration form actions and preference dialogs for the TUI.

Owns the form, persisted preferences and preview state. The App supplies
widgets, lifecycle admission and its project/interpreter paths.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Collapsible, ContentSwitcher, Label, Select, Static

from acprof.experiment import ConfigIssue, RunConfig, RunConfigError, build_run_command
from acprof.messages import join_messages, message
from acprof.tui import run_form, run_planning
from acprof.tui.commands import format_command
from acprof.tui.i18n import error_message, message_id
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog
from acprof.tui.scrollbar import SolidScrollBarRender
from acprof.tui.settings import (
    UiPreferences,
    default_settings_path,
    load_settings,
    save_settings,
)
from acprof.tui.views import LogPanel

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

class ConfigurationController:
    """Own configuration state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui, initial_config: RunConfig | None = None, *, settings_path: Path | None = None) -> None:
        self.app = app
        self.settings_path = settings_path or default_settings_path(app.project_dir)
        self.saved_settings, self.settings_warning = load_settings(self.settings_path, app.project_dir)
        self.preferences = self.saved_settings.ui
        config = self.saved_settings.run_defaults or RunConfig.smoke()
        if self.saved_settings.last_model:
            config = replace(config, model=self.saved_settings.last_model,
                             revision=config.revision if config.model == self.saved_settings.last_model else "")
        self.initial_config = initial_config if initial_config is not None else config
        self.extra_options = dict(self.initial_config.extra_options)
        self.planned_input: dict | None = None
        self.planned_input_identity: tuple[object, ...] | None = None
        self.environment_open: bool = False
        self.config_issues: tuple[ConfigIssue, ...] = ()
        self.field_errors_visible = False
        self.applying_config = False
        self.preview_timer: Timer | None = None
        self.selected_preset: str | None = None
        self.initial_preset: str = run_form.infer_preset(self.initial_config)

    def use_process_inherited_resources(self) -> None:
        if self.app.is_busy():
            return
        self.app.query_one("#runtime", Select).value = "process"
        self.app.query_one("#cpus", Input).value = "auto"
        self.app.query_one("#mems", Input).value = "auto"
        self.app.query_one("#prune-startup-oom", Checkbox).value = False
        self.app.call_after_refresh(self.app.preflight.quick_check)

    def apply_ui_preferences(self) -> None:
        self.app.localization.apply_language()
        self.app.theme = self.preferences.theme
        log = self.app.query_one("#run-log", SelectableLog)
        log.wrap = self.preferences.log_wrap
        log.max_lines = self.preferences.log_max_lines
        self.app.query_one("#bottom-panel").set_class(
            not self.preferences.show_command_bar, "command-hidden",
        )

    def ui_preference_changed(self) -> None:
        if not self.app.form_ready or self.app.is_busy():
            return
        preferences = UiPreferences(
            language=self.select("ui-language"),
            theme=self.select("ui-theme"),
            log_max_lines=int(str(self.app.query_one("#ui-log-lines", Select).value or UiPreferences().log_max_lines)),
            log_wrap=self.checked("ui-log-wrap"),
            show_command_bar=self.checked("ui-command-bar"),
        )
        if preferences == self.preferences:
            return
        self.preferences = preferences
        self.apply_ui_preferences()
        self.app.localization.set_text(self.app.query_one('#settings-status', Static), '已应用 · 点击保存设置可在下次启动时沿用')
        self.app.query_one('#settings-status').set_classes('page-summary stage-running')

    def restore_ui_defaults(self) -> None:
        if self.app.is_busy():
            return
        defaults = UiPreferences()
        with self.app.prevent(Select.Changed, Checkbox.Changed):
            self.app.query_one("#ui-language", Select).value = defaults.language
            self.app.query_one("#ui-theme", Select).value = defaults.theme
            self.app.query_one("#ui-log-lines", Select).value = defaults.log_max_lines
            self.app.query_one("#ui-log-wrap", Checkbox).value = defaults.log_wrap
            self.app.query_one("#ui-command-bar", Checkbox).value = defaults.show_command_bar
        self.preferences = defaults
        self.apply_ui_preferences()
        self.app.localization.set_text(self.app.query_one('#settings-status', Static), '界面已恢复默认 · 点击保存设置可保留')
        self.app.query_one('#settings-status').set_classes('page-summary stage-running')

    def save_settings(self, *, remember_run: bool) -> None:
        if self.app.is_busy():
            self.app.notify("任务完成后可保存设置", severity="warning")
            return
        try:
            config = (
                replace(self.collect_config(allow_empty_model=True), rebuild_image=False)
                if remember_run else self.saved_settings.run_defaults
            )
            settings = replace(
                self.saved_settings,
                ui=self.saved_settings.ui if remember_run else self.preferences,
                run_defaults=config,
            )
            save_settings(self.settings_path, settings, self.app.project_dir)
        except (OSError, ValueError, RunConfigError) as exc:
            self.app.notify(error_message(exc), title="设置未保存", severity="error")
            self.app.localization.set_text(self.app.query_one('#settings-status', Static), '保存失败 · 请检查配置或文件权限')
            self.app.query_one('#settings-status').set_classes('page-summary stage-error')
            return
        self.saved_settings = settings
        self.settings_warning = ""
        message = "已记住当前实验配置" if remember_run else "界面设置已保存"
        if not remember_run:
            self.app.localization.set_text(self.app.query_one('#settings-status', Static), message)
            self.app.query_one('#settings-status').set_classes('page-summary stage-success')
        self.app.notify(message, timeout=3)

    def save_ui_settings(self) -> None:
        self.save_settings(remember_run=False)

    def save_run_default(self) -> None:
        self.save_settings(remember_run=True)

    def show_settings(self) -> None:
        self.app.activate_tab("settings-tab")

    def open_environment_settings(self) -> None:
        if self.app.is_busy() or self.app.preflight.check_running:
            return
        from acprof.tui.environment import EnvironmentSettingsScreen
        try:
            screen = EnvironmentSettingsScreen(
                self.app.project_dir, self.settings_path.parent, sniff_iface=self.input("sniff-iface"),
            )
        except (OSError, ValueError) as read_error:
            self.app.notify(message_id('ui.0293', type(read_error).__name__), severity="error")
            return
        self.environment_open = True
        self.app.set_busy(True)
        self.app.push_screen(screen, self.environment_closed)

    def environment_closed(self, _saved: bool | None) -> None:
        self.environment_open = False
        self.app.set_busy(False)

    def open_run_settings(self) -> None:
        self.app.activate_tab("run-tab")
        pages = self.app.query_one("#experiment-pages", ContentSwitcher)
        show_advanced = pages.current != "advanced-form"
        self.show_run_form("advanced-form" if show_advanced else "run-form")

    def show_run_form(self, page: str) -> None:
        self.app.query_one("#experiment-pages", ContentSwitcher).current = page
        show_advanced = page == "advanced-form"
        self.app.localization.set_text(
            self.app.query_one("#open-run-settings", Button),
            "返回基本配置" if show_advanced else "高级参数", "label",
        )

    def gpu_options(self) -> list[tuple[str, str]]:
        options = [("仅 CPU", "off"), ("仅 GPU", "on"), ("CPU + GPU", "off,on")]
        if self.initial_config.gpus not in {value for _, value in options}:
            options.append((message_id('ui.0740', self.initial_config.gpus), self.initial_config.gpus))
        return options

    def input(self, widget_id: str) -> str:
        return self.app.query_one(f"#{widget_id}", Input).value.strip()

    def select(self, widget_id: str) -> str:
        value = self.app.query_one(f"#{widget_id}", Select).value
        return "" if value is Select.NULL else str(value)

    def checked(self, widget_id: str) -> bool:
        return bool(self.app.query_one(f"#{widget_id}", Checkbox).value)

    def configure_interaction(self) -> None:
        # Textual ignores another click while a button's active effect lasts
        # (200 ms by default). Use focus/hover styling for immediate feedback.
        for button in self.app.query(Button):
            button.active_effect_duration = 0
        for field in self.app.query(Input):
            # The app toggles the native caret without Input's repaint timer.
            field.cursor_blink = False

    def configure_scrollbars(self) -> None:
        # Instance-level renderers keep this behavior local to this TUI.
        for widget in self.app.query(Widget):
            if widget.is_scrollable:
                setattr(widget.vertical_scrollbar, "renderer", SolidScrollBarRender)
                setattr(widget.horizontal_scrollbar, "renderer", SolidScrollBarRender)

    def follow_latest_log(self) -> None:
        self.app.query_one("#run-log", SelectableLog).follow_tail()

    def toggle_log_view_button(self) -> None:
        self.toggle_log_view()

    def toggle_log_view(self) -> None:
        if isinstance(self.app.screen, ModalScreen):
            return
        panel = self.app.query_one("#log-panel", LogPanel)
        if self.app.screen.maximized is panel:
            self.app.screen.minimize()
        else:
            self.app.activate_tab("monitor-tab")
            self.app.screen.maximize(panel, container=False)
        self.app.query_one("#run-log", SelectableLog).focus()

    def cancel_preview_timer(self) -> None:
        if self.preview_timer is not None:
            self.preview_timer.stop()
            self.preview_timer = None

    def configuration_changed(self) -> None:
        if not self.app.is_running or not self.app.form_ready or self.applying_config or self.app.is_busy():
            return
        self.cancel_preview_timer()
        # Coalesce typing and preset field updates into one validation/render.
        self.preview_timer = self.app.set_timer(0.05, self.sync_form_state)

    def sync_form_state(self) -> None:
        self.preview_timer = None
        # Timer callbacks already queued before shutdown can run after the
        # form has been removed; do not query widgets during that phase.
        if not self.app.is_running or not self.app.form_ready or self.app.is_busy() or self.applying_config:
            return
        self.refresh_command_preview(notify=False, sync_preset=True)

    def run_preset_changed(self, event: Select.Changed) -> None:
        if not self.app.form_ready or self.applying_config or self.app.is_busy():
            return
        preset = str(event.value)
        if self.selected_preset == preset:
            return
        self.selected_preset = preset
        if preset == "smoke":
            self.app.experiments.preset_smoke()
        elif preset == "main":
            self.app.experiments.preset_main()

    def collect_config(self, *, allow_empty_model: bool = False) -> RunConfig:
        return run_form.collect_config(
            {key: self.input(key) for key in run_form.INPUT_FIELDS},
            {key: self.select(key) for key in run_form.SELECT_FIELDS},
            {key: self.checked(key) for key in (*run_form.CHECKED_FIELDS, *run_form.PROFILER_CHECKBOXES)},
            project_dir=self.app.project_dir, allow_empty_model=allow_empty_model, extra_options=self.extra_options,
        )

    def apply_config(self, config: RunConfig, *, preset: str = "custom") -> None:
        self.cancel_preview_timer()
        self.applying_config = True
        self.extra_options = dict(config.extra_options)
        values, selects, checks = run_form.config_values(config)
        # Value watchers post Changed messages asynchronously. Suppressing
        # them here avoids dozens of queued debounce timers after a preset.
        try:
            with self.app.prevent(Input.Changed, Select.Changed, Checkbox.Changed):
                with self.app.batch_update():
                    for widget_id, value in values.items():
                        self.app.query_one(f"#{widget_id}", Input).value = value
                    for widget_id, value in selects.items():
                        self.app.query_one(f"#{widget_id}", Select).value = value
                    for widget_id, checked_value in checks.items():
                        self.app.query_one(f"#{widget_id}", Checkbox).value = checked_value
                    self.app.query_one("#run-preset", Select).value = preset
                    self.selected_preset = preset
        finally:
            self.applying_config = False
        if self.app.form_ready:
            self.refresh_command_preview(notify=False, sync_preset=True)

    def show_config_error(self, exc: RunConfigError) -> None:
        text = join_messages("\n", (message("• {0}", issue.reason) for issue in exc.issues))
        self.app.notify(text, title="配置有误", severity="error", timeout=8)
        if not any(issue.field for issue in exc.issues):
            return
        self.field_errors_visible = True
        control = self.update_field_errors(exc.issues)
        if control is None:
            return
        ancestors = list(control.ancestors)
        page = next((item.id for item in ancestors if item.id in {"run-form", "advanced-form"}), None)
        if page is not None:
            self.app.activate_tab("run-tab")
            self.show_run_form(page)
        for ancestor in ancestors:
            if isinstance(ancestor, Collapsible):
                ancestor.collapsed = False
        self.app.call_after_refresh(self.focus_config_field, control)

    def update_field_errors(self, issues):
        from acprof.tui.field_validation import render_field_issues
        self.config_issues = tuple(issues)
        return render_field_issues(self.app, self.config_issues, self.app.localization.tr)

    def focus_config_field(self, control: Widget) -> None:
        if (control.is_mounted and not control.is_disabled and not self.app.is_busy()
                and not self.app.latest_snapshot.measurement_active):
            control.focus(scroll_visible=False)
            control.scroll_visible(animate=False, immediate=True)

    def refresh_command_preview(
        self, *, notify: bool = True, sync_preset: bool = False
    ) -> bool:
        self.app.preflight.refresh_preflight_state()
        config = None
        try:
            config = self.collect_config(allow_empty_model=True)
            self.update_plan_summary(config)
            command = build_run_command(
                config,
                project_dir=self.app.project_dir,
                python_executable=self.app.python_executable,
            )
        except RunConfigError as exc:
            if self.field_errors_visible:
                self.update_field_errors(exc.issues)
            self.app.localization.set_text(
                self.app.query_one("#config-summary", Static),
                message_id('ui.0744', join_messages("; ", (item.reason for item in exc.issues[:2]))),
            )
            self.app.localization.set_text(
                self.app.query_one("#command-preview", Static),
                message_id('ui.0745', join_messages("; ", (item.reason for item in exc.issues))),
            )
            if notify:
                self.show_config_error(exc)
            return False
        finally:
            if sync_preset and config is not None:
                selected_preset = self.select("run-preset")
                if selected_preset != "custom":
                    adjusted = not run_form.matches_preset(config, selected_preset)
                    options = tuple((message_id('ui.0187', message(label)) if key == selected_preset and adjusted else label, key)
                        for label, key in run_form.PRESET_OPTIONS)
                    widget = self.app.query_one("#run-preset", Select)
                    self.app.localization.localized_selects[widget] = options
                    with self.app.prevent(Select.Changed):
                        widget.set_options((self.app.localization.tr(label), key) for label, key in options)
                        widget.value = selected_preset
        if self.field_errors_visible:
            self.update_field_errors(())
        effective = replace(config, runtime=self.app.preflight.selected_runtime) if config.runtime == "auto" and self.app.preflight.selected_runtime else config
        case_count = run_planning.estimate_run(effective).cases
        scale_summary = (
            message_id('ui.0746', len(config.input_scales.split(',')))
            if config.input_scales
            else message("最小单一尺度" if config.input_scale_policy == "minimal" else "自动规划")
        )
        profiler_summary = (
            message_id('ui.0748')
            if config.compute_profile_tool == "none"
            and config.execution_profile_tool == "none"
            else (
                message_id('ui.0749', config.compute_profile_tool, config.execution_profile_tool)
            )
        )
        self.app.localization.set_text(self.app.query_one("#config-summary", Static), message_id(
            'ui.0750',
            case_count, scale_summary, config.request_timeout_seconds,
            profiler_summary, config.output_dir,
        ))
        self.app.localization.set_text(self.app.query_one('#command-preview', Static), format_command(command))
        if notify:
            self.app.notify("命令预览已更新", timeout=2)
        return True

    def update_plan_summary(self, config: RunConfig) -> None:
        if config.runtime == "auto" and self.app.preflight.selected_runtime:
            config = replace(config, runtime=self.app.preflight.selected_runtime)
        planned = self.planned_input if self.planned_input_identity == run_planning.input_identity(config) else None
        summary = self.app.query_one("#run-plan-summary", Static)
        self.app.localization.set_text(summary, run_planning.plan_summary(config, planned))
        self.app.localization.set_text(summary, run_planning.preparation_details(config, planned), "tooltip")
        unit = run_planning.input_unit(config, planned)
        self.app.localization.set_text(self.app.query_one("#input-scales-label", Label), message_id('ui.0627', unit))
        if planned and planned.get("scales"):
            placeholder = ",".join(f"{value:g}" for value in planned["scales"]) + f" {unit}"
        else:
            placeholder = message_id('ui.0628', message("最小单一尺度" if config.input_scale_policy == "minimal" else "自动范围"), unit)
        self.app.localization.set_text(self.app.query_one("#input-scales", Input), placeholder, "placeholder")

    def remember_last_used(
        self, *, model: str = "", result_dir: str = "", result_csv: str = "", html_report: str = "",
        log_export_dir: str = "",
    ) -> None:
        """Remember confirmed inputs, preserving explicitly saved preferences."""
        updates = {}
        report_source = self.app.query_one("#report-source", Input)
        if result_csv and report_source.value in {"", self.saved_settings.last_result_csv}:
            report_source.value = result_csv
        if model.strip():
            updates["last_model"] = model.strip()
        if result_dir:
            updates["last_result_dir"] = result_dir
        if result_csv:
            self.app.query_one("#result-csv", Input).value = result_csv
            updates["last_result_csv"] = result_csv
        settings = replace(
            self.saved_settings,
            last_model=updates.get("last_model", self.saved_settings.last_model),
            last_result_dir=updates.get("last_result_dir", self.saved_settings.last_result_dir),
            last_result_csv=updates.get("last_result_csv", self.saved_settings.last_result_csv),
            last_html_report=html_report or self.saved_settings.last_html_report,
            last_log_export_dir=log_export_dir or self.saved_settings.last_log_export_dir,
        )
        if settings == self.saved_settings:
            return
        if self.settings_warning:
            self.app.notify(
                "本地设置无法读取，已保留原文件。可在设置页主动保存后恢复自动记忆。",
                title="自动记忆未保存", severity="warning",
            )
            return
        try:
            save_settings(self.settings_path, settings, self.app.project_dir)
        except (OSError, ValueError, RunConfigError) as exc:
            self.app.notify(error_message(exc), title="自动记忆未保存", severity="warning")
            return
        self.saved_settings = settings
