"""Configuration form actions and preference dialogs for the TUI.

No subprocess/measurement owner lives in this module. The App supplies shared
widgets, persisted state, and its project/venv path settings.
"""
from __future__ import annotations

from dataclasses import replace

from textual import on
from textual.message_pump import MessagePump
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Collapsible, ContentSwitcher, Label, Select, Static

from acprof.experiment import RunConfig, RunConfigError, build_run_command
from acprof.messages import join_messages, message
from acprof.tui import run_form, run_planning
from acprof.tui.commands import format_command
from acprof.tui.i18n import error_message
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog
from acprof.tui.scrollbar import SolidScrollBarRender
from acprof.tui.settings import UiPreferences
from acprof.tui.views import LogPanel


class ConfigurationActions(MessagePump):
    def _apply_ui_preferences(self) -> None:
        self._apply_language()
        self.theme = self.ui_preferences.theme
        log = self.query_one("#run-log", SelectableLog)
        log.wrap = self.ui_preferences.log_wrap
        log.max_lines = self.ui_preferences.log_max_lines
        self.query_one("#bottom-panel").set_class(
            not self.ui_preferences.show_command_bar, "command-hidden",
        )

    @on(Select.Changed, ".ui-preference")
    @on(Checkbox.Changed, ".ui-preference")
    def _ui_preference_changed(self) -> None:
        if not self._form_ready or self._is_busy():
            return
        preferences = UiPreferences(
            language=self._select("ui-language"),
            theme=self._select("ui-theme"),
            log_max_lines=int(self.query_one("#ui-log-lines", Select).value),
            log_wrap=self._checked("ui-log-wrap"),
            show_command_bar=self._checked("ui-command-bar"),
        )
        if preferences == self.ui_preferences:
            return
        self.ui_preferences = preferences
        self._apply_ui_preferences()
        self._set_text(self.query_one('#settings-status', Static), '已应用 · 点击保存设置可在下次启动时沿用')
        self.query_one('#settings-status').set_classes('page-summary stage-running')

    @on(Button.Pressed, "#restore-ui-defaults")
    def restore_ui_defaults(self) -> None:
        if self._is_busy():
            return
        defaults = UiPreferences()
        with self.prevent(Select.Changed, Checkbox.Changed):
            self.query_one("#ui-language", Select).value = defaults.language
            self.query_one("#ui-theme", Select).value = defaults.theme
            self.query_one("#ui-log-lines", Select).value = defaults.log_max_lines
            self.query_one("#ui-log-wrap", Checkbox).value = defaults.log_wrap
            self.query_one("#ui-command-bar", Checkbox).value = defaults.show_command_bar
        self.ui_preferences = defaults
        self._apply_ui_preferences()
        self._set_text(self.query_one('#settings-status', Static), '界面已恢复默认 · 点击保存设置可保留')
        self.query_one('#settings-status').set_classes('page-summary stage-running')

    def _save_settings(self, *, remember_run: bool) -> None:
        if self._is_busy():
            self.notify("任务完成后可保存设置", severity="warning")
            return
        try:
            config = (
                self._collect_config(allow_empty_model=True)
                if remember_run else self._saved_settings.run_defaults
            )
            settings = replace(
                self._saved_settings,
                ui=self._saved_settings.ui if remember_run else self.ui_preferences,
                run_defaults=config,
            )
            self._save_local_settings(self.settings_path, settings, self._project_dir)
        except (OSError, ValueError, RunConfigError) as exc:
            self.notify(error_message(exc), title="设置未保存", severity="error")
            self._set_text(self.query_one('#settings-status', Static), '保存失败 · 请检查配置或文件权限')
            self.query_one('#settings-status').set_classes('page-summary stage-error')
            return
        self._saved_settings = settings
        self._settings_warning = ""
        message = "已记住当前实验配置" if remember_run else "界面设置已保存"
        if not remember_run:
            self._set_text(self.query_one('#settings-status', Static), message)
            self.query_one('#settings-status').set_classes('page-summary stage-success')
        self.notify(message, timeout=3)

    @on(Button.Pressed, "#save-ui-settings")
    def save_ui_settings(self) -> None:
        self._save_settings(remember_run=False)

    @on(Button.Pressed, "#save-run-default")
    def save_run_default(self) -> None:
        self._save_settings(remember_run=True)

    def action_show_settings(self) -> None:
        self._activate_tab("settings-tab")

    @on(Button.Pressed, "#open-environment-settings")
    def open_environment_settings(self) -> None:
        if self._is_busy() or self._check_running:
            return
        from acprof.tui.environment import EnvironmentSettingsScreen
        try:
            screen = EnvironmentSettingsScreen(
                self._project_dir, self.settings_path.parent, sniff_iface=self._input("sniff-iface"),
            )
        except (OSError, ValueError) as read_error:
            self.notify(message('无法读取连接配置：{0}', type(read_error).__name__), severity="error")
            return
        self._environment_open = True
        self._set_busy(True)
        self.push_screen(screen, self._environment_closed)

    def _environment_closed(self, _saved: bool | None) -> None:
        self._environment_open = False
        self._set_busy(False)

    @on(Button.Pressed, "#open-run-settings")
    def open_run_settings(self) -> None:
        self._activate_tab("run-tab")
        pages = self.query_one("#experiment-pages", ContentSwitcher)
        show_advanced = pages.current != "advanced-form"
        self._show_run_form("advanced-form" if show_advanced else "run-form")

    def _show_run_form(self, page: str) -> None:
        self.query_one("#experiment-pages", ContentSwitcher).current = page
        show_advanced = page == "advanced-form"
        self._set_text(
            self.query_one("#open-run-settings", Button),
            "返回基本配置" if show_advanced else "高级参数", "label",
        )


    def _gpu_options(self) -> list[tuple[str, str]]:
        options = [("仅 CPU", "off"), ("仅 GPU", "on"), ("CPU + GPU", "off,on")]
        if self.initial_config.gpus not in {value for _, value in options}:
            options.append((message('自定义：{0}', self.initial_config.gpus), self.initial_config.gpus))
        return options


    def _input(self, widget_id: str) -> str:
        return self.query_one(f"#{widget_id}", Input).value.strip()

    def _select(self, widget_id: str) -> str:
        value = self.query_one(f"#{widget_id}", Select).value
        return "" if value is Select.NULL else str(value)

    def _checked(self, widget_id: str) -> bool:
        return bool(self.query_one(f"#{widget_id}", Checkbox).value)

    def _configure_interaction(self) -> None:
        # Textual ignores another click while a button's active effect lasts
        # (200 ms by default). Use focus/hover styling for immediate feedback.
        for button in self.query(Button):
            button.active_effect_duration = 0
        for field in self.query(Input):
            # The app toggles the native caret without Input's repaint timer.
            field.cursor_blink = False

    def _configure_scrollbars(self) -> None:
        # Instance-level renderers keep this behavior local to this TUI.
        for widget in self.query(Widget):
            if widget.is_scrollable:
                widget.vertical_scrollbar.renderer = SolidScrollBarRender
                widget.horizontal_scrollbar.renderer = SolidScrollBarRender

    @on(Button.Pressed, "#copy-log")
    def copy_log_selection(self) -> None:
        log = self.query_one("#run-log", SelectableLog)
        if log.copy_selection():
            self.notify("已复制日志选区", timeout=2)
        else:
            self.notify("先在日志中拖动选择文字，Ctrl+A 可全选", timeout=3)

    @on(Button.Pressed, "#follow-log")
    def follow_latest_log(self) -> None:
        self.query_one("#run-log", SelectableLog).follow_tail()

    @on(Button.Pressed, "#expand-log")
    @on(Button.Pressed, "#restore-log")
    def toggle_log_view_button(self) -> None:
        self.action_toggle_log_view()

    def action_toggle_log_view(self) -> None:
        if isinstance(self.screen, ModalScreen):
            return
        panel = self.query_one("#log-panel", LogPanel)
        if self.screen.maximized is panel:
            self.screen.minimize()
        else:
            self._activate_tab("monitor-tab")
            self.screen.maximize(panel, container=False)
        self.query_one("#run-log", SelectableLog).focus()

    def _cancel_preview_timer(self) -> None:
        if self._preview_timer is not None:
            self._preview_timer.stop()
            self._preview_timer = None

    @on(Input.Changed, ".config-control")
    @on(Select.Changed, ".config-control")
    @on(Checkbox.Changed, ".config-control")
    def _configuration_changed(self) -> None:
        if not self.is_running or not self._form_ready or self._applying_config or self._is_busy():
            return
        self._cancel_preview_timer()
        # Coalesce typing and preset field updates into one validation/render.
        self._preview_timer = self.set_timer(0.05, self._sync_form_state)

    def _sync_form_state(self) -> None:
        self._preview_timer = None
        # Timer callbacks already queued before shutdown can run after the
        # form has been removed; do not query widgets during that phase.
        if not self.is_running or not self._form_ready or self._is_busy() or self._applying_config:
            return
        self._refresh_command_preview(notify=False, sync_preset=True)

    @on(Select.Changed, "#run-preset")
    def _run_preset_changed(self, event: Select.Changed) -> None:
        if not self._form_ready or self._applying_config or self._is_busy():
            return
        preset = str(event.value)
        if self._selected_preset == preset:
            return
        self._selected_preset = preset
        if preset == "smoke":
            self.preset_smoke()
        elif preset == "main":
            self.preset_main()

    def _collect_config(self, *, allow_empty_model: bool = False) -> RunConfig:
        return run_form.collect_config(
            {key: self._input(key) for key in run_form.INPUT_FIELDS},
            {key: self._select(key) for key in run_form.SELECT_FIELDS},
            {key: self._checked(key) for key in (*run_form.CHECKED_FIELDS, *run_form.PROFILER_CHECKBOXES)},
            project_dir=self._project_dir, allow_empty_model=allow_empty_model, extra_options=self._extra_run_options,
        )

    def _apply_config(self, config: RunConfig, *, preset: str = "custom") -> None:
        self._cancel_preview_timer()
        self._applying_config = True
        self._extra_run_options = dict(config.extra_options)
        values, selects, checks = run_form.config_values(config)
        # Value watchers post Changed messages asynchronously. Suppressing
        # them here avoids dozens of queued debounce timers after a preset.
        try:
            with self.prevent(Input.Changed, Select.Changed, Checkbox.Changed):
                with self.batch_update():
                    for widget_id, value in values.items():
                        self.query_one(f"#{widget_id}", Input).value = value
                    for widget_id, value in selects.items():
                        self.query_one(f"#{widget_id}", Select).value = value
                    for widget_id, value in checks.items():
                        self.query_one(f"#{widget_id}", Checkbox).value = value
                    self.query_one("#run-preset", Select).value = preset
                    self._selected_preset = preset
        finally:
            self._applying_config = False
        if self._form_ready:
            self._refresh_command_preview(notify=False, sync_preset=True)

    def _show_config_error(self, exc: RunConfigError) -> None:
        text = join_messages("\n", (message("• {0}", issue.reason) for issue in exc.issues))
        self.notify(text, title="配置有误", severity="error", timeout=8)
        if not any(issue.field for issue in exc.issues):
            return
        self._field_errors_visible = True
        control = self._update_field_errors(exc.issues)
        if control is None:
            return
        ancestors = list(control.ancestors)
        page = next((item.id for item in ancestors if item.id in {"run-form", "advanced-form"}), None)
        if page is not None:
            self._activate_tab("run-tab")
            self._show_run_form(page)
        for ancestor in ancestors:
            if isinstance(ancestor, Collapsible):
                ancestor.collapsed = False
        self.call_after_refresh(self._focus_config_field, control)

    def _update_field_errors(self, issues):
        from acprof.tui.field_validation import render_field_issues
        self._config_issues = tuple(issues)
        return render_field_issues(self, self._config_issues, self.tr)

    def _focus_config_field(self, control: Widget) -> None:
        if (control.is_mounted and not control.is_disabled and not self._is_busy()
                and not self._latest_snapshot.measurement_active):
            control.focus(scroll_visible=False)
            control.scroll_visible(animate=False, immediate=True)

    def _refresh_command_preview(
        self, *, notify: bool = True, sync_preset: bool = False
    ) -> bool:
        self._refresh_preflight_state()
        config = None
        try:
            config = self._collect_config(allow_empty_model=True)
            self._update_plan_summary(config)
            command = build_run_command(
                config,
                project_dir=self._project_dir,
                python_executable=self._python_executable,
            )
        except RunConfigError as exc:
            if self._field_errors_visible:
                self._update_field_errors(exc.issues)
            self._set_text(
                self.query_one("#config-summary", Static),
                message("配置待完善 · {0}", join_messages("; ", (item.reason for item in exc.issues[:2]))),
            )
            self._set_text(
                self.query_one("#command-preview", Static),
                message("配置尚未完成：{0}", join_messages("; ", (item.reason for item in exc.issues))),
            )
            if notify:
                self._show_config_error(exc)
            return False
        finally:
            if sync_preset and config is not None:
                selected_preset = self._select("run-preset")
                if selected_preset != "custom":
                    adjusted = not run_form.matches_preset(config, selected_preset)
                    options = tuple((message("{0} · 已调整", message(label)) if key == selected_preset and adjusted else label, key)
                        for label, key in run_form.PRESET_OPTIONS)
                    widget = self.query_one("#run-preset", Select)
                    self._localized_selects[widget] = options
                    with self.prevent(Select.Changed):
                        widget.set_options((self.tr(label), key) for label, key in options)
                        widget.value = selected_preset
        if self._field_errors_visible:
            self._update_field_errors(())
        case_count = (
            len(config.cpus.split(","))
            * len(config.mems.split(","))
            * len(config.gpus.split(","))
        )
        scale_summary = (
            message('{0} 档', len(config.input_scales.split(',')))
            if config.input_scales
            else message("最小单一尺度" if config.input_scale_policy == "minimal" else "自动规划")
        )
        profiler_summary = (
            message("分析器关闭")
            if config.compute_profile_tool == "none"
            and config.execution_profile_tool == "none"
            else (
                message('计算={0} · 执行={1}', config.compute_profile_tool, config.execution_profile_tool)
            )
        )
        self._set_text(self.query_one("#config-summary", Static), message(
            "{0} 个资源 case · 输入规模 {1} · 单请求超时 {2:g}s · {3} · 输出 {4}",
            case_count, scale_summary, config.request_timeout_seconds,
            profiler_summary, config.output_dir,
        ))
        self._set_text(self.query_one('#command-preview', Static), format_command(command))
        if notify:
            self.notify("命令预览已更新", timeout=2)
        return True

    def _update_plan_summary(self, config: RunConfig) -> None:
        planned = self._planned_input if self._planned_input_identity == run_planning.input_identity(config) else None
        summary = self.query_one("#run-plan-summary", Static)
        self._set_text(summary, run_planning.plan_summary(config, planned))
        self._set_text(summary, run_planning.preparation_details(config, planned), "tooltip")
        unit = run_planning.input_unit(config, planned)
        self._set_text(self.query_one("#input-scales-label", Label), message("输入规模 ({0})", unit))
        if planned and planned.get("scales"):
            placeholder = ",".join(f"{value:g}" for value in planned["scales"]) + f" {unit}"
        else:
            placeholder = message("留空：{0} · {1}", message("最小单一尺度" if config.input_scale_policy == "minimal" else "自动范围"), unit)
        self._set_text(self.query_one("#input-scales", Input), placeholder, "placeholder")
