"""Configuration form actions and preference dialogs for the TUI.

Owns the form, persisted preferences and preview state. The App supplies
widgets, lifecycle admission and its project/interpreter paths.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Collapsible, ContentSwitcher, Label, Select, Static

from acprof.experiments.configuration import (
    ConfigIssue,
    RunConfig,
    RunConfigError,
    build_run_command,
)
from acprof.messages import join_messages, message
from acprof.tui.commands import format_command
from acprof.tui.i18n import error_message, message_id
from acprof.tui.logging.panel import LogPanel
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run import (
    form as run_form,
    planning as run_planning,
)
from acprof.tui.run.plan_panel import RunPlanPanel
from acprof.tui.settings.preferences import (
    UiPreferences,
    default_settings_path,
    load_settings,
    save_settings,
)
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.scrollbar import configure_solid_scrollbars
from acprof.tui.ui.structured import StatusBanner
from acprof.tui.ui.view_models import StatusSummary

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
        self.initial_config = run_form.runtime_form_config(initial_config if initial_config is not None else config)
        self.recovery_policy = self.initial_config.profiling_mode if self.initial_config.resume else "best_effort"
        self.extra_options = dict(self.initial_config.extra_options)
        self.planned_input: dict | None = None
        self.planned_input_identity: tuple[object, ...] | None = None
        self.environment_open: bool = False
        self.config_issues: tuple[ConfigIssue, ...] = ()
        self.config_warnings: tuple[ConfigIssue, ...] = ()
        self.validation_pending = True
        self.field_errors_visible = False
        self.applying_config = False
        self.preview_timer: Timer | None = None

    def use_process_inherited_resources(self) -> None:
        if self.app.is_busy() or self.app.preflight.check_running or self.app.execution.session.snapshot.measurement_active:
            return
        from acprof.tui.ui.dialogs import ScrollConfirmActionScreen
        current = (self.select('runtime'), self.input('cpus'), self.input('mems'), self.checked('prune-startup-oom'))
        self.app.push_screen(ScrollConfirmActionScreen(
            message_id('run.resources.inherit'),
            message_id('run.resources.inherit_confirm', current[1], current[2]),
            message_id('run.resources.inherit_apply'),
        ), lambda confirmed: self._apply_inherited_resources(confirmed, current))

    def _apply_inherited_resources(self, confirmed: bool | None, expected: tuple[str, str, str, bool]) -> None:
        if (not confirmed or self.app.is_busy() or self.app.preflight.check_running
                or self.app.execution.session.snapshot.measurement_active
                or expected != (self.select('runtime'), self.input('cpus'), self.input('mems'), self.checked('prune-startup-oom'))):
            return
        self.app.query_one("#runtime", Select).value = "process"
        self.app.query_one("#cpus", Input).value = "auto"
        self.app.query_one("#mems", Input).value = "auto"
        self.app.query_one("#prune-startup-oom", Checkbox).value = False
        self.configuration_changed()
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

    def _autosave_ui_preferences(self, *, restored_defaults: bool = False) -> None:
        if self.settings_warning:
            self.app.query_one("#settings-banner", StatusBanner).update_summary(
                StatusSummary("warning", message_id("settings.preferences.autosave_skipped_warning")),
            )
            return
        try:
            settings = replace(self.saved_settings, ui=self.preferences)
            save_settings(self.settings_path, settings, self.app.project_dir)
        except (OSError, ValueError, RunConfigError):
            self.app.query_one("#settings-banner", StatusBanner).update_summary(
                StatusSummary("error", "保存失败 · 请检查配置或文件权限"),
            )
            return
        self.saved_settings = settings
        key = "settings.preferences.defaults_autosaved" if restored_defaults else "settings.preferences.autosaved"
        self.app.query_one("#settings-banner", StatusBanner).update_summary(
            StatusSummary("success", message_id(key)),
        )

    def refresh_connection_badges(self) -> None:
        if not self.app.is_running or self.app.ui_closing:
            return
        from acprof.host.env_utils import ProjectEnvSnapshot

        try:
            values = ProjectEnvSnapshot.read(self.app.project_dir).configurable_values()
        except (OSError, ValueError):
            values = {}
        hf_badge = message_id(
            "settings.connections.badge_configured" if values.get("HF_TOKEN", "").strip()
            else "settings.connections.badge_unset",
        )
        wecom_badge = message_id(
            "settings.connections.badge_configured" if values.get("ACPROF_WECOM_WEBHOOK_URL", "").strip()
            else "settings.connections.badge_unset",
        )
        self.app.localization.set_text(
            self.app.query_one("#settings-connection-status", Static),
            message_id("settings.connections.status_summary", hf_badge, wecom_badge),
        )

    def ui_preference_changed(self) -> None:
        if not self.app.is_running or self.app.ui_closing or not self.app.form_ready or self.app.is_busy():
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
        self._autosave_ui_preferences(restored_defaults=False)

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
        self._autosave_ui_preferences(restored_defaults=True)

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
            self.app.query_one("#settings-banner", StatusBanner).update_summary(
                StatusSummary("error", "保存失败 · 请检查配置或文件权限"),
            )
            return
        self.saved_settings = settings
        self.settings_warning = ""
        message = "已记住当前实验配置" if remember_run else "界面设置已保存"
        if not remember_run:
            self.app.query_one("#settings-banner", StatusBanner).update_summary(
                StatusSummary("success", message),
            )
        self.app.notify(message, timeout=3)

    def save_ui_settings(self) -> None:
        self.save_settings(remember_run=False)

    def save_run_default(self) -> None:
        self.save_settings(remember_run=True)

    def show_settings(self) -> None:
        self.app.activate_tab("settings-tab")

    def open_environment_settings(self, *, permissions: bool = False, permission_target: str = '',
                                  on_closed: Callable[[], None] | None = None) -> None:
        if self.app.is_busy() or self.app.preflight.check_running or self.app.execution.session.snapshot.measurement_active:
            return
        from acprof.tui.settings.environment import EnvironmentSettingsScreen
        try:
            screen = EnvironmentSettingsScreen(
                self.app.project_dir, self.settings_path.parent, sniff_iface=self.input("sniff-iface"),
                initial_tab='permissions-tab' if permissions else 'connections-tab',
                permission_target=permission_target, preflight=self.app.preflight,
            )
        except (OSError, ValueError) as read_error:
            self.app.notify(message_id('settings.environment.read_failed', type(read_error).__name__), severity="error")
            return
        self.environment_open = True
        self.app.set_busy(True)
        def closed(saved: bool | None) -> None:
            self.environment_closed(saved)
            if on_closed is not None:
                on_closed()

        self.app.push_screen(screen, closed)

    def environment_closed(self, _saved: bool | None) -> None:
        self.environment_open = False
        self.refresh_connection_badges()
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
        self.app.query_one("#run-mode-basic", Button).set_class(not show_advanced, "-active")
        self.app.query_one("#run-mode-advanced", Button).set_class(show_advanced, "-active")

    def set_minimum_repeats(self) -> None:
        if self.app.is_busy() or self.app.execution.session.snapshot.measurement_active:
            return
        self.app.query_one("#repeat", Input).value = "3"
        self.refresh_command_preview(notify=False)

    def _active_advanced_overrides(self, config: RunConfig) -> list[str]:
        overrides: list[str] = []
        if config.revision:
            overrides.append(f"revision={config.revision[:8]}")
        if config.task:
            overrides.append(f"task={config.task}")
        if config.task_family:
            overrides.append(f"family={config.task_family}")
        if config.backend:
            overrides.append(f"backend={config.backend}")
        if config.model_spec:
            overrides.append("model-spec")
        if config.workload_spec:
            overrides.append("workload-spec")
        if config.inference_config:
            overrides.append("inference-config")
        if config.confirm_inference:
            overrides.append("confirm-inference")
        if config.cpuset_cpus:
            overrides.append(f"cpuset={config.cpuset_cpus}")
        if config.max_download:
            overrides.append(f"max-download={config.max_download}")
        if config.rebuild_image:
            overrides.append("rebuild")
        return overrides

    def _update_section_summaries(self, config: RunConfig, planned: dict | None) -> None:
        gpu_modes = {part.strip().lower() for part in config.gpus.split(",") if part.strip()}
        if len(gpu_modes) > 1 or ("off" in gpu_modes and "on" in gpu_modes):
            gpu_label: str = "CPU + GPU"
        elif "on" in gpu_modes or "yes" in gpu_modes or "gpu" in gpu_modes:
            gpu_label = message_id("run.resources.gpu_only")
        else:
            gpu_label = message_id("run.resources.cpu_only")
        if planned and planned.get("scales"):
            scales_label: str = ",".join(f"{value:g}" for value in planned["scales"])
        elif config.input_scales:
            scales_label = config.input_scales
        else:
            scales_label = message_id("common.option.auto")
        mem_label = config.mems if config.mems == "auto" else f"{config.mems}GB"
        self.app.localization.set_text(
            self.app.query_one("#resource-matrix", Collapsible),
            message_id("run.form.resources_summary", config.cpus, mem_label, gpu_label, scales_label),
            "title",
        )
        window_label = (
            message_id("run.windows.auto_duration") if config.repeat_in_window == 0
            else message_id("run.form.window_requests", config.repeat_in_window)
        )
        self.app.localization.set_text(
            self.app.query_one("#measurement-parameters", Collapsible),
            message_id("run.form.measurement_summary", config.warmup, config.repeat, config.batch_size, window_label),
            "title",
        )
        profilers: list[str] = []
        if config.compute_profile_tool in {"torch", "both"}:
            profilers.append("Torch")
        if config.compute_profile_tool in {"ncu", "both"}:
            profilers.append("NCU")
        if config.execution_profile_tool in {"massif", "both"}:
            profilers.append("Massif")
        if config.execution_profile_tool in {"nsys", "both"}:
            profilers.append("Nsys")
        profiler_title = (
            message_id("run.form.profilers_selected", len(profilers), ", ".join(profilers))
            if profilers else message_id("run.form.profilers_off")
        )
        self.app.localization.set_text(
            self.app.query_one("#profiling-options", Collapsible),
            profiler_title,
            "title",
        )
        overrides = self._active_advanced_overrides(config)
        adv_tab = self.app.query_one("#run-mode-advanced", Button)
        self.app.localization.set_text(
            adv_tab,
            message_id("run.form.advanced_count", len(overrides)) if overrides else message_id("run.form.advanced"),
            "label",
        )
        banner = self.app.query_one("#advanced-overrides-banner", Button)
        banner.display = bool(overrides)
        if overrides:
            preview = ", ".join(overrides[:3]) + ("…" if len(overrides) > 3 else "")
            self.app.localization.set_text(
                banner, message_id("run.form.overrides_active", len(overrides), preview), "label",
            )

    def gpu_options(self) -> list[tuple[str, str]]:
        options = [("仅 CPU", "off"), ("仅 GPU", "on"), ("CPU + GPU", "off,on")]
        if self.initial_config.gpus not in {value for _, value in options}:
            options.append((message_id('run.resources.custom_gpu_modes', self.initial_config.gpus), self.initial_config.gpus))
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
        configure_solid_scrollbars(self.app)

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
        self.validation_pending = True
        self.app.preflight.refresh_preflight_state()
        # Coalesce typing and preset field updates into one validation/render.
        self.preview_timer = self.app.set_timer(0.05, self.sync_form_state)

    def sync_form_state(self) -> None:
        self.preview_timer = None
        # Timer callbacks already queued before shutdown can run after the
        # form has been removed; do not query widgets during that phase.
        if not self.app.is_running or not self.app.form_ready or self.app.is_busy() or self.applying_config:
            return
        self.refresh_command_preview(notify=False)

    def collect_config(self, *, allow_empty_model: bool = False) -> RunConfig:
        return run_form.collect_config(
            {key: self.input(key) for key in run_form.INPUT_FIELDS},
            {key: self.select(key) for key in run_form.SELECT_FIELDS},
            {key: self.checked(key) for key in (*run_form.CHECKED_FIELDS, *run_form.PROFILER_CHECKBOXES)},
            project_dir=self.app.project_dir, allow_empty_model=allow_empty_model, extra_options=self.extra_options,
            profiling_mode=self.recovery_policy,
        )

    def apply_config(self, config: RunConfig) -> None:
        config = run_form.runtime_form_config(config)
        self.cancel_preview_timer()
        self.applying_config = True
        self.recovery_policy = config.profiling_mode if config.resume else "best_effort"
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
        finally:
            self.applying_config = False
        if self.app.form_ready:
            self.refresh_command_preview(notify=False)

    def show_config_error(self, exc: RunConfigError) -> None:
        text = join_messages("\n", (message("• {0}", issue.reason) for issue in exc.issues))
        self.app.notify(text, title="配置有误", severity="error", timeout=8)
        if not any(issue.field for issue in exc.issues):
            return
        self.field_errors_visible = True
        control = self.update_field_errors(exc.issues)
        self.validation_pending = False
        self.app.preflight.refresh_preflight_state()
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
        from acprof.tui.run.resource_view import field_issues
        from acprof.tui.settings.field_validation import render_field_issues
        self.config_issues = tuple(issues)
        if self.config_issues:
            self.config_warnings = ()
        return render_field_issues(self.app, (*self.config_issues, *field_issues(self.app.preflight)), self.app.localization.tr)

    def focus_config_field(self, control: Widget) -> None:
        if (control.is_mounted and not control.is_disabled and not self.app.is_busy()
                and not self.app.execution.session.snapshot.measurement_active):
            control.focus(scroll_visible=False)
            control.scroll_visible(animate=False, immediate=True)

    def open_config_field(self, field: str) -> None:
        """Reveal the existing control, preserving its local validation feedback."""
        controls = {value: key for key, value in
                    {**run_form.INPUT_FIELDS, **run_form.SELECT_FIELDS, **run_form.CHECKED_FIELDS}.items()}
        controls.update(compute_profile_tool='torch-profiler', execution_profile_tool='massif-profiler')
        controls.update((key, key) for key in run_form.PROFILER_CHECKBOXES)
        if field not in controls:
            return
        control = self.app.query_one('#' + controls[field])
        ancestors = list(control.ancestors)
        page = next((item.id for item in ancestors if item.id in {'run-form', 'advanced-form'}), 'run-form')
        self.app.activate_tab('run-tab')
        self.show_run_form(page)
        for ancestor in ancestors:
            if isinstance(ancestor, Collapsible):
                ancestor.collapsed = False
        self.app.call_after_refresh(self.focus_config_field, control)

    def refresh_command_preview(self, *, notify: bool = True) -> bool:
        self.cancel_preview_timer()
        self.validation_pending = False
        try:
            config = self.collect_config()
            self.config_warnings = run_planning.configuration_warnings(config)
            command = build_run_command(
                config,
                project_dir=self.app.project_dir,
                python_executable=self.app.python_executable,
            )
            self.update_field_errors(())
        except RunConfigError as exc:
            self.update_field_errors(exc.issues)
            if not self.app.execution.session.snapshot.measurement_active:
                self.app.query_one(RunPlanPanel).show_plan(run_planning.invalid_plan_view(exc.issues))
                self.app.localization.set_text(
                    self.app.query_one("#input-scales-label", Label), message_id("run.inputs.scales"),
                )
                self.app.localization.set_text(
                    self.app.query_one("#input-scales", Input), message_id("common.state.pending"), "placeholder",
                )
            self.app.localization.set_text(
                self.app.query_one("#command-preview", Static),
                message_id('run.command.incomplete_detail', join_messages("; ", (item.reason for item in exc.issues))),
            )
            if notify:
                self.show_config_error(exc)
            return False
        finally:
            self.app.preflight.schedule_auto_check()
            self.app.preflight.refresh_preflight_state()
        self.update_plan_summary(config)
        self.app.localization.set_text(self.app.query_one('#command-preview', Static), format_command(command))
        if notify:
            self.app.notify("命令预览已更新", timeout=2)
        return True

    def update_plan_summary(self, config: RunConfig) -> None:
        # Preparation can update resolved scales while busy; formal measurement cannot.
        if self.app.execution.session.snapshot.measurement_active:
            return
        if config.runtime == "auto" and self.app.preflight.selected_runtime:
            config = replace(config, runtime=self.app.preflight.selected_runtime)
        planned = self.planned_input if self.planned_input_identity == run_planning.input_identity(config) else None
        view = run_planning.run_plan_view(
            config, planned,
            download_budget=config.max_download or os.environ.get("ACPROF_MAX_DOWNLOAD", ""),
            model_store=config.model_store or os.environ.get("ACPROF_MODEL_STORE")
            or str(Path.home() / ".cache" / "acprof" / "model-store"),
        )
        self.app.query_one(RunPlanPanel).show_plan(view)
        self._update_section_summaries(config, planned)
        unit = run_planning.input_unit(config, planned)
        self.app.localization.set_text(self.app.query_one("#input-scales-label", Label), message_id('run.inputs.scales_with_unit', unit))
        if planned and planned.get("scales"):
            placeholder = ",".join(f"{value:g}" for value in planned["scales"]) + f" {unit}"
        else:
            placeholder = message_id('run.inputs.blank_plan', message("最小单一尺度" if config.input_scale_policy == "minimal" else "自动范围"), unit)
        self.app.localization.set_text(self.app.query_one("#input-scales", Input), placeholder, "placeholder")

    def remember_last_used(
        self, *, model: str = "", result_dir: str = "", result_csv: str = "", html_report: str = "",
        log_export_dir: str = "",
    ) -> None:
        """Remember confirmed inputs, preserving explicitly saved preferences."""
        updates = {}
        if model.strip():
            updates["last_model"] = model.strip()
        if result_dir:
            updates["last_result_dir"] = result_dir
        if result_csv:
            from acprof.tui.commands import resolve_result_path
            from acprof.tui.results.overview import selection_path
            result_csv = str(selection_path(resolve_result_path(result_csv, self.app.project_dir)))
            if self.input("experiment-source") != result_csv:
                self.app.results.select_result(result_csv, load=False)
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
