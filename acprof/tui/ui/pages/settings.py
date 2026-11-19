"""AC-Prof TUI ui/pages/settings.py components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Label, Static, TabPane

from acprof import __version__
from acprof.tui.i18n import LANGUAGE_OPTIONS, message_id
from acprof.tui.ui.checkbox import StatusCheckbox
from acprof.tui.ui.structured import StatusBanner
from acprof.tui.ui.themes import THEME_OPTIONS
from acprof.tui.ui.view_models import StatusSummary

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

def compose_settings_tab(app: AcprofTui) -> ComposeResult:
    with TabPane(message_id("settings.app.tooltip"), id="settings-tab"):
        with Vertical(classes="page-header"):
            yield app.localization.localized_widget(Static("应用设置", classes="page-title"))
            yield StatusBanner(
                StatusSummary("neutral", message_id("updates.ui.settings_hint")),
                id="settings-banner", headline_id="settings-status", include_actions=False,
                classes="page-summary",
            )
        with VerticalScroll(id="settings-body", classes="pane-scroll"):
            with Vertical(id="settings-interface", classes="settings-section") as section:
                app.localization.set_text(section, "界面与交互", "border_title")
                with Grid(classes="form-grid"):
                    yield app.localization.localized_widget(Label("界面语言"))
                    yield app.localization.localized_select(
                        LANGUAGE_OPTIONS,
                        value=app.configuration.preferences.language, allow_blank=False,
                        id="ui-language", classes="ui-preference",
                    )
                    yield app.localization.localized_widget(Label("界面主题"))
                    yield app.localization.localized_select(
                        THEME_OPTIONS,
                        value=app.configuration.preferences.theme, allow_blank=False,
                        id="ui-theme", classes="ui-preference",
                    )
                yield app.localization.localized_widget(StatusCheckbox(
                    "显示底部快捷命令框",
                    value=app.configuration.preferences.show_command_bar,
                    id="ui-command-bar", classes="ui-preference option-checkbox",
                ))
            with Vertical(id="settings-logs", classes="settings-section") as section:
                app.localization.set_text(section, "日志", "border_title")
                with Grid(classes="form-grid"):
                    yield app.localization.localized_widget(Label("保留日志行数"))
                    yield app.localization.localized_select(
                        ((str(n), n) for n in (500, 1000, 3000, 10000)),
                        value=app.configuration.preferences.log_max_lines, allow_blank=False,
                        id="ui-log-lines", classes="ui-preference",
                    )
                yield app.localization.localized_widget(StatusCheckbox(
                    "日志自动换行", value=app.configuration.preferences.log_wrap,
                    id="ui-log-wrap", classes="ui-preference option-checkbox",
                ))
            with Vertical(id="settings-connections", classes="settings-section") as section:
                app.localization.set_text(section, "连接与权限", "border_title")
                yield app.localization.localized_widget(Static(
                    message_id("settings.environment.description"),
                    classes="page-hint", markup=False,
                ))
                yield app.localization.localized_widget(Static(
                    "", id="settings-connection-status", classes="connection-badges", markup=False,
                ))
                yield app.localization.localized_widget(Button("管理连接与权限", id="open-environment-settings"))
            with Vertical(id="settings-version", classes="settings-section") as section:
                app.localization.set_text(section, message_id("common.updates.title"), "border_title")
                yield app.localization.localized_widget(Static(
                    message_id("common.updates.checking", __version__),
                    id="version-status", markup=False,
                ))
                with Grid(classes="form-grid"):
                    yield app.localization.localized_widget(Label(message_id("updates.ui.channel")))
                    yield app.localization.localized_select(
                        ((message_id("updates.ui.stable"), "stable"), (message_id("updates.ui.preview"), "preview")),
                        value=app.updates.channel, allow_blank=False, id="update-channel",
                    )
                with Horizontal(classes="version-actions"):
                    yield app.localization.localized_widget(Button(message_id("updates.ui.check"), id="version-check"))
                    yield app.localization.localized_widget(Button(message_id("updates.ui.apply"), id="version-apply", variant="primary", disabled=True))
                with Horizontal(classes="version-actions"):
                    yield app.localization.localized_widget(Button(message_id("updates.ui.logs"), id="version-log", disabled=True))
                    yield app.localization.localized_widget(Button(message_id("updates.ui.restart"), id="version-restart"))
                yield app.localization.localized_widget(Static("", id="version-installation", markup=False, classes="page-hint"))
                yield app.localization.localized_widget(Button(message_id("updates.ui.manual_copy"), id="version-manual"))
            yield app.localization.localized_widget(Static("", id="settings-location", classes="page-hint", markup=False))
        with Horizontal(id="settings-actions", classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                yield app.localization.localized_widget(Button("恢复界面默认", id="restore-ui-defaults"))
            with Horizontal(classes="action-primary"):
                yield app.localization.localized_widget(Button("保存设置", id="save-ui-settings", variant="primary"))
