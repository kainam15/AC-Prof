"""AC-Prof TUI ui/pages/settings.py components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Label, Static, TabPane

from acprof.tui.i18n import LANGUAGE_OPTIONS
from acprof.tui.ui.checkbox import StatusCheckbox
from acprof.tui.ui.themes import THEME_OPTIONS

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

def compose_settings_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("应用设置", id="settings-tab"):
        with Vertical(classes="page-header"):
            yield app.localization.localized_widget(Static("应用设置", classes="page-title"))
            yield app.localization.localized_widget(Static(
                "修改立即生效，点击保存后下次启动沿用。",
                id="settings-status", classes="page-summary", markup=False,
            ))
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
                    "配置 Hugging Face、代理、企业微信通知和采集权限。",
                    classes="page-hint", markup=False,
                ))
                yield app.localization.localized_widget(Button("管理连接与权限", id="open-environment-settings"))
            yield app.localization.localized_widget(Static("", id="settings-location", classes="page-hint", markup=False))
        with Horizontal(id="settings-actions", classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                yield app.localization.localized_widget(Button("恢复界面默认", id="restore-ui-defaults"))
            with Horizontal(classes="action-primary"):
                yield app.localization.localized_widget(Button("保存设置", id="save-ui-settings", variant="primary"))
