"""AC-Prof TUI ui/dialogs.py components."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static
from textual.widgets.button import ButtonVariant

from acprof.tui.i18n import message_id
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.rendering import CjkCompositor


class EnvironmentPreflightScreen(ModalScreen[bool]):
    """Show cached issues without running probes while the dialog is open."""

    AUTO_FOCUS = "#preflight-close"
    BINDINGS = [("escape", "close", "关闭详情")]
    CSS = """
    EnvironmentPreflightScreen { align: center middle; }
    #preflight-dialog {
        width: 92%; max-width: 100; height: auto; max-height: 85%;
        border: round $accent; background: $surface; padding: 1 2;
    }
    #preflight-title { height: auto; text-style: bold; margin-bottom: 1; }
    #preflight-scroll { height: auto; max-height: 45vh; margin-bottom: 1; }
    #preflight-details { height: auto; }
    #preflight-buttons { height: auto; align-horizontal: right; }
    #preflight-buttons Button { margin-left: 1; }
    """

    def __init__(self, details: str, *, retry_disabled: bool):
        super().__init__()
        self._compositor = CjkCompositor()
        self.details = details
        self.retry_disabled = retry_disabled

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        with Vertical(id="preflight-dialog"):
            yield Static(tr("环境状态"), id="preflight-title")
            with VerticalScroll(id="preflight-scroll"):
                yield Static(tr(self.details), id="preflight-details", markup=False)
            with Horizontal(id="preflight-buttons"):
                yield Button(tr("关闭详情"), id="preflight-close")
                yield Button(tr("重新检查"), id="preflight-retry", variant="primary",
                             disabled=self.retry_disabled)

    @on(Button.Pressed, "#preflight-close")
    def action_close(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#preflight-retry")
    def retry(self) -> None:
        self.dismiss(True)

class ConfirmActionScreen(ModalScreen[bool]):
    """Small confirmation screen for long-running or mutating actions."""

    # 打开时不预选按钮；鼠标悬停与主动使用 Tab 聚焦仍保留高亮。
    AUTO_FOCUS = ""

    BINDINGS = [("escape", "cancel", "取消")]

    CSS = """
    ConfirmActionScreen {
        align: center middle;
    }

    #confirm-dialog {
        width: 92%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }

    #confirm-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #confirm-message {
        height: auto;
        max-height: 40vh;
        overflow-y: auto;
        margin-bottom: 1;
    }

    #confirm-buttons {
        height: auto;
        align-horizontal: right;
    }

    #confirm-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self, title: str, message: str, confirm_label: str = "确认", *,
        variant: ButtonVariant = "primary",
    ):
        super().__init__()
        self._compositor = CjkCompositor()
        self.dialog_title = title
        self.message = message
        self.confirm_label = confirm_label
        self.confirm_variant = variant

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        with Vertical(id="confirm-dialog"):
            yield Static(tr(self.dialog_title), id="confirm-title", markup=False)
            yield Static(tr(self.message), id="confirm-message", markup=False)
            with Horizontal(id="confirm-buttons"):
                yield Button(tr("取消"), id="confirm-no")
                yield Button(
                    tr(self.confirm_label),
                    id="confirm-yes",
                    variant=self.confirm_variant,
                )

    @on(Button.Pressed, "#confirm-yes")
    def confirm(self) -> None:
        self.dismiss(True)

    def on_mount(self) -> None:
        for button in self.query(Button):
            button.active_effect_duration = 0

    @on(Button.Pressed, "#confirm-no")
    def cancel(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)

class ScrollConfirmActionScreen(ConfirmActionScreen):
    """Keep approval actions reachable while the complete action plan scrolls."""

    CSS = ConfirmActionScreen.CSS + """
    ScrollConfirmActionScreen { align: center middle; }
    ScrollConfirmActionScreen #confirm-dialog { height: 80%; }
    ScrollConfirmActionScreen #confirm-message { height: 1fr; max-height: 100%; }
    """

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        with Vertical(id="confirm-dialog"):
            yield Static(tr(self.dialog_title), id="confirm-title", markup=False)
            with VerticalScroll(id="confirm-message"):
                yield Static(tr(self.message), markup=False)
            with Horizontal(id="confirm-buttons"):
                yield Button(tr(message_id('common.action.cancel')), id="confirm-no")
                yield Button(tr(self.confirm_label), id="confirm-yes", variant=self.confirm_variant)
