"""Keyboard shortcut reference overlay; purely descriptive with no host I/O."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from acprof.tui.i18n import message_id
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.dialogs import focus_horizontal_action
from acprof.tui.ui.rendering import CjkCompositor
from acprof.tui.ui.structured import FactGrid, StatusBanner
from acprof.tui.ui.view_models import FactRow, StatusSummary, ViewSection


class ShortcutsScreen(ModalScreen[None]):
    """Display active TUI keyboard shortcuts without disrupting measurements."""

    AUTO_FOCUS = "#shortcuts-close"
    BINDINGS = [("escape", "close", "关闭"), ("f1", "close", "关闭")]

    def __init__(self) -> None:
        super().__init__()
        self._compositor = CjkCompositor()

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        global_rows = ViewSection(
            message_id("common.shortcuts.app_title"),
            (
                FactRow("F1 / ?", message_id("common.shortcuts.binding_help")),
                FactRow("Ctrl+K / /", message_id("common.shortcuts.binding_palette")),
                FactRow("Alt+1..5", message_id("common.shortcuts.binding_pages")),
                FactRow("Alt+E", message_id("common.shortcuts.binding_status")),
                FactRow("F2", message_id("common.shortcuts.binding_settings")),
                FactRow("F3", message_id("common.shortcuts.binding_form_toggle")),
                FactRow("F4", message_id("common.shortcuts.binding_model_candidates")),
                FactRow("F5", message_id("common.shortcuts.binding_run")),
                FactRow("F6", message_id("common.shortcuts.binding_environment")),
                FactRow("F8", message_id("common.shortcuts.binding_log_expand")),
                FactRow("Alt+Left", message_id("run.status.return_hint")),
                FactRow("Ctrl+X", message_id("common.shortcuts.binding_stop")),
                FactRow("Ctrl+L", message_id("common.shortcuts.binding_clear_log")),
                FactRow("Ctrl+Q", message_id("common.shortcuts.binding_quit")),
            ),
        )
        control_rows = ViewSection(
            message_id("common.shortcuts.controls_title"),
            (
                FactRow("Enter / Shift+Enter", message_id("common.shortcuts.log_search")),
                FactRow("↓ / Enter", message_id("common.shortcuts.picker_nav")),
                FactRow("← / →", message_id("common.shortcuts.dialog_buttons")),
            ),
        )
        with Vertical(id="shortcuts-dialog"):
            yield Static(tr(message_id("common.shortcuts.title")), id="shortcuts-title", markup=False)
            with VerticalScroll(id="shortcuts-scroll"):
                yield StatusBanner(
                    StatusSummary("neutral", message_id("common.shortcuts.summary")),
                    id="shortcuts-banner",
                    include_actions=False,
                )
                yield FactGrid(global_rows, id="shortcuts-global", collapsible=False)
                yield FactGrid(control_rows, id="shortcuts-controls", collapsible=False)
            with Horizontal(id="shortcuts-actions", classes="action-bar"):
                yield Button(tr(message_id("common.action.close_dialog")), id="shortcuts-close", variant="primary")

    def on_mount(self) -> None:
        for button in self.query(Button):
            button.active_effect_duration = 0

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "shortcuts-close":
            self.dismiss()

    def action_close(self) -> None:
        self.dismiss()

    def on_key(self, event) -> None:
        if event.key == "question_mark":
            event.stop()
            self.dismiss()
            return
        if focus_horizontal_action(self, "#shortcuts-actions", event.key):
            event.stop()
