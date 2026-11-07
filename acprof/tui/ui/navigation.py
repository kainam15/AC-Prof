"""Sidebar presentation for the existing mounted page stack."""
from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static, TabbedContent


class PageNavigation(VerticalScroll, can_focus=False):
    """Project tab labels and selection without owning a second active page."""

    BINDINGS = [("up", "move(-1)", ""), ("down", "move(1)", "")]
    ICONS = {
        "run-tab": "≡", "monitor-tab": "◷", "results-tab": "▥",
        "backfill-tab": "↻", "images-tab": "▣", "settings-tab": "⚙",
    }

    class Selected(Message):
        def __init__(self, pane_id: str) -> None:
            super().__init__()
            self.pane_id = pane_id

    def compose(self) -> ComposeResult:
        for pane_id in self.ICONS:
            if pane_id == "settings-tab":
                yield Static(id="navigation-divider")
            yield Button("", id=f"nav-{pane_id}", name=pane_id, compact=True)

    def sync(self, tabs: TabbedContent) -> None:
        for button in self.query(Button):
            pane_id = button.name
            assert pane_id is not None
            tab = tabs.get_tab(pane_id)
            button.label = f"{self.ICONS[pane_id]}  {tab.label.plain}"
            button.tooltip = tab.tooltip
            button.set_class(tabs.active == pane_id, "selected-page")

    @on(Button.Pressed)
    def select_page(self, event: Button.Pressed) -> None:
        event.stop()
        pane_id = event.button.name
        assert pane_id is not None
        self.post_message(self.Selected(pane_id))

    def action_move(self, step: int) -> None:
        buttons = list(self.query(Button))
        if self.app.focused in buttons:
            index = buttons.index(self.app.focused)
            buttons[(index + step) % len(buttons)].focus()
