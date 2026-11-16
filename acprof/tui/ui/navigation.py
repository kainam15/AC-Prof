"""Sidebar presentation for the existing mounted page stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Offset
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, ContentSwitcher, Static, TabbedContent

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


@dataclass(frozen=True)
class NavigationContext:
    """One return location for explicit issue navigation, without owning page state."""

    tab: str
    form: str | None
    focus: Widget | None
    scrolls: tuple[tuple[Widget, Offset], ...]
    maximized: Widget | None

    @classmethod
    def capture(cls, app: AcprofTui) -> NavigationContext:
        return cls(app.query_one('#main-tabs', TabbedContent).active,
                   app.query_one('#experiment-pages', ContentSwitcher).current,
                   app.focused,
                   tuple((widget, widget.scroll_offset) for widget in app.screen.query(Widget)
                         if widget.is_scrollable), app.screen.maximized)

    def restore(self, app: AcprofTui) -> None:
        app.activate_tab(self.tab)
        if self.form is not None:
            app.configuration.show_run_form(self.form)

        def finish() -> None:
            if self.maximized is not None and self.maximized.is_mounted:
                app.screen.maximize(self.maximized, container=False)
            if self.focus is not None and self.focus.is_mounted and self.focus.is_on_screen and not self.focus.is_disabled:
                app.screen.set_focus(self.focus, scroll_visible=False)
            for widget, position in self.scrolls:
                if widget.is_mounted:
                    widget.scroll_to(position.x, position.y, animate=False, immediate=True, force=True)

        app.call_after_refresh(finish)


class PageNavigation(VerticalScroll, can_focus=False):
    """Project tab labels and selection without owning a second active page."""

    BINDINGS = [("up", "move(-1)", ""), ("down", "move(1)", "")]
    ICONS = {
        "run-tab": "☰", "monitor-tab": "◉", "results-tab": "▥",
        "backfill-tab": "↻", "images-tab": "⬢", "settings-tab": "⚙︎",
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
        collapsed = self.app.has_class("collapsed-navigation")
        for button in self.query(Button):
            # Match the actual sidebar width without Button's implicit line pad.
            button.styles.line_pad = 0
            pane_id = button.name
            assert pane_id is not None
            tab = tabs.get_tab(pane_id)
            icon = self.ICONS[pane_id]
            button.label = icon if collapsed else f"{icon}  {tab.label.plain}"
            button.tooltip = tab.tooltip or tab.label.plain
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
