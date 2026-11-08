"""Shared, event-driven structured views for the terminal UI."""
from __future__ import annotations

from dataclasses import replace

from rich.cells import cell_len
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Static

from acprof.messages import Message as CatalogText
from acprof.tui.i18n import message_id
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.presentation import COMPACT_COLLAPSED_SYMBOL, COMPACT_EXPANDED_SYMBOL
from acprof.tui.ui.rendering import CjkCompositor
from acprof.tui.ui.view_models import (
    CalloutContent,
    FactRow,
    StatusSummary,
    ViewAction,
    ViewSection,
    middle_ellipsis,
    status_presentation,
    status_symbol,
)


def _value_text(value: str, app: object) -> str:
    """Translate structured messages only; user paths and values stay verbatim."""
    return translated_app(app).tr(value) if isinstance(value, CatalogText) else value


class ActionRequested(Message):
    """Route a stable action to its existing controller, never launch work here."""

    def __init__(self, action_id: str) -> None:
        super().__init__()
        self.action_id = action_id


class StructuredWidget(Vertical):
    """Localize mounted children without rebuilding them or registering stale widgets."""

    def tr(self, source: str) -> str:
        return translated_app(self.app).tr(source)

    def refresh_language(self) -> None:
        pass


class ActionButtons(Horizontal):
    def __init__(self, actions: tuple[ViewAction, ...] = (), **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions = actions

    def compose(self) -> ComposeResult:
        for index in range(2):
            yield Button("", name=str(index), classes="structured-action")

    def on_mount(self) -> None:
        self.update_actions(self.actions)

    def update_actions(self, actions: tuple[ViewAction, ...]) -> None:
        if len(actions) > 2:
            raise ValueError("Structured action rows support one primary and one secondary action")
        self.actions = actions
        if not self.is_mounted:
            return
        tr = translated_app(self.app).tr
        self.display = bool(actions)
        for index, button in enumerate(self.query(Button)):
            button.active_effect_duration = 0
            button.display = index < len(actions)
            if index < len(actions):
                action = actions[index]
                label = tr(action.label)
                if str(button.label) != label:
                    button.label = Text(label)
                    button.refresh(layout=True)
                button.variant = "primary" if action.primary else "default"
                button.disabled = action.disabled
                button.tooltip = label

    @on(Button.Pressed)
    def requested(self, event: Button.Pressed) -> None:
        event.stop()
        index = int(event.button.name or "0")
        if index < len(self.actions) and not self.actions[index].disabled:
            self.post_message(ActionRequested(self.actions[index].id))


class StatusBanner(StructuredWidget):
    def __init__(self, summary: StatusSummary, *, headline_id: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.summary = summary
        self.headline_id = headline_id

    def compose(self) -> ComposeResult:
        with Horizontal(classes="status-banner-line"):
            yield Static("", id=self.headline_id, classes="status-headline", markup=False)
            yield ActionButtons(self.summary.actions, classes="status-actions")

    def on_mount(self) -> None:
        self.refresh_language()

    def on_resize(self) -> None:
        self.set_class(self.size.width < 76, "compact")

    def update_summary(self, summary: StatusSummary) -> None:
        self.summary = summary
        if self.is_mounted:
            self.refresh_language()

    def refresh_language(self) -> None:
        for level in ("success", "warning", "error", "info", "neutral"):
            self.set_class(self.summary.level == level, "status-" + level)
        self.query_one(".status-headline", Static).update(
            Text(f"{status_symbol(self.summary.level)} {self.tr(self.summary.headline)}")
        )
        self.query_one(ActionButtons).update_actions(self.summary.actions)

    def set_action_disabled(self, action_id: str, disabled: bool) -> None:
        actions = tuple(replace(action, disabled=disabled) if action.id == action_id else action
                        for action in self.summary.actions)
        if actions != self.summary.actions:
            self.update_summary(replace(self.summary, actions=actions))


class FactDetailScreen(ModalScreen[None]):
    BINDINGS = [("escape", "close", "Close")]

    def __init__(self, row: FactRow) -> None:
        super().__init__()
        self.row = row
        self._compositor = CjkCompositor()

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        with Vertical(id="fact-detail-dialog"):
            yield Static(Text(tr(self.row.label)), classes="fact-detail-title")
            with VerticalScroll(id="fact-detail-body"):
                yield Static(Text(_value_text(self.row.value, self.app)), markup=False)
                if self.row.detail and self.row.detail != self.row.value:
                    yield Static(Text(self.row.detail), markup=False)
            with Horizontal(classes="fact-detail-actions"):
                yield Button(tr(message_id("structured.action.copy")), id="fact-copy",
                             disabled=not self.row.copyable, variant="primary")
                yield Button(tr(message_id("structured.action.close")), id="fact-close")

    @on(Button.Pressed, "#fact-copy")
    def copy_value(self, event: Button.Pressed) -> None:
        event.stop()
        self.app.copy_to_clipboard(_value_text(self.row.value, self.app))
        self.app.notify(translated_app(self.app).tr(message_id("structured.action.copied")))

    @on(Button.Pressed, "#fact-close")
    def close_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_close()

    def action_close(self) -> None:
        self.dismiss(None)


class FactValue(Static, can_focus=True):
    BINDINGS = [
        Binding("enter", "details", "Details"),
        Binding("c", "copy_value", "Copy full value"),
    ]

    def __init__(self, row: FactRow) -> None:
        super().__init__("", classes="fact-value", markup=False)
        self.row = row
        self.can_focus = bool(row.copyable or row.detail)

    def on_mount(self) -> None:
        self.refresh_value()

    def on_focus(self) -> None:
        self.refresh(layout=True)

    def on_blur(self) -> None:
        self.refresh(layout=True)

    def update_row(self, row: FactRow) -> None:
        self.row = row
        self.can_focus = bool(row.copyable or row.detail)
        self.refresh_value()

    def refresh_value(self) -> None:
        level = status_presentation(self.row.status).level if self.row.status else None
        for candidate in ("success", "warning", "error", "info", "neutral"):
            self.set_class(level == candidate, "status-" + candidate)
        self.tooltip = Text(_value_text(self.row.value, self.app) + ("\n" + self.row.detail
                           if self.row.detail and self.row.detail != self.row.value else ""))
        self.refresh(layout=True)

    def render(self) -> Text:
        value = _value_text(self.row.value, self.app)
        prefix = status_presentation(self.row.status).symbol + " " if self.row.status else ""
        if self.row.copyable and self.content_size.width:
            value = middle_ellipsis(value, max(0, self.content_size.width - cell_len(prefix)))
        value = prefix + value
        if self.has_focus and self.can_focus:
            key = "structured.fact.hint" if self.row.copyable else "structured.fact.details_hint"
            value += "\n" + translated_app(self.app).tr(message_id(key))
        return Text(value, overflow="fold", no_wrap=False)

    def action_details(self) -> None:
        if self.can_focus:
            self.app.push_screen(FactDetailScreen(self.row))

    def action_copy_value(self) -> None:
        if self.row.copyable:
            self.app.copy_to_clipboard(_value_text(self.row.value, self.app))
            self.app.notify(translated_app(self.app).tr(message_id("structured.action.copied")))

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "copy_value":
            return self.row.copyable
        return True


class FactGrid(StructuredWidget):
    def __init__(self, section: ViewSection, **kwargs) -> None:
        super().__init__(**kwargs)
        self.section = section
        self._composed_shape: tuple[int, int] | None = None
        self._recompose_pending = False

    def _rows(self, rows: tuple[FactRow, ...]) -> ComposeResult:
        for row in rows:
            yield Static(Text(self.tr(row.label)), classes="fact-label", markup=False)
            yield FactValue(row)

    def compose(self) -> ComposeResult:
        self._composed_shape = (len(self.section.rows), len(self.section.detail_rows))
        with Collapsible(title=self.tr(self.section.title), collapsed=self.section.collapsed,
                         collapsed_symbol=COMPACT_COLLAPSED_SYMBOL,
                         expanded_symbol=COMPACT_EXPANDED_SYMBOL, classes="fact-section"):
            with Grid(classes="fact-grid"):
                yield from self._rows(self.section.rows)
            if self.section.detail_rows:
                with Collapsible(title=self.tr(message_id("structured.fact.details")),
                                 collapsed=True, collapsed_symbol=COMPACT_COLLAPSED_SYMBOL,
                                 expanded_symbol=COMPACT_EXPANDED_SYMBOL, classes="fact-more"):
                    with Grid(classes="fact-grid"):
                        yield from self._rows(self.section.detail_rows)

    def update_section(self, section: ViewSection) -> None:
        self.section = section
        if not self.is_mounted or self._recompose_pending:
            return
        shape = (len(section.rows), len(section.detail_rows))
        if shape != self._composed_shape:
            # Metadata and row readers may both finish before the first layout pass.
            # Coalesce their projections; never zip fresh data against the old DOM.
            self._recompose_pending = True
            self.call_after_refresh(self._recompose_section)
        else:
            self.refresh_language()

    async def _recompose_section(self) -> None:
        try:
            await self.recompose()
        finally:
            self._recompose_pending = False
        self.update_section(self.section)

    def refresh_language(self) -> None:
        if self._recompose_pending:
            return
        self.query_one(".fact-section", Collapsible).title = self.tr(self.section.title)
        for details in self.query(".fact-more").results(Collapsible):
            details.title = self.tr(message_id("structured.fact.details"))
        rows = (*self.section.rows, *self.section.detail_rows)
        for label, value, row in zip(self.query(".fact-label").results(Static), self.query(FactValue), rows, strict=True):
            label.update(Text(self.tr(row.label)))
            value.update_row(row)


class Callout(StructuredWidget):
    def __init__(self, content: CalloutContent, **kwargs) -> None:
        super().__init__(**kwargs)
        self.content = content

    def compose(self) -> ComposeResult:
        with Grid(classes="callout-grid"):
            for kind in ("problem", "impact", "next"):
                yield Static("", classes="callout-label callout-label-" + kind, markup=False)
                yield Static("", classes="callout-value callout-" + kind, markup=False)
        yield ActionButtons(self.content.actions, classes="callout-actions")
        with Collapsible(title="", collapsed=True, collapsed_symbol=COMPACT_COLLAPSED_SYMBOL,
                         expanded_symbol=COMPACT_EXPANDED_SYMBOL, classes="callout-details"):
            yield Static("", classes="callout-raw", markup=False)
            yield Button("", classes="callout-copy")

    def on_mount(self) -> None:
        self.refresh_language()

    def update_content(self, content: CalloutContent) -> None:
        self.content = content
        if self.is_mounted:
            self.refresh_language()

    def refresh_language(self) -> None:
        for level in ("success", "warning", "error", "info", "neutral"):
            self.set_class(self.content.level == level, "status-" + level)
        for kind, value in (("problem", self.content.problem), ("impact", self.content.impact),
                            ("next", self.content.next_step)):
            self.query_one(".callout-label-" + kind, Static).update(
                Text(self.tr(message_id("structured.callout." + kind))))
            self.query_one(".callout-" + kind, Static).update(Text(self.tr(value)))
        details = self.query_one(".callout-details", Collapsible)
        details.title = self.tr(message_id("structured.detail.title"))
        details.display = bool(self.content.detail)
        self.query_one(".callout-raw", Static).update(Text(self.content.detail))
        self.query_one(".callout-copy", Button).label = Text(self.tr(message_id("structured.action.copy")))
        self.query_one(ActionButtons).update_actions(self.content.actions)

    def show_details(self) -> None:
        details = self.query_one(".callout-details", Collapsible)
        if details.display:
            details.collapsed = False
            details.scroll_visible(animate=False)
            self.query_one(".callout-copy", Button).focus()

    @on(Button.Pressed, ".callout-copy")
    def copy_details(self, event: Button.Pressed) -> None:
        event.stop()
        self.app.copy_to_clipboard(self.content.detail)
        self.app.notify(self.tr(message_id("structured.action.copied")))
