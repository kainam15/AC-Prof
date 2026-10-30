"""Presentation-only localization actions; no measurement or process ownership."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from rich.text import Text
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Collapsible, Select, Static, Tab

from acprof import platform
from acprof.tui.i18n import translate
from acprof.tui.ui.input import BarCursorInput as Input

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class LocalizationController:
    """Own localization state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.localized_text: dict[tuple[Widget, str], str] = {}
        self.localized_selects: dict[Select, tuple] = {}
        self.applied_language: str | None = None
        self.source_bindings: dict[str, list[Binding]] = {}

    def tr(self, source: str) -> str:
        return translate(source, self.app.configuration.preferences.language)

    def localized_widget(self, widget: Widget) -> Widget:
        """Register only our own widgets, before Textual creates their children."""
        if isinstance(widget, (Button, Checkbox)):
            label = widget.label
            self.localized_text[widget, "label"] = label if isinstance(label, str) else label.plain
        elif isinstance(widget, Static):
            content = widget.content
            self.localized_text[widget, "content"] = (
                content if isinstance(content, str) else
                content.plain if isinstance(content, Text) else str(content)
            )
        elif isinstance(widget, Input):
            self.localized_text[widget, "placeholder"] = widget.placeholder
        elif isinstance(widget, Collapsible):
            self.localized_text[widget, "title"] = widget.title
        if isinstance(widget.tooltip, str):
            self.localized_text[widget, "tooltip"] = widget.tooltip
        return widget

    def localized_select(self, options, **kwargs) -> Select:
        sources = tuple(options)
        widget = Select(sources, **kwargs)
        self.localized_selects[widget] = sources
        return widget

    def set_text(self, widget: Widget, source: str, attribute: str = "content", *, layout: bool = True) -> None:
        self.localized_text[widget, attribute] = source
        self.render_text(widget, attribute, source, layout=layout)

    def render_text(self, widget: Widget, attribute: str, source: str, *, layout: bool = True) -> None:
        rendered = self.tr(source)
        if attribute == "content":
            assert isinstance(widget, Static)
            widget.update(rendered, layout=layout)
        else:
            setattr(widget, attribute, rendered)
            # Button labels do not invalidate cached content widths themselves.
            # Recompute geometry when switching a visible page back and forth.
            widget.refresh(layout=True)

    def capture_language_text(self) -> None:
        # Nested result sections have their own TabbedContent owner.
        for tab in self.app.query(Tab):
            self.localized_text[tab, "label"] = tab.label.plain
        self.source_bindings = {
            key: list(bindings) for key, bindings in self.app._bindings.key_to_bindings.items()
        }

    def apply_language(self) -> None:
        if self.applied_language == self.app.configuration.preferences.language:
            return
        self.applied_language = self.app.configuration.preferences.language
        # Keep mounted widgets, drafts, selected values, log text/selection,
        # scroll positions and progress state. This runs only on a UI change.
        with self.app.prevent(Select.Changed), self.app.batch_update():
            self.app.sub_title = f"{platform.detect_environment().label} · {self.tr(self.app.SUB_TITLE)}"
            for (widget, attribute), source in self.localized_text.items():
                self.render_text(widget, attribute, source)
            for widget, sources in self.localized_selects.items():
                value = widget.value
                widget.set_options((self.tr(label), key) for label, key in sources)
                widget.value = value
                # If the selection is the first option, its value did not
                # change. Still refresh the displayed prompt from the catalog.
                # Use the actual reactive descriptor: a redraw alone leaves the
                # current option label stale when Select.value is unchanged.
                widget.mutate_reactive(getattr(Select, "value"))
            self.app._bindings.key_to_bindings = {
                key: [replace(binding, description=self.tr(binding.description)) for binding in bindings]
                for key, bindings in self.source_bindings.items()
            }
            self.app.refresh_bindings()
            self.app.logs.refresh_status()
            self.app.query_one("#status-preparation").refresh(layout=True)
            if self.app.preflight.latest_download is not None and not self.app.execution.session.snapshot.measurement_active:
                self.app.preflight.download_progress(self.app.preflight.latest_download)
            if self.app.results.report_view is not None:
                self.app.results.render_report_view()
            self.app.images.render_images()
            if self.app.configuration.field_errors_visible:
                self.app.configuration.update_field_errors(self.app.configuration.config_issues)
