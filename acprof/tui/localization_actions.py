"""Presentation-only localization actions; no measurement or process ownership."""
from __future__ import annotations

from dataclasses import replace

from textual.widget import Widget
from textual.widgets import Button, Checkbox, Collapsible, Select, Static, TabbedContent, TabPane

from acprof.tui.i18n import translate
from acprof.tui.input import BarCursorInput as Input


class LocalizationActions:
    def tr(self, source: str) -> str:
        return translate(source, self.ui_preferences.language)

    def notify(self, message: str, *, title: str = "", **kwargs) -> None:
        super().notify(self.tr(message), title=self.tr(title), **kwargs)

    def _localized_widget(self, widget: Widget) -> Widget:
        """Register only our own widgets, before Textual creates their children."""
        if isinstance(widget, (Button, Checkbox)):
            self._localized_text[widget, "label"] = widget.label.plain
        elif isinstance(widget, Static):
            self._localized_text[widget, "content"] = widget.content
        elif isinstance(widget, Input):
            self._localized_text[widget, "placeholder"] = widget.placeholder
        elif isinstance(widget, Collapsible):
            self._localized_text[widget, "title"] = widget.title
        if isinstance(widget.tooltip, str):
            self._localized_text[widget, "tooltip"] = widget.tooltip
        return widget

    def _localized_select(self, options, **kwargs) -> Select:
        sources = tuple(options)
        widget = Select(sources, **kwargs)
        self._localized_selects[widget] = sources
        return widget

    def _set_text(self, widget: Widget, source: str, attribute: str = "content") -> None:
        self._localized_text[widget, attribute] = source
        self._render_text(widget, attribute, source)

    def _render_text(self, widget: Widget, attribute: str, source: str) -> None:
        rendered = self.tr(source)
        if attribute == "content":
            assert isinstance(widget, Static)
            widget.update(rendered)
        else:
            setattr(widget, attribute, rendered)
            # Button labels do not invalidate cached content widths themselves.
            # Recompute geometry when switching a visible page back and forth.
            widget.refresh(layout=True)

    def _capture_language_text(self) -> None:
        tabs = self.query_one("#main-tabs", TabbedContent)
        for pane in tabs.query(TabPane):
            tab = tabs.get_tab(pane)
            self._localized_text[tab, "label"] = tab.label.plain
        self._source_bindings = {
            key: list(bindings) for key, bindings in self._bindings.key_to_bindings.items()
        }

    def _apply_language(self) -> None:
        if self._applied_language == self.ui_preferences.language:
            return
        self._applied_language = self.ui_preferences.language
        # Keep mounted widgets, drafts, selected values, log text/selection,
        # scroll positions and progress state. This runs only on a UI change.
        with self.prevent(Select.Changed), self.batch_update():
            self.sub_title = f"{self._current_environment().label} · {self.tr(self.SUB_TITLE)}"
            for (widget, attribute), source in self._localized_text.items():
                self._render_text(widget, attribute, source)
            for widget, sources in self._localized_selects.items():
                value = widget.value
                widget.set_options((self.tr(label), key) for label, key in sources)
                widget.value = value
                # If the selection is the first option, its value did not
                # change. Still refresh the displayed prompt from the catalog.
                widget.mutate_reactive(Select.value)
            self._bindings.key_to_bindings = {
                key: [replace(binding, description=self.tr(binding.description)) for binding in bindings]
                for key, bindings in self._source_bindings.items()
            }
            self.refresh_bindings()
            if self._report_view is not None:
                self._render_report_view()
            self._render_images()
            if self._field_errors_visible:
                self._update_field_errors(self._config_issues)
