"""Edit runtime intent while displaying the cached automatic resolution."""
from __future__ import annotations

from textual import on
from textual.widgets import Select
from textual.widgets._select import NoSelection, SelectOverlay

from acprof.tui.i18n import message_id
from acprof.tui.ui.contracts import translated_app


class RuntimeSelect(Select[str]):
    """Keep Auto explicitly selectable while displaying its resolved runtime."""

    def __init__(self, requested: str, *, id: str, classes: str) -> None:
        self.requested = requested if requested in {"auto", "docker", "process"} else "auto"
        self._resolved = ""
        self._projecting = False
        super().__init__(
            self._build_options(translate=str),
            value=self.requested,
            allow_blank=False,
            tooltip=message_id("run.runtime.selection_help"),
            id=id, classes=classes,
        )

    def _auto_label(self, translate) -> str:
        if self._resolved in {"docker", "process"}:
            resolved_name = "Docker" if self._resolved == "docker" else "Process"
            return translate(message_id("run.runtime.option_auto_resolved", resolved_name))
        return translate(message_id("run.runtime.option_auto_pending"))

    def _build_options(self, *, translate) -> tuple[tuple[str, str], ...]:
        return (
            (self._auto_label(translate), "auto"),
            ("Docker", "docker"),
            ("Process", "process"),
        )

    def _watch_value(self, value: str | NoSelection) -> None:
        if not self._projecting:
            self.requested = value if isinstance(value, str) else "auto"
        super()._watch_value(value)

    @on(SelectOverlay.UpdateSelection)
    def _update_selection(self, event: SelectOverlay.UpdateSelection) -> None:
        value = self._options[event.option_index][1]
        assert isinstance(value, str)
        self.requested = value
        unchanged = self.value == value
        super()._update_selection(event)
        if unchanged:
            self.post_message(self.Changed(self, value))

    def _watch_expanded(self, expanded: bool) -> None:
        super()._watch_expanded(expanded)
        if not expanded:
            self._refresh_display()

    def on_mount(self) -> None:
        self._refresh_display()

    def set_requested(self, requested: str) -> None:
        self.requested = requested if requested in {"auto", "docker", "process"} else "auto"
        self._resolved = ""
        self.expanded = False
        self._refresh_display()

    def show_resolved(self, resolved: str) -> None:
        self._resolved = resolved
        self._refresh_display()

    def _refresh_display(self) -> None:
        if self.expanded or not self.is_mounted:
            return  # A background result must not move the user's highlighted option.
        translate = translated_app(self.app).tr
        self._projecting = True
        try:
            with self.prevent(Select.Changed):
                self.set_options(self._build_options(translate=translate))
                self.value = self.requested
                self.mutate_reactive(getattr(Select, "value"))
        finally:
            self._projecting = False
