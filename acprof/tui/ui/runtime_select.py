"""Edit runtime intent while displaying the cached automatic resolution."""
from __future__ import annotations

from collections.abc import Iterable

from rich.console import RenderableType
from textual import on
from textual.widgets import Select
from textual.widgets._select import NoSelection, SelectOverlay

from acprof.tui.i18n import message_id


class RuntimeSelect(Select[str]):
    """Only explicit selection changes the request; preflight changes its display."""

    def __init__(self, requested: str, *, id: str, classes: str) -> None:
        self.requested = requested
        self._resolved = ""
        self._projecting = False
        super().__init__(
            (("Docker", "docker"), ("Process", "process")),
            value=self.NULL if requested == "auto" else requested,
            prompt=message_id("run.runtime.pending_selection"),
            tooltip=message_id("run.runtime.selection_help"),
            id=id, classes=classes,
        )

    def _setup_variables_for_options(self, options: Iterable[tuple[RenderableType, str]]) -> None:
        super()._setup_variables_for_options(options)
        # NULL represents unresolved Auto, but is never a third menu choice.
        # Retain it in _legal_values for Select's normal pending-value rendering.
        self._options = [(label, value) for label, value in self._options if value is not self.NULL]

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
        # Confirming the displayed automatic choice is still an explicit request,
        # although Textual skips Changed when the value stays the same.
        if unchanged:
            self.post_message(self.Changed(self, value))

    def _watch_expanded(self, expanded: bool) -> None:
        super()._watch_expanded(expanded)
        if not expanded:
            self._refresh_display()

    def set_requested(self, requested: str) -> None:
        self.requested = requested
        self._resolved = ""
        self.expanded = False
        self._refresh_display()

    def show_resolved(self, resolved: str) -> None:
        self._resolved = resolved
        self._refresh_display()

    def _refresh_display(self) -> None:
        if self.expanded:
            return  # A background result must not move the user's highlighted option.
        value = self.requested if self.requested != "auto" else self._resolved
        self._projecting = True
        try:
            with self.prevent(Select.Changed):
                self.value = value or self.NULL
        finally:
            self._projecting = False
