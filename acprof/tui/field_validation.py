"""Present shared configuration failures next to their existing form controls."""
from __future__ import annotations

from collections.abc import Callable, Iterable

from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Static

from acprof.experiment import ConfigIssue
from acprof.tui.run_form import CHECKED_FIELDS, INPUT_FIELDS, SELECT_FIELDS


class ConfigField(Vertical):
    """Keep the normal control geometry until a persistent error is displayed."""

    DEFAULT_CSS = """
    ConfigField { height: auto; width: 1fr; }
    ConfigField > .field-error { display: none; height: auto; color: $error; padding: 0 1; }
    """

    def __init__(self, control: Widget, *, field_id: str):
        super().__init__(classes="config-field")
        self.control, self.field_id = control, field_id

    def compose(self):
        yield self.control
        yield Static("", id=f"{self.field_id}-error", classes="field-error", markup=False)


def render_field_issues(root: Widget, issues: Iterable[ConfigIssue], translate: Callable[[str], str]) -> Widget | None:
    """Refresh existing slots and return the first invalid, addressable control."""
    controls = {field: identifier for identifier, field in
                {**INPUT_FIELDS, **SELECT_FIELDS, **CHECKED_FIELDS}.items()}
    grouped: dict[str, list[str]] = {}
    for issue in issues:
        if issue.field in controls:
            grouped.setdefault(controls[issue.field], []).append(translate(issue.reason))
    first = None
    for identifier in controls.values():
        try:
            control = root.query_one(f"#{identifier}")
        except NoMatches:
            continue
        reasons = grouped.get(identifier, [])
        control.set_class(bool(reasons), "config-invalid")
        try:
            error = root.query_one(f"#{identifier}-error", Static)
        except NoMatches:
            pass
        else:
            error.update("\n".join(reasons))
            error.display = bool(reasons)
    for identifier in grouped:
        try:
            first = root.query_one(f"#{identifier}")
        except NoMatches:
            continue
        break
    return first
