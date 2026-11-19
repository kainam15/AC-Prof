"""Checkboxes whose marks remain meaningful without terminal colors."""

from textual.widgets import Checkbox


class StatusCheckbox(Checkbox):
    """Show a check only when selected, preserving the toggle's cell width."""

    BUTTON_LEFT = "["
    BUTTON_INNER = " "
    BUTTON_RIGHT = "]"

    def watch_value(self) -> None:
        self.BUTTON_INNER = "✓" if self.value else " "
        super().watch_value()
