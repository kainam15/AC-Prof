"""Checkboxes whose marks remain meaningful without terminal colors."""

from dataclasses import replace

from textual.content import Content
from textual.widgets import Checkbox


class StatusCheckbox(Checkbox):
    """Show a check only when selected, preserving the toggle's cell width."""

    DEFAULT_CSS = """
    StatusCheckbox {
        & > .toggle--button {
            background: transparent;
            color: $text-muted;
        }
        &.-on > .toggle--button {
            background: transparent;
            color: $accent;
        }
        &:disabled > .toggle--button {
            background: transparent;
            color: $text-disabled;
        }
        &:ansi > .toggle--button {
            background: ansi_default;
            color: $ansi-foreground;
            text-style: dim;
        }
        &:ansi.-on > .toggle--button {
            background: ansi_default;
            color: $accent;
            text-style: bold not dim;
        }
        &:ansi:disabled > .toggle--button {
            background: ansi_default;
            color: $ansi-foreground;
            text-style: dim;
        }
    }
    """

    BUTTON_LEFT = "["
    BUTTON_INNER = " "
    BUTTON_RIGHT = "]"

    @property
    def _button(self) -> Content:
        """Render bracketed marks directly on the checkbox background."""
        button_style = replace(
            self.get_visual_style("toggle--button"),
            background=self.background_colors[1],
        )
        return Content.assemble(
            (f"{self.BUTTON_LEFT}{self.BUTTON_INNER}{self.BUTTON_RIGHT}", button_style),
        )

    def watch_value(self) -> None:
        self.BUTTON_INNER = "✓" if self.value else " "
        super().watch_value()
