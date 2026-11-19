"""Floating slash-command palette; no timers, polling, or host I/O."""
from __future__ import annotations

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from acprof.tui.i18n import message_id
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.rendering import CjkCompositor

PALETTE_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/run", "common.palette.cmd.run"),
    ("/smoke", "common.palette.cmd.smoke"),
    ("/main", "common.palette.cmd.main"),
    ("/probe", "common.palette.cmd.probe"),
    ("/check", "common.palette.cmd.check"),
    ("/status", "common.palette.cmd.status"),
    ("/stop", "common.palette.cmd.stop"),
    ("/results", "common.palette.cmd.results"),
    ("/plot", "common.palette.cmd.plot"),
    ("/profile", "common.palette.cmd.profile"),
    ("/profile-run", "common.palette.cmd.profile_run"),
    ("/images", "common.palette.cmd.images"),
    ("/log", "common.palette.cmd.log"),
    ("/clear", "common.palette.cmd.clear"),
    ("/preview", "common.palette.cmd.preview"),
    ("/settings", "common.palette.cmd.settings"),
    ("/help", "common.palette.cmd.help"),
    ("/quit", "common.palette.cmd.quit"),
)


class CommandPaletteScreen(ModalScreen[str | None]):
    """On-demand command launcher triggered by '/' or Ctrl+K."""

    BINDINGS = [("escape", "close", "关闭")]
    DEFAULT_CSS = """
    CommandPaletteScreen { align: center middle; }
    #palette-dialog {
        width: 82%;
        max-width: 84;
        height: auto;
        max-height: 82%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    #palette-title { height: 1; text-style: bold; margin-bottom: 1; }
    #palette-search { height: 3; margin-bottom: 1; }
    #palette-list {
        height: auto;
        max-height: 16;
        min-height: 4;
        background: transparent;
        border: none;
    }
    #palette-empty { display: none; height: 2; color: $text-muted; content-align: center middle; }
    #palette-hint { height: 1; color: $text-muted; margin-top: 1; }
    """

    def __init__(self, initial_query: str = "") -> None:
        super().__init__()
        self._compositor = CjkCompositor()
        self._initial_query = initial_query
        self._filtered: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        with Vertical(id="palette-dialog"):
            yield Static(tr(message_id("common.palette.title")), id="palette-title", markup=False)
            yield Input(
                value=self._initial_query,
                placeholder=tr(message_id("common.palette.hint")),
                id="palette-search",
            )
            yield OptionList(id="palette-list")
            yield Static(tr(message_id("common.palette.empty")), id="palette-empty", markup=False)
            yield Static(tr(message_id("common.palette.hint")), id="palette-hint", markup=False)

    def on_mount(self) -> None:
        self._refresh_options("")
        self.query_one("#palette-search", Input).focus()

    def _refresh_options(self, raw_query: str) -> None:
        tr = translated_app(self.app).tr
        query = raw_query.strip().casefold()
        command_head = query.split()[0] if query else ""
        option_list = self.query_one("#palette-list", OptionList)
        option_list.clear_options()
        self._filtered = []
        for command, key in PALETTE_COMMANDS:
            desc = tr(message_id(key))
            haystack = f"{command} {desc}".casefold()
            if not query or query in haystack or (command_head and command_head in command.casefold()):
                self._filtered.append((command, desc))
                prompt = Text.assemble((f"{command:<14}", "bold"), " ", desc)
                option_list.add_option(Option(prompt, id=command))
        empty = self.query_one("#palette-empty", Static)
        empty.display = not self._filtered
        option_list.display = bool(self._filtered)
        if self._filtered:
            option_list.highlighted = 0

    @on(Input.Changed, "#palette-search")
    def search_changed(self, event: Input.Changed) -> None:
        self._refresh_options(event.value)

    @on(Input.Submitted, "#palette-search")
    def submit_search(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        if raw.startswith("/") and " " in raw:
            self.dismiss(raw)
            return
        option_list = self.query_one("#palette-list", OptionList)
        if self._filtered:
            index = option_list.highlighted if option_list.highlighted is not None else 0
            if 0 <= index < len(self._filtered):
                self.dismiss(self._filtered[index][0])
                return
        if raw:
            self.dismiss(raw if raw.startswith("/") else "/" + raw)

    @on(OptionList.OptionSelected, "#palette-list")
    def option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self.dismiss(event.option_id)

    def on_key(self, event: events.Key) -> None:
        if event.key in {"down", "up"} and self.focused is self.query_one("#palette-search", Input):
            option_list = self.query_one("#palette-list", OptionList)
            if self._filtered:
                current = option_list.highlighted if option_list.highlighted is not None else 0
                step = 1 if event.key == "down" else -1
                option_list.highlighted = (current + step) % len(self._filtered)
                event.prevent_default()
                event.stop()

    def action_close(self) -> None:
        self.dismiss(None)
