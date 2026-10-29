"""Save dialog and explicit server-directory navigation for manual log export."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from socket import gethostname
from threading import Event

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList, Static

from acprof.tui.app_contracts import translated_app
from acprof.tui.i18n import message_id
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log_export import export_directory, log_export_error, log_export_path
from acprof.tui.rendering import CjkCompositor

type DirectoryLoader = Callable[[ExportDirectoryScreen, Path, bool], None]


class LogExportScreen(ModalScreen[Path | None]):
    BINDINGS = [('escape', 'cancel', '取消')]
    CSS = """
    LogExportScreen { align: center middle; }
    #log-export-dialog { width: 94%; max-width: 96; height: auto; max-height: 100%;
        border: round $accent; background: $surface; padding: 0 1; }
    #log-export-body { height: auto; max-height: 75vh; }
    #log-export-title { text-style: bold; height: 1; }
    #log-export-host, #log-export-hint { height: auto; color: $text-muted; }
    .log-export-label { height: 1; }
    #log-export-directory-row { height: 3; }
    #log-export-directory { width: 1fr; }
    #log-export-browse { min-width: 8; width: auto; margin-left: 1; }
    #log-export-error { display: none; height: auto; color: $error; }
    #log-export-actions { height: 3; align-horizontal: right; }
    #log-export-actions Button { min-width: 10; margin-left: 1; }
    """

    def __init__(self, directory: Path, filename: str, load_directory: DirectoryLoader) -> None:
        super().__init__(id='log-export-screen')
        self._compositor = CjkCompositor()
        self.directory, self.filename = directory, filename
        self.load_directory = load_directory

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        with Vertical(id='log-export-dialog'):
            with VerticalScroll(id='log-export-body'):
                yield Static(tr(message_id('logs.export.title')), id='log-export-title', markup=False)
                yield Static(tr(message_id('logs.export.host_location', gethostname())), id='log-export-host', markup=False)
                yield Static(tr(message_id('logs.export.directory_label')), classes='log-export-label')
                with Horizontal(id='log-export-directory-row'):
                    yield Input(str(self.directory), id='log-export-directory')
                    yield Button(tr(message_id('logs.export.browse')), id='log-export-browse')
                yield Static(tr(message_id('logs.export.filename_label')), classes='log-export-label')
                yield Input(self.filename, id='log-export-filename')
                yield Static(tr(message_id('logs.export.save_hint')), id='log-export-hint', markup=False)
                yield Static('', id='log-export-error', markup=False)
            with Horizontal(id='log-export-actions'):
                yield Button(tr('取消'), id='log-export-cancel')
                yield Button(tr(message_id('logs.export.save')), id='log-export-save', variant='primary')

    def on_mount(self) -> None:
        self.query_one('#log-export-filename', Input).focus()

    def show_error(self, error: Exception) -> None:
        widget = self.query_one('#log-export-error', Static)
        widget.update(translated_app(self.app).tr(log_export_error(error)))
        widget.display = True
        widget.scroll_visible()

    @on(Button.Pressed, '#log-export-browse')
    def browse(self) -> None:
        try:
            directory = export_directory(self.query_one('#log-export-directory', Input).value)
        except (ValueError, RuntimeError) as exc:
            self.show_error(exc)
            return
        self.app.push_screen(ExportDirectoryScreen(directory, self.load_directory), self.directory_chosen)

    def directory_chosen(self, path: Path | None) -> None:
        if path is not None:
            self.query_one('#log-export-directory', Input).value = str(path)
            self.query_one('#log-export-error').display = False
        self.query_one('#log-export-directory', Input).focus()

    @on(Input.Submitted)
    @on(Button.Pressed, '#log-export-save')
    def save(self) -> None:
        try:
            path = log_export_path(self.query_one('#log-export-directory', Input).value,
                                   self.query_one('#log-export-filename', Input).value)
        except (ValueError, RuntimeError) as exc:
            self.show_error(exc)
            return
        self.dismiss(path)

    @on(Button.Pressed, '#log-export-cancel')
    def action_cancel(self) -> None:
        self.dismiss(None)


class ExportDirectoryScreen(ModalScreen[Path | None]):
    BINDINGS = [('escape', 'cancel', '取消')]
    CSS = """
    ExportDirectoryScreen { align: center middle; }
    #export-directory-dialog { width: 94%; max-width: 96; height: 90%;
        border: round $accent; background: $surface; padding: 0 1; }
    #export-directory-title { height: 1; text-style: bold; }
    #export-directory-host { height: auto; color: $text-muted; }
    #export-directory-nav, #export-directory-actions { height: 3; }
    #export-directory-location { width: 1fr; }
    #export-directory-nav Button { min-width: 8; width: auto; margin-left: 1; }
    #export-directory-list { height: 1fr; min-height: 2; }
    #export-directory-status { height: auto; max-height: 3; overflow-y: auto; }
    #export-directory-actions { align-horizontal: right; }
    #export-directory-actions Button { min-width: 10; margin-left: 1; }
    """

    def __init__(self, directory: Path, load_directory: DirectoryLoader) -> None:
        super().__init__(id='export-directory-screen')
        self._compositor = CjkCompositor()
        self.initial_directory = directory
        self.load_directory = load_directory
        self.current: Path | None = None
        self.entries: tuple[Path, ...] = ()
        self.loading = False
        self.cancelled = Event()

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        with Vertical(id='export-directory-dialog'):
            yield Static(tr(message_id('logs.directory.title')), id='export-directory-title', markup=False)
            yield Static(tr(message_id('logs.export.host_location', gethostname())), id='export-directory-host', markup=False)
            with Horizontal(id='export-directory-nav'):
                yield Input(str(self.initial_directory), id='export-directory-location')
                yield Button(tr(message_id('logs.directory.parent')), id='export-directory-parent')
                yield Button(tr(message_id('logs.directory.home')), id='export-directory-home')
            yield OptionList(id='export-directory-list')
            yield Static('', id='export-directory-status', markup=False)
            with Horizontal(id='export-directory-actions'):
                yield Button(tr('取消'), id='export-directory-cancel')
                yield Button(tr(message_id('logs.directory.select')), id='export-directory-use', variant='primary')

    def on_mount(self) -> None:
        self.navigate(self.initial_directory, nearest_parent=True)

    def on_unmount(self) -> None:
        self.cancelled.set()

    def navigate(self, directory: Path, *, nearest_parent: bool = False) -> None:
        if self.loading:
            return
        self.loading = True
        self.update_controls()
        self.query_one('#export-directory-status', Static).update(
            translated_app(self.app).tr(message_id('logs.directory.loading')))
        self.load_directory(self, directory, nearest_parent)

    def loaded(self, directory: Path, entries: tuple[Path, ...], error: str) -> None:
        if self.cancelled.is_set() or not self.is_mounted:
            return
        self.loading = False
        if not error:
            self.current, self.entries = directory, entries
            self.query_one('#export-directory-location', Input).value = str(directory)
            options = self.query_one('#export-directory-list', OptionList)
            options.clear_options().add_options(Text(path.name) for path in entries)
            options.focus()
        elif self.current is not None:
            # The list still describes the last successfully opened directory.
            self.query_one('#export-directory-location', Input).value = str(self.current)
        self.query_one('#export-directory-status', Static).update(
            translated_app(self.app).tr(error or message_id('logs.directory.navigation_hint')))
        self.update_controls()

    def update_controls(self) -> None:
        for widget in self.query('#export-directory-nav Button, #export-directory-location, #export-directory-list'):
            widget.disabled = self.loading
        self.query_one('#export-directory-use', Button).disabled = self.loading or self.current is None

    @on(Input.Submitted, '#export-directory-location')
    def enter_directory(self, event: Input.Submitted) -> None:
        try:
            path = export_directory(event.value)
        except (ValueError, RuntimeError) as exc:
            self.query_one('#export-directory-status', Static).update(
                translated_app(self.app).tr(log_export_error(exc)))
            return
        self.navigate(path)

    @on(OptionList.OptionSelected, '#export-directory-list')
    def open_directory(self, event: OptionList.OptionSelected) -> None:
        if not self.loading:
            self.navigate(self.entries[event.option_index])

    @on(Button.Pressed, '#export-directory-parent')
    def parent_directory(self) -> None:
        if self.current is not None:
            self.navigate(self.current.parent)

    @on(Button.Pressed, '#export-directory-home')
    def home_directory(self) -> None:
        self.navigate(Path.home())

    @on(Button.Pressed, '#export-directory-use')
    def choose(self) -> None:
        if not self.loading and self.current is not None:
            self.dismiss(self.current)

    @on(Button.Pressed, '#export-directory-cancel')
    def action_cancel(self) -> None:
        self.cancelled.set()
        self.dismiss(None)
