"""Search existing local evidence; cancellation never starts or changes an experiment."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from textual import on, work
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from acprof.tui.experiment_catalog import ExperimentRecord
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.rendering import CjkCompositor

ACTION_LABELS = {'select': '选入路径', 'view': '查看结果', 'reuse': '复用配置', 'resume': '继续实验', 'use': '使用模型'}



@dataclass(frozen=True)
class PickerChoice:
    value: object
    cells: tuple[str, ...]
    detail: str
    search_text: str
    actions: frozenset[str]


def experiment_choice(record: ExperimentRecord) -> PickerChoice:
    import json
    actions = {'select', 'view'}
    if record.options and not record.issues:
        actions.add('reuse')
    if record.can_resume:
        actions.add('resume')
    detail = json.dumps({'run_id': record.run_id or 'unknown', 'directories': [str(path) for path in record.aliases],
        'issues': record.issues, 'options': record.options, 'failures': record.failures,
        'runtime_validation': record.validation}, ensure_ascii=False, indent=2)
    return PickerChoice(record, (record.model_id, record.created_at, record.device, {'complete': '已完成', 'failed': '失败',
        'interrupted': '已停止', 'running': '运行中', 'preparing': '准备中'}.get(record.status, record.status),
        record.run_id or 'unknown'), detail, record.search_text, frozenset(actions))




class SearchPickerScreen(ModalScreen[tuple[str, object] | None]):
    BINDINGS = [('escape', 'close', '关闭')]
    CSS = """
    SearchPickerScreen { align: center middle; }
    #picker-dialog { width: 96%; max-width: 140; height: 92%; border: round $accent; background: $surface; padding: 0 1; }
    #picker-title { height: 1; text-style: bold; }
    #picker-scope { height: 1; color: $text-muted; }
    #picker-status { height: auto; max-height: 2; color: $text-muted; }
    #picker-search { height: 3; }
    #picker-table { height: 1fr; min-height: 3; }
    #picker-detail { height: 5; overflow: auto; color: $text-muted; }
    #picker-actions { height: 3; }
    #picker-actions Button { min-width: 7; width: 1fr; margin-right: 1; }
    """

    def __init__(self, title: str, columns: tuple[str, ...], loader: Callable,
                 *, query: str = '', actions: tuple[str, ...] = ('view', 'reuse', 'resume'),
                 loading_changed: Callable[[bool], None] = lambda busy: None, scope: tuple = ()):
        super().__init__()
        self._compositor = CjkCompositor()
        self.title_text, self.columns, self.loader = title, columns, loader
        self.initial_query, self.actions, self.loading_changed = query, actions, loading_changed
        self.cancelled = threading.Event()
        self.choices: tuple[PickerChoice, ...] = ()
        self.filtered: tuple[PickerChoice, ...] = ()
        self.warnings: tuple[str, ...] = ()
        self._owner = None
        self.scope = scope

    def compose(self):
        tr = self.app.tr
        with Vertical(id='picker-dialog'):
            yield Static(tr(self.title_text), id='picker-title', markup=False)
            scope = tr('搜索范围') + ': ' + ' · '.join(map(str, self.scope))
            scope_widget = Static(scope, id='picker-scope', markup=False)
            scope_widget.tooltip = scope
            yield scope_widget
            yield Input(self.initial_query, placeholder=tr('搜索模型、日期、设备或状态'), id='picker-search')
            yield Static(tr('正在读取已知目录内的实验……'), id='picker-status', markup=False)
            yield DataTable(id='picker-table', cursor_type='row', fixed_columns=1)
            yield Static('', id='picker-detail', markup=False)
            with Horizontal(id='picker-actions'):
                for action in self.actions:
                    yield Button(tr(ACTION_LABELS[action]), id='picker-' + action, classes='picker-action', disabled=True)
                yield Button(tr('关闭弹窗'), id='picker-close')

    def on_mount(self):
        self._owner = self.app
        self.query_one('#picker-table', DataTable).add_columns(*(self.app.tr(column) for column in self.columns))
        self.loading_changed(True)
        self.load_choices()
        self.query_one('#picker-search', Input).focus()

    @work(thread=True, exclusive=True, exit_on_error=False)
    def load_choices(self):
        try:
            choices, warnings = self.loader(self.cancelled.is_set)
        except InterruptedError:
            choices, warnings = (), ()
        except Exception as exc:
            choices, warnings = (), (str(exc),)
        try:
            self._owner.call_from_thread(self.loaded, choices, warnings)
        except RuntimeError:
            # The app exited while an already-cancelled bounded read completed.
            pass

    def on_unmount(self):
        self.cancelled.set()

    def loaded(self, choices, warnings):
        self.loading_changed(False)
        if not self.is_mounted or self.cancelled.is_set():
            return
        self.choices, self.warnings = tuple(choices), tuple(warnings)
        self.filter_choices()

    @on(Input.Changed, '#picker-search')
    def search_changed(self):
        self.filter_choices()

    def filter_choices(self):
        terms = self.query_one('#picker-search', Input).value.casefold().split()
        self.filtered = tuple(choice for choice in self.choices if all(
            term in (choice.search_text + ' ' + ' '.join(self.app.tr(cell) for cell in choice.cells)).casefold()
            for term in terms))
        table = self.query_one('#picker-table', DataTable)
        table.clear()
        for index, choice in enumerate(self.filtered):
            table.add_row(*(self.app.tr(cell) for cell in choice.cells), key=str(index))
        self.query_one('#picker-status', Static).update(f'{len(self.filtered)} / {len(self.choices)}' +
            (' · ' + '; '.join(self.warnings) if self.warnings else ''))
        self.show_choice()

    def selected(self) -> PickerChoice | None:
        row = self.query_one('#picker-table', DataTable).cursor_row
        return self.filtered[row] if 0 <= row < len(self.filtered) else None

    @on(DataTable.RowHighlighted, '#picker-table')
    def row_highlighted(self):
        self.show_choice()

    def show_choice(self):
        choice = self.selected()
        self.query_one('#picker-detail', Static).update(choice.detail if choice else self.app.tr('没有匹配的本地记录'))
        for action in self.actions:
            self.query_one('#picker-' + action, Button).disabled = choice is None or action not in choice.actions

    @on(Button.Pressed, '.picker-action')
    def choose(self, event: Button.Pressed):
        choice = self.selected()
        action = event.button.id.removeprefix('picker-')
        if choice and action in choice.actions:
            self.dismiss((action, choice.value))

    @on(Button.Pressed, '#picker-close')
    def action_close(self):
        self.cancelled.set()
        self.dismiss(None)
