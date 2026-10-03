"""Search existing local evidence; cancellation never starts or changes an experiment."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.text import Text
from textual import on, work
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from acprof.messages import message
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
    #picker-scope-note { display: none; height: 1; color: $text-muted; }
    #picker-status { height: auto; max-height: 2; color: $text-muted; }
    #picker-search { height: 3; }
    #picker-table { height: 1fr; min-height: 3; }
    #picker-detail { height: 5; overflow: auto; color: $text-muted; }
    #picker-actions { height: 3; }
    #picker-actions Button { min-width: 7; width: 1fr; margin-right: 1; }
    """

    def __init__(self, title: str, columns: tuple[str, ...], loader: Callable,
                 *, query: str = '', actions: tuple[str, ...] = ('view', 'reuse', 'resume'),
                 loading_changed: Callable[[bool], None] = lambda busy: None, scope: tuple = (),
                 project_dir: Path | None = None, empty_message: str = '尚无本地实验记录，请检查搜索目录。'):
        super().__init__()
        self._compositor = CjkCompositor()
        self.title_text, self.columns, self.loader = title, columns, loader
        self.initial_query, self.actions, self.loading_changed = query, actions, loading_changed
        self.cancelled = threading.Event()
        self.choices: tuple[PickerChoice, ...] = ()
        self.filtered: tuple[PickerChoice, ...] = ()
        self.warnings: tuple[str, ...] = ()
        self.empty_message = empty_message
        self._choices_loading = True
        self._owner = None
        self.scope = scope
        self.project_dir = project_dir if project_dir is not None else Path.cwd()

    def compose(self):
        tr = self.app.tr
        with Vertical(id='picker-dialog'):
            yield Static(tr(self.title_text), id='picker-title', markup=False)
            yield Static(tr('正在检查搜索目录……'), id='picker-scope', markup=False)
            yield Static('', id='picker-scope-note', markup=False)
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
        roots, skipped = (), ()
        try:
            roots, skipped = self.read_scope()
            choices, warnings = self.loader(self.cancelled.is_set)
        except InterruptedError:
            choices, warnings = (), ()
        except Exception as exc:
            choices, warnings = (), (str(exc),)
        try:
            self._owner.call_from_thread(self.loaded, choices, warnings, roots, skipped)
        except RuntimeError:
            # The app exited while an already-cancelled bounded read completed.
            pass

    def on_unmount(self):
        self.cancelled.set()

    def read_scope(self) -> tuple[tuple[Path, ...], tuple[str, ...]]:
        """Inspect display paths in the worker; keep explicit scan roots intact."""
        roots, skipped, seen = [], [], set()
        for source in self.scope:
            if self.cancelled.is_set():
                raise InterruptedError('search scope inspection cancelled')
            path = Path(source)
            try:
                path = path.expanduser()
                if not path.is_absolute():
                    path = self.project_dir / path
                path = path.resolve()
                if path in seen:
                    continue
                seen.add(path)
                if path.is_dir():
                    roots.append(path)
                else:
                    skipped.append(message('目录不存在或不是目录：{0}', str(path)))
            except (OSError, RuntimeError, ValueError) as exc:
                skipped.append(message('无法检查目录 {0}：{1}', str(path), str(exc)))
        return tuple(roots), tuple(skipped)

    def show_scope(self, roots: tuple[Path, ...], skipped: tuple[str, ...]) -> None:
        tr = self.app.tr
        # Parent roots summarize the display only: bounded scans may need an
        # explicitly selected child beyond the parent's depth or record boundary.
        summary = [path for path in roots if not any(parent in path.parents for parent in roots)]
        labels = [str(path.relative_to(self.project_dir)) if path.is_relative_to(self.project_dir)
                  else str(path) for path in summary]
        scope = self.query_one('#picker-scope', Static)
        scope.update(tr('搜索范围') + ': ' + (' · '.join(labels) or tr('无可用目录')))
        details = [tr('搜索范围') + ':', *(str(path) for path in roots)]
        if skipped:
            details.extend(('', tr('已跳过的搜索路径') + ':', *(tr(reason) for reason in skipped)))
        scope.tooltip = Text('\n'.join(details))
        note = self.query_one('#picker-scope-note', Static)
        note.display = bool(skipped)
        note.update(tr(message('已跳过 {0} 个不可用目录；悬停查看详情。', len(skipped))))
        note.tooltip = scope.tooltip

    def loaded(self, choices, warnings, roots, skipped):
        self.loading_changed(False)
        if not self.is_mounted or self.cancelled.is_set():
            return
        self.choices, self.warnings = tuple(choices), tuple(warnings)
        self._choices_loading = False
        self.show_scope(roots, skipped)
        self.filter_choices()

    @on(Input.Changed, '#picker-search')
    def search_changed(self):
        self.filter_choices()

    def filter_choices(self):
        if self._choices_loading:
            self.show_choice()
            return
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
        if choice:
            detail = choice.detail
        elif self._choices_loading:
            detail = self.app.tr('正在读取已知目录内的实验……')
        elif self.choices:
            detail = self.app.tr('没有匹配项，请修改搜索词。')
        elif self.warnings:
            detail = self.app.tr('未读取到本地记录，请查看上方提示。')
        else:
            detail = self.app.tr(self.empty_message)
        self.query_one('#picker-detail', Static).update(detail)
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
