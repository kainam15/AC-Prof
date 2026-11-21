"""Search existing local evidence; cancellation never starts or changes an experiment."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.cells import cell_len
from rich.text import Text
from textual import events, on, work
from textual.app import App
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from acprof.messages import join_messages
from acprof.tui.experiments.catalog import ExperimentRecord, format_experiment_created_at
from acprof.tui.i18n import message_id
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.dialogs import focus_horizontal_action
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.rendering import CjkCompositor
from acprof.tui.ui.structured import Callout, FactGrid, StatusBanner
from acprof.tui.ui.table import ResizableDataTable
from acprof.tui.ui.view_models import (
    CalloutContent,
    FactRow,
    StatusSummary,
    ViewSection,
    reason_presentation,
)

ACTION_LABELS = {'select': '选入路径', 'view': '查看结果', 'reuse': '复用配置', 'resume': '恢复 / 重试', 'use': '使用模型'}



@dataclass(frozen=True)
class PickerChoice:
    value: object
    cells: tuple[str, ...]
    detail: str
    search_text: str
    actions: frozenset[str]
    cell_tooltips: tuple[str | None, ...] = ()
    summary: StatusSummary | None = None
    callout: CalloutContent | None = None
    section: ViewSection | None = None


class PickerTable(ResizableDataTable):
    """Use rendered cell coordinates so tooltips follow scrolling and filtering."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_tooltips: dict[tuple[int, int], str] = {}

    def on_mouse_move(self, event: events.MouseMove) -> None:
        meta = event.style.meta
        text = None if meta.get('out_of_bounds') else self.cell_tooltips.get(
            (meta.get('row', -1), meta.get('column', -1)))
        self.tooltip = Text(text) if text else None

    @on(events.Leave)
    @on(events.MouseScrollUp)
    @on(events.MouseScrollDown)
    def clear_tooltip(self) -> None:
        self.tooltip = None


def _experiment_status_presentation(status: str) -> tuple[str, str]:
    labels = {'complete': '已完成', 'failed': '失败', 'interrupted': '已停止', 'running': '运行中', 'preparing': '准备中'}
    levels = {'complete': 'success', 'failed': 'error', 'interrupted': 'warning', 'running': 'info', 'preparing': 'info'}
    return labels.get(status, status), levels.get(status, 'neutral')


def experiment_choice(record: ExperimentRecord) -> PickerChoice:
    actions = {'select', 'view'}
    if record.options and not record.issues:
        actions.add('reuse')
    if record.has_recovery_state:
        actions.add('resume')
    # Keep missing or invalid historical timestamps visible as recorded.
    date_text = format_experiment_created_at(record.created_at) or record.created_at
    detail = json.dumps({'run_id': record.run_id or 'unknown', 'directories': [str(path) for path in record.aliases],
        'issues': record.issues, 'options': record.options, 'failures': record.failures,
        'runtime_validation': record.validation}, ensure_ascii=False, indent=2)
    device = str((record.metadata.get('gpu_device') or {}).get('name') or record.device)
    device_uuid = record.device if record.device.startswith('GPU-') else None
    status_label, banner_level = _experiment_status_presentation(record.status)
    summary = StatusSummary(banner_level, join_messages(' · ', (record.model_id, status_label)))
    callout: CalloutContent | None = None
    if record.issues or record.failures:
        code = str(record.failures[0].get('reason_code') or '') if record.failures else str(record.issues[0])
        tech = json.dumps({'issues': record.issues, 'failures': record.failures}, ensure_ascii=False, indent=2)
        callout = reason_presentation(code, tech)
    rows = (
        FactRow(message_id('results.view.run_id'), record.run_id or 'unknown', copyable=True),
        FactRow(message_id('common.label.status'), status_label, status=record.status),
        FactRow(message_id('common.field.date'), date_text or message_id('common.value.unknown')),
        FactRow(message_id('common.field.device'), device, detail=device_uuid or '', copyable=bool(device_uuid)),
        FactRow(message_id('results.view.directory'), str(record.directory), copyable=True),
    )
    detail_rows = (
        FactRow(message_id('results.view.profile'), record.runtime_profile or message_id('common.value.unknown')),
        FactRow(message_id('results.picker.validation'),
                str((record.validation or {}).get('status') or message_id('common.value.unknown')),
                detail=json.dumps(record.validation, ensure_ascii=False, indent=2)),
        FactRow(message_id('results.picker.aliases'), str(len(record.aliases)),
                detail='\n'.join(str(path) for path in record.aliases), copyable=True),
        FactRow(message_id('results.picker.raw_record'), 'JSON', detail=detail, copyable=True),
    )
    section = ViewSection(message_id('results.picker.details_title'), rows, detail_rows)
    return PickerChoice(record, (record.model_id, date_text, device, status_label,
        record.run_id or 'unknown'), detail, record.search_text, frozenset(actions),
        cell_tooltips=(None, None, device_uuid), summary=summary, callout=callout, section=section)




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
    #picker-filters { height: 1; margin-bottom: 1; }
    .picker-filter-pill {
        height: 1; min-height: 1; min-width: 0; width: auto;
        border: none; padding: 0 1; margin-right: 1;
        background: transparent; color: $text-muted;
    }
    .picker-filter-pill:hover, .picker-filter-pill:focus {
        border: none; background: $boost; color: $foreground; text-style: bold;
    }
    .picker-filter-pill.-active, .picker-filter-pill.-active:hover, .picker-filter-pill.-active:focus {
        border: none; background: $accent; color: $background; text-style: bold;
    }
    #picker-main { layout: vertical; height: 1fr; }
    #picker-table { height: 1fr; min-height: 5; }
    #picker-detail-scroll { height: auto; min-height: 3; max-height: 6; overflow-y: auto; }
    #picker-dialog.wide-picker #picker-main { layout: horizontal; }
    #picker-dialog.wide-picker #picker-table { width: 3fr; height: 1fr; }
    #picker-dialog.wide-picker #picker-detail-scroll {
        width: 2fr; height: 1fr; max-height: 100%;
        border-left: solid $panel; padding-left: 1; margin-left: 1;
    }
    #picker-detail-banner, #picker-detail-callout, #picker-detail-facts { display: none; }
    #picker-detail-banner { padding: 0 1; }
    #picker-detail { height: auto; color: $text-muted; }
    #picker-actions { height: 3; }
    #picker-actions Button { min-width: 7; width: 1fr; margin-right: 1; }
    """

    def __init__(self, title: str, columns: tuple[str, ...], loader: Callable,
                 *, query: str = '', actions: tuple[str, ...] = ('view', 'reuse', 'resume'),
                 loading_changed: Callable[[bool], None] = lambda busy: None, scope: tuple = (),
                 project_dir: Path | None = None, empty_message: str = '尚无本地实验记录，请检查搜索目录。',
                 quick_filters: tuple[tuple[str, str, Callable[[PickerChoice], bool]], ...] = ()):
        super().__init__()
        self._compositor = CjkCompositor()
        self.title_text, self.columns, self.loader = title, columns, loader
        self.initial_query, self.actions, self.loading_changed = query, actions, loading_changed
        self.cancelled = threading.Event()
        self.choices: tuple[PickerChoice, ...] = ()
        self.filtered: tuple[PickerChoice, ...] = ()
        self.warnings: tuple[str, ...] = ()
        self.empty_message = empty_message
        self.quick_filters = quick_filters
        self._active_filter = quick_filters[0][0] if quick_filters else ''
        self._choices_loading = True
        self._owner: App[object] | None = None
        self.scope = scope
        self.project_dir = project_dir if project_dir is not None else Path.cwd()
        self.sort_state: tuple[int, bool] | None = None

    def compose(self):
        tr = translated_app(self.app).tr
        with Vertical(id='picker-dialog'):
            yield Static(tr(self.title_text), id='picker-title', markup=False)
            yield Static(tr('正在检查搜索目录……'), id='picker-scope', markup=False)
            yield Static('', id='picker-scope-note', markup=False)
            yield Input(self.initial_query, placeholder=tr('搜索模型、日期、设备或状态'), id='picker-search')
            if self.quick_filters:
                with Horizontal(id='picker-filters'):
                    for filter_id, label, _ in self.quick_filters:
                        yield Button(
                            tr(label),
                            id=f'picker-filter-{filter_id}',
                            name=filter_id,
                            classes='picker-filter-pill' + (' -active' if filter_id == self._active_filter else ''),
                            compact=True,
                        )
            yield Static(tr('正在读取已知目录内的实验……'), id='picker-status', markup=False)
            with Vertical(id='picker-main'):
                yield PickerTable(id='picker-table', cursor_type='row', fixed_columns=1)
                with VerticalScroll(id='picker-detail-scroll'):
                    yield StatusBanner(StatusSummary('neutral', ''), id='picker-detail-banner')
                    yield Callout(CalloutContent('neutral', '', '', ''), id='picker-detail-callout')
                    yield FactGrid(ViewSection('', ()), id='picker-detail-facts', collapsible=False)
                    yield Static('', id='picker-detail', markup=False)
            with Horizontal(id='picker-actions'):
                for index, action in enumerate(self.actions):
                    yield Button(
                        tr(ACTION_LABELS[action]),
                        id='picker-' + action,
                        classes='picker-action',
                        variant='primary' if index == 0 else 'default',
                        disabled=True,
                    )
                yield Button(tr('关闭弹窗'), id='picker-close')

    def on_mount(self):
        self._owner = self.app
        self._fit_detail_height()
        table = self.query_one('#picker-table', PickerTable)
        tr = translated_app(self.app).tr
        for index, column in enumerate(self.columns):
            table.add_column(tr(column), key=str(index))
        self.loading_changed(True)
        self.load_choices()
        self.query_one('#picker-search', Input).focus()

    def on_resize(self) -> None:
        self._fit_detail_height()

    def _fit_detail_height(self) -> None:
        if not self.is_attached or self._pruning or self.cancelled.is_set():
            return
        wide = self.app.size.width >= 105
        self.query_one('#picker-dialog').set_class(wide, 'wide-picker')
        detail = self.query_one('#picker-detail-scroll')
        if wide:
            detail.styles.max_height = None
        else:
            detail.styles.max_height = max(5, min(11, self.app.size.height // 3 - 2))

    @work(thread=True, exclusive=True, exit_on_error=False)
    def load_choices(self):
        from acprof.host.run_state import RunStateError, offline_analysis_lock

        roots: tuple[Path, ...] = ()
        skipped: tuple[str, ...] = ()
        try:
            with offline_analysis_lock():
                roots, skipped = self.read_scope()
                choices, warnings = self.loader(self.cancelled.is_set)
        except RunStateError as exc:
            choices, warnings = (), (message_id("results.picker.lock_unavailable", str(exc)),)
        except InterruptedError:
            choices, warnings = (), ()
        except Exception as exc:
            choices, warnings = (), (str(exc),)
        try:
            owner = self._owner
            if owner is not None:
                owner.call_from_thread(self.loaded, choices, warnings, roots, skipped)
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
                    skipped.append(message_id('results.picker.invalid_directory', str(path)))
            except (OSError, RuntimeError, ValueError) as exc:
                skipped.append(message_id('results.picker.inspection_failed', str(path), str(exc)))
        return tuple(roots), tuple(skipped)

    def show_scope(self, roots: tuple[Path, ...], skipped: tuple[str, ...]) -> None:
        tr = translated_app(self.app).tr
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
        note.update(tr(message_id('results.picker.skipped_roots_count', len(skipped))))
        note.tooltip = scope.tooltip

    def loaded(self, choices, warnings, roots, skipped):
        self.loading_changed(False)
        if not self.is_attached or self._pruning or self.cancelled.is_set():
            return
        self.choices, self.warnings = tuple(choices), tuple(warnings)
        self._choices_loading = False
        self.show_scope(roots, skipped)
        self.filter_choices()

    @on(Input.Changed, '#picker-search')
    def search_changed(self):
        self.filter_choices()

    @on(Button.Pressed, '.picker-filter-pill')
    def quick_filter_selected(self, event: Button.Pressed) -> None:
        selected_id = event.button.name or ''
        if selected_id == self._active_filter:
            return
        self._active_filter = selected_id
        for pill in self.query('.picker-filter-pill').results(Button):
            pill.set_class(pill.name == selected_id, '-active')
        self.filter_choices()

    def _update_column_headers(self) -> None:
        tr = translated_app(self.app).tr
        table = self.query_one('#picker-table', PickerTable)
        for index, title in enumerate(self.columns):
            if index >= len(table.ordered_columns):
                continue
            label = tr(title)
            if self.sort_state and self.sort_state[0] == index:
                label = f"{label} {'↓' if self.sort_state[1] else '↑'}"
            column = table.ordered_columns[index]
            column.label = Text(label)
            if column.auto_width:
                row_width = max(
                    (
                        cell_len(line)
                        for choice in self.filtered
                        if index < len(choice.cells)
                        for line in tr(choice.cells[index]).splitlines()
                    ),
                    default=0,
                )
                column.content_width = max(cell_len(label), row_width)
        table._update_count += 1
        table._clear_caches()
        table._update_dimensions(())
        table.refresh(layout=True)

    @on(DataTable.HeaderSelected, '#picker-table')
    def sort_column(self, event: DataTable.HeaderSelected) -> None:
        if not self.is_attached or self._pruning or self.cancelled.is_set():
            return
        if self._choices_loading:
            return
        table = self.query_one('#picker-table', PickerTable)
        column_index = table.get_column_index(event.column_key)
        reverse = not self.sort_state[1] if self.sort_state and self.sort_state[0] == column_index else False
        self.sort_state = (column_index, reverse)
        self.filter_choices()

    def filter_choices(self):
        if not self.is_attached or self._pruning or self.cancelled.is_set():
            return
        if self._choices_loading:
            self.show_choice()
            return
        tr = translated_app(self.app).tr
        current = self.selected()
        current_value = current.value if current is not None else None
        terms = self.query_one('#picker-search', Input).value.casefold().split()
        predicate = next((fn for fid, _, fn in self.quick_filters if fid == self._active_filter), None)
        pool = [choice for choice in self.choices if predicate(choice)] if predicate is not None else list(self.choices)
        matched = [choice for choice in pool if all(
            term in (choice.search_text + ' ' + ' '.join(tr(cell) for cell in choice.cells)).casefold()
            for term in terms)]
        if self.sort_state is not None:
            column_index, reverse = self.sort_state
            matched.sort(
                key=lambda choice: tr(choice.cells[column_index]).casefold() if column_index < len(choice.cells) else '',
                reverse=reverse,
            )
        self.filtered = tuple(matched)
        table = self.query_one('#picker-table', PickerTable)
        table.clear()
        table.clear_tooltip()
        table.cell_tooltips = {(row, column): tooltip for row, choice in enumerate(self.filtered)
                               for column, tooltip in enumerate(choice.cell_tooltips) if tooltip}
        for index, choice in enumerate(self.filtered):
            table.add_row(*(tr(cell) for cell in choice.cells), key=str(index))
        if self.filtered and current_value is not None:
            target = next((index for index, choice in enumerate(self.filtered) if choice.value == current_value), 0)
            table.move_cursor(row=target, animate=False)
        self.query_one('#picker-status', Static).update(f'{len(self.filtered)} / {len(self.choices)}' +
            (' · ' + '; '.join(tr(warning) for warning in self.warnings) if self.warnings else ''))
        self._update_column_headers()
        self.show_choice()

    def selected(self) -> PickerChoice | None:
        if not self.is_attached or self._pruning or self.cancelled.is_set():
            return None
        row = self.query_one('#picker-table', DataTable).cursor_row
        return self.filtered[row] if 0 <= row < len(self.filtered) else None

    @on(DataTable.RowHighlighted, '#picker-table')
    def row_highlighted(self):
        self.show_choice()

    @on(DataTable.RowSelected, '#picker-table')
    def activate_selected_row(self) -> None:
        choice = self.selected()
        if choice is None:
            return
        for action in self.actions:
            if action in choice.actions and not self.query_one('#picker-' + action, Button).disabled:
                self.dismiss((action, choice.value))
                return

    def show_choice(self):
        if not self.is_attached or self._pruning or self.cancelled.is_set():
            return
        tr = translated_app(self.app).tr
        choice = self.selected()
        banner = self.query_one('#picker-detail-banner', StatusBanner)
        callout = self.query_one('#picker-detail-callout', Callout)
        facts = self.query_one('#picker-detail-facts', FactGrid)
        plain = self.query_one('#picker-detail', Static)
        has_structured = choice is not None and (
            choice.summary is not None or choice.callout is not None or choice.section is not None
        )
        if choice is not None and has_structured:
            plain.update('')
            plain.display = False
            banner.display = choice.summary is not None
            if choice.summary is not None:
                banner.update_summary(choice.summary)
            callout.display = choice.callout is not None
            if choice.callout is not None:
                callout.update_content(choice.callout)
            facts.display = choice.section is not None
            if choice.section is not None:
                facts.update_section(choice.section)
        else:
            banner.display = False
            callout.display = False
            facts.display = False
            plain.display = True
            if choice:
                detail = choice.detail
            elif self._choices_loading:
                detail = tr(message_id('results.picker.loading'))
            elif self.choices:
                detail = tr(message_id('common.search.no_matches'))
            elif self.warnings:
                detail = tr(message_id('results.picker.read_failed'))
            else:
                detail = tr(self.empty_message)
            plain.update(detail)
        for action in self.actions:
            self.query_one('#picker-' + action, Button).disabled = choice is None or action not in choice.actions

    def on_key(self, event: events.Key) -> None:
        if not self.is_attached or self._pruning or self.cancelled.is_set():
            return
        if self.focused is self.query_one('#picker-search', Input) and event.key == 'down':
            if self.filtered:
                self.query_one('#picker-table', PickerTable).focus()
                event.prevent_default()
                event.stop()
        elif focus_horizontal_action(self, '#picker-actions', event.key):
            event.prevent_default()
            event.stop()

    @on(Button.Pressed, '.picker-action')
    def choose(self, event: Button.Pressed):
        choice = self.selected()
        if event.button.id is None:
            return
        action = event.button.id.removeprefix('picker-')
        if choice and action in choice.actions:
            self.dismiss((action, choice.value))

    @on(Button.Pressed, '#picker-close')
    def action_close(self):
        self.cancelled.set()
        self.dismiss(None)
