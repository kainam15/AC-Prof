"""Search and reuse local experiments through the existing pages and run launcher."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING

from textual import on
from textual.message_pump import MessagePump
from textual.widgets import Button, Static

from acprof.messages import message
from acprof.tui.commands import PendingLaunch, format_command, resolve_result_path
from acprof.tui.experiment_catalog import (
    ExperimentRecord,
    config_from_record,
    resume_command,
    scan_experiments,
)
from acprof.tui.experiment_picker import SearchPickerScreen, experiment_choice
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.views import ConfirmActionScreen

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class CatalogActions(MessagePump):
    """Each refresh is bounded and remains owned until its worker has finished."""

    def _catalog_roots(self: AcprofTui) -> tuple[Path, ...]:
        from acprof.tui.app import PROJECT_DIR
        roots = []
        for source in (self._input('output-dir'), self._saved_settings.last_result_dir,
                       self._input('result-csv'), self._input('result-dir'), self._input('report-source')):
            if not source.strip():
                continue
            path = resolve_result_path(source, PROJECT_DIR)
            roots.append(path.parent if path.suffix.lower() in {'.csv', '.json'} else path)
        roots.append(PROJECT_DIR / 'results')
        return tuple(dict.fromkeys(path.resolve() for path in roots))

    def _picker_loading(self: AcprofTui, token: object, busy: bool) -> None:
        if busy:
            self._read_jobs.add(token)
        else:
            self._read_jobs.discard(token)
        if self.is_running and not self._ui_closing:
            self._set_busy(self._is_busy())

    @on(Button.Pressed, '.experiment-picker')
    def select_experiment(self: AcprofTui, event: Button.Pressed) -> None:
        self._open_experiment_picker(event.button.name or 'result-csv')

    def _open_experiment_picker(self: AcprofTui, target: str) -> None:
        if not self._allow_operation('catalog'):
            return
        roots, token = self._catalog_roots(), object()
        def loader(cancelled):
            catalog = scan_experiments(roots, cancelled=cancelled)
            return tuple(experiment_choice(record) for record in catalog.records), catalog.warnings
        self._picker_open = True
        self._set_busy(True)
        self.push_screen(SearchPickerScreen('选择实验', ('模型', '日期', '设备', '状态', 'Run ID'), loader,
            actions=('select', 'view', 'reuse', 'resume'), scope=roots,
            loading_changed=lambda busy: self._picker_loading(token, busy)),
            lambda choice: self._experiment_selected(target, choice))

    def _picker_closed(self: AcprofTui) -> None:
        self._picker_open = False
        self._set_busy(self._is_busy())

    def _experiment_selected(self: AcprofTui, target: str, choice) -> None:
        self._picker_closed()
        if choice is None or not self._allow_operation('catalog'):
            return
        action, record = choice
        if action == 'view':
            self._view_experiment(record)
        elif action == 'select':
            path = record.directory / 'result_all.csv' if target == 'result-csv' else record.directory
            value = str(path)
            if target in {'comparison-left', 'comparison-right'}:
                # Keep CLI's quoted-semicolon grammar for directories containing separators.
                output = io.StringIO()
                csv.writer(output, delimiter=';', lineterminator='').writerow([value])
                value = output.getvalue()
            self.query_one('#' + target, Input).value = value
        elif action == 'reuse':
            try:
                config = config_from_record(record, reuse=True)
                self._apply_config(config)
                self._show_run_form('run-form')
                self.notify('已复用冻结配置；开始采集会创建新实验身份。')
            except ValueError as exc:
                self.notify(str(exc), severity='error')
        elif action == 'resume':
            self._resume_experiment(record)

    def _view_experiment(self: AcprofTui, record: ExperimentRecord) -> None:
        self._remember_last_used(result_dir=str(record.directory))
        self._activate_tab('plot-tab')
        csv_path = record.directory / 'result_all.csv'
        if csv_path.is_file():
            self.query_one('#result-csv', Input).value = str(csv_path)
            self._update_result_summary(str(csv_path), notify=False)
        else:
            self._set_text(self.query_one('#result-summary', Static),
                message('尚无结果 CSV；保留的实验状态与失败证据：\n{0}', experiment_choice(record).detail))

    def _resume_experiment(self: AcprofTui, record: ExperimentRecord) -> None:
        from acprof.tui.app import PROJECT_DIR, PYTHON_EXECUTABLE
        try:
            command = resume_command(record, python_executable=PYTHON_EXECUTABLE)
            config = config_from_record(record, reuse=False)
            self._apply_config(config)
            self._show_run_form('run-form')
        except ValueError as exc:
            self.notify(str(exc), severity='error')
            return
        self._pending_launch = PendingLaunch(command, 'run', config, result_dir=str(record.directory))
        preview = format_command(command, project_dir=PROJECT_DIR)
        self._set_text(self.query_one('#command-preview', Static), preview)
        self.push_screen(ConfirmActionScreen('继续未完成实验',
            message('将使用冻结配置续跑。后端会核对主机、依赖、源码、输入与已有产物；不兼容时必须创建新实验。\n\n{0}', preview),
            '继续未完成实验'), self._confirmed_launch)
