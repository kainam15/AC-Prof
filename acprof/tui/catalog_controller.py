"""Search and reuse local experiments through the existing pages and run launcher."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Button, Static

from acprof.tui.commands import PendingLaunch, resolve_result_path
from acprof.tui.experiment_catalog import (
    ExperimentRecord,
    config_from_record,
    scan_experiments,
)
from acprof.tui.experiment_picker import SearchPickerScreen, experiment_choice
from acprof.tui.i18n import message_id
from acprof.tui.input import BarCursorInput as Input

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class CatalogController:
    """Own catalog state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.picker_open = False

    def catalog_roots(self) -> tuple[Path, ...]:
        roots = []
        for source in (self.app.configuration.input('output-dir'), self.app.configuration.saved_settings.last_result_dir,
                       self.app.configuration.input('result-csv'), self.app.configuration.saved_settings.last_result_dir):
            if not source.strip():
                continue
            path = resolve_result_path(source, self.app.project_dir)
            roots.append(path.parent if path.suffix.lower() in {'.csv', '.json'} else path)
        roots.append(self.app.project_dir / 'results')
        return tuple(dict.fromkeys(path.resolve() for path in roots))

    def picker_loading(self, token: object, busy: bool) -> None:
        if busy:
            self.app.read_jobs.add(token)
        else:
            self.app.read_jobs.discard(token)
        if self.app.is_running and not self.app.ui_closing:
            self.app.set_busy(self.app.is_busy())

    def select_experiment(self, event: Button.Pressed) -> None:
        self.open_experiment_picker(event.button.name or 'result-csv')

    def open_experiment_picker(self, target: str) -> None:
        if not self.app.allow_operation('catalog'):
            return
        roots, token = self.catalog_roots(), object()
        def loader(cancelled):
            catalog = scan_experiments(roots, cancelled=cancelled)
            return tuple(experiment_choice(record) for record in catalog.records), catalog.warnings
        self.picker_open = True
        self.app.set_busy(True)
        self.app.push_screen(SearchPickerScreen('选择实验', ('模型', '日期', '设备', '状态', 'Run ID'), loader,
            actions=('select', 'view', 'reuse', 'resume'), scope=roots, project_dir=self.app.project_dir,
            loading_changed=lambda busy: self.picker_loading(token, busy)),
            lambda choice: self.experiment_selected(target, choice))

    def picker_closed(self) -> None:
        self.picker_open = False
        self.app.set_busy(self.app.is_busy())

    def experiment_selected(self, target: str, choice) -> None:
        self.picker_closed()
        if choice is None or not self.app.allow_operation('catalog'):
            return
        action, record = choice
        if action == 'view':
            self.view_experiment(record)
        elif action == 'select':
            path = record.directory / 'result_layers.json' if target == 'result-csv' else record.directory
            value = str(path)
            if target in {'comparison-left', 'comparison-right'}:
                # Keep CLI's quoted-semicolon grammar for directories containing separators.
                output = io.StringIO()
                csv.writer(output, delimiter=';', lineterminator='').writerow([value])
                value = output.getvalue()
            if target == 'result-csv':
                self.app.results.select_result(value)
            else:
                self.app.query_one('#' + target, Input).value = value
        elif action == 'reuse':
            try:
                config = config_from_record(record, reuse=True)
                self.app.configuration.apply_config(config)
                self.app.configuration.show_run_form('run-form')
                self.app.notify('已复用冻结配置；开始采集会创建新实验身份。')
            except ValueError as exc:
                self.app.notify(str(exc), severity='error')
        elif action == 'resume':
            self.resume_experiment(record)

    def view_experiment(self, record: ExperimentRecord) -> None:
        self.app.configuration.remember_last_used(result_dir=str(record.directory))
        self.app.results.show_result_section('overview-view')
        csv_path = record.directory / 'result_layers.json'
        if csv_path.is_file():
            self.app.results.select_result(str(csv_path))
        else:
            self.app.results.select_result(str(csv_path), summarize=False)
            self.app.localization.set_text(self.app.query_one('#result-summary', Static),
                message_id('results.evidence.no_csv', experiment_choice(record).detail))

    def resume_experiment(self, record: ExperimentRecord) -> None:
        from acprof.experiment import build_run_command
        try:
            config = config_from_record(record, reuse=False)
            command = build_run_command(config, project_dir=self.app.project_dir,
                                        python_executable=self.app.python_executable)
        except ValueError as exc:
            self.app.notify(str(exc), severity='error')
            return
        self.app.recovery.review_run_destination(PendingLaunch(tuple(command), 'run', config,
                                                   result_dir=str(record.directory)), record=record)
