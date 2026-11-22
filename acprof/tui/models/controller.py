"""Connect the existing model input to local evidence, only while idle."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from textual.suggester import SuggestFromList
from textual.widgets import Select, Static

from acprof.experiments.configuration import RunConfigError
from acprof.models.evidence import pinned_revision
from acprof.tui.experiments.catalog import scan_experiments
from acprof.tui.experiments.picker import SearchPickerScreen
from acprof.tui.i18n import message_id
from acprof.tui.models.preview import MODEL_OPTIONS, inspect_summary, preview_text
from acprof.tui.ui.input import BarCursorInput as Input

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class ModelController:
    """Own models state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.preview_arguments: tuple[str, ...] = ()
        self.gguf_revision = ""
        self.preview_pending = False
        self.preview_edited = False
        self.preview_worker: asyncio.Task[None] | None = None

    @property
    def _form_active(self) -> bool:
        return self.app.is_running and self.app.form_ready and not self.app.ui_closing

    def current_arguments(self) -> tuple[str, ...]:
        form = self.app.configuration
        arguments = [form.input("model"), "--model-source", form.select("model-source")]
        for option in MODEL_OPTIONS:
            if value := form.input(option):
                arguments.extend((f"--{option}", value))
        for select in ("task-family", "gpus"):
            if value := form.select(select):
                arguments.extend((f"--{select}", value))
        return tuple(arguments)

    def choose_gguf_file(self, value) -> None:
        if (self.gguf_revision and self._form_active and self.current_arguments() == self.preview_arguments
                and self.app.operation_state().allows("check")):
            form = self.app.configuration
            form.select_model_candidate(form.input("model"), form.select("model-source"),
                                        self.gguf_revision, "" if value is Select.NULL else str(value))

    def _show_preview(self, text: str, summary: dict | None = None) -> None:
        summary = summary or {}
        self.gguf_revision = ""
        selector = self.app.query_one("#gguf-file", Select)
        paths = [item["path"] for item in summary.get("gguf_candidates", []) if item["selectable"]]
        selector.display = bool(paths and pinned_revision(summary.get("revision") or ""))
        if selector.display:
            self.gguf_revision = summary["revision"]
            with selector.prevent(Select.Changed):
                selector.set_options([(Text(path), path) for path in paths])
                explicit = self.app.configuration.input("model-variant")
                selector.value = explicit if explicit in paths else Select.NULL

        widget = self.app.query_one("#model-preview", Static)
        self.app.localization.set_text(widget, text)
        widget.display = bool(text)

    def model_edited(self) -> None:
        if not self._form_active:
            return
        arguments = self.current_arguments()
        if arguments == self.preview_arguments:
            return
        self.preview_arguments = arguments
        self.preview_pending = False
        self.preview_edited = True
        self._show_preview("")
        if self.preview_worker and not self.preview_worker.cancelling():
            self.preview_worker.cancel()

    def inspect_model(self, *, submitted: bool = False) -> None:
        """Submit completed edits, preserving a typed but unfinished ID during other checks."""
        if not self._form_active or self.app.execution.session.kind:
            return
        self.model_edited()
        if submitted and self.preview_worker and not self.preview_edited:
            return
        if not (self.preview_edited or submitted) or not self.preview_arguments[0]:
            return
        self.preview_edited = False
        self.preview_pending = True
        self.resume_preview()

    def resume_preview(self) -> None:
        if (not self.preview_pending or self.preview_worker or not self._form_active
                or not self.app.operation_state().allows("check")):
            return
        self.preview_pending = False
        self._show_preview(message_id("models.preview.loading"))
        self.preview_worker = asyncio.create_task(self._read_preview(self.preview_arguments))
        self.preview_worker.add_done_callback(self._preview_finished)
        self.app.set_busy(False)

    async def _read_preview(self, arguments: tuple[str, ...]) -> None:
        from acprof.host.run_state import RunStateError, offline_analysis_lock
        summary = None
        try:
            with offline_analysis_lock():
                summary = await inspect_summary(arguments, self.app.python_executable, self.app.project_dir)
            text = preview_text(summary)
        except TimeoutError:
            text = message_id("models.preview.timeout")
        except (OSError, RunStateError, ValueError) as exc:
            text = message_id("models.preview.failure", str(exc))
        if self._form_active and arguments == self.current_arguments():
            self._show_preview(text, summary)

    def _preview_finished(self, worker: asyncio.Task[None]) -> None:
        self.preview_worker = None
        if not worker.cancelled():
            worker.result()
        if self._form_active:
            self.app.set_busy(False)

    async def close_preview(self) -> None:
        self.preview_pending = False
        if worker := self.preview_worker:
            if not worker.cancelling():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    def open_model_candidates(self) -> None:
        if not self.app.allow_operation('catalog'):
            return
        try:
            config = self.app.configuration.collect_config(allow_empty_model=True)
        except RunConfigError as exc:
            self.app.configuration.show_config_error(exc)
            return
        from acprof.host.model_store.store import store_root
        roots, token = self.app.catalog.catalog_roots(), object()
        cache_root = Path(config.model_store).expanduser() if config.model_store else store_root()
        if not cache_root.is_absolute():
            cache_root = self.app.project_dir / cache_root
        models: set[str] = set()
        def loader(cancelled):
            from acprof.tui.models.candidates import (
                cached_candidates,
                candidate_from_record,
                coverage_candidates,
                current_conditions,
                current_context,
                model_choice,
            )
            catalog = scan_experiments(roots, cancelled=cancelled)
            warnings, candidates = list(catalog.warnings), []
            if catalog.records:
                from acprof.installation import resource_root
                from acprof.runtime.profiles import PROFILES
                needs_platform = any(profile.adapter == 'family-default' and profile.environment.platform.torch_version
                                     for record in catalog.records if (profile := PROFILES.get(record.runtime_profile)))
                context = current_context(config, resource_root(), needs_torch_platform=needs_platform, cancelled=cancelled)
                warnings.extend(context['warnings'])
                for record in catalog.records:
                    if cancelled():
                        raise InterruptedError('model candidate scan cancelled')
                    try:
                        candidates.append(candidate_from_record(record, current_conditions(record, config, context, resource_root())))
                    except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
                        warnings.append(f'{record.directory}: {exc}')
            for report in catalog.coverage_reports:
                if cancelled():
                    raise InterruptedError('model candidate scan cancelled')
                try:
                    candidates.extend(coverage_candidates(report))
                except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
                    warnings.append(f'{report}: {exc}')
            cached, cache_warnings = cached_candidates(cache_root, cancelled=cancelled)
            warnings.extend(cache_warnings)
            candidates.extend(cached)
            models.update(candidate.model_id for candidate in candidates)
            return tuple(model_choice(candidate) for candidate in candidates), tuple(warnings)
        self.app.catalog.picker_open = True
        self.app.set_busy(True)
        quick_filters = (
            ('all', message_id('results.picker.filter_all'), lambda _choice: True),
            ('validated', message_id('models.candidates.filter_validated'), lambda choice: getattr(choice.value, 'status', '') == 'verified_current'),
            ('cached', message_id('models.candidates.filter_cached'), lambda choice: getattr(choice.value, 'status', '') == 'cached_unverified'),
            ('revalidate', message_id('models.candidates.filter_revalidate'),
             lambda choice: getattr(choice.value, 'status', '') not in {'verified_current', 'cached_unverified'}),
        )
        self.app.push_screen(SearchPickerScreen('模型候选', ('模型', 'Revision', '验证状态', '设备'), loader,
            query=self.app.configuration.input('model'), actions=('use', 'view'), scope=(*roots, cache_root), project_dir=self.app.project_dir,
            empty_message='尚无本地模型记录，可直接输入 Hugging Face 模型 ID。',
            quick_filters=quick_filters,
            loading_changed=lambda busy: self.app.catalog.picker_loading(token, busy)),
            lambda choice: self.model_candidate_selected(choice, models))

    def model_candidate_selected(self, choice, models: set[str]) -> None:
        self.app.catalog.picker_closed()
        if choice is None or not self.app.allow_operation('catalog'):
            return
        self.app.query_one('#model', Input).suggester = SuggestFromList(sorted(models), case_sensitive=False)
        action, candidate = choice
        if action == 'use':
            revision = candidate.revision if pinned_revision(candidate.revision) else ''
            self.app.configuration.select_model_candidate(candidate.model_id, candidate.model_source, revision, candidate.model_variant)
            self.app.notify(message_id('models.candidates.selected_notice', candidate.reason))
        elif action == 'view' and candidate.record:
            self.app.catalog.view_experiment(candidate.record)
        elif action == 'view' and candidate.report:
            self.app.activate_tab('results-tab')
            self.app.results.open_report(str(candidate.report))
