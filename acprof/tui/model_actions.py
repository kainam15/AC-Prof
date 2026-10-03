"""Connect the existing model input to local evidence, only while idle."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual import on
from textual.message_pump import MessagePump
from textual.suggester import SuggestFromList
from textual.widgets import Button

from acprof.experiment import RunConfigError
from acprof.messages import message
from acprof.model_evidence import pinned_revision
from acprof.tui.experiment_catalog import scan_experiments
from acprof.tui.experiment_picker import SearchPickerScreen
from acprof.tui.input import BarCursorInput as Input

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class ModelActions(MessagePump):
    @on(Button.Pressed, '#model-candidates')
    def open_model_candidates(self: AcprofTui) -> None:
        if not self._allow_operation('catalog'):
            return
        from acprof.tui.app import PROJECT_DIR
        try:
            config = self._collect_config(allow_empty_model=True)
        except RunConfigError as exc:
            self._show_config_error(exc)
            return
        from acprof.host.model_store import store_root
        roots, token = self._catalog_roots(), object()
        cache_root = Path(config.model_store).expanduser() if config.model_store else store_root()
        if not cache_root.is_absolute():
            cache_root = PROJECT_DIR / cache_root
        models: set[str] = set()
        def loader(cancelled):
            from acprof.tui.model_candidates import (
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
                from acprof.runtime_profiles import PROFILES
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
        self._picker_open = True
        self._set_busy(True)
        self.push_screen(SearchPickerScreen('模型候选', ('模型', 'Revision', '验证状态', '设备'), loader,
            query=self._input('model'), actions=('use', 'view'), scope=(*roots, cache_root), project_dir=PROJECT_DIR,
            empty_message='尚无本地模型记录，可直接输入 Hugging Face 模型 ID。',
            loading_changed=lambda busy: self._picker_loading(token, busy)),
            lambda choice: self._model_candidate_selected(choice, models))

    def _model_candidate_selected(self: AcprofTui, choice, models: set[str]) -> None:
        self._picker_closed()
        if choice is None or not self._allow_operation('catalog'):
            return
        self.query_one('#model', Input).suggester = SuggestFromList(sorted(models), case_sensitive=False)
        action, candidate = choice
        if action == 'use':
            self._candidate_pin = (candidate.model_id, candidate.revision)
            self.query_one('#model', Input).value = candidate.model_id
            self.query_one('#revision', Input).value = candidate.revision if pinned_revision(candidate.revision) else ''
            self._refresh_command_preview()
            self.notify(message('模型候选已填入；开始前仍会执行当前条件验证。{0}', candidate.reason))
        elif action == 'view' and candidate.record:
            self._view_experiment(candidate.record)
        elif action == 'view' and candidate.report:
            self._activate_tab('reports-tab')
            self._open_report(str(candidate.report))

    @on(Input.Changed, '#model')
    def candidate_model_changed(self: AcprofTui, event: Input.Changed) -> None:
        selected = getattr(self, '_candidate_pin', None)
        if selected and event.value != selected[0]:
            if self._input('revision') == selected[1]:
                self.query_one('#revision', Input).value = ''
            self._candidate_pin = None
