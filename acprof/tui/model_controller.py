"""Connect the existing model input to local evidence, only while idle."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.suggester import SuggestFromList
from textual.widgets import Input as TextualInput, Select

from acprof.experiment import RunConfigError
from acprof.models.evidence import pinned_revision
from acprof.tui.experiment_catalog import scan_experiments
from acprof.tui.experiment_picker import SearchPickerScreen
from acprof.tui.i18n import message_id
from acprof.tui.input import BarCursorInput as Input

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class ModelController:
    """Own models state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.candidate_pin: tuple[str, str, str] | None = None

    def open_model_candidates(self) -> None:
        if not self.app.allow_operation('catalog'):
            return
        try:
            config = self.app.configuration.collect_config(allow_empty_model=True)
        except RunConfigError as exc:
            self.app.configuration.show_config_error(exc)
            return
        from acprof.host.model_store import store_root
        roots, token = self.app.catalog.catalog_roots(), object()
        cache_root = Path(config.model_store).expanduser() if config.model_store else store_root()
        if not cache_root.is_absolute():
            cache_root = self.app.project_dir / cache_root
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
        self.app.catalog.picker_open = True
        self.app.set_busy(True)
        self.app.push_screen(SearchPickerScreen('模型候选', ('模型', 'Revision', '验证状态', '设备'), loader,
            query=self.app.configuration.input('model'), actions=('use', 'view'), scope=(*roots, cache_root), project_dir=self.app.project_dir,
            empty_message='尚无本地模型记录，可直接输入 Hugging Face 模型 ID。',
            loading_changed=lambda busy: self.app.catalog.picker_loading(token, busy)),
            lambda choice: self.model_candidate_selected(choice, models))

    def model_candidate_selected(self, choice, models: set[str]) -> None:
        self.app.catalog.picker_closed()
        if choice is None or not self.app.allow_operation('catalog'):
            return
        self.app.query_one('#model', Input).suggester = SuggestFromList(sorted(models), case_sensitive=False)
        action, candidate = choice
        if action == 'use':
            self.candidate_pin = (candidate.model_id, candidate.model_source, candidate.revision)
            self.app.query_one('#model', Input).value = candidate.model_id
            self.app.query_one('#model-source', Select).value = candidate.model_source
            self.app.query_one('#revision', Input).value = candidate.revision if pinned_revision(candidate.revision) else ''
            self.app.configuration.refresh_command_preview()
            self.app.notify(message_id('models.candidates.selected_notice', candidate.reason))
        elif action == 'view' and candidate.record:
            self.app.catalog.view_experiment(candidate.record)
        elif action == 'view' and candidate.report:
            self.app.activate_tab('results-tab')
            self.app.results.open_report(str(candidate.report))

    def candidate_identity_changed(self, event: TextualInput.Changed | Select.Changed) -> None:
        selected = self.candidate_pin
        if selected and (self.app.configuration.input('model'), self.app.configuration.select('model-source')) != selected[:2]:
            if self.app.configuration.input('revision') == selected[2]:
                self.app.query_one('#revision', Input).value = ''
            self.candidate_pin = None
