"""Local model candidates with explicit revision/runtime/device evidence."""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acprof.hub.repository import MODEL_SOURCES
from acprof.models.evidence import pinned_revision
from acprof.tui.experiments.catalog import ExperimentRecord

CONDITION_KEYS = ('revision', 'runtime_profile', 'runtime_environment', 'device',
                  'cpus', 'mems', 'gpus', 'batch_size', 'source_sha256', 'host_id', 'execution_options', 'tasks')


@dataclass(frozen=True)
class ModelCandidate:
    model_id: str
    revision: str
    status: str
    reason: str
    conditions: dict
    detail: str
    experiment: Path | None = None
    report: Path | None = None
    history_status: str = ""
    record: ExperimentRecord | None = None
    model_source: str = "huggingface"
    reason_code: str = ""

    @property
    def search_text(self) -> str:
        return ' '.join((self.model_id, self.model_source, self.revision, self.status, self.reason,
                         json.dumps(self.conditions, ensure_ascii=False))).casefold()


def execution_options(options: dict) -> dict | None:
    """Project the existing run-state option protocol, retaining explicit unset thread values."""
    from acprof.host.execution_conditions import MEASUREMENT_ENV_NAMES
    environment = options.get('measurement_environment')
    if (not isinstance(environment, dict) or set(MEASUREMENT_ENV_NAMES) - environment.keys()
            or 'input_scales' not in options or 'request_timeout_seconds' not in options):
        return None
    # RunState v1 deliberately omits blank cpuset and the default auto policy.
    return {'cpuset_cpus': options.get('cpuset_cpus', ''), 'input_scales': options['input_scales'] or None,
        'input_scale_policy': options.get('input_scale_policy', 'auto'),
        'model_spec_sha256': options.get('model_spec_sha256', 'none') if not options.get('model_spec') else options.get('model_spec_sha256'),
        'workload_spec_sha256': options.get('workload_spec_sha256', 'none') if not options.get('workload_spec') else options.get('workload_spec_sha256'),
        'request_timeout_seconds': options['request_timeout_seconds'], 'measurement_environment': environment}


def _semantic_ids(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value or not all(isinstance(part, str) and part for part in value):
        return None
    return tuple(sorted(value)) if len(set(value)) == len(value) else None


def _discovered_groups(discovery: object, revision: str) -> dict[str, tuple[str, ...]]:
    """Only a complete pinned inventory can expand a group ID, 'all' or the default."""
    if (not isinstance(discovery, dict) or discovery.get("model_revision") != revision
            or discovery.get("status") not in {"ready", "selection_required"}
            or discovery.get("conflicts") or discovery.get("unresolved")):
        return {}
    profiles = discovery.get("profiles")
    if not isinstance(profiles, list):
        return {}
    groups = {}
    for profile in profiles:
        if not isinstance(profile, dict):
            return {}
        name = profile.get("profile_id")
        members = _semantic_ids(profile.get("semantic_task_ids"))
        if not isinstance(name, str) or not name or name in groups or not members:
            return {}
        groups[name] = members
    return groups


def _task_selection(record: ExperimentRecord, request: object, *, recorded: bool = False,
                    interface: str = "") -> tuple[str, ...] | None:
    """Identify the semantic workloads covered by one minimum-input validation."""
    if not isinstance(request, str):
        return None
    ids = _semantic_ids([part.strip() for part in request.split(",")]) if request else ()
    if ids is None:
        return None
    task = record.state.get("runtime", {}).get("task", {})
    selected = task.get("selected_tasks", [])
    if not isinstance(selected, list) or len(selected) > 1 or (selected and not isinstance(selected[0], dict)):
        return None
    if recorded and selected:
        workload = selected[0]
        if not isinstance(workload.get("task_id"), str) or not isinstance(workload.get("execution_profile", {}), dict):
            return None
        group_id = workload.get("task_id")
        members = _semantic_ids(workload.get("execution_profile", {}).get("semantic_task_ids", [group_id]))
        # Child options name the group; the actual snapshot can cover a semantic subset.
        return members if not ids or ids in {("all",), (group_id,), members} else None

    discovery = task.get("task_discovery", {})
    groups = _discovered_groups(discovery, record.revision)
    if not ids or ids == ("all",):
        if len(groups) == 1:
            return next(iter(groups.values()))
        # Legacy single-interface runs predate task discovery and omit default --tasks.
        tag = task.get("pipeline_tag")
        if (not ids and not selected and discovery == {} and isinstance(tag, str) and tag not in {"", "unknown"}
                and (not interface or interface == tag)):
            return ()
        return None
    if "all" in ids or (recorded and len(ids) > 1):
        return None
    if not groups and (discovery or (selected and selected[0].get("task_id") in ids)):
        return None
    if groups and set(ids) - groups.keys() - {member for members in groups.values() for member in members}:
        return None
    return tuple(sorted({member for name in ids for member in groups.get(name, (name,))}))


def record_conditions(record: ExperimentRecord) -> dict:
    options, runtime = record.options, record.state.get('runtime', {})
    host = record.state.get('host', {})
    environment = record.metadata.get('runtime_environment') or runtime.get('image', {}).get('runtime_environment', {})
    device = ((record.metadata.get('gpu_device') or {}).get('uuid') or options.get('gpu_device')) if 'on' in options.get('gpus', '') else (
        'cpu:' + host['machine_id_sha256'] if host.get('machine_id_sha256') else None)
    return {'host_id': host.get('machine_id_sha256'), 'execution_options': execution_options(options),
            'tasks': _task_selection(record, options.get('tasks', ''), recorded=True),
            'revision': record.revision, 'runtime_profile': record.runtime_profile,
            'runtime_environment': environment.get('environment_id'), 'device': device,
            **{key: options.get(key) for key in ('cpus', 'mems', 'gpus', 'batch_size')},
            'source_sha256': host.get('source_sha256')}


def changed_conditions(recorded: dict, current: dict) -> tuple[str, ...]:
    reasons = []
    for key in CONDITION_KEYS:
        old, new = recorded.get(key), current.get(key)
        missing = old in (None, '', 'unknown') or new in (None, '', 'unknown')
        if key == 'revision' and (not isinstance(old, str) or not pinned_revision(old)
                                  or not isinstance(new, str) or not pinned_revision(new)):
            missing = True
        if missing:
            reasons.append(f'{key}: unknown')
        elif old != new:
            if isinstance(old, dict) and isinstance(new, dict):
                reasons.extend(f'{key}.{name}: changed' for name in sorted(old.keys() | new.keys()) if old.get(name) != new.get(name))
            else:
                reasons.append(f'{key}: changed')
    return tuple(reasons)


def candidate_from_record(record: ExperimentRecord, current: dict) -> ModelCandidate:
    conditions = record_conditions(record)
    reasons = list(changed_conditions(conditions, current))
    if record.model_source != current.get('model_source', 'huggingface'):
        reasons.append('model_source: changed')
    primary_code = str(record.failures[0].get('reason_code') or 'unknown') if record.failures else ''
    if primary_code == 'access_denied':
        status, reason = 'access_required', 'access_denied'
    elif primary_code == 'resource_limit':
        status, reason = 'resource_limited', 'resource_limit'
    elif record.issues:
        status, reason = 'revalidate', ', '.join(record.issues)
    elif record.failures:
        status, reason = 'previous_failure', primary_code
    elif (record.validation.get('status') == 'ok'
          and record.validation.get('scope') == 'isolated_minimum_scale_predict_and_postprocess_before_measurement'):
        status, reason = 'verified_current', 'minimum input predict/postprocess passed under recorded conditions'
    else:
        status, reason = 'revalidate', 'full runtime evidence unavailable; ' + '; '.join(reasons)
    history_status = 'previous_success' if status == 'verified_current' and reasons else status
    if reasons and status in {'verified_current', 'access_required', 'resource_limited', 'previous_failure'}:
        reason += '; current conditions require revalidation: ' + '; '.join(reasons)
        status = 'revalidate'
    detail = json.dumps({'conditions': conditions, 'current_conditions': current, 'condition_changes': reasons,
        'run_id': record.run_id, 'aliases': [str(path) for path in record.aliases],
        'validation': record.validation, 'failures': record.failures, 'status': record.status}, ensure_ascii=False, indent=2)
    return ModelCandidate(record.model_id, record.revision, status, reason, conditions, detail, record.directory,
                          history_status=history_status, record=record, model_source=record.model_source,
                          reason_code=primary_code)


def cached_candidates(root: Path, *, cancelled: Callable[[], bool] = lambda: False,
                      max_entries: int = 2048) -> tuple[tuple[ModelCandidate, ...], tuple[str, ...]]:
    """Read verified entry plans and file presence; do not lock, hash weights or download."""
    from acprof.host.model_store.store import model_sources, read_entry
    from acprof.models.download_plan import PLAN_FILENAME
    values, warnings, identities = [], [], set()
    entries = root / 'entries'
    if not entries.is_dir():
        return (), ()
    with os.scandir(entries) as iterator:
        paths = []
        for index, entry in enumerate(iterator):
            if cancelled():
                raise InterruptedError('model candidate scan cancelled')
            if index >= max_entries:
                warnings.append('cache_scan_limit')
                break
            if entry.is_dir(follow_symlinks=False):
                paths.append(Path(entry.path) / PLAN_FILENAME)
    for path in sorted(paths):
        if cancelled():
            raise InterruptedError('model candidate scan cancelled')
        try:
            plan = read_entry(path.parent.name, root)
            if not plan or not all(source.cache_status == 'hit' for source in model_sources(plan, root, entry_id=path.parent.name)):
                continue
            identity = (plan['model_id'], plan['model_revision'], plan.get('source', 'huggingface'))
            if identity in identities or not pinned_revision(identity[1]):
                continue
            identities.add(identity)
            detail = json.dumps({'cache_entry': str(path.parent), 'plan_sha256': plan.get('plan_sha256'),
                'model_id': identity[0], 'revision': identity[1], 'source': identity[2], 'inference_verified': False}, ensure_ascii=False, indent=2)
            values.append(ModelCandidate(*identity[:2], 'cached_unverified', 'verified cache entry; runtime validation required', {}, detail,
                                         model_source=identity[2]))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            warnings.append(f'{path.parent.name}: {exc}')
    return tuple(values), tuple(warnings)


def coverage_candidates(path: Path) -> tuple[ModelCandidate, ...]:
    """Read recorded coverage attempts without upgrading missing runtime identity."""
    from acprof.tui.experiments.catalog import read_artifact
    data = read_artifact(path)
    if data.get('schema_version') not in {1, 2} or data.get('scope') not in {
        'selected_sample_only; no_formal_measurement', 'recorded_results; no_reexecution',
    }:
        raise ValueError('unsupported coverage evidence')
    attempts = {attempt['attempt_id']: attempt for attempt in data.get('attempts', [])}
    values = []
    for row in data.get('rows', []):
        attempt = attempts.get(row.get('attempt_id'), {})
        config = attempt.get('configuration', data.get('configuration', {}))
        resources, host = config.get('resources', data.get('resources', {})), config.get('host', {})
        gpu = resources.get('gpu')
        device = config.get('device', {}).get('uuid') if gpu else ('cpu:' + host['machine_id_sha256'] if host.get('machine_id_sha256') else None)
        conditions = {'revision': row.get('revision'), 'runtime_profile': row.get('runtime_profile'),
            'runtime_environment': None, 'device': device, 'cpus': str(resources['cpus']) if 'cpus' in resources else None,
            'mems': str(resources['memory_gb']) if 'memory_gb' in resources else None,
            'gpus': 'on' if gpu else 'off' if gpu is False else None, 'batch_size': 1 if config.get('probe') == 'full' else None,
            'source_sha256': host.get('source_sha256')}
        failure = row.get('failure') or {}
        code = str(failure.get('reason_code') or '')
        reason = 'runtime environment and current conditions require revalidation'
        if failure:
            reason = (code or 'previous failure') + '; ' + reason
        history_status = {'access_denied': 'access_required', 'resource_limit': 'resource_limited'}.get(code,
            'previous_failure' if failure else 'revalidate')
        detail = json.dumps({'row': row, 'attempt_configuration': config, 'conditions': conditions,
                             'source': str(path)}, ensure_ascii=False, indent=2)
        source = row.get('source', 'huggingface')
        values.append(ModelCandidate(str(row['model_id']), str(row.get('revision') or 'unknown'), 'revalidate',
                                     reason, conditions, detail, report=path, history_status=history_status,
                                     model_source=source if source in MODEL_SOURCES else 'unknown', reason_code=code))
    return tuple(values)


def current_context(config, project_dir: Path, *, needs_torch_platform: bool = True,
                    cancelled: Callable[[], bool] = lambda: False) -> dict:
    """Capture shared host/platform facts once, outside measurement; never start Docker."""
    import subprocess

    from acprof.host.images.runtime import select_nlp_torch_index_url
    from acprof.host.run_state import host_identity
    context: dict[str, Any] = {'host': host_identity(project_dir), 'gpu_uuid': None,
                               'torch_index_url': None, 'warnings': []}
    gpu_requested = 'on' in config.gpus.split(',')
    # Resolve the current matrix's platform once for the whole idle scan.
    # CPU-only requests use the CPU lock without querying the GPU driver.
    if cancelled():
        raise InterruptedError('model candidate scan cancelled')
    if needs_torch_platform:
        try:
            context['torch_index_url'] = select_nlp_torch_index_url(devices=config.gpus.split(','))
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            context['warnings'].append(str(exc))
    if cancelled():
        raise InterruptedError('model candidate scan cancelled')
    if gpu_requested:
        from acprof.host.gpu_device import resolve_gpu_device
        try:
            context['gpu_uuid'] = resolve_gpu_device(config.extra_options.get('gpu_device')).get('uuid')
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            context['warnings'].append(str(exc))
    return context


def current_conditions(record: ExperimentRecord, config, context: dict, project_dir: Path) -> dict:
    from acprof.host.detect import TaskInfo
    from acprof.host.images.runtime import configure_runtime_profile
    from acprof.runtime.profiles import environment_id, select_runtime_profile
    revision = config.revision or record.revision
    from acprof.host.execution_conditions import measurement_environment
    host = context['host']
    options = {name: getattr(config, name) for name in ('cpuset_cpus', 'input_scales', 'input_scale_policy',
               'model_spec', 'workload_spec', 'request_timeout_seconds')}
    options['measurement_environment'] = measurement_environment()
    current = {'revision': revision, 'model_source': config.model_source, 'runtime_profile': None, 'runtime_environment': None,
        'device': context['gpu_uuid'] if 'on' in config.gpus.split(',') else (
            'cpu:' + host['machine_id_sha256'] if host.get('machine_id_sha256') else None),
        'cpus': config.cpus, 'mems': config.mems, 'gpus': config.gpus, 'batch_size': config.batch_size,
        'source_sha256': host.get('source_sha256'), 'host_id': host.get('machine_id_sha256'),
        'execution_options': execution_options(options), 'tasks': _task_selection(record, config.tasks, interface=config.task)}
    if revision != record.revision or config.model_spec or config.workload_spec:
        return current
    try:
        task = TaskInfo(**deepcopy(record.state['runtime']['task']))
        task.runtime_profile_id = ''
        task.pipeline_tag = config.task or task.pipeline_tag
        task.task_family = config.task_family or task.task_family
        task.runtime_backend = config.backend or task.runtime_backend
        profile = select_runtime_profile(task)
        if profile.environment is None:
            raise ValueError('selected runtime profile is missing an environment')
        if profile.adapter == 'family-default' and profile.environment.platform.torch_version:
            if context['torch_index_url'] is None:
                return current
            task.model_resolution['requested_devices'] = config.gpus.split(',')
            profile = configure_runtime_profile(task, torch_index_url=context['torch_index_url'])
            if profile.environment is None:
                raise ValueError('resolved runtime profile is missing an environment')
        selected_environment = profile.environment
        if selected_environment is None:
            raise ValueError('resolved runtime profile is missing an environment')
        current.update(runtime_profile=profile.profile_id,
                       runtime_environment=environment_id(selected_environment, project_dir))
    except (KeyError, ValueError, TypeError, OSError, RuntimeError) as exc:
        current['resolution_error'] = str(exc)
    return current


MODEL_STATUS_LABELS = {'verified_current': '当前条件有成功验证记录', 'cached_unverified': '已缓存、待验证',
    'access_required': '需要授权', 'resource_limited': '上次资源不足', 'previous_failure': '保留历史失败',
    'revalidate': '需要重新验证', 'previous_success': '历史最小输入验证成功'}
_CANDIDATE_LEVELS = {
    'verified_current': 'success', 'cached_unverified': 'neutral', 'revalidate': 'warning',
    'access_required': 'error', 'resource_limited': 'error', 'previous_failure': 'error',
}
_CANDIDATE_FACT_STATUS = {
    'verified_current': 'ok', 'cached_unverified': 'unverified', 'revalidate': 'warning',
    'access_required': 'error', 'resource_limited': 'warning', 'previous_failure': 'error',
}


def _candidate_views(candidate: ModelCandidate, status_message: Any) -> tuple[Any, Any, Any]:
    from acprof.messages import join_messages
    from acprof.tui.i18n import message_id
    from acprof.tui.ui.view_models import (
        CalloutContent,
        FactRow,
        StatusSummary,
        ViewSection,
        reason_presentation,
    )

    summary = StatusSummary(
        _CANDIDATE_LEVELS.get(candidate.status, 'neutral'),
        join_messages(' · ', (candidate.model_id, status_message)),
    )
    callout: CalloutContent | None = None
    reason_key = candidate.history_status or candidate.status
    if reason_key in {'access_required', 'resource_limited', 'previous_failure'}:
        code = candidate.reason_code or {'access_required': 'access_denied', 'resource_limited': 'resource_limit'}.get(reason_key, 'unknown')
        callout = reason_presentation(code, candidate.reason + '\n' + candidate.detail)
    rows = (
        FactRow(message_id('models.source.label'), candidate.model_source),
        FactRow(message_id('models.revision.label'), candidate.revision, copyable=True),
        FactRow(
            message_id('models.candidates.validation_status'),
            status_message,
            status=_CANDIDATE_FACT_STATUS.get(candidate.status, 'unknown'),
            detail=candidate.reason,
        ),
        FactRow(
            message_id('common.field.device'),
            str(candidate.conditions.get('device') or message_id('common.value.unknown')),
            detail=candidate.reason,
        ),
    )
    evidence_path = candidate.experiment or candidate.report
    detail_rows = [
        FactRow(message_id('models.candidates.reason_label'), candidate.reason, copyable=True),
        *(
            (FactRow(message_id('models.candidates.evidence_path'), str(evidence_path), copyable=True),)
            if evidence_path is not None else ()
        ),
        FactRow(message_id('models.candidates.raw_evidence'), 'JSON', detail=candidate.detail, copyable=True),
    ]
    section = ViewSection(message_id('models.candidates.details_title'), rows, tuple(detail_rows))
    return summary, callout, section


def model_choice(candidate: ModelCandidate):
    from acprof.messages import join_messages, message
    from acprof.tui.experiments.picker import PickerChoice
    actions = {'use'} if candidate.model_source in MODEL_SOURCES else set()
    if candidate.experiment or candidate.report:
        actions.add('view')
    status = MODEL_STATUS_LABELS[candidate.status]
    status_value: Any = message(status)
    if candidate.history_status and candidate.history_status != candidate.status:
        status_value = join_messages(' · ', (message(status), message(MODEL_STATUS_LABELS[candidate.history_status])))
    summary, callout, section = _candidate_views(candidate, status_value)
    return PickerChoice(candidate, (f'{candidate.model_id} ({candidate.model_source})', candidate.revision[:12], status_value,
        str(candidate.conditions.get('device') or 'unknown')), candidate.reason + '\n' + candidate.detail, candidate.search_text,
        frozenset(actions), summary=summary, callout=callout, section=section)
