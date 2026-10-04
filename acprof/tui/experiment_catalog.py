"""Rebuildable, bounded discovery of recorded experiments within explicit roots."""
from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable
from uuid import uuid4

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import read_json_object
from acprof.experiment import RunConfig
from acprof.installation import cli_command
from acprof.run_args import flatten_run_options

MAX_DUPLICATE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class ExperimentRecord:
    directory: Path
    aliases: tuple[Path, ...]
    run_id: str | None
    model_id: str
    revision: str
    runtime_profile: str
    created_at: str
    status: str
    device: str
    options: dict
    state: dict
    metadata: dict
    validation: dict
    failures: tuple[dict, ...]
    issues: tuple[str, ...] = ()

    @property
    def has_recovery_state(self) -> bool:
        return bool(self.run_id and self.options and self.status != 'complete' and not self.issues)

    @property
    def search_text(self) -> str:
        return ' '.join((self.model_id, self.revision, self.runtime_profile, self.created_at,
            self.status, self.device, self.run_id or 'unknown', str(self.directory),
            json.dumps(self.failures, ensure_ascii=False))).casefold()


@dataclass(frozen=True)
class ExperimentCatalog:
    records: tuple[ExperimentRecord, ...]
    warnings: tuple[str, ...] = ()
    coverage_reports: tuple[Path, ...] = ()

    def search(self, query: str) -> tuple[ExperimentRecord, ...]:
        terms = query.casefold().split()
        return tuple(record for record in self.records if all(term in record.search_text for term in terms))


def read_artifact(path: Path) -> dict:
    """Read bounded JSON objects; missing evidence remains empty, never invented."""
    if not path.is_file():
        return {}
    return read_json_object(path)


def _record(layout: ArtifactLayout, warnings: list[str]) -> ExperimentRecord:
    directory = layout.root
    values = {}
    for name in ('run_state.json', 'static_meta.json', 'runtime_validation.json', 'runtime_failures.json', 'model_resolution.json'):
        path = layout.path(name)
        try:
            if not path.resolve().is_relative_to(directory):
                raise ValueError(f'{name}: artifact_outside_experiment')
            values[name] = read_artifact(path)
        except (OSError, ValueError) as exc:
            warnings.append(f'{directory}: {name}: {exc}')
            values[name] = {}
    state, metadata, validation = (values[name] for name in ('run_state.json', 'static_meta.json', 'runtime_validation.json'))
    resolution = values['model_resolution.json']
    task = state.get('runtime', {}).get('task', {})
    options = state.get('options', {})
    if not isinstance(options, dict):
        options = {}
    failures = list(values['runtime_failures.json'].get('failures', []))
    failures.extend(value.get('failure') for value in validation.get('devices', {}).values() if isinstance(value, dict))
    failures.append(resolution.get('failure'))
    failures = [value for value in failures if isinstance(value, dict)]
    gpu = metadata.get('gpu_device') or {}
    device = str(gpu.get('uuid') or options.get('gpu_device') or (
        'cpu ' + str(metadata.get('cpu_model', 'unknown')) if options.get('gpus') == 'off' else options.get('gpus', 'unknown')))
    run_id = state.get('run_id')
    issues = ('unsupported_run_state',) if state and state.get('schema_version') != 1 else ()
    return ExperimentRecord(directory, (directory,), run_id if isinstance(run_id, str) and run_id else None,
        str(metadata.get('model_name') or metadata.get('model_id') or task.get('model_id') or resolution.get('model_id') or options.get('model') or 'unknown'),
        str(metadata.get('model_revision') or task.get('model_revision') or resolution.get('model_revision') or 'unknown'),
        str(metadata.get('runtime_profile_id') or task.get('runtime_profile_id') or resolution.get('runtime_profile') or 'unknown'),
        str(state.get('created_at') or metadata.get('created_at') or 'unknown'),
        str(state.get('status') or ('failed' if failures else 'unknown')), device,
        options, state, metadata, validation, tuple(failures), issues)


def _duplicate_digest(record: ExperimentRecord, cancelled: Callable[[], bool], budget: list[int]) -> str:
    # Only duplicate run IDs need data hashing; ordinary browsing never scans CSV contents.
    digest = hashlib.sha256(json.dumps({'state': record.state, 'metadata': record.metadata, 'validation': record.validation, 'failures': record.failures}, sort_keys=True).encode())
    csv = record.directory / 'result_all.csv'
    if csv.is_file() and csv.resolve().is_relative_to(record.directory):
        with csv.open('rb') as stream:
            while chunk := stream.read(min(1024 * 1024, budget[0] + 1)):
                budget[0] -= len(chunk)
                if budget[0] < 0:
                    raise ValueError('duplicate_evidence_unverified: CSV byte budget exceeded')
                if cancelled():
                    raise InterruptedError('experiment scan cancelled')
                digest.update(chunk)
    return digest.hexdigest()


def scan_experiments(roots: Iterable[Path], *, cancelled: Callable[[], bool] = lambda: False,
                     max_directories: int = 4096, max_depth: int = 6,
                     max_entries: int = 16384, max_duplicate_bytes: int = MAX_DUPLICATE_BYTES) -> ExperimentCatalog:
    """No global filesystem search, model imports, lock waits or network requests."""
    if max_directories < 1 or max_depth < 0 or max_entries < 1 or max_duplicate_bytes < 1:
        raise ValueError('invalid experiment scan limits')
    queue = deque((Path(root).expanduser().resolve(), 0) for root in roots)
    seen: set[Path] = set()
    records: list[ExperimentRecord] = []
    warnings: list[str] = []
    coverage_reports: list[Path] = []
    identities: dict[str, int] = {}
    entries_read = 0
    duplicate_budget = [max_duplicate_bytes]
    while queue:
        if cancelled():
            raise InterruptedError('experiment scan cancelled')
        directory, depth = queue.popleft()
        if directory in seen or not directory.is_dir():
            continue
        if len(seen) >= max_directories:
            warnings.append('scan_limit: too many directories; choose a narrower results root')
            break
        seen.add(directory)
        if (directory / 'coverage.json').is_file() and not (directory / 'coverage.json').is_symlink():
            coverage_reports.append(directory / 'coverage.json')
        try:
            layout = ArtifactLayout.discover(directory)
            is_experiment = any(layout.path(name).is_file() for name in (
                'run_state.json', 'static_meta.json', 'model_resolution.json', 'result_all.csv'))
            record = _record(layout, warnings) if is_experiment else None
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            warnings.append(f'{directory}: invalid_experiment: {exc}')
            continue
        if record is not None:
            if record.run_id and record.run_id in identities:
                index = identities[record.run_id]
                previous = records[index]
                try:
                    conflicting = _duplicate_digest(previous, cancelled, duplicate_budget) != _duplicate_digest(record, cancelled, duplicate_budget)
                    issue = 'run_id_content_conflict' if conflicting else ''
                except InterruptedError:
                    raise
                except (OSError, ValueError):
                    issue = 'duplicate_evidence_unverified'
                issues = tuple(dict.fromkeys((*previous.issues, *((issue,) if issue else ()))))
                records[index] = replace(previous, aliases=(*previous.aliases, directory), issues=issues)
            else:
                if record.run_id:
                    identities[record.run_id] = len(records)
                records.append(record)
            continue
        try:
            children = []
            with os.scandir(directory) as entries:
                for entry in entries:
                    if cancelled():
                        raise InterruptedError('experiment scan cancelled')
                    entries_read += 1
                    if entries_read > max_entries:
                        warnings.append('scan_entry_limit: choose a narrower results root')
                        queue.clear()
                        break
                    if not entry.name.startswith('.') and entry.is_dir(follow_symlinks=False):
                        children.append(Path(entry.path))
            children.sort()
        except InterruptedError:
            raise
        except OSError as exc:
            warnings.append(f'{directory}: {exc}')
            continue
        if depth >= max_depth:
            if children:
                warnings.append(f'{directory}: scan_depth_limit')
            continue
        if entries_read <= max_entries:
            queue.extend((child, depth + 1) for child in children)
    return ExperimentCatalog(tuple(sorted(records, key=lambda record: (record.created_at, record.run_id or ''), reverse=True)),
                             tuple(warnings), tuple(coverage_reports))


def new_experiment_config(config: RunConfig, output_root: Path) -> RunConfig:
    """Allocate a proposed sibling attempt without creating any files."""
    root = output_root
    if root.name.startswith("experiment-") and len(root.name) == 43:
        try:
            int(root.name[11:], 16)
        except ValueError:
            pass
        else:
            root = root.parent
    return replace(config, output_dir=str(root / ("experiment-" + uuid4().hex)), resume=False)


def config_from_record(record: ExperimentRecord, *, reuse: bool) -> RunConfig:
    if not record.options or record.issues:
        raise ValueError('frozen configuration unavailable: ' + ', '.join(record.issues))
    options = flatten_run_options(record.options)
    config = RunConfig.from_namespace(SimpleNamespace(**options))
    explicit = {item.name for item in fields(RunConfig)}
    derived = {'measurement_environment', 'model_spec_sha256', 'workload_spec_sha256'}
    extra = {name: value for name, value in options.items() if name not in explicit | derived}
    config = replace(config, extra_options=extra)
    from acprof.model_evidence import pinned_revision
    if reuse and pinned_revision(record.revision):
        config = replace(config, revision=record.revision)
    if reuse:
        return new_experiment_config(config, record.directory.parent)
    return replace(config, output_dir=str(record.directory.parent), resume=True)


def resume_command(record: ExperimentRecord, *, python_executable: Path) -> tuple[str, ...]:
    """Restore all frozen CLI options, including options not exposed in the form."""
    from acprof.run_args import build_parser
    if not record.has_recovery_state:
        raise ValueError('run_id/frozen configuration cannot resume: ' + ', '.join(record.issues))
    config = config_from_record(record, reuse=False)
    if record.directory.name != config.model.replace('/', '--'):
        raise ValueError('resume requires the recorded model directory name; reuse configuration for a new experiment')
    if read_artifact(ArtifactLayout.discover(record.directory).path('run_state.json')) != record.state:
        raise ValueError('experiment changed since discovery; refresh before resuming')
    from acprof.run_args import arguments_from_options
    derived = {'measurement_environment', 'model_spec_sha256', 'workload_spec_sha256'}
    options = {name: value for name, value in flatten_run_options(record.options).items()
               if name not in derived | {'resume', 'output_dir', 'notify', 'skip_build'}}
    arguments = arguments_from_options(options)
    arguments.extend(('--resume', '--output-dir', str(record.directory.parent)))
    build_parser().parse_args(arguments)
    return tuple([*cli_command('run', python_executable=python_executable), *arguments])
