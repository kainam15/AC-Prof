"""Rebuildable, bounded discovery of recorded experiments within explicit roots."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from acprof.analysis.compatibility import (
    RecordedModelIdentity,
    RecordedSourceError,
    recorded_model_identity,
)
from acprof.artifacts import read_json_object, recorded_artifact_present
from acprof.cancellation import check_cancelled
from acprof.experiments.arguments import flatten_run_options
from acprof.experiments.configuration import RunConfig
from acprof.failures import ordered_validation_failures
from acprof.installation import cli_command
from acprof.results.layout import (
    ArtifactLayout,
    experiment_result_file,
    resolve_single_result_source,
)

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
    identity: RecordedModelIdentity
    issues: tuple[str, ...] = ()

    @property
    def model_source(self) -> str:
        return self.identity.source

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


def read_artifact(path: Path, *, cancelled: Callable[[], bool] | None = None) -> dict:
    """Read bounded JSON objects; missing evidence remains empty, never invented."""
    check_cancelled(cancelled)
    if not path.is_file():
        return {}
    return read_json_object(path, cancelled=cancelled)


def _read_standalone_or_embedded(path: Path, parent: dict, parent_name: str,
                                 cancelled: Callable[[], bool] | None) -> tuple[dict, str]:
    check_cancelled(cancelled)
    if recorded_artifact_present(path):
        record = read_json_object(path, cancelled=cancelled)
        origin = path.name
    else:
        record = parent.get(path.stem, {})
        origin = parent_name
    if not isinstance(record, dict):
        raise ValueError(f"{origin} {path.stem}: expected an object")
    if path.name == "runtime_validation.json":
        devices = record.get("devices", {})
        if not isinstance(devices, dict) or not all(isinstance(item, dict) for item in devices.values()):
            raise ValueError(f"{origin}: devices must be an object of device records")
    return record, origin


def format_experiment_created_at(value: str) -> str | None:
    """Present saved timestamps in host-local time, matching the experiment picker."""
    try:
        timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone()
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class ExperimentSourceDisplay:
    """Lightweight result-selection state, not proof of successful measurement."""

    title: str
    run_id: str | None
    recorded_status: str
    availability: str  # selected | no_data | missing | unsupported

    @property
    def can_read(self) -> bool:
        return self.availability == 'selected'


def experiment_source_display(source: Path) -> ExperimentSourceDisplay:
    """Describe one source without scanning measured rows or changing saved metadata."""
    fallback = source.parent.name if source.name in {'result_layers.json', 'result_all.csv'} else source.name
    if re.fullmatch(r'[0-9a-fA-F]{32}', fallback):
        fallback = ''  # Internal Run ID is not a human-readable experiment name.
    try:
        source = resolve_single_result_source(source)
        if source.is_dir():
            root, result_file = source, experiment_result_file(source)
        elif source.name == 'result_layers.json' or source.suffix.lower() == '.csv':
            root, result_file = source.parent, source
        else:
            return ExperimentSourceDisplay(fallback, None, '', 'unsupported')
        available = result_file.is_file() and result_file.stat().st_size > 0
        present = source.exists()
    except (OSError, ValueError):
        return ExperimentSourceDisplay(fallback, None, '', 'missing')

    title, run_id, status = fallback, None, ''
    if root.is_dir():
        try:
            layout = ArtifactLayout.discover(root)
            state = read_artifact(layout.path('run_state.json'))
            metadata = read_artifact(layout.path('static_meta.json'))
            if state and state.get('schema_version') != 1:
                state = {}
            options = state.get('options')
            options = options if isinstance(options, dict) else {}
            runtime = state.get('runtime')
            runtime = runtime if isinstance(runtime, dict) else {}
            task = runtime.get('task')
            task = task if isinstance(task, dict) else {}
            model = metadata.get('model_name') or metadata.get('model_id') or task.get('model_id') or options.get('model')
            created_at = state.get('created_at') or metadata.get('created_at')
            if isinstance(model, str) and model.strip():
                date_text = format_experiment_created_at(created_at) if isinstance(created_at, str) else None
                title = f'{model} · {date_text}' if date_text else model
            identity = state.get('run_id')
            run_id = identity if isinstance(identity, str) and identity else None
            raw_status = state.get('status')
            status = raw_status if isinstance(raw_status, str) else ''
        except (OSError, ValueError):
            # A corrupt or unsupported identity must not be presented as a completed run.
            pass

    availability = ('selected' if available else 'no_data' if present or (run_id is not None)
                    else 'missing')
    return ExperimentSourceDisplay(title, run_id, status, availability)


def experiment_source_title(source: Path) -> tuple[str, str | None]:
    """Human-readable label and separate stable internal Run ID."""
    display = experiment_source_display(source)
    return display.title, display.run_id


def _record(layout: ArtifactLayout, warnings: list[str], *,
            cancelled: Callable[[], bool] | None) -> ExperimentRecord:
    directory = layout.root
    values: dict[str, dict] = {}
    issues: list[str] = []
    embedded_parents = {
        "runtime_validation.json": "static_meta.json",
        "model_resolution.json": "static_meta.json",
        "interface_validation.json": "model_resolution.json",
    }
    origins: dict[str, str] = {}
    for name in ('run_state.json', 'static_meta.json', 'runtime_failures.json', *embedded_parents):
        origins[name] = name
        try:
            path = layout.path(name)
            if not path.resolve().is_relative_to(directory):
                raise ValueError(f'{name}: artifact_outside_experiment')
            parent_name = embedded_parents.get(name)
            if parent_name is not None:
                values[name], origins[name] = _read_standalone_or_embedded(
                    path, values[parent_name], origins[parent_name], cancelled,
                )
            else:
                values[name] = read_artifact(path, cancelled=cancelled)
        except (OSError, ValueError) as exc:
            detail = f'{directory}: {name}: {exc}'
            warnings.append(detail)
            if name in {'static_meta.json', 'model_resolution.json', 'interface_validation.json', 'run_state.json'}:
                issues.append(detail)
            values[name] = {}
    state, metadata, validation = (values[name] for name in ('run_state.json', 'static_meta.json', 'runtime_validation.json'))
    resolution = values['model_resolution.json']
    interface = values['interface_validation.json']
    task = state.get('runtime', {}).get('task', {})
    options = state.get('options', {})
    if not isinstance(options, dict):
        options = {}
    failures = list(values['runtime_failures.json'].get('failures', []))
    failures.extend(ordered_validation_failures(validation, [
        (device, device["failure"]) for device in validation.get("devices", {}).values()
        if isinstance(device, dict) and isinstance(device.get("failure"), dict)
    ]))
    failures.append(resolution.get('failure'))
    failures = [value for value in failures if isinstance(value, dict)]
    gpu = metadata.get('gpu_device') or {}
    device = str(gpu.get('uuid') or options.get('gpu_device') or (
        'cpu ' + str(metadata.get('cpu_model', 'unknown')) if options.get('gpus') == 'off' else options.get('gpus', 'unknown')))
    run_id = state.get('run_id')
    try:
        identity = RecordedModelIdentity() if issues else recorded_model_identity(
            metadata, resolution, state, resolution_name=origins['model_resolution.json'],
            interface=interface, interface_name=origins['interface_validation.json'],
        )
    except RecordedSourceError as exc:
        identity = RecordedModelIdentity()
        issues.extend(f"{artifact}: {detail}" for artifact, detail in exc.problems)
    if state and state.get('schema_version') != 1:
        issues.append('unsupported_run_state')
    recorded_runtime_env = metadata.get('runtime_environment')
    runtime_env = recorded_runtime_env if isinstance(recorded_runtime_env, dict) else {}
    return ExperimentRecord(directory, (directory,), run_id if isinstance(run_id, str) and run_id else None,
        str(identity.model_id or metadata.get('model_name') or metadata.get('model_id') or task.get('model_id') or resolution.get('model_id') or options.get('model') or 'unknown'),
        identity.revision(metadata.get('model_revision'), task.get('model_revision'), resolution.get('model_revision'), interface.get('revision')),
        str(metadata.get('runtime_profile_id') or runtime_env.get('profile_id') or task.get('runtime_profile_id') or resolution.get('runtime_profile') or interface.get('runtime_profile') or 'unknown'),
        str(state.get('created_at') or metadata.get('created_at') or 'unknown'),
        str(state.get('status') or ('failed' if failures else 'unknown')), device,
        options, state, metadata, validation, tuple(failures), identity, tuple(issues))


def read_experiment_record(directory: Path | ArtifactLayout, *,
                           cancelled: Callable[[], bool] | None = None) -> ExperimentRecord:
    """Read one experiment's bounded metadata without catalog or CSV scanning."""
    layout = (directory if isinstance(directory, ArtifactLayout)
              else ArtifactLayout.discover(directory, cancelled=cancelled))
    warnings: list[str] = []
    record = _record(layout, warnings, cancelled=cancelled)
    return replace(record, issues=tuple(dict.fromkeys((*record.issues, *warnings))))


def _duplicate_digest(record: ExperimentRecord, cancelled: Callable[[], bool], budget: list[int]) -> str:
    # Only duplicate run IDs need data hashing; ordinary browsing never scans CSV contents.
    digest = hashlib.sha256(json.dumps({'state': record.state, 'metadata': record.metadata, 'validation': record.validation, 'failures': record.failures}, sort_keys=True).encode())
    manifest_path = record.directory / 'result_layers.json'
    if manifest_path.is_file():
        if manifest_path.is_symlink() or not manifest_path.resolve().is_relative_to(record.directory):
            raise ValueError('duplicate_evidence_unverified: unsafe result layer')
        from acprof.artifacts import read_json_object_snapshot
        snapshot = read_json_object_snapshot(manifest_path, cancelled=cancelled, byte_budget=budget)
        manifest = snapshot.payload
        if manifest.get('schema_version') == 3:
            from acprof.result_store.reader import committed_content_digest
            digest.update(committed_content_digest(record.directory, cancelled=cancelled, budget=budget).encode())
            return digest.hexdigest()
        # Hash the full immutable snapshot, not just the manifest: equal
        # metadata must never mask a changed or missing result layer.
        from acprof.results.layers import result_layer_paths

        def consume(path: Path) -> str:
            if budget[0] <= 0:
                raise ValueError('duplicate_evidence_unverified: CSV byte budget exceeded')
            if path.is_symlink() or not path.resolve().is_relative_to(record.directory):
                raise ValueError('duplicate_evidence_unverified: unsafe result layer')
            file_digest = hashlib.sha256()
            with path.open('rb', buffering=0) as stream:
                while True:
                    if cancelled():
                        raise InterruptedError('experiment scan cancelled')
                    chunk = stream.read(min(1024 * 1024, budget[0] + 1))
                    if not chunk:
                        break
                    budget[0] -= len(chunk)
                    if budget[0] < 0:
                        raise ValueError('duplicate_evidence_unverified: CSV byte budget exceeded')
                    if cancelled():
                        raise InterruptedError('experiment scan cancelled')
                    digest.update(chunk)
                    file_digest.update(chunk)
            return file_digest.hexdigest()

        digest.update(snapshot.content)
        try:
            paths = result_layer_paths(record.directory, manifest)
        except ValueError as exc:
            raise ValueError(f'duplicate_evidence_unverified: {exc}') from exc
        for layer, path in sorted(paths.items()):
            digest.update(layer.encode())
            layer_digest = consume(path)
            if layer_digest != manifest['layers'][layer]['sha256']:
                raise ValueError('duplicate_evidence_unverified: modified result layer')
    flat_path = record.directory / 'result_all.csv'
    if experiment_result_file(record.directory) == flat_path:
        if flat_path.is_symlink() or not flat_path.resolve().is_relative_to(record.directory):
            raise ValueError('duplicate_evidence_unverified: unsafe legacy CSV')
        digest.update(b'legacy-flat-csv\0')
        with flat_path.open('rb', buffering=0) as stream:
            while True:
                if cancelled():
                    raise InterruptedError('experiment scan cancelled')
                chunk = stream.read(min(1024 * 1024, budget[0] + 1))
                if not chunk:
                    break
                budget[0] -= len(chunk)
                if budget[0] < 0:
                    raise ValueError('duplicate_evidence_unverified: CSV byte budget exceeded')
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
            layout = ArtifactLayout.discover(directory, cancelled=cancelled)
            is_collection = (directory / 'metadata' / 'task_runs').is_dir()
            is_experiment = not is_collection and any(layout.path(name).is_file() for name in (
                'run_state.json', 'static_meta.json', 'model_resolution.json', 'interface_validation.json', 'result_layers.json', 'result_all.csv'))
            record = _record(layout, warnings, cancelled=cancelled) if is_experiment else None
        except CancelledError as exc:
            raise InterruptedError('experiment scan cancelled') from exc
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            warnings.append(f'{directory}: invalid_experiment: {exc}')
            continue
        if record is not None:
            if record.run_id and record.run_id in identities:
                index = identities[record.run_id]
                previous = records[index]
                prior, current = previous.identity, record.identity
                identity_conflict = prior.source != current.source or any(
                    left and right and left != right
                    for left, right in ((prior.model_id, current.model_id), (prior.commit, current.commit))
                )
                identity = RecordedModelIdentity() if identity_conflict else RecordedModelIdentity(
                    prior.source, prior.model_id or current.model_id, prior.commit or current.commit,
                )
                try:
                    conflicting = identity_conflict or (
                        _duplicate_digest(previous, cancelled, duplicate_budget) != _duplicate_digest(record, cancelled, duplicate_budget)
                    )
                    issue = 'run_id_content_conflict' if conflicting else ''
                except InterruptedError:
                    raise
                except CancelledError as exc:
                    raise InterruptedError('experiment scan cancelled') from exc
                except (OSError, ValueError):
                    issue = 'duplicate_evidence_unverified'
                issues = tuple(dict.fromkeys((*previous.issues, *record.issues, *((issue,) if issue else ()))))
                records[index] = replace(previous, aliases=(*previous.aliases, directory), issues=issues,
                                         identity=identity, model_id=identity.model_id or previous.model_id,
                                         revision=identity.revision(previous.revision))
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
    return replace(config, output_dir=str(root / ("experiment-" + uuid4().hex)), resume=False, task_run_id="")


def config_from_record(record: ExperimentRecord, *, reuse: bool) -> RunConfig:
    if not record.options or record.issues:
        raise ValueError('frozen configuration unavailable: ' + ', '.join(record.issues))
    options = flatten_run_options(record.options)
    config = RunConfig.from_namespace(SimpleNamespace(**options))
    explicit = {item.name for item in fields(RunConfig)}
    derived = {'measurement_environment', 'model_spec_sha256', 'workload_spec_sha256'}
    extra = {name: value for name, value in options.items() if name not in explicit | derived}
    config = replace(config, extra_options=extra, model_source=record.model_source)
    from acprof.models.evidence import pinned_revision
    if reuse and pinned_revision(record.revision):
        config = replace(config, revision=record.revision)
    root = record.directory.parents[3] if config.task_run_id else record.directory.parent
    if reuse:
        return new_experiment_config(config, root)
    return replace(config, output_dir=str(root), resume=True)


def resume_command(record: ExperimentRecord, *, python_executable: Path) -> tuple[str, ...]:
    """Restore all frozen CLI options, including options not exposed in the form."""
    from acprof.experiments.arguments import build_parser
    if not record.has_recovery_state:
        raise ValueError('run_id/frozen configuration cannot resume: ' + ', '.join(record.issues))
    config = config_from_record(record, reuse=False)
    if record.directory != config.result_dir(Path.cwd()):
        raise ValueError('resume requires the recorded model directory name; reuse configuration for a new experiment')
    if read_artifact(ArtifactLayout.discover(record.directory).path('run_state.json')) != record.state:
        raise ValueError('experiment changed since discovery; refresh before resuming')
    from acprof.experiments.arguments import arguments_from_options
    derived = {'measurement_environment', 'model_spec_sha256', 'workload_spec_sha256'}
    options = {name: value for name, value in flatten_run_options(record.options).items()
               if name not in derived | {'resume', 'output_dir', 'notify', 'rebuild_image'}}
    options['model_source'] = config.model_source
    arguments = arguments_from_options(options)
    arguments.extend(('--resume', '--output-dir', config.output_dir))
    build_parser().parse_args(arguments)
    return tuple([*cli_command('run', python_executable=python_executable), *arguments])
