"""Statistics and profiler status share one cancellable result snapshot."""
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from test_layer_read_cancellation import source_hashes, watch_cancelled_read
from test_posthoc import TestPosthocProfile as _PosthocFixture

from acprof import result_csv
from acprof.artifact_layout import ArtifactLayout
from acprof.host.posthoc import backfill, context as posthoc_context
from acprof.result_layers import LAYER_FILES, layer_fields
from acprof.result_store import protocol
from acprof.tui import diagnostics, profile_status
from acprof.tui.result_controller import ResultController


def profile_store(root, version=2):
    _PosthocFixture()._write_fixture(root)
    path = root / "result_layers.json"
    manifest = json.loads(path.read_bytes())
    if version == 2:
        return manifest
    for layer, info in manifest["layers"].items():
        target = root / LAYER_FILES[layer]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / info["path"], target)
        info["path"] = LAYER_FILES[layer]
    manifest.pop("generation")
    manifest["schema_version"] = version
    if version == 3:
        layout = ArtifactLayout(root)
        assigned = layer_fields(manifest["wide_fields"])
        for layer, fields in assigned.items():
            target = root / LAYER_FILES[layer]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(protocol.csv_bytes(fields, header=True))
                manifest["layers"][layer] = {"path": LAYER_FILES[layer], "fields": fields, "row_count": 0,
                                             "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            manifest["layers"][layer].update(committed_bytes=target.stat().st_size, revision=0)
        manifest.update(run_id="profile-overview-fixture", sequence=0, state="complete",
                        schema_sha256=protocol.digest(protocol.json_bytes(assigned)))
        layout.result_store_path("commits").mkdir(parents=True)
        layout.result_store_path("lock").touch()
        manifest = protocol.save_commit(layout, None, manifest, [], transaction="fixture")
    path.write_text(json.dumps(manifest))
    return manifest


def run_summary_worker(path, cancelled=None):
    callbacks = []
    app = SimpleNamespace(process=SimpleNamespace(safe_process_callback=lambda *args: callbacks.append(args)))
    controller = ResultController(app)
    controller._execute_summary_read_worker(path, cancelled or Event(), False)
    assert len(callbacks) == 1
    return callbacks[0]


@pytest.mark.parametrize("version", [1, 2, 3])
def test_complete_summary_worker_parses_each_layer_once(tmp_path, monkeypatch, version):
    manifest = profile_store(tmp_path, version)
    before = source_hashes(tmp_path)
    parses = Counter()
    original = result_csv.parse_result_stream

    def observe(stream, path, expected, **kwargs):
        parses[Path(path)] += 1
        return original(stream, path, expected, **kwargs)

    monkeypatch.setattr(result_csv, "parse_result_stream", observe)
    monkeypatch.setattr(protocol, "parse_result_stream", observe)
    _, _, _, summary, error, _, overview, profile_error = run_summary_worker(tmp_path / "result_layers.json")
    assert summary is not None, error
    assert overview is not None, profile_error
    assert parses == Counter({tmp_path / info["path"]: 1 for info in manifest["layers"].values()})
    assert summary.profiler_overview is overview
    assert summary.profiler_error == profile_error == ""
    assert source_hashes(tmp_path) == before


def test_audit_and_posthoc_context_borrow_the_same_snapshot_without_mutating_rows(tmp_path, monkeypatch):
    from acprof.analysis import audit

    profile_store(tmp_path)
    snapshots = []
    originals = []
    audit_original = audit.audit_result
    context_original = posthoc_context._load_result_csv

    def audit_snapshot(*args, **kwargs):
        snapshot = kwargs["result_snapshot"]
        snapshots.append(snapshot)
        originals.append([dict(row) for row in snapshot.rows])
        return audit_original(*args, **kwargs)

    def context_snapshot(*args, **kwargs):
        snapshots.append(kwargs["result_snapshot"])
        return context_original(*args, **kwargs)

    monkeypatch.setattr(audit, "audit_result", audit_snapshot)
    monkeypatch.setattr(posthoc_context, "_load_result_csv", context_snapshot)
    summary = diagnostics.summarize_result_csv(tmp_path)
    assert summary.profiler_overview is not None
    assert len(snapshots) == 2
    assert snapshots[0] is snapshots[1]
    assert snapshots[0].rows == originals[0]


@pytest.mark.parametrize("invalid", ["path", "duplicate"])
def test_borrowed_snapshot_still_requires_correct_path_and_unique_rows(tmp_path, invalid):
    profile_store(tmp_path)
    snapshot = result_csv.read_result_csv_snapshot(tmp_path / "result_layers.json")
    if invalid == "path":
        snapshot = replace(snapshot, path=tmp_path / "other" / "result_layers.json")
        error, message = posthoc_context.PosthocError, "snapshot path"
    else:
        snapshot = replace(snapshot, rows=[*snapshot.rows, snapshot.rows[0]])
        error, message = result_csv.ResultValidationError, "duplicate measurement"
    with pytest.raises(error, match=message):
        posthoc_context.load_result_context(tmp_path, result_snapshot=snapshot)


def test_invalid_profiler_metadata_keeps_verified_statistics_and_reports_its_error(tmp_path):
    profile_store(tmp_path)
    path = tmp_path / "static_meta.json"
    metadata = json.loads(path.read_bytes())
    path.write_text(json.dumps({**metadata, "image_id": "mutable-tag:latest"}))
    before = source_hashes(tmp_path)
    summary = diagnostics.summarize_result_csv(tmp_path)
    assert summary.rows == summary.ok_rows == 2
    assert summary.profiler_overview is None
    assert "immutable image_id" in summary.profiler_error
    assert source_hashes(tmp_path) == before


def test_profiler_counts_each_applicable_row_once(tmp_path, monkeypatch):
    rows = [{"gpu_mode": mode} for mode in ("off", "on")]
    for row in rows:
        for tool in posthoc_context.SUPPORTED_TOOLS:
            row.update({name: "1" for name in posthoc_context.TOOL_METRIC_FIELDS[tool]})
            row[posthoc_context.TOOL_ERROR_FIELD[tool]] = ""
    context = SimpleNamespace(result_dir=tmp_path, rows=rows)
    monkeypatch.setattr(profile_status, "load_result_context", lambda *_args, **_kwargs: context)
    calls = Counter()
    original = backfill.row_tool_complete

    def observe(row, tool):
        calls[tool] += 1
        return original(row, tool)

    monkeypatch.setattr(profile_status, "row_tool_complete", observe)
    monkeypatch.setattr(backfill, "row_tool_complete", observe)
    overview = profile_status.inspect_profile_result(tmp_path)
    assert overview.complete == 4
    assert calls == {"torch": 2, "ncu": 1, "nsys": 1, "massif": 1}


@pytest.mark.parametrize("changed", ["manifest", "layer"])
def test_source_change_during_profiler_inspection_rejects_the_whole_summary(tmp_path, monkeypatch, changed):
    manifest = profile_store(tmp_path)
    path = tmp_path / "result_layers.json"
    original = profile_status.row_tool_complete
    modified = False

    def change_source(row, tool):
        nonlocal modified
        if not modified:
            if changed == "manifest":
                path.write_text(json.dumps({**manifest, "source_note": "changed during overview"}))
            else:
                layer = tmp_path / manifest["layers"]["summary"]["path"]
                layer.write_bytes(b"X" + layer.read_bytes()[1:])
            modified = True
        return original(row, tool)

    monkeypatch.setattr(profile_status, "row_tool_complete", change_source)
    _, _, _, summary, error, _, overview, _ = run_summary_worker(path)
    assert modified
    assert summary is None
    assert overview is None
    assert "快照" in error or "snapshot" in error


def test_unexpected_profiler_failure_still_completes_worker_callback(tmp_path, monkeypatch):
    profile_store(tmp_path)

    def fail(*_args, **_kwargs):
        raise KeyError("broken profiler inspection")

    monkeypatch.setattr(profile_status, "row_tool_complete", fail)
    _, _, _, summary, error, _, overview, _ = run_summary_worker(tmp_path / "result_layers.json")
    assert summary is None
    assert overview is None
    assert "broken profiler inspection" in error


def test_cancelled_profiler_count_discards_summary_and_still_completes_worker_callback(tmp_path, monkeypatch):
    profile_store(tmp_path)
    cancelled = Event()
    original = profile_status.row_tool_complete
    processed = []

    def cancel_after_first(row, tool):
        processed.append(id(row))
        cancelled.set()
        return original(row, tool)

    monkeypatch.setattr(profile_status, "row_tool_complete", cancel_after_first)
    _, _, _, summary, _, _, overview, _ = run_summary_worker(tmp_path / "result_layers.json", cancelled)
    assert summary is None
    assert overview is None
    assert len(set(processed)) == 1


@pytest.mark.parametrize("operation", ["context", "overview"])
def test_precancelled_inspection_performs_no_file_io(tmp_path, monkeypatch, operation):
    def forbidden(*_args, **_kwargs):
        pytest.fail("cancelled profiler inspection opened a file")

    monkeypatch.setattr(Path, "open", forbidden)
    read = posthoc_context.load_result_context if operation == "context" else profile_status.inspect_profile_result
    with pytest.raises(CancelledError):
        read(tmp_path, cancelled=lambda: True)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_context_propagates_cancellation_into_layer_read(tmp_path, monkeypatch, version):
    manifest = profile_store(tmp_path, version)
    cancelled = Event()
    before = source_hashes(tmp_path)
    target = tmp_path / manifest["layers"]["summary"]["path"]
    with monkeypatch.context() as scoped:
        opened, received = watch_cancelled_read(scoped, target, cancelled)
        with pytest.raises(CancelledError):
            posthoc_context.load_result_context(tmp_path, cancelled=cancelled.is_set)
    assert len(opened) == 1
    assert opened[0].closed
    assert opened[0].raw.closed
    assert 0 < sum(received) <= 256 * 1024
    assert source_hashes(tmp_path) == before


@pytest.mark.parametrize("version", [1, 2, 3])
def test_standalone_context_still_reads_each_layer_once(tmp_path, monkeypatch, version):
    manifest = profile_store(tmp_path, version)
    paths = {tmp_path / info["path"] for info in manifest["layers"].values()}
    opened = Counter()
    original_open = Path.open

    def observe(path, *args, **kwargs):
        if path in paths:
            opened[path] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe)
    context = posthoc_context.load_result_context(tmp_path)
    assert len(context.rows) == 2
    assert opened == Counter(dict.fromkeys(paths, 1))


@pytest.mark.parametrize("phase", ["metadata", "resources"])
def test_context_cancellation_during_validation_does_not_return_partial_context(tmp_path, monkeypatch, phase):
    from acprof import artifacts

    profile_store(tmp_path)
    cancelled = Event()
    processed = []
    owner, name = (artifacts, "read_static_metadata") if phase == "metadata" else (posthoc_context, "_integer")
    original = getattr(owner, name)

    def cancel_after(*args, **kwargs):
        result = original(*args, **kwargs)
        processed.append(args)
        cancelled.set()
        return result

    monkeypatch.setattr(owner, name, cancel_after)
    with pytest.raises(CancelledError):
        posthoc_context.load_result_context(tmp_path, cancelled=cancelled.is_set)
    assert len(processed) == (1 if phase == "metadata" else 2)
