"""Posthoc backups preserve paths, readable snapshots, and directory durability."""
from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from acprof.artifact_layout import ArtifactLayout
from acprof.host.posthoc import storage
from acprof.host.posthoc.context import PosthocError, load_result_context
from acprof.result_csv import read_result_csv
from acprof.result_layers import publish_result_rows, read_result_layers
from acprof.result_store.protocol import publication_lock
from tests.measurement.profiling.test_posthoc import TestPosthocProfile as _PosthocFixture
from tests.results.storage.test_projection_recovery import committed_store
from tests.results.storage.test_result_generations import _write_legacy_snapshot


def backup_context(tmp_path, version):
    template = tmp_path / "template"
    _PosthocFixture()._write_fixture(template)
    context = load_result_context(template)
    root = tmp_path / "result"
    if version == "v3":
        layout = committed_store(root)
        fields, rows = read_result_layers(root)
    elif version == "v1":
        layout = ArtifactLayout(root)
        root.mkdir()
        _write_legacy_snapshot(root)
        fields, rows = read_result_layers(root)
    else:
        layout = ArtifactLayout.for_new_run(root) if version == "v2" else ArtifactLayout(root)
        if version == "v2":
            layout.initialize()
            publish_result_rows(context.fieldnames, context.rows, root)
        else:
            root.mkdir()
            with (root / "result_all.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=context.fieldnames)
                writer.writeheader()
                writer.writerows(context.rows)
        fields, rows = context.fieldnames, context.rows
    result_path = layout.path("result_all.csv" if version == "flat" else "result_layers.json")
    meta_path = layout.path("static_meta.json")
    shutil.copy2(context.static_meta_path, meta_path)
    history_path = layout.path("collection_history.json")
    history_path.write_text(json.dumps(context.collection_history), encoding="utf-8")
    return replace(context, result_dir=root, result_csv=result_path, static_meta_path=meta_path,
                   collection_history_path=history_path, collection_history_existed=True,
                   input_scale_plan_path=layout.path("input_scale_plan.json"), fieldnames=fields, rows=rows)


@pytest.mark.parametrize("version", ["flat", "v1", "v2", "v3"])
def test_backup_can_be_read_without_the_original_experiment(tmp_path, version):
    context = backup_context(tmp_path, version)
    relative = context.result_csv.relative_to(context.result_dir)
    original = context.result_csv.read_bytes()
    expected = read_result_csv(context.result_csv)
    backup = storage.create_backup(context)
    isolated = tmp_path / "isolated-backup"
    backup.rename(isolated)
    context.result_dir.rename(tmp_path / "source-moved-away")
    assert (isolated / relative).read_bytes() == original
    assert read_result_csv(isolated / relative) == expected
    layout = ArtifactLayout.discover(isolated)
    assert json.loads(layout.path("collection_history.json").read_bytes()) == context.collection_history


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is a POSIX durability contract")
@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_backup_syncs_every_nested_directory_after_its_contents(tmp_path, monkeypatch, version):
    context = backup_context(tmp_path, version)
    events = []
    sync = storage.os.fsync

    def record_sync(fd):
        info = os.fstat(fd)
        events.append((info.st_dev, info.st_ino))
        sync(fd)

    def identity(entry):
        info = entry.stat()
        return info.st_dev, info.st_ino

    monkeypatch.setattr(storage.os, "fsync", record_sync)
    backup = storage.create_backup(context)
    for path in backup.rglob("*"):
        assert identity(path) in events, f"backup entry was never synced: {path.relative_to(backup)}"
        assert identity(path.parent) in events
        assert events.index(identity(path)) < max(index for index, item in enumerate(events) if item == identity(path.parent))
    assert identity(backup.parent) in events


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_corrupt_copied_layer_is_rejected_before_backup_is_returned(tmp_path, monkeypatch, version):
    context = backup_context(tmp_path, version)
    before = context.result_csv.read_bytes()
    copy = storage.copy_backup_file

    def corrupt_copy(source, destination):
        copy(source, destination)
        if Path(source).name == "summary.csv":
            Path(destination).write_bytes(b"broken copied layer\n")

    monkeypatch.setattr("acprof.host.posthoc.storage.copy_backup_file", corrupt_copy)
    with pytest.raises(ValueError, match="missing or modified|hash mismatch|invalid.*layer"):
        storage.create_backup(context)
    assert context.result_csv.read_bytes() == before
    backup_root = ArtifactLayout.discover(context.result_dir).path("posthoc_backups")
    assert list(backup_root.iterdir()) == []


@pytest.mark.parametrize("attribute", ["static_meta_path", "collection_history_path"])
def test_copied_metadata_must_remain_readable(tmp_path, monkeypatch, attribute):
    context = backup_context(tmp_path, "v3")
    copy = storage.copy_backup_file
    target = getattr(context, attribute)
    original = target.read_bytes()

    def corrupt_copy(source, destination):
        copy(source, destination)
        if source == target:
            Path(destination).write_bytes(b"{")

    monkeypatch.setattr("acprof.host.posthoc.storage.copy_backup_file", corrupt_copy)
    with pytest.raises(PosthocError, match="cannot read backup"):
        storage.create_backup(context)
    assert target.read_bytes() == original
    assert list(ArtifactLayout.discover(context.result_dir).path("posthoc_backups").iterdir()) == []


def test_incremental_copy_holds_one_publication_lock(tmp_path, monkeypatch):
    context = backup_context(tmp_path, "v3")
    layout = ArtifactLayout.discover(context.result_dir)
    copy = storage.copy_backup_file
    copied = []

    def locked_copy(source, destination):
        with pytest.raises(BlockingIOError):
            with publication_lock(layout, exclusive=True, blocking=False):
                pytest.fail("writer acquired the publication lock during backup")
        copied.append(source)
        copy(source, destination)

    monkeypatch.setattr("acprof.host.posthoc.storage.copy_backup_file", locked_copy)
    backup = storage.create_backup(context)
    assert context.result_csv in copied
    assert layout.layer_csv("summary") in copied
    assert read_result_csv(backup / "result_layers.json") == read_result_csv(context.result_csv)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


def test_pending_module_replacement_requires_source_recovery_before_backup(tmp_path):
    context = backup_context(tmp_path, "v3")
    layout = ArtifactLayout.discover(context.result_dir)
    manifest = json.loads(context.result_csv.read_bytes())
    pending = layout.result_store_path("pending.json")
    pending.write_text(json.dumps({"before": manifest, "replacements": {"summary": "pending"}}))
    with pytest.raises(ValueError, match="requires recovery"):
        storage.create_backup(context)
    assert list(layout.path("posthoc_backups").iterdir()) == []
    assert pending.is_file()


def test_incremental_backup_keeps_prior_result_after_posthoc_commit(tmp_path):
    context = backup_context(tmp_path, "v3")
    before = read_result_csv(context.result_csv)
    metadata = context.static_meta_path.read_bytes()
    backup = storage.create_backup(context)
    rows = [dict(row, latency_s="0.5") for row in context.rows]
    storage.commit_result_files(context, fieldnames=context.fieldnames, rows=rows,
                                static_meta={"changed": True}, collection_history=context.collection_history,
                                backup_dir=backup)
    assert read_result_csv(context.result_csv)[1][0]["latency_s"] == "0.5"
    assert read_result_csv(backup / "result_layers.json") == before
    assert (backup / "static_meta.json").read_bytes() == metadata


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_failed_incremental_copy_releases_lock_and_preserves_prior_backup(tmp_path, monkeypatch, failure):
    context = backup_context(tmp_path, "v3")
    previous = storage.create_backup(context)
    before = {path.relative_to(previous): path.read_bytes() for path in previous.rglob("*") if path.is_file()}
    original = context.result_csv.read_bytes()

    def fail_copy(_source, destination):
        Path(destination).write_bytes(b"partial backup")
        raise failure("interrupted backup copy")

    monkeypatch.setattr("acprof.host.posthoc.storage.copy_backup_file", fail_copy)
    with pytest.raises(failure, match="interrupted backup copy"):
        storage.create_backup(context)
    assert list(previous.parent.iterdir()) == [previous]
    assert {path.relative_to(previous): path.read_bytes() for path in previous.rglob("*") if path.is_file()} == before
    assert context.result_csv.read_bytes() == original
    with publication_lock(ArtifactLayout.discover(context.result_dir), exclusive=True, blocking=False):
        pass


def test_nested_directory_sync_failure_does_not_return_a_backup(tmp_path, monkeypatch):
    context = backup_context(tmp_path, "v2")
    sync = storage._fsync_directory

    def fail_metadata_sync(path):
        if path.name == "metadata":
            raise OSError("backup directory sync failed")
        sync(path)

    monkeypatch.setattr(storage, "_fsync_directory", fail_metadata_sync)
    with pytest.raises(OSError, match="backup directory sync failed"):
        storage.create_backup(context)
    assert list(ArtifactLayout.discover(context.result_dir).path("posthoc_backups").iterdir()) == []


@pytest.mark.parametrize("unsafe", ["outside", "symlink"])
def test_backup_rejects_source_paths_outside_the_artifact_contract(tmp_path, unsafe):
    context = backup_context(tmp_path, "v3")
    outside = tmp_path / "outside.json"
    outside.write_bytes(context.static_meta_path.read_bytes())
    if unsafe == "outside":
        context = replace(context, static_meta_path=outside)
    else:
        context.static_meta_path.unlink()
        context.static_meta_path.symlink_to(outside)
    original = outside.read_bytes()
    with pytest.raises(ValueError, match="not in the subpath|escapes|symlink"):
        storage.create_backup(context)
    assert outside.read_bytes() == original
    assert list(ArtifactLayout.discover(context.result_dir).path("posthoc_backups").iterdir()) == []


@pytest.mark.parametrize("history_existed", [False, True])
def test_wide_csv_rollback_uses_the_same_relative_paths_as_backup(tmp_path, monkeypatch, history_existed):
    context = backup_context(tmp_path, "flat")
    nested = context.result_dir / "legacy"
    nested.mkdir()
    for name in ("result_csv", "static_meta_path", "collection_history_path"):
        path = getattr(context, name)
        destination = nested / path.name
        path.rename(destination)
        setattr(context, name, destination)
    if not history_existed:
        context.collection_history_path.unlink()
        context.collection_history_existed = False
    paths = (context.result_csv, context.static_meta_path, context.collection_history_path)
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    backup = storage.create_backup(context)

    def fail_publication(*_paths):
        raise OSError("publication sync failed")

    monkeypatch.setattr(storage, "_fsync_parent_directories", fail_publication)
    rows = [dict(row, marker="changed") for row in context.rows]
    with pytest.raises(OSError, match="publication sync failed"):
        storage.commit_result_files(context, fieldnames=context.fieldnames, rows=rows,
                                    static_meta={"changed": True}, collection_history=context.collection_history,
                                    backup_dir=backup)
    for path, raw in before.items():
        assert (path.read_bytes() if path.exists() else None) == raw
        if raw is not None:
            assert (backup / path.relative_to(context.result_dir)).read_bytes() == raw
