"""Cancellation must not leave a partially published post-hoc transaction."""
import asyncio
import json
from pathlib import Path

import pytest
from test_posthoc import TestPosthocProfile as _PosthocFixture
from test_posthoc_rollback import _transaction

from acprof.host.posthoc import context as posthoc_context, storage

CANCELLATIONS = (KeyboardInterrupt, SystemExit, asyncio.CancelledError)


@pytest.mark.parametrize("history_existed", [False, True])
def test_v2_cancellation_restores_files_across_directories(tmp_path, monkeypatch, history_existed):
    from acprof.artifact_layout import ArtifactLayout

    layout = ArtifactLayout.for_new_run(tmp_path)
    layout.initialize()
    _PosthocFixture()._write_fixture(tmp_path)
    (tmp_path / posthoc_context.INPUT_SCALE_PLAN_NAME).rename(layout.path(posthoc_context.INPUT_SCALE_PLAN_NAME))
    if history_existed:
        layout.path(storage.COLLECTION_HISTORY_NAME).write_text(json.dumps({
            "schema_version": 1, "posthoc_profile_history": [],
            "timeout_retry_history": [], "quality_retry_history": [],
        }), encoding="utf-8")
    context = posthoc_context.load_result_context(tmp_path)
    backup = storage.create_backup(context)
    paths = (context.result_csv, context.static_meta_path, context.collection_history_path)
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    if history_existed:
        relative = context.collection_history_path.relative_to(context.result_dir)
        assert (backup / relative).read_bytes() == before[context.collection_history_path]
    rows = [dict(row) for row in context.rows]
    rows[0]["marker"] = "changed"
    original_replace = storage.os.replace
    cancellation = KeyboardInterrupt("cancel after v2 history publication")

    def interrupt(source, destination):
        result = original_replace(source, destination)
        if destination == context.collection_history_path and ".restore." not in Path(source).name:
            raise cancellation
        return result

    monkeypatch.setattr(storage.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        storage.commit_result_files(
            context, fieldnames=context.fieldnames, rows=rows,
            static_meta={**context.static_meta, "run_command": "changed"},
            collection_history=context.collection_history, backup_dir=backup,
        )
    assert caught.value is cancellation
    assert {path: path.read_bytes() if path.exists() else None for path in paths} == before
    assert backup.is_relative_to(layout.path(storage.BACKUP_DIRNAME))
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("cancellation_type", CANCELLATIONS)
@pytest.mark.parametrize("history_existed", [False, True])
@pytest.mark.parametrize("stage", ["before-meta", "after-csv", "after-meta", "after-history", "sync"])
def test_cancellation_restores_all_attempted_publications(tmp_path, monkeypatch, cancellation_type, history_existed, stage):
    context, arguments, before = _transaction(tmp_path, history_existed=history_existed)
    cancellation = cancellation_type("intentional transaction cancellation")
    replace = storage.os.replace
    destinations = {"after-csv": context.result_csv, "after-meta": context.static_meta_path,
                    "after-history": context.collection_history_path}
    interrupted = False

    def interrupt_replace(source, destination):
        nonlocal interrupted
        if interrupted or ".restore." in Path(source).name:
            return replace(source, destination)
        if stage == "before-meta" and destination == context.static_meta_path:
            interrupted = True
            raise cancellation
        result = replace(source, destination)
        if destination == destinations.get(stage):
            interrupted = True
            # Model a signal after rename but before the next Python statement.
            raise cancellation
        return result

    def interrupt_sync(*_paths):
        raise cancellation

    monkeypatch.setattr(storage.os, "replace", interrupt_replace)
    if stage == "sync":
        monkeypatch.setattr(storage, "_fsync_parent_directories", interrupt_sync)
    with pytest.raises(cancellation_type) as caught:
        storage.commit_result_files(context, **arguments)
    assert caught.value is cancellation
    for path, content in before.items():
        assert (path.read_bytes() if path.exists() else None) == content
        if content is not None:
            assert (arguments["backup_dir"] / path.name).read_bytes() == content
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("cancellation_type", CANCELLATIONS)
@pytest.mark.parametrize("writer", ["csv", "json"])
def test_cancelled_temporary_writer_removes_its_partial_file(tmp_path, monkeypatch, cancellation_type, writer):
    path = tmp_path / ("result.csv" if writer == "csv" else "metadata.json")
    path.write_bytes(b"original artifact")
    cancellation = cancellation_type("intentional writer cancellation")

    def interrupt_sync(_fd):
        raise cancellation

    monkeypatch.setattr(storage.os, "fsync", interrupt_sync)
    with pytest.raises(cancellation_type) as caught:
        if writer == "csv":
            storage._write_csv_temporary(path, fieldnames=["value"], rows=[{"value": "new"}], encoding="utf-8")
        else:
            storage._write_json_temporary(path, {"value": "new"})
    assert caught.value is cancellation
    assert path.read_bytes() == b"original artifact"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("cancellation_type", CANCELLATIONS)
def test_cancelled_backup_copy_removes_only_the_incomplete_backup(tmp_path, monkeypatch, cancellation_type):
    _PosthocFixture()._write_fixture(tmp_path)
    context = posthoc_context.load_result_context(tmp_path)
    previous = storage.create_backup(context)
    saved = {path.relative_to(previous): path.read_bytes()
             for path in previous.rglob("*") if path.is_file()}
    cancellation = cancellation_type("intentional backup cancellation")

    def interrupt_copy(_source, destination):
        Path(destination).write_bytes(b"partial backup")
        raise cancellation

    monkeypatch.setattr(storage.shutil, "copy2", interrupt_copy)
    with pytest.raises(cancellation_type) as caught:
        storage.create_backup(context)
    assert caught.value is cancellation
    assert list(previous.parent.iterdir()) == [previous]
    assert {path.relative_to(previous): path.read_bytes()
            for path in previous.rglob("*") if path.is_file()} == saved


@pytest.mark.parametrize("cancellation_type", CANCELLATIONS)
def test_rollback_cancellation_does_not_skip_other_files(tmp_path, monkeypatch, cancellation_type):
    context, arguments, before = _transaction(tmp_path, history_existed=True)
    primary = OSError("publication sync failed")
    cancellation = cancellation_type("cancelled during CSV restore")
    restore = storage._restore_from_backup

    def fail_publication(*_paths):
        raise primary

    def interrupt_csv(destination, backup):
        if destination == context.result_csv:
            raise cancellation
        return restore(destination, backup)

    monkeypatch.setattr(storage, "_fsync_parent_directories", fail_publication)
    monkeypatch.setattr(storage, "_restore_from_backup", interrupt_csv)
    with pytest.raises(cancellation_type) as caught:
        storage.commit_result_files(context, **arguments)
    assert caught.value is cancellation
    assert isinstance(caught.value.__cause__, posthoc_context.PosthocError)
    assert "recovery incomplete" in str(caught.value.__cause__)
    assert "publication sync failed" in str(caught.value.__cause__)
    for path in (context.static_meta_path, context.collection_history_path):
        assert path.read_bytes() == before[path]
    assert (arguments["backup_dir"] / context.result_csv.name).read_bytes() == before[context.result_csv]
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("cancellation_type", CANCELLATIONS)
def test_primary_cancellation_retains_secondary_recovery_diagnostics(tmp_path, monkeypatch, cancellation_type):
    context, arguments, before = _transaction(tmp_path, history_existed=True)
    cancellation = cancellation_type("original transaction cancellation")
    restore = storage._restore_from_backup

    def interrupt_publication(*_paths):
        raise cancellation

    def fail_csv(destination, backup):
        if destination == context.result_csv:
            raise OSError("CSV restore failed")
        return restore(destination, backup)

    monkeypatch.setattr(storage, "_fsync_parent_directories", interrupt_publication)
    monkeypatch.setattr(storage, "_restore_from_backup", fail_csv)
    with pytest.raises(cancellation_type) as caught:
        storage.commit_result_files(context, **arguments)
    assert caught.value is cancellation
    assert isinstance(caught.value.__cause__, posthoc_context.PosthocError)
    assert "CSV restore failed" in str(caught.value.__cause__)
    assert str(arguments["backup_dir"]) in str(caught.value.__cause__)
    assert caught.value.__cause__.__cause__ is not caught.value
    for path in (context.static_meta_path, context.collection_history_path):
        assert path.read_bytes() == before[path]
    assert list(tmp_path.glob(".*.tmp")) == []
