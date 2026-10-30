"""A secondary cleanup failure must not hide publication failures or cancellation."""
import asyncio
import traceback
from functools import partial
from pathlib import Path

import pytest

from acprof.host.posthoc import storage
from tests.measurement.profiling.test_posthoc_backup import backup_context
from tests.measurement.profiling.test_posthoc_rollback import _transaction

FAILURES = (OSError, KeyboardInterrupt, SystemExit, asyncio.CancelledError)


@pytest.mark.parametrize("failure_type", FAILURES)
@pytest.mark.parametrize("writer", ["csv", "json"])
def test_failed_temporary_cleanup_preserves_writer_exception(tmp_path, monkeypatch, failure_type, writer):
    target = tmp_path / ("result.csv" if writer == "csv" else "metadata.json")
    target.write_bytes(b"original artifact")
    failure = failure_type("original writer failure")
    leftovers = []

    def fail_sync(_fd):
        raise failure

    def fail_unlink(path, *args, **kwargs):
        leftovers.append(path)
        raise PermissionError("temporary removal denied")

    monkeypatch.setattr(storage.os, "fsync", fail_sync)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    write = (partial(storage._write_csv_temporary, target, fieldnames=["value"],
                     rows=[{"value": "new"}], encoding="utf-8") if writer == "csv"
             else partial(storage._write_json_temporary, target, {"value": "new"}))
    with pytest.raises(failure_type) as caught:
        write()
    assert caught.value is failure
    notes = "\n".join(failure.__notes__)
    assert "temporary removal denied" in notes
    assert len(leftovers) == 1
    assert str(leftovers[0]) in notes
    assert leftovers[0].exists()
    assert target.read_bytes() == b"original artifact"


@pytest.mark.parametrize("failure_type", FAILURES)
@pytest.mark.parametrize("cleanup_stage", ["remove", "sync"])
def test_failed_backup_cleanup_keeps_original_error_and_prior_backup(tmp_path, monkeypatch, failure_type, cleanup_stage):
    context = backup_context(tmp_path, "v3")
    previous = storage.create_backup(context)
    saved = {path.relative_to(previous): path.read_bytes()
             for path in previous.rglob("*") if path.is_file()}
    original = context.result_csv.read_bytes()
    failure = failure_type("original backup failure")

    def fail_copy(_source, destination):
        Path(destination).write_bytes(b"partial backup")
        raise failure

    def fail_cleanup(_path):
        raise PermissionError("backup cleanup denied")

    monkeypatch.setattr(storage, "copy_backup_file", fail_copy)
    monkeypatch.setattr(storage.shutil if cleanup_stage == "remove" else storage,
                        "rmtree" if cleanup_stage == "remove" else "_fsync_directory", fail_cleanup)
    with pytest.raises(failure_type) as caught:
        storage.create_backup(context)
    assert caught.value is failure
    notes = "\n".join(failure.__notes__)
    assert "backup cleanup denied" in notes
    assert str(previous.parent) in notes
    assert {path.relative_to(previous): path.read_bytes()
            for path in previous.rglob("*") if path.is_file()} == saved
    assert context.result_csv.read_bytes() == original


@pytest.mark.parametrize("failure_type", FAILURES)
def test_transaction_cleanup_attempts_remaining_files_after_one_failure(tmp_path, monkeypatch, failure_type):
    context = backup_context(tmp_path, "flat")
    arguments = dict(fieldnames=context.fieldnames, rows=context.rows, static_meta=context.static_meta,
                     collection_history=context.collection_history, backup_dir=storage.create_backup(context))
    before = {path: path.read_bytes() for path in (
        context.result_csv, context.static_meta_path, context.collection_history_path)}
    failure = failure_type("original publication failure")
    replace, unlink = storage.os.replace, Path.unlink
    attempted = []

    def fail_publication(source, destination):
        if ".restore." not in Path(source).name:
            raise failure
        return replace(source, destination)

    def fail_first_unlink(path, *args, **kwargs):
        if path.suffix == ".tmp" and ".restore." not in path.name:
            attempted.append(path)
            if len(attempted) == 1:
                raise PermissionError("first temporary removal denied")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(storage.os, "replace", fail_publication)
    monkeypatch.setattr(Path, "unlink", fail_first_unlink)
    with pytest.raises(failure_type) as caught:
        storage.commit_result_files(context, **arguments)
    assert caught.value is failure
    assert len(attempted) == 3
    assert [path.exists() for path in attempted] == [True, False, False]
    assert "first temporary removal denied" in "\n".join(failure.__notes__)
    assert {path: path.read_bytes() for path in before} == before


def test_restore_error_and_cleanup_error_both_reach_transaction_diagnostic(tmp_path, monkeypatch):
    context, arguments, before = _transaction(tmp_path, history_existed=True)
    copy, unlink = storage.copy_backup_file, Path.unlink

    def fail_publication(*_paths):
        raise OSError("publication sync failed")

    def fail_csv_restore(source, destination):
        if Path(source).name == context.result_csv.name:
            raise OSError("original restore failed")
        return copy(source, destination)

    def fail_restore_unlink(path, *args, **kwargs):
        if ".restore." in path.name and context.result_csv.name in path.name:
            raise PermissionError("restore temporary removal denied")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(storage, "_fsync_parent_directories", fail_publication)
    monkeypatch.setattr(storage, "copy_backup_file", fail_csv_restore)
    monkeypatch.setattr(Path, "unlink", fail_restore_unlink)
    with pytest.raises(storage.PosthocError, match="recovery incomplete") as caught:
        storage.commit_result_files(context, **arguments)
    assert "original restore failed" in str(caught.value)
    assert "restore temporary removal denied" in str(caught.value)
    for path in (context.static_meta_path, context.collection_history_path):
        assert path.read_bytes() == before[path]


def test_profile_cli_includes_cleanup_notes(tmp_path, monkeypatch, capsys):
    from acprof.cli import posthoc

    failure = OSError("original backup failure")
    failure.add_note("Cleanup failed for incomplete-backup: removal denied")

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(posthoc, "bootstrap_project_env", lambda _path: None)
    monkeypatch.setattr(posthoc, "run_posthoc", fail)
    with pytest.raises(SystemExit) as caught:
        posthoc.main([str(tmp_path)])
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "original backup failure" in error
    assert "incomplete-backup: removal denied" in error


@pytest.mark.parametrize("cleanup_type", [PermissionError, asyncio.CancelledError])
def test_layered_staging_cleanup_keeps_cancellation(tmp_path, monkeypatch, cleanup_type):
    from acprof.results import layers

    context = backup_context(tmp_path, "v2")
    backup = storage.create_backup(context)
    original = context.result_csv.read_bytes()
    failure = KeyboardInterrupt("cancelled staging publication")
    stages = []

    def fail_publication(_fields, _rows, root, **_kwargs):
        stages.append(Path(root))
        raise failure

    def fail_cleanup(path, *_args, **_kwargs):
        raise cleanup_type(f"staging cleanup failed: {path}")

    monkeypatch.setattr(layers, "publish_result_rows", fail_publication)
    monkeypatch.setattr(storage.shutil, "rmtree", fail_cleanup)
    with pytest.raises(KeyboardInterrupt) as caught:
        storage.commit_result_files(context, fieldnames=context.fieldnames, rows=context.rows,
                                    static_meta=context.static_meta, collection_history=context.collection_history,
                                    backup_dir=backup)
    assert caught.value is failure
    assert str(stages[0]) in "\n".join(failure.__notes__)
    assert stages[0].is_dir()
    assert context.result_csv.read_bytes() == original


def test_successful_publication_reports_failed_staging_cleanup(tmp_path, monkeypatch):
    from acprof.results.layers import read_result_layers

    context, arguments, before = _transaction(tmp_path, history_existed=True)
    failure = PermissionError("staging removal denied")
    stages = []

    def fail_cleanup(path, *_args, **_kwargs):
        stages.append(path)
        raise failure

    monkeypatch.setattr(storage.shutil, "rmtree", fail_cleanup)
    with pytest.raises(PermissionError) as caught:
        storage.commit_result_files(context, **arguments)
    assert caught.value is failure
    assert len(stages) == 1
    assert stages[0].is_dir()
    assert str(stages[0]) in "\n".join(failure.__notes__)
    assert read_result_layers(context.result_dir)[1][0]["marker"] == "new-generation"
    for path, original in before.items():
        assert (arguments["backup_dir"] / path.relative_to(context.result_dir)).read_bytes() == original


@pytest.mark.parametrize("cancellation_type", FAILURES[1:])
def test_cleanup_interruption_preserves_all_failures_and_attempts_every_file(tmp_path, monkeypatch, cancellation_type):
    context = backup_context(tmp_path, "flat")
    backup = storage.create_backup(context)
    before = {path: path.read_bytes() for path in (
        context.result_csv, context.static_meta_path, context.collection_history_path)}
    failure = OSError("original publication failure")
    cancellation = cancellation_type("temporary cleanup cancelled")
    errors = [PermissionError("first cleanup failed"), cancellation, OSError("last cleanup failed")]
    replace, unlink = storage.os.replace, Path.unlink
    attempted = []

    def fail_publication(source, destination):
        if ".restore." not in Path(source).name:
            raise failure
        return replace(source, destination)

    def fail_cleanup(path, *args, **kwargs):
        if path.suffix == ".tmp" and ".restore." not in path.name:
            attempted.append(path)
            raise errors[len(attempted) - 1]
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(storage.os, "replace", fail_publication)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(cancellation_type) as caught:
        storage.commit_result_files(context, fieldnames=context.fieldnames, rows=context.rows,
                                    static_meta=context.static_meta, collection_history=context.collection_history,
                                    backup_dir=backup)
    assert caught.value is cancellation
    assert caught.value.__cause__ is failure
    assert len(attempted) == 3
    diagnostic = "".join(traceback.format_exception(caught.value))
    for message in ("original publication failure", "first cleanup failed", "last cleanup failed"):
        assert message in diagnostic
    assert {path: path.read_bytes() for path in before} == before
