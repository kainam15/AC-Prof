"""Cross-file recovery must continue after a secondary storage failure."""
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from acprof.host import collection_history
from acprof.host.posthoc import context as posthoc_context, storage
from tests.measurement.profiling.test_posthoc import TestPosthocProfile as _PosthocFixture
from tests.support.storage_faults import fail_directory_sync


def _transaction(root: Path, *, history_existed: bool, legacy_csv: bool = False):
    _PosthocFixture()._write_fixture(root)
    history_path = root / collection_history.COLLECTION_HISTORY_NAME
    if history_existed:
        history_path.write_text(json.dumps({
            "schema_version": 1, "posthoc_profile_history": [],
            "timeout_retry_history": [], "quality_retry_history": [],
        }), encoding="utf-8")
    context = posthoc_context.load_result_context(root)
    if legacy_csv:
        context = replace(context, result_csv=root / "result_all.csv")
    backup = storage.create_backup(context)
    paths = (context.result_csv, context.static_meta_path, history_path)
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    rows = [dict(row) for row in context.rows]
    rows[0]["marker"] = "new-generation"
    arguments = {
        "fieldnames": context.fieldnames, "rows": rows,
        "static_meta": {**context.static_meta, "run_command": "new-generation"},
        "collection_history": collection_history.append_collection_record(
            context.collection_history, "posthoc_profile_history",
            {"completed_at": "2026-10-07T00:00:00Z"},
        ),
        "backup_dir": backup,
    }
    return context, arguments, before


@pytest.mark.parametrize("history_existed", [False, True])
def test_rollback_continues_after_directory_sync_failure(tmp_path, monkeypatch, history_existed):
    context, arguments, before = _transaction(tmp_path, history_existed=history_existed)
    primary = OSError("publication directory sync failed")

    fail_directory_sync(monkeypatch, context.result_csv, before[context.result_csv], primary,
                        later_error=OSError("rollback directory sync failed"))
    with pytest.raises(posthoc_context.PosthocError, match="recovery incomplete") as caught:
        storage.commit_result_files(context, **arguments)

    assert caught.value.__cause__ is primary
    assert str(arguments["backup_dir"]) in str(caught.value)
    for path, original in before.items():
        assert (path.read_bytes() if path.exists() else None) == original
        assert path.name in str(caught.value)
        if original is not None:
            assert (arguments["backup_dir"] / path.name).read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("history_existed", [False, True])
def test_failed_csv_restore_does_not_skip_other_files(tmp_path, monkeypatch, history_existed):
    context, arguments, before = _transaction(tmp_path, history_existed=history_existed)
    primary = OSError("publication directory sync failed")
    replace = os.replace

    def fail_csv(source, destination):
        if destination == context.result_csv and Path(source).read_bytes() == before[destination]:
            raise PermissionError("CSV restore denied")
        return replace(source, destination)

    fail_directory_sync(monkeypatch, context.result_csv, before[context.result_csv], primary)
    monkeypatch.setattr(os, "replace", fail_csv)
    with pytest.raises(posthoc_context.PosthocError, match="CSV restore denied") as caught:
        storage.commit_result_files(context, **arguments)

    assert caught.value.__cause__ is primary
    assert context.result_csv.read_bytes() != before[context.result_csv]
    for path in (context.static_meta_path, context.collection_history_path):
        assert (path.read_bytes() if path.exists() else None) == before[path]
    for path, original in before.items():
        if original is not None:
            assert (arguments["backup_dir"] / path.name).read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_missing_history_backup_is_not_treated_as_new_history(tmp_path, monkeypatch):
    context, arguments, before = _transaction(tmp_path, history_existed=True)
    (arguments["backup_dir"] / collection_history.COLLECTION_HISTORY_NAME).unlink()

    fail_directory_sync(monkeypatch, context.result_csv, before[context.result_csv],
                        OSError("publication directory sync failed"))
    with pytest.raises(posthoc_context.PosthocError, match="collection_history.json"):
        storage.commit_result_files(context, **arguments)

    assert context.result_csv.read_bytes() == before[context.result_csv]
    assert context.static_meta_path.read_bytes() == before[context.static_meta_path]
    assert context.collection_history_path.is_file()
    assert context.collection_history_path.read_bytes() != before[context.collection_history_path]
    assert list(tmp_path.glob(".*.tmp")) == []
