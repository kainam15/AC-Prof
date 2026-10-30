"""A failed publication cannot reuse partially updated writer state."""
from __future__ import annotations

import asyncio
import fcntl
import json
import os

import pytest
from tests.results.storage.test_incremental_results import initialized, receipt
from tests.results.storage.test_recovery_hash_state import assert_committed_hashes
from tests.results.storage.test_result_writer_lifecycle import evidence_bytes, fault_streams
from tests.results.storage.test_sequential_result_reads import artifact_hashes
from tests.support.result_store_fixtures import assert_writer_lease_available

from acprof.result_store.protocol import publication_lock
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import ResultValidationError
from acprof.results.layers import read_result_layers


def operation_for(writer, source, operation):
    if operation == "append":
        return lambda: writer.append_window(source)
    if operation == "freeze":
        return writer.freeze
    rows = [dict(row) for row in writer.rows.values()]
    if operation == "replace":
        rows[0]["latency_s"] = "0.75"
    return lambda: writer.update_rows(rows, reason="fixture publication", state="processing")


def assert_publication_rejected(writer, source):
    before = artifact_hashes(writer.layout)
    for operation in (lambda: writer.append_window(source),
                      lambda: writer.update_rows([], reason="must not publish"),
                      writer.freeze):
        with pytest.raises(ResultValidationError, match="requires recovery"):
            operation()
    assert artifact_hashes(writer.layout) == before


@pytest.mark.parametrize("operation", ["append", "replace", "state", "freeze"])
@pytest.mark.parametrize("phase", ["acquire", "unlock", "close"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, asyncio.CancelledError])
def test_publication_lock_failures_require_recovery(
        tmp_path, monkeypatch, operation, phase, error_type):
    layout = initialized(tmp_path)
    first, second = receipt(tmp_path), receipt(tmp_path, 1)
    original_evidence = evidence_bytes(layout)
    error = error_type(f"fixture publication {phase} failed")
    writer = ResultWriter(tmp_path)
    with writer:
        assert writer.append_window(first)
        before = json.loads(layout.result_csv.read_bytes())
        publish = operation_for(writer, second, operation)
        lock = layout.result_store_path("lock")
        lock_info = lock.stat()
        flock = fcntl.flock
        calls = []

        def fail_flock(fd, action):
            info = os.fstat(fd)
            target = (info.st_dev, info.st_ino) == (lock_info.st_dev, lock_info.st_ino)
            if target and phase == "acquire" and action == fcntl.LOCK_EX:
                calls.append(action)
                raise error
            result = flock(fd, action)
            if target and phase == "unlock" and action == fcntl.LOCK_UN:
                calls.append(action)
                raise error
            return result

        with monkeypatch.context() as guard:
            if phase == "close":
                opened = fault_streams(guard, {(lock, "a+b"): {"close_error": error}})
            else:
                guard.setattr(fcntl, "flock", fail_flock)
            with pytest.raises(BaseException) as caught:
                publish()
        assert caught.value is error
        if phase == "close":
            assert len(opened) == 1
            assert opened[0].closed and opened[0].close_calls == 1
        else:
            assert len(calls) == 1
        with publication_lock(layout, exclusive=True, blocking=False):
            pass
        committed = phase != "acquire"
        current = json.loads(layout.result_csv.read_bytes())
        assert current["sequence"] == before["sequence"] + int(committed)
        assert current["row_count"] == 1 + int(committed and operation == "append")
        assert len(read_result_layers(tmp_path)[1]) == current["row_count"]
        assert_publication_rejected(writer, second)
        assert writer.failed
    assert_writer_lease_available(layout)
    with writer:
        assert not writer.failed
        assert_committed_hashes(writer)
        result = publish()
        if operation in {"append", "replace"}:
            assert result is (not committed)
            assert not publish()
        elif operation == "freeze":
            assert result["row_count"] == 1
        assert writer.manifest["state"] == {"append": "collecting", "freeze": "complete"}.get(
            operation, "processing")
        assert writer.append_window(second) is (operation != "append")
        assert not writer.append_window(second)
        assert_committed_hashes(writer)
    assert evidence_bytes(layout) == original_evidence
    rows = read_result_layers(tmp_path)[1]
    assert [row["repeat_idx"] for row in rows] == ["0", "1"]
    assert rows[0]["latency_s"] == ("0.75" if operation == "replace" else "nan")


class InterruptedRows(dict):
    def __init__(self, original, error):
        super().__init__(original)
        self.error = error

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        raise self.error


class InterruptedWindows(dict):
    def __init__(self, original, error):
        super().__init__(original)
        self.error = error

    def update(self, other):
        super().update(other)
        raise self.error


@pytest.mark.parametrize("phase", ["commit-return", "row-index", "window-index"])
@pytest.mark.parametrize("error_type", [MemoryError, KeyboardInterrupt, asyncio.CancelledError])
def test_interrupted_bookkeeping_requires_recovery(tmp_path, monkeypatch, phase, error_type):
    layout = initialized(tmp_path)
    first, second = receipt(tmp_path), receipt(tmp_path, 1)
    original_evidence = evidence_bytes(layout)
    writer = ResultWriter(tmp_path)
    error = error_type(f"fixture interrupted {phase}")
    with writer:
        commit = writer._commit

        def interrupted_commit(*args, **kwargs):
            commit(*args, **kwargs)
            raise error

        if phase == "row-index":
            writer.rows = InterruptedRows(writer.rows, error)
        elif phase == "window-index":
            writer.windows = InterruptedWindows(writer.windows, error)
        with monkeypatch.context() as guard:
            if phase == "commit-return":
                guard.setattr(writer, "_commit", interrupted_commit)
            with pytest.raises(BaseException) as caught:
                writer.append_window(first)
        assert caught.value is error
        assert [row["repeat_idx"] for row in read_result_layers(tmp_path)[1]] == ["0"]
        assert_publication_rejected(writer, first)
        assert writer.failed
    assert_writer_lease_available(layout)
    with writer:
        assert not writer.append_window(first)
        assert writer.append_window(second)
        assert not writer.append_window(second)
        assert_committed_hashes(writer)
    assert evidence_bytes(layout) == original_evidence
    assert [row["repeat_idx"] for row in read_result_layers(tmp_path)[1]] == ["0", "1"]
