"""Writer ownership survives recovery, nested entry and cleanup failures."""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import pytest
from tests.results.storage.test_incremental_results import initialized, receipt
from tests.results.storage.test_recovery_hash_state import assert_committed_hashes, hash_digests
from tests.support.result_store_fixtures import assert_writer_lease_available

from acprof.result_store import writer as writer_module
from acprof.result_store.protocol import publication_lock
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import ResultValidationError
from acprof.results.layers import read_result_layers


class FaultStream:
    def __init__(self, path, stream, *, close_error=None, operation=None, operation_error=None):
        self.path, self.stream = path, stream
        self.close_error = close_error
        self.operation, self.operation_error = operation, operation_error
        self.close_calls = 0

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.close_calls += 1
        # An I/O error reported by close must not make this test leak real FDs.
        self.stream.close()
        if self.close_error is not None:
            raise self.close_error

    def write(self, raw):
        count = self.stream.write(raw)
        if self.operation == "write":
            raise self.operation_error
        return count

    def flush(self):
        self.stream.flush()
        if self.operation == "flush":
            raise self.operation_error


def fault_streams(monkeypatch, faults):
    opened = []
    original = Path.open

    def open_stream(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        fault = faults.get((path, mode))
        if fault is None:
            return stream
        wrapped = FaultStream(path, stream, **fault)
        opened.append(wrapped)
        return wrapped

    monkeypatch.setattr(Path, "open", open_stream)
    return opened


def evidence_bytes(layout):
    return {path.relative_to(layout.root): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("raw/windows", "raw/telemetry")
            for path in layout.path(directory).rglob("*") if path.is_file()}


def assert_released(layout, opened):
    assert opened
    assert all(stream.closed and stream.close_calls == 1 for stream in opened)
    assert_writer_lease_available(layout)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


def test_nested_entry_preserves_outer_lease_and_writer(tmp_path):
    layout = initialized(tmp_path)
    first, second = receipt(tmp_path), receipt(tmp_path, 1)
    writer = ResultWriter(tmp_path)
    with writer:
        with pytest.raises(ResultValidationError, match="already active"):
            with writer:
                pass
        with pytest.raises(BlockingIOError):
            with ResultWriter(tmp_path):
                pytest.fail("nested entry released the outer writer lease")
        assert writer.append_window(first)
    assert_writer_lease_available(layout)
    with writer:
        assert writer.append_window(second)
    assert [row["repeat_idx"] for row in read_result_layers(tmp_path)[1]] == ["0", "1"]


@pytest.mark.parametrize("failure_point", ["append", "clear_pending"])
def test_same_writer_recovers_a_failed_transaction_before_reuse(tmp_path, monkeypatch, failure_point):
    layout = initialized(tmp_path)
    first, second = receipt(tmp_path), receipt(tmp_path, 1)
    before = evidence_bytes(layout)
    assert before
    writer = ResultWriter(tmp_path)
    primary = OSError("fixture transaction interrupted")
    with writer:
        append = writer._append_bytes

        def fail_append(layer, raw):
            append(layer, raw)
            raise primary

        def fail_clear(_layout):
            raise primary

        with monkeypatch.context() as guard:
            if failure_point == "append":
                guard.setattr(writer, "_append_bytes", fail_append)
            else:
                guard.setattr(writer_module, "clear_pending", fail_clear)
            with pytest.raises(OSError) as caught:
                writer.append_window(first)
        assert caught.value is primary
        assert writer.failed
        with pytest.raises(ResultValidationError, match="requires recovery"):
            writer.append_window(second)
    assert_writer_lease_available(layout)
    assert layout.result_store_path("pending.json").is_file()
    with writer:
        assert not writer.failed
        assert not layout.result_store_path("pending.json").exists()
        assert writer.append_window(first) is (failure_point == "append")
        assert not writer.append_window(first)
        assert writer.append_window(second)
        assert_committed_hashes(writer)
    assert evidence_bytes(layout) == before
    assert [row["repeat_idx"] for row in read_result_layers(tmp_path)[1]] == ["0", "1"]


ERROR_PAIRS = [
    (OSError, OSError),
    (KeyboardInterrupt, OSError),
    (asyncio.CancelledError, OSError),
    (OSError, KeyboardInterrupt),
    (OSError, asyncio.CancelledError),
    (KeyboardInterrupt, asyncio.CancelledError),
]


@pytest.mark.parametrize("phase", ["recovery", "body"])
@pytest.mark.parametrize("primary_type,close_type", ERROR_PAIRS)
def test_writer_close_preserves_failures_and_cancellation(
        tmp_path, monkeypatch, phase, primary_type, close_type):
    layout = initialized(tmp_path)
    source = receipt(tmp_path)
    before = layout.result_csv.read_bytes(), evidence_bytes(layout)
    primary, secondary = primary_type("fixture operation failed"), close_type("fixture lease close failed")
    writer = ResultWriter(tmp_path)

    def fail_recovery(_layout):
        raise primary

    with monkeypatch.context() as guard:
        opened = fault_streams(guard, {(layout.result_store_path("writer.lock"), "a+b"):
                                      {"close_error": secondary}})
        if phase == "recovery":
            guard.setattr(writer_module, "recover_projection", fail_recovery)
        with pytest.raises(BaseException) as caught:
            with writer:
                raise primary
    if isinstance(primary, Exception) and not isinstance(secondary, Exception):
        assert caught.value is secondary
        assert caught.value.__cause__ is primary
        assert "fixture operation failed" in "\n".join(getattr(caught.value, "__notes__", ()))
    else:
        assert caught.value is primary
        assert "fixture lease close failed" in "\n".join(getattr(caught.value, "__notes__", ()))
    assert_released(layout, opened)
    assert (layout.result_csv.read_bytes(), evidence_bytes(layout)) == before
    writer.__exit__(None, None, None)
    assert all(stream.close_calls == 1 for stream in opened)
    with pytest.raises(ResultValidationError, match="requires recovery"):
        writer.append_window(source)
    with writer:
        assert writer.append_window(source)


@pytest.mark.parametrize("close_type", [OSError, asyncio.CancelledError])
def test_successful_body_still_reports_lease_close_failure(tmp_path, monkeypatch, close_type):
    layout = initialized(tmp_path)
    source = receipt(tmp_path)
    error = close_type("fixture lease close failed")
    writer = ResultWriter(tmp_path)
    with monkeypatch.context() as guard:
        opened = fault_streams(guard, {(layout.result_store_path("writer.lock"), "a+b"):
                                      {"close_error": error}})
        with pytest.raises(BaseException) as caught:
            with writer:
                assert writer.append_window(source)
    assert caught.value is error
    assert "result writer lease" in "\n".join(getattr(caught.value, "__notes__", ()))
    assert_released(layout, opened)
    writer.__exit__(None, None, None)
    with pytest.raises(ResultValidationError, match="requires recovery"):
        writer.append_window(source)
    with writer:
        assert not writer.append_window(source)
    assert len(read_result_layers(tmp_path)[1]) == 1


@pytest.mark.parametrize("operation", ["write", "flush", "fsync"])
@pytest.mark.parametrize("primary_type", [OSError, KeyboardInterrupt, asyncio.CancelledError])
def test_append_failure_survives_both_stream_and_lease_close_errors(
        tmp_path, monkeypatch, operation, primary_type):
    layout = initialized(tmp_path)
    source = receipt(tmp_path)
    target = layout.layer_csv("summary")
    before = layout.result_csv.read_bytes(), evidence_bytes(layout)
    primary = primary_type(f"fixture append {operation} failed")
    writer = ResultWriter(tmp_path)
    fsync = os.fsync

    with monkeypatch.context() as guard:
        opened = fault_streams(guard, {
            (layout.result_store_path("writer.lock"), "a+b"): {"close_error": OSError("fixture lease close failed")},
            (target, "ab"): {"close_error": OSError("fixture CSV close failed"),
                             "operation": operation, "operation_error": primary},
        })

        def fail_fsync(fd):
            fsync(fd)
            if any(stream.path == target and not stream.closed and stream.fileno() == fd for stream in opened):
                raise primary

        if operation == "fsync":
            guard.setattr(os, "fsync", fail_fsync)
        with pytest.raises(BaseException) as caught:
            with writer:
                previous_hashes = hash_digests(writer.hashes)
                writer.append_window(source)
    assert caught.value is primary
    notes = "\n".join(getattr(primary, "__notes__", ()))
    assert "fixture CSV close failed" in notes
    assert "fixture lease close failed" in notes
    assert writer.failed
    assert hash_digests(writer.hashes) == previous_hashes
    assert (layout.result_csv.read_bytes(), evidence_bytes(layout)) == before
    assert layout.result_store_path("pending.json").is_file()
    assert_released(layout, opened)
    with writer:
        assert writer.append_window(source)
        assert not writer.append_window(source)
        assert_committed_hashes(writer)
    assert evidence_bytes(layout) == before[1]


@pytest.mark.parametrize("close_type", [KeyboardInterrupt, asyncio.CancelledError])
def test_append_close_cancellation_retains_original_io_failure(tmp_path, monkeypatch, close_type):
    layout = initialized(tmp_path)
    source = receipt(tmp_path)
    primary, interrupted = OSError("fixture write failed"), close_type("fixture close interrupted")
    with monkeypatch.context() as guard:
        opened = fault_streams(guard, {
            (layout.layer_csv("summary"), "ab"): {"operation": "write", "operation_error": primary,
                                              "close_error": interrupted},
            (layout.result_store_path("writer.lock"), "a+b"): {"close_error": OSError("fixture lease close failed")},
        })
        with pytest.raises(BaseException) as caught:
            with ResultWriter(tmp_path) as writer:
                writer.append_window(source)
    assert caught.value is interrupted
    assert caught.value.__cause__ is primary
    notes = "\n".join(getattr(caught.value, "__notes__", ()))
    assert "fixture write failed" in notes
    assert "fixture lease close failed" in notes
    assert_released(layout, opened)


def test_append_close_failure_requires_recovery_even_after_successful_fsync(tmp_path, monkeypatch):
    layout = initialized(tmp_path)
    source = receipt(tmp_path)
    before = layout.result_csv.read_bytes()
    primary = OSError("fixture CSV close failed")
    writer = ResultWriter(tmp_path)
    with monkeypatch.context() as guard:
        opened = fault_streams(guard, {(layout.layer_csv("summary"), "ab"): {"close_error": primary}})
        with pytest.raises(OSError) as caught:
            with writer:
                writer.append_window(source)
    assert caught.value is primary
    assert "result CSV append stream" in "\n".join(getattr(primary, "__notes__", ()))
    assert writer.failed
    assert layout.result_csv.read_bytes() == before
    assert_released(layout, opened)
    with writer:
        assert writer.append_window(source)
        assert not writer.append_window(source)
