"""Publication cleanup preserves failures, cancellation and actual lock release."""
import asyncio
import fcntl
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

import pytest

from acprof.result_store.protocol import publication_lock
from acprof.results.layout import ArtifactLayout

FAILURES = [OSError, CancelledError, asyncio.CancelledError, KeyboardInterrupt, SystemExit]


@pytest.fixture
def lock_faults(tmp_path, monkeypatch):
    layout = ArtifactLayout(tmp_path)
    path = layout.result_store_path("lock")
    path.parent.mkdir(parents=True)
    path.touch()
    open_path, flock = Path.open, fcntl.flock
    state = {"close_error": OSError("lock close failed"), "unlock_error": OSError("unlock failed"),
             "acquire_error": None, "cancel_after_acquire": Event(), "close_calls": 0,
             "unlock_calls": 0, "streams": []}

    class LockStream:
        def __init__(self, stream):
            self.stream = stream

        def fileno(self):
            return self.stream.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.close()

        def close(self):
            state["close_calls"] += 1
            self.stream.close()
            if state["close_error"] is not None:
                raise state["close_error"]

    def open_lock(candidate, *args, **kwargs):
        stream = open_path(candidate, *args, **kwargs)
        if candidate == path:
            state["streams"].append(stream)
            return LockStream(stream)
        return stream

    def controlled_flock(descriptor, flags):
        if flags == fcntl.LOCK_UN:
            state["unlock_calls"] += 1
            if state["unlock_error"] is not None:
                raise state["unlock_error"]
        elif state["acquire_error"] is not None:
            raise state["acquire_error"]
        result = flock(descriptor, flags)
        if flags != fcntl.LOCK_UN:
            state["cancel_after_acquire"].set()
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", open_lock)
        scoped.setattr(fcntl, "flock", controlled_flock)
        yield layout, state
    assert state["close_calls"] == 1
    assert all(stream.closed for stream in state["streams"])
    # Even a failed explicit unlock must leave no owned lock after close.
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("failure_type", FAILURES)
@pytest.mark.parametrize("cleanup_failure", ["unlock", "close", "both"])
def test_lock_body_failure_survives_cleanup(lock_faults, exclusive, failure_type, cleanup_failure):
    layout, state = lock_faults
    if cleanup_failure == "unlock":
        state["close_error"] = None
    elif cleanup_failure == "close":
        state["unlock_error"] = None
    failure = failure_type("publication failed")
    with pytest.raises(failure_type) as caught:
        with publication_lock(layout, exclusive=exclusive):
            raise failure
    assert caught.value is failure
    assert state["unlock_calls"] == 1
    notes = "\n".join(failure.__notes__)
    for key in ("unlock_error", "close_error"):
        if state[key] is not None:
            assert str(state[key]) in notes
    assert str(layout.result_store_path("lock")) in notes


@pytest.mark.parametrize("failure_type", FAILURES)
def test_failed_acquisition_closes_without_unlocking(lock_faults, failure_type):
    layout, state = lock_faults
    failure = failure_type("lock acquisition failed")
    state["acquire_error"] = failure
    with pytest.raises(failure_type) as caught:
        with publication_lock(layout, exclusive=False):
            pytest.fail("failed acquisition entered the body")
    assert caught.value is failure
    assert state["unlock_calls"] == 0
    assert "lock close failed" in "\n".join(failure.__notes__)


def test_cancellation_after_acquisition_survives_both_cleanup_errors(lock_faults):
    layout, state = lock_faults
    with pytest.raises(CancelledError) as caught:
        with publication_lock(layout, exclusive=False, cancelled=state["cancel_after_acquire"].is_set):
            pytest.fail("cancelled acquisition entered the body")
    assert state["unlock_calls"] == 1
    notes = "\n".join(caught.value.__notes__)
    assert "unlock failed" in notes
    assert "lock close failed" in notes


@pytest.mark.parametrize("cleanup_failure", ["unlock", "close", "both"])
def test_successful_body_reports_cleanup_failure(lock_faults, cleanup_failure):
    layout, state = lock_faults
    if cleanup_failure == "unlock":
        state["close_error"] = None
    elif cleanup_failure == "close":
        state["unlock_error"] = None
    expected = state["unlock_error"] or state["close_error"]
    with pytest.raises(OSError, match=str(expected)) as caught:
        with publication_lock(layout, exclusive=True):
            pass
    assert caught.value is expected
    assert state["unlock_calls"] == 1
    if cleanup_failure == "both":
        assert "lock close failed" in "\n".join(caught.value.__notes__)


@pytest.mark.parametrize("cancellation_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
def test_cleanup_cancellation_keeps_preceding_failure_and_attempts_close(lock_faults, cancellation_type):
    layout, state = lock_faults
    failure = OSError("publication failed")
    cancellation = cancellation_type("unlock interrupted")
    state["unlock_error"] = cancellation
    with pytest.raises(cancellation_type) as caught:
        with publication_lock(layout, exclusive=True):
            raise failure
    assert caught.value is cancellation
    assert cancellation.__cause__ is failure
    assert "lock close failed" in "\n".join(cancellation.__notes__)
