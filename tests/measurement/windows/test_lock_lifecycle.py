"""Owned lock resources are released on failed acquisition, cancellation and close."""
import asyncio
import errno
import fcntl
import os
from concurrent.futures import CancelledError

import pytest

from acprof.host import run_state

FAILURES = [OSError, RuntimeError, CancelledError, asyncio.CancelledError, KeyboardInterrupt, SystemExit]


@pytest.fixture(params=["result", "measurement"])
def lock_factory(request, tmp_path):
    return (lambda: run_state.ResultDirectoryLock(tmp_path)) if request.param == "result" else run_state.MeasurementLock


def assert_released(factory):
    with factory():
        pass


@pytest.mark.parametrize("failure_type", FAILURES)
def test_validation_failure_releases_owned_descriptor(lock_factory, monkeypatch, failure_type):
    failure = failure_type("lock validation failed")
    descriptors = []
    lock = lock_factory()

    def fail(descriptor, *_args, **_kwargs):
        descriptors.append(descriptor)
        raise failure

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(run_state, "_validate_lock_descriptor", fail)
            expected = run_state.RunStateError if failure_type is OSError else failure_type
            with pytest.raises(expected) as caught:
                lock.__enter__()
        assert (caught.value.__cause__ if failure_type is OSError else caught.value) is failure
        assert len(descriptors) == 1
        with pytest.raises(OSError, match="Bad file descriptor") as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
        assert_released(lock_factory)
    finally:
        lock.__exit__()
        # Keep the failing baseline test from leaking its unowned raw descriptor.
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise


@pytest.mark.parametrize("failure_type", FAILURES)
def test_failure_after_flock_releases_every_acquired_resource(lock_factory, monkeypatch, failure_type):
    failure = failure_type("post-acquisition failure")
    flock = fcntl.flock
    lock = lock_factory()

    def fail(descriptor, flags):
        flock(descriptor, flags)
        raise failure

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(fcntl, "flock", fail)
            expected = run_state.RunStateError if failure_type is OSError else failure_type
            with pytest.raises(expected) as caught:
                lock.__enter__()
        assert (caught.value.__cause__ if failure_type is OSError else caught.value) is failure
        assert_released(lock_factory)
    finally:
        lock.__exit__()


def test_reentry_does_not_replace_or_release_the_owned_lock(lock_factory):
    lock = lock_factory()
    with lock:
        with pytest.raises(run_state.RunStateError, match="持有"):
            lock.__enter__()
        with pytest.raises(run_state.RunStateError):
            with lock_factory():
                pytest.fail("reentry released an owned lock")
    assert_released(lock_factory)
    with lock:
        pass


class FailingClose:
    def __init__(self, resource, error):
        self.resource = resource
        self.error = error
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.resource, name)

    def close(self):
        self.calls += 1
        self.resource.close()
        if self.error is not None:
            raise self.error


class ClosingDescriptor:
    def __init__(self, descriptor, close, error):
        self.descriptor = descriptor
        self.close_fd = close
        self.error = error
        self.calls = 0
        self.closed = False

    def close(self):
        self.calls += 1
        self.close_fd(self.descriptor)
        self.closed = True
        if self.error is not None:
            raise self.error


@pytest.fixture
def close_faults(monkeypatch):
    open_fd, close_fd, socket = os.open, os.close, run_state.socket.socket
    faults = {"fd_error": OSError("descriptor close failed"), "socket_error": OSError("socket close failed"),
              "descriptors": [], "sockets": []}
    owned = {}

    def open_descriptor(*args, **kwargs):
        descriptor = open_fd(*args, **kwargs)
        resource = ClosingDescriptor(descriptor, close_fd, faults["fd_error"])
        owned[descriptor] = resource
        faults["descriptors"].append(resource)
        return descriptor

    def close_descriptor(descriptor):
        if descriptor in owned:
            return owned[descriptor].close()
        return close_fd(descriptor)

    def open_socket(*args, **kwargs):
        resource = FailingClose(socket(*args, **kwargs), faults["socket_error"])
        faults["sockets"].append(resource)
        return resource

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "open", open_descriptor)
        scoped.setattr(os, "close", close_descriptor)
        scoped.setattr(run_state.socket, "socket", open_socket)
        yield faults
    for resource in faults["sockets"]:
        resource.resource.close()
    for descriptor, resource in owned.items():
        if not resource.closed:
            try:
                close_fd(descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise


@pytest.mark.parametrize("failure_type", FAILURES)
def test_body_failure_survives_close_errors(lock_factory, close_faults, failure_type):
    lock = lock_factory()
    failure = failure_type("protected operation failed")
    with pytest.raises(failure_type) as caught:
        with lock:
            raise failure
    assert caught.value is failure
    notes = "\n".join(failure.__notes__)
    assert "descriptor close failed" in notes
    if close_faults["sockets"]:
        assert "socket close failed" in notes
    # Cleanup is consumed even if close reported an error; do not close twice.
    lock.__exit__()
    assert all(resource.calls == 1 for resource in [*close_faults["descriptors"], *close_faults["sockets"]])


def test_close_only_failure_keeps_all_diagnostics_and_allows_reuse(lock_factory, close_faults):
    lock = lock_factory()
    with pytest.raises(OSError, match="descriptor close failed") as caught:
        with lock:
            pass
    assert caught.value is close_faults["fd_error"]
    if close_faults["sockets"]:
        assert "socket close failed" in "\n".join(caught.value.__notes__)
    lock.__exit__()
    assert all(resource.calls == 1 for resource in [*close_faults["descriptors"], *close_faults["sockets"]])
    close_faults["fd_error"] = close_faults["socket_error"] = None
    with lock:
        pass


def test_failed_close_cannot_release_a_later_lock_owner(lock_factory, close_faults):
    close_faults["socket_error"] = None
    old = lock_factory()
    with pytest.raises(OSError, match="descriptor close failed"):
        with old:
            pass
    released = close_faults["descriptors"][0]
    assert released.calls == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(released.descriptor)
    close_faults["fd_error"] = None
    with lock_factory():
        old.__exit__()
        assert released.calls == 1
        with pytest.raises(run_state.RunStateError):
            with lock_factory():
                pytest.fail("repeated close released the new owner's lock")


@pytest.mark.parametrize("failure_type", [BlockingIOError, KeyboardInterrupt])
def test_idle_probe_preserves_contention_or_cancellation_when_close_fails(tmp_path, monkeypatch, failure_type):
    with run_state.ResultDirectoryLock(tmp_path):
        pass
    failure = failure_type("probe failed")
    close = os.close
    descriptors = []

    def fail_flock(*_args):
        raise failure

    def fail_close(descriptor):
        descriptors.append(descriptor)
        close(descriptor)
        raise OSError("probe close failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(fcntl, "flock", fail_flock)
        scoped.setattr(os, "close", fail_close)
        expected = run_state.RunStateError if failure_type is BlockingIOError else failure_type
        with pytest.raises(expected) as caught:
            run_state.check_result_directory_idle(tmp_path)
    assert (caught.value.__cause__ if failure_type is BlockingIOError else caught.value) is failure
    assert "probe close failed" in "\n".join(caught.value.__notes__)
    assert len(descriptors) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptors[0])


@pytest.mark.parametrize("cancellation_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
def test_cleanup_interruption_preserves_cause_and_releases_remaining_resources(lock_factory, close_faults,
                                                                             cancellation_type):
    failure = ValueError("protected operation failed")
    cancellation = cancellation_type("descriptor close interrupted")
    close_faults["fd_error"] = cancellation
    with pytest.raises(cancellation_type) as caught:
        with lock_factory():
            raise failure
    assert caught.value is cancellation
    assert caught.value.__cause__ is failure
    assert all(resource.calls == 1 for resource in [*close_faults["descriptors"], *close_faults["sockets"]])
    close_faults["fd_error"] = close_faults["socket_error"] = None
    assert_released(lock_factory)


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_run_state_constructor_releases_both_locks_when_cleanup_fails(tmp_path, monkeypatch, close_faults,
                                                                    failure_type):
    failure = failure_type("run state load failed")

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(run_state, "load_run_state", fail)
    with pytest.raises(failure_type) as caught:
        run_state.RunState(tmp_path, {}, resume=True, project_dir=str(tmp_path))
    assert caught.value is failure
    assert len(close_faults["descriptors"]) == 2
    assert all(resource.calls == 1 for resource in [*close_faults["descriptors"], *close_faults["sockets"]])
    assert "descriptor close failed" in "\n".join(failure.__notes__)
    assert "socket close failed" in "\n".join(failure.__notes__)
    close_faults["fd_error"] = close_faults["socket_error"] = None
    assert_released(run_state.MeasurementLock)
    assert_released(lambda: run_state.ResultDirectoryLock(tmp_path))


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_run_state_close_preserves_save_failure_and_releases_both_locks(tmp_path, monkeypatch, close_faults,
                                                                     failure_type):
    failure = failure_type("run state save failed")
    calls = []

    def save(_state):
        calls.append(None)
        if len(calls) > 1:
            raise failure

    monkeypatch.setattr(run_state, "load_run_state", lambda _path: {"attempts": [], "status": "collecting"})
    monkeypatch.setattr(run_state, "validate_resume", lambda *_args, **_kwargs: "collection")
    monkeypatch.setattr(run_state.RunState, "save", save)
    state = run_state.RunState(tmp_path, {}, resume=True, project_dir=str(tmp_path))
    with pytest.raises(failure_type) as caught:
        state.close()
    assert caught.value is failure
    assert all(resource.calls == 1 for resource in [*close_faults["descriptors"], *close_faults["sockets"]])
    notes = "\n".join(failure.__notes__)
    assert "descriptor close failed" in notes
    assert "socket close failed" in notes
    close_faults["fd_error"] = close_faults["socket_error"] = None
    assert_released(run_state.MeasurementLock)
    assert_released(lambda: run_state.ResultDirectoryLock(tmp_path))
