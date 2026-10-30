"""A cancelled result reader need not wait for another publication to finish."""
import fcntl
from concurrent.futures import CancelledError, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Event

import pytest

from acprof.result_store.protocol import publication_lock
from acprof.results import csv as result_csv
from acprof.results.layout import ArtifactLayout
from tests.results.storage.test_result_row_assembly import row_store
from tests.runtime.containers.test_layer_read_cancellation import source_hashes


@pytest.mark.parametrize("operation", ["read", "verify"])
def test_cancellation_releases_waiting_reader_before_publisher(tmp_path, monkeypatch, operation):
    row_store(tmp_path, 3, count=1, width=1)
    layout = ArtifactLayout(tmp_path)
    before = source_hashes(tmp_path)
    cancelled, attempted = Event(), Event()
    original_flock = fcntl.flock

    def observe(descriptor, flags):
        if flags & fcntl.LOCK_SH:
            attempted.set()
        return original_flock(descriptor, flags)

    monkeypatch.setattr(fcntl, "flock", observe)
    operation_call = (partial(result_csv.read_result_csv_snapshot, layout.result_csv) if operation == "read" else
                      partial(result_csv.verify_result_source, layout.result_csv, before["result_layers.json"]))
    # The publisher is released even if an old reader ignores cancellation.
    with ThreadPoolExecutor(max_workers=1) as pool:
        with publication_lock(layout, exclusive=True):
            future = pool.submit(operation_call, cancelled=cancelled.is_set)
            assert attempted.wait(2), "reader did not reach the contended publication lock"
            cancelled.set()
            with pytest.raises(CancelledError):
                future.result(timeout=2)
            # Cancelling the reader must leave the publisher's exclusive lock intact.
            with pytest.raises(BlockingIOError):
                with publication_lock(layout, exclusive=False, blocking=False):
                    pytest.fail("cancelled reader released the publisher's lock")
    with publication_lock(layout, exclusive=True, blocking=False):
        pass
    assert source_hashes(tmp_path) == before


@pytest.mark.parametrize("exclusive", [False, True])
def test_precancelled_lock_does_not_open_or_create_files(tmp_path, monkeypatch, exclusive):
    def forbidden(*_args, **_kwargs):
        pytest.fail("cancelled lock acquisition performed I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(CancelledError):
        with publication_lock(ArtifactLayout(tmp_path), exclusive=exclusive, cancelled=lambda: True):
            pytest.fail("cancelled lock body ran")
    assert not list(tmp_path.iterdir())


def test_cancellation_racing_with_acquisition_releases_the_acquired_lock(tmp_path, monkeypatch):
    row_store(tmp_path, 3, count=1, width=1)
    layout = ArtifactLayout(tmp_path)
    original_flock = fcntl.flock
    cancelled = Event()

    def cancel_after_acquiring(descriptor, flags):
        original_flock(descriptor, flags)
        if flags & fcntl.LOCK_SH:
            cancelled.set()

    with monkeypatch.context() as scoped:
        scoped.setattr(fcntl, "flock", cancel_after_acquiring)
        with pytest.raises(CancelledError):
            with publication_lock(layout, exclusive=False, cancelled=cancelled.is_set):
                pytest.fail("cancelled lock body ran")
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


def test_cancellable_lock_propagates_non_contention_errors(tmp_path, monkeypatch):
    row_store(tmp_path, 3, count=1, width=1)
    failure = OSError("publication lock unavailable")
    attempts = []

    def fail(*args):
        attempts.append(args)
        raise failure

    monkeypatch.setattr(fcntl, "flock", fail)
    with pytest.raises(OSError, match="publication lock unavailable") as caught:
        with publication_lock(ArtifactLayout(tmp_path), exclusive=False, cancelled=lambda: False):
            pytest.fail("failed lock body ran")
    assert caught.value is failure
    assert len(attempts) == 1
