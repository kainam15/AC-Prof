"""Committed pending evidence must name the actual previous transaction."""
import json
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

import pytest

from acprof.result_store.protocol import publication_lock, read_pending
from acprof.results.csv import read_result_csv_snapshot
from tests.results.storage.test_committed_digest import make_store
from tests.results.storage.test_pending_contract import OPERATIONS, assert_pending_rejected
from tests.results.storage.test_sequential_result_reads import artifact_hashes
from tests.support.result_store_fixtures import assert_writer_lease_available, leave_committed_pending_replacement


@pytest.fixture
def committed_pending(tmp_path, monkeypatch):
    layout = make_store(tmp_path)
    leave_committed_pending_replacement(layout, monkeypatch)
    snapshot = read_result_csv_snapshot(layout.result_csv)
    return layout, snapshot


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("field", ["path", "sha256"])
def test_committed_pending_must_reference_its_actual_previous_commit(committed_pending, operation, field):
    layout, snapshot = committed_pending
    pending = layout.result_store_path("pending.json")
    transaction = json.loads(pending.read_bytes())
    manifest = json.loads(layout.result_csv.read_bytes())
    if field == "path":
        transaction["before"]["commit"][field] = manifest["commit"]["path"]
    else:
        transaction["before"]["commit"][field] = "0" * 64
    pending.write_text(json.dumps(transaction))
    assert_pending_rejected(layout, snapshot, operation)


def test_committed_pending_association_read_honors_cancellation(committed_pending, monkeypatch):
    layout, _snapshot = committed_pending
    manifest = json.loads(layout.result_csv.read_bytes())
    commit_path = layout.contained(manifest["commit"]["path"])
    before = artifact_hashes(layout)
    cancelled = Event()
    opened = []
    original = Path.open

    def cancel_commit(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        if path == commit_path:
            opened.append(stream)
            cancelled.set()
        return stream

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", cancel_commit)
        with publication_lock(layout, exclusive=False), pytest.raises(CancelledError):
            read_pending(layout, manifest, cancelled=cancelled.is_set)
    assert opened and all(stream.closed for stream in opened)
    assert artifact_hashes(layout) == before
    assert_writer_lease_available(layout)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


@pytest.mark.parametrize("shortfall", [0, 1])
def test_committed_pending_association_read_charges_actual_bytes(committed_pending, shortfall):
    layout, _snapshot = committed_pending
    manifest = json.loads(layout.result_csv.read_bytes())
    pending = layout.result_store_path("pending.json")
    transaction = json.loads(pending.read_bytes())
    commit_path = layout.contained(manifest["commit"]["path"])
    budget = [pending.stat().st_size + commit_path.stat().st_size - shortfall]
    before = artifact_hashes(layout)
    with publication_lock(layout, exclusive=False):
        if shortfall:
            with pytest.raises(ValueError, match="byte budget"):
                read_pending(layout, manifest, byte_budget=budget)
            assert budget[0] <= 0
        else:
            assert read_pending(layout, manifest, byte_budget=budget) == transaction
            assert budget == [0]
    assert artifact_hashes(layout) == before
    assert_writer_lease_available(layout)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass
