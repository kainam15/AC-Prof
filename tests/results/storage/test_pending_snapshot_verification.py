"""Every committed reader observes interrupted replacement transactions."""
import json
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

import pytest

from acprof.results.csv import ResultValidationError, read_result_csv_snapshot, verify_result_source
from acprof.result_store import writer as writer_module
from acprof.result_store.protocol import publication_lock
from acprof.result_store.reader import committed_content_digest, read_snapshot
from acprof.result_store.writer import ResultWriter
from tests.results.storage.test_committed_digest import expected_digest, make_store
from tests.results.storage.test_sequential_result_reads import artifact_hashes


@pytest.fixture
def interrupted_store(tmp_path, monkeypatch):
    layout = make_store(tmp_path)
    snapshot = read_result_csv_snapshot(layout.result_csv)
    with ResultWriter(tmp_path) as writer:
        def fail_replacement(*_args):
            raise OSError("interrupted before module replacement")

        with monkeypatch.context() as scoped:
            scoped.setattr(writer_module, "write_bytes", fail_replacement)
            with pytest.raises(OSError, match="before module replacement"):
                writer.update_rows([{**row, "latency_app_s": "9"} for row in writer.rows.values()],
                                   reason="fixture replacement")
    assert layout.result_store_path("pending.json").exists()
    return layout, snapshot


@pytest.mark.parametrize("operation", ["snapshot", "verify", "digest"])
def test_interrupted_replacement_requires_recovery_even_before_bytes_change(interrupted_store, operation):
    layout, snapshot = interrupted_store
    before = artifact_hashes(layout)
    calls = {
        "snapshot": lambda: read_snapshot(layout.root),
        "verify": lambda: verify_result_source(layout.result_csv, snapshot.sha256),
        "digest": lambda: committed_content_digest(layout.root, cancelled=lambda: False, budget=[32 * 1024 * 1024]),
    }
    with pytest.raises(ResultValidationError, match="replacement requires recovery"):
        calls[operation]()
    assert artifact_hashes(layout) == before
    with publication_lock(layout, exclusive=True, blocking=False):
        pass
    with ResultWriter(layout.root):
        pass
    assert read_snapshot(layout.root) == snapshot
    calls[operation]()


@pytest.mark.parametrize("operation", ["snapshot", "verify", "digest"])
def test_committed_replacement_with_leftover_journal_is_readable(tmp_path, monkeypatch, operation):
    layout = make_store(tmp_path)
    with ResultWriter(tmp_path) as writer:
        def fail_cleanup(*_args):
            raise OSError("journal cleanup interrupted")

        with monkeypatch.context() as scoped:
            scoped.setattr(writer_module, "clear_pending", fail_cleanup)
            with pytest.raises(OSError, match="journal cleanup interrupted"):
                writer.update_rows([{**row, "latency_app_s": "9"} for row in writer.rows.values()],
                                   reason="fixture replacement")
    pending = layout.result_store_path("pending.json")
    expected, consumed = expected_digest(layout)
    budget = [consumed + pending.stat().st_size]
    snapshot = read_result_csv_snapshot(layout.result_csv)
    before = artifact_hashes(layout)
    if operation == "snapshot":
        assert read_snapshot(layout.root).rows[0]["latency_app_s"] == "9"
    elif operation == "verify":
        verify_result_source(layout.result_csv, snapshot.sha256)
    else:
        assert committed_content_digest(layout.root, cancelled=lambda: False, budget=budget) == expected
        assert budget == [0]
    assert artifact_hashes(layout) == before


def test_pending_append_remains_a_readable_committed_prefix(tmp_path):
    layout = make_store(tmp_path)
    snapshot = read_result_csv_snapshot(layout.result_csv)
    manifest = json.loads(layout.result_csv.read_text())
    pending = layout.result_store_path("pending.json")
    pending.write_text(json.dumps({"before": manifest, "after": {}, "replacements": []}))
    with layout.layer_csv("summary").open("ab") as stream:
        stream.write(b'\xff,"unfinished append')
    expected, consumed = expected_digest(layout)
    budget = [consumed + pending.stat().st_size]
    before = artifact_hashes(layout)
    assert read_snapshot(layout.root) == snapshot
    verify_result_source(layout.result_csv, snapshot.sha256)
    assert committed_content_digest(layout.root, cancelled=lambda: False, budget=budget) == expected
    assert budget == [0]
    assert artifact_hashes(layout) == before


@pytest.mark.parametrize("operation", ["verify", "digest"])
def test_pending_read_can_be_cancelled_without_returning_verified_content(interrupted_store, monkeypatch, operation):
    layout, snapshot = interrupted_store
    pending = layout.result_store_path("pending.json")
    cancelled = Event()
    opened = []
    original = Path.open

    def cancel_pending(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        if path == pending:
            opened.append(stream)
            cancelled.set()
        return stream

    monkeypatch.setattr(Path, "open", cancel_pending)
    with pytest.raises((CancelledError, InterruptedError)):
        if operation == "verify":
            verify_result_source(layout.result_csv, snapshot.sha256, cancelled=cancelled.is_set)
        else:
            committed_content_digest(layout.root, cancelled=cancelled.is_set, budget=[32 * 1024 * 1024])
    assert opened and all(stream.closed for stream in opened)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


def test_duplicate_digest_pending_read_respects_the_shared_byte_budget(interrupted_store):
    layout, _snapshot = interrupted_store
    pending = layout.result_store_path("pending.json")
    manifest = json.loads(layout.result_csv.read_text())
    # Leave enough for all regular metadata but not the large pending document.
    pending.write_text(json.dumps({"before": manifest, "replacements": [], "detail": "x" * (1024 * 1024)}))
    budget = [expected_digest(layout)[1] + 4096]
    with pytest.raises(ValueError, match="byte budget"):
        committed_content_digest(layout.root, cancelled=lambda: False, budget=budget)
    assert budget[0] <= 0
    with publication_lock(layout, exclusive=True, blocking=False):
        pass
