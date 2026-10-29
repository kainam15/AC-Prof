"""Model Store GC plans scale linearly and retain current ownership evidence."""
import errno
import fcntl
import os
import threading
from unittest.mock import patch

import pytest
from model_store_fixtures import ModelStoreFixture, locked_store

from acprof.host import model_store


class TestModelStoreGc:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.root = tmp_path
        self.store = ModelStoreFixture(tmp_path)

    @pytest.mark.parametrize("entry_count", [8, 80])
    def test_target_preview_has_fixed_candidate_scan_count_as_entries_grow(self, entry_count):
        for index in range(1, entry_count + 1):
            self.store.entry(index, [self.store.blob(str(index), 4096)])
        with patch.object(model_store, "prune_candidates", wraps=model_store.prune_candidates) as scans:
            result = model_store.prune_store(root=self.root, target_bytes=0)
        assert len(result["entries"]) == entry_count
        assert result["reclaimable_bytes"] == entry_count * 4096
        assert scans.call_count == 1

    def test_target_lru_frees_shared_blob_only_after_its_last_reference(self):
        shared = self.store.blob('shared', 100)
        first = self.store.entry(1, [shared, self.store.blob('first', 40)])
        second = self.store.entry(2, [shared, self.store.blob('second', 40)])
        used = model_store.disk_report(self.root)['total_bytes']
        one = model_store.prune_store(root=self.root, target_bytes=used - 40)
        assert (one) == ({'entries': [first], 'reclaimable_bytes': 40})
        two = model_store.prune_store(root=self.root, target_bytes=used - 41)
        assert (two) == ({'entries': [first, second], 'reclaimable_bytes': 180})

    def test_store_lock_rejects_symlink_lock_file(self):
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / "lock-target"
        target.write_text("sentinel")
        (self.root / ".lock").symlink_to(target)
        with pytest.raises(OSError, check=lambda error: error.errno == errno.ELOOP):
            with model_store.store_lock(self.root):
                pass
        assert target.read_text() == "sentinel"

    def test_store_lock_rejects_hardlinked_lock_file(self):
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / "lock-target"
        target.write_text("sentinel")
        os.link(target, self.root / ".lock")
        with pytest.raises(OSError, match="lock is not a regular file owned only by this user"):
            with model_store.store_lock(self.root):
                pass
        assert target.read_text() == "sentinel"

    @pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
    def test_prune_rejects_linked_lease_file(self, link_kind):
        key = self.store.entry(1, [self.store.blob("live", 100)])
        target = self.root / "lease-target"
        target.write_text("sentinel")
        lease_path = self.root / f"{key}.lease"
        if link_kind == "symlink":
            lease_path.symlink_to(target)
        else:
            os.link(target, lease_path)
        with pytest.raises(OSError, check=lambda error: (
            error.errno == errno.ELOOP if link_kind == "symlink"
            else str(error) == "Model Store lease lock is not a regular file owned only by this user"
        )):
            model_store.prune_store(root=self.root)
        assert target.read_text() == "sentinel"

    def test_apply_rechecks_new_leases_and_preserves_the_callers_keep_set(self):
        blob = self.store.blob('live', 100)
        key = self.store.entry(1, [blob])
        approved = set(model_store.prune_store(root=self.root)['entries'])
        with (self.root / (key + '.lease')).open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_SH)
            keep = set()
            result = model_store.prune_store(root=self.root, apply=True, keep=keep, approved_entries=approved)
        assert (result['entries']) == ([])
        assert (keep) == (set())
        assert (blob.exists())
        assert ((self.root / 'entries' / key).exists())

    def test_apply_protects_new_entries_and_their_shared_weights(self):
        shared, old = self.store.blob('shared', 100), self.store.blob('old', 40)
        first = self.store.entry(1, [shared, old])
        approved = set(model_store.prune_store(root=self.root)['entries'])
        second = self.store.entry(2, [shared])
        result = model_store.prune_store(root=self.root, apply=True, approved_entries=approved)
        assert (result) == ({'entries': [first], 'reclaimable_bytes': 40})
        assert ((self.root / 'entries' / second).exists())
        assert (shared.exists())
        assert not (old.exists())

    def test_prune_reclaims_abandoned_hf_incomplete_blobs(self):
        partial = self.store.blob('deadbeef.12345678.incomplete', 37)
        lock_file = self.store.blob('deadbeef.lock', 11)
        preview = model_store.prune_store(root=self.root)
        assert preview == {'entries': [], 'reclaimable_bytes': 37}
        assert partial.exists()
        assert lock_file.exists()
        applied = model_store.prune_store(root=self.root, apply=True)
        assert applied == preview
        assert not partial.exists()
        assert lock_file.exists()

    def test_waiting_for_another_process_lock_is_cancelled_before_mutation(self):
        blob = self.store.blob('pending', 100)
        key = self.store.entry(1, [blob])
        cancel, waiting = threading.Event(), threading.Event()
        errors = []

        def prune():
            try:
                model_store.prune_store(root=self.root, apply=True, cancel=cancel, on_wait=waiting.set)
            except Exception as exc:
                errors.append(exc)

        with locked_store(self.root):
            thread = threading.Thread(target=prune, daemon=True)
            thread.start()
            try:
                assert (waiting.wait(2)), 'preview must report the held lock'
                cancel.set()
                thread.join(timeout=2)
                assert not (thread.is_alive()), 'cancel must not wait for the other process to unlock'
            finally:
                cancel.set()
        thread.join(timeout=2)
        assert (type(errors[0]).__name__) == ('ModelStoreCancelled')
        assert (blob.exists())
        assert ((self.root / 'entries' / key).exists())
