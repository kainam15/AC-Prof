"""Model Store GC plans scale linearly and retain current ownership evidence."""
import errno
import fcntl
import os
import threading
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.container.model_files import PLAN_FILENAME
from acprof.host import model_store
from tests.support.model_store_fixtures import ModelStoreFixture, locked_store


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('extra_required', [-1, 0, 1])
def test_capacity_target_counts_entry_files_before_evicting_another_model(tmp_path, shared, extra_required):
    store = ModelStoreFixture(tmp_path)
    first_blob = store.blob('first', 10_000)
    second_blob = first_blob if shared else store.blob('second', 10_000)
    first, second = store.entry(1, [first_blob]), store.entry(2, [second_blob])
    entry_sizes = {}
    for key in (first, second):
        entry = tmp_path / 'entries' / key
        refs = entry / 'hf/models--example--test/refs'
        refs.mkdir()
        (refs / 'main').write_bytes(b'a' * 40)
        entry_sizes[key] = len((entry / PLAN_FILENAME).read_bytes()) + 40
    before = sum(entry_sizes.values()) + (10_000 if shared else 20_000)
    first_freed = entry_sizes[first] + (0 if shared else 10_000)
    target = before - first_freed - extra_required
    expected = [first, second] if extra_required > 0 else [first]
    expected_freed = before if extra_required > 0 else first_freed

    preview = model_store.prune_store(root=tmp_path, target_bytes=target)
    assert preview == {'entries': expected, 'reclaimable_bytes': expected_freed}
    assert all((tmp_path / 'entries' / key).exists() for key in (first, second))
    applied = model_store.prune_store(root=tmp_path, target_bytes=target, apply=True)
    assert applied == preview
    remaining = {path.name for path in (tmp_path / 'entries').iterdir()}
    assert remaining == ({second} if extra_required <= 0 else set())
    if remaining:
        assert second_blob.read_bytes() == b'x' * 10_000
    actual_remaining = sum(len(path.read_bytes()) for path in tmp_path.rglob('*')
                           if not path.is_symlink() and path.is_file())
    assert actual_remaining <= target
    assert before - actual_remaining == applied['reclaimable_bytes']


@pytest.mark.parametrize('protection', ['keep', 'lease', 'unapproved'])
def test_capacity_accounting_excludes_protected_entry_files(tmp_path, protection):
    store = ModelStoreFixture(tmp_path)
    shared = store.blob('shared', 10_000)
    first, protected = store.entry(1, [shared]), store.entry(2, [shared])
    first_bytes = (tmp_path / 'entries' / first / PLAN_FILENAME).read_bytes()
    protected_plan = tmp_path / 'entries' / protected / PLAN_FILENAME
    protected_bytes = protected_plan.read_bytes()
    options = {}
    with ExitStack() as leases:
        if protection == 'keep':
            options['keep'] = {protected}
        elif protection == 'unapproved':
            options['approved_entries'] = {first}
        else:
            plan = model_store.read_entry(protected, tmp_path)
            manifest = {'model_store': {'entry_id': protected, 'plan_sha256': plan['plan_sha256']}}
            leases.enter_context(model_store.acquire_mount(manifest, tmp_path))
        result = model_store.prune_store(root=tmp_path, apply=True, target_bytes=0, **options)
    assert result == {'entries': [first], 'reclaimable_bytes': len(first_bytes)}
    assert not (tmp_path / 'entries' / first).exists()
    assert protected_plan.read_bytes() == protected_bytes
    assert shared.read_bytes() == b'x' * 10_000


@pytest.mark.parametrize('apply', [False, True])
@pytest.mark.parametrize('protection', ['keep', 'lease', 'unapproved'])
def test_missing_plan_does_not_remove_entry_protection(tmp_path, protection, apply):
    store = ModelStoreFixture(tmp_path)
    shared = store.blob('shared', 100)
    removable = store.blob('removable', 40)
    retained = store.blob('retained', 60)
    orphan = store.blob('orphan', 20)
    first = store.entry(1, [shared, removable])
    protected = store.entry(2, [shared, retained])
    first_size = len((tmp_path / 'entries' / first / PLAN_FILENAME).read_bytes())
    protected_entry = tmp_path / 'entries' / protected
    options = {}
    with ExitStack() as leases:
        if protection == 'keep':
            options['keep'] = {protected}
        elif protection == 'unapproved':
            options['approved_entries'] = {first}
        else:
            plan = model_store.read_entry(protected, tmp_path)
            manifest = {'model_store': {'entry_id': protected, 'plan_sha256': plan['plan_sha256']}}
            leases.enter_context(model_store.acquire_mount(manifest, tmp_path))
        (protected_entry / PLAN_FILENAME).unlink()

        result = model_store.prune_store(root=tmp_path, apply=apply, target_bytes=0, **options)

        assert shared.read_bytes() == b'x' * 100
        assert retained.read_bytes() == b'x' * 60
        assert protected_entry.is_dir()
        assert result == {'entries': [first], 'reclaimable_bytes': first_size + 40 + 20}
        assert (tmp_path / 'entries' / first).exists() is not apply
        assert removable.exists() is not apply
        assert orphan.exists() is not apply
    if protection == 'lease' and apply:
        assert model_store.prune_store(root=tmp_path, apply=True) == {
            'entries': [protected], 'reclaimable_bytes': 160}
        assert not protected_entry.exists()
        assert not shared.exists()
        assert not retained.exists()


@pytest.mark.parametrize('staging', [False, True])
def test_unprotected_entry_without_plan_is_reclaimed_as_a_whole(tmp_path, staging):
    store = ModelStoreFixture(tmp_path)
    blob = store.blob('abandoned', 100)
    key = store.entry(1, [blob])
    entry = tmp_path / 'entries' / key
    (entry / PLAN_FILENAME).unlink()
    refs = entry / 'hf/models--example--test/refs'
    refs.mkdir()
    (refs / 'main').write_bytes(b'a' * 40)
    if staging:
        key = '.preparing-interrupted'
        entry = entry.rename(entry.with_name(key))
    expected = {'entries': [key], 'reclaimable_bytes': 140}

    preview = model_store.prune_store(root=tmp_path, target_bytes=0)
    assert preview == expected
    assert entry.is_dir()
    assert blob.read_bytes() == b'x' * 100
    assert model_store.prune_store(root=tmp_path, target_bytes=0, apply=True) == expected
    assert not entry.exists()
    assert not blob.exists()


class TestModelStoreGc:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.root = tmp_path
        self.store = ModelStoreFixture(tmp_path)

    @pytest.mark.parametrize("entry_count", [8, 80])
    def test_target_preview_has_fixed_candidate_scan_count_as_entries_grow(self, entry_count, monkeypatch):
        entries = [self.store.entry(index, [self.store.blob(str(index), 4096)])
                   for index in range(1, entry_count + 1)]
        metadata = sum(len((self.root / 'entries' / key / PLAN_FILENAME).read_bytes()) for key in entries)
        traversals = Counter()
        scandir = os.scandir
        expected = {str(path): 1 for path in (self.root / 'entries').rglob('*') if path.is_dir()}
        expected[str(self.root / 'entries')] = 1

        def track_scan(path):
            if isinstance(path, int):
                directory = Path(os.readlink(f'/proc/self/fd/{path}'))
                if directory.is_relative_to(self.root / 'entries'):
                    traversals[str(directory)] += 1
            return scandir(path)

        monkeypatch.setattr(os, 'scandir', track_scan)
        with patch.object(model_store, 'disk_report', wraps=model_store.disk_report) as totals:
            result = model_store.prune_store(root=self.root, target_bytes=0)
        assert len(result["entries"]) == entry_count
        assert result["reclaimable_bytes"] == entry_count * 4096 + metadata
        assert totals.call_count == 1
        assert traversals == expected

    def test_target_lru_frees_shared_blob_only_after_its_last_reference(self):
        shared = self.store.blob('shared', 100)
        first = self.store.entry(1, [shared, self.store.blob('first', 40)])
        second = self.store.entry(2, [shared, self.store.blob('second', 40)])
        first_metadata = len((self.root / 'entries' / first / PLAN_FILENAME).read_bytes())
        second_metadata = len((self.root / 'entries' / second / PLAN_FILENAME).read_bytes())
        used = model_store.disk_report(self.root)['total_bytes']
        one = model_store.prune_store(root=self.root, target_bytes=used - first_metadata - 40)
        assert one == {'entries': [first], 'reclaimable_bytes': first_metadata + 40}
        two = model_store.prune_store(root=self.root, target_bytes=used - first_metadata - 41)
        assert two == {'entries': [first, second], 'reclaimable_bytes': first_metadata + second_metadata + 180}

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
        metadata = len((self.root / 'entries' / first / PLAN_FILENAME).read_bytes())
        result = model_store.prune_store(root=self.root, apply=True, approved_entries=approved)
        assert result == {'entries': [first], 'reclaimable_bytes': metadata + 40}
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
