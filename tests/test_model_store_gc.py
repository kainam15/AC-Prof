"""Model Store GC plans scale linearly and retain current ownership evidence."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.container.model_files import PLAN_FILENAME, plan_download, seal_plan
from acprof.host import model_store


@contextmanager
def locked_store(root):
    process = subprocess.Popen([sys.executable, '-I', '-c',
        "import fcntl,sys; stream=open(sys.argv[1], 'a'); fcntl.flock(stream, fcntl.LOCK_EX); "
        "print('locked', flush=True); sys.stdin.read()", str(root / '.lock')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if process.stdout.readline().strip() != 'locked':
            raise AssertionError('lock holder did not start')
        yield process
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


class TestModelStoreGc:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        directory = tmp_path
        self.root = Path(str(directory))

    def blob(self, name, size):
        path = self.root / 'hf/models--example--test/blobs' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * size)
        return path

    def entry(self, index, blobs):
        key = f'{index:064x}'
        path = self.root / 'entries' / key
        snapshot = path / 'hf/models--example--test/snapshots' / ('a' * 40)
        snapshot.mkdir(parents=True)
        files = {}
        for index, blob in enumerate(blobs):
            name = f'file-{index}.bin'
            (snapshot / name).symlink_to(os.path.relpath(blob, snapshot))
            files[name] = {'size': blob.stat().st_size, 'sha256': hashlib.sha256(blob.read_bytes()).hexdigest()}
        plan = plan_download(model_id='example/test', revision='a' * 40, family='nlp',
            backend='transformers_pipeline', policy='full', files=files, read_json=lambda _: {})
        plan['verification'] = 'sha256'
        (path / PLAN_FILENAME).write_text(json.dumps(seal_plan(plan)))
        stamp = path / 'last-used'
        stamp.touch()
        os.utime(stamp, (int(key, 16), int(key, 16)))
        return key

    @pytest.mark.parametrize("entry_count", [8, 80])
    def test_target_preview_has_fixed_candidate_scan_count_as_entries_grow(self, entry_count):
        for index in range(1, entry_count + 1):
            self.entry(index, [self.blob(str(index), 4096)])
        with patch.object(model_store, "prune_candidates", wraps=model_store.prune_candidates) as scans:
            result = model_store.prune_store(root=self.root, target_bytes=0)
        assert len(result["entries"]) == entry_count
        assert result["reclaimable_bytes"] == entry_count * 4096
        assert scans.call_count == 1

    def test_target_lru_frees_shared_blob_only_after_its_last_reference(self):
        shared = self.blob('shared', 100)
        first = self.entry(1, [shared, self.blob('first', 40)])
        second = self.entry(2, [shared, self.blob('second', 40)])
        used = model_store.disk_report(self.root)['total_bytes']
        one = model_store.prune_store(root=self.root, target_bytes=used - 40)
        assert (one) == ({'entries': [first], 'reclaimable_bytes': 40})
        two = model_store.prune_store(root=self.root, target_bytes=used - 41)
        assert (two) == ({'entries': [first, second], 'reclaimable_bytes': 180})

    def test_apply_rechecks_new_leases_and_preserves_the_callers_keep_set(self):
        blob = self.blob('live', 100)
        key = self.entry(1, [blob])
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
        shared, old = self.blob('shared', 100), self.blob('old', 40)
        first = self.entry(1, [shared, old])
        approved = set(model_store.prune_store(root=self.root)['entries'])
        second = self.entry(2, [shared])
        result = model_store.prune_store(root=self.root, apply=True, approved_entries=approved)
        assert (result) == ({'entries': [first], 'reclaimable_bytes': 40})
        assert ((self.root / 'entries' / second).exists())
        assert (shared.exists())
        assert not (old.exists())

    def test_prune_reclaims_abandoned_hf_incomplete_blobs(self):
        partial = self.blob('deadbeef.12345678.incomplete', 37)
        lock_file = self.blob('deadbeef.lock', 11)
        preview = model_store.prune_store(root=self.root)
        assert preview == {'entries': [], 'reclaimable_bytes': 37}
        assert partial.exists()
        assert lock_file.exists()
        applied = model_store.prune_store(root=self.root, apply=True)
        assert applied == preview
        assert not partial.exists()
        assert lock_file.exists()

    def test_waiting_for_another_process_lock_is_cancelled_before_mutation(self):
        blob = self.blob('pending', 100)
        key = self.entry(1, [blob])
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
