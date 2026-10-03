"""Model Store GC plans scale linearly and retain current ownership evidence."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

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


class ModelStoreGcTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

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

    def test_target_preview_has_fixed_candidate_scan_count_as_entries_grow(self):
        for index in range(1, 81):
            self.entry(index, [self.blob(str(index), 4096)])
            if index in (8, 80):
                with self.subTest(entries=index), patch.object(
                    model_store, 'prune_candidates', wraps=model_store.prune_candidates,
                ) as scans:
                    result = model_store.prune_store(root=self.root, target_bytes=0)
                    self.assertEqual(len(result['entries']), index)
                    self.assertEqual(result['reclaimable_bytes'], index * 4096)
                    self.assertEqual(scans.call_count, 1)

    def test_target_lru_frees_shared_blob_only_after_its_last_reference(self):
        shared = self.blob('shared', 100)
        first = self.entry(1, [shared, self.blob('first', 40)])
        second = self.entry(2, [shared, self.blob('second', 40)])
        used = model_store.disk_report(self.root)['total_bytes']
        one = model_store.prune_store(root=self.root, target_bytes=used - 40)
        self.assertEqual(one, {'entries': [first], 'reclaimable_bytes': 40})
        two = model_store.prune_store(root=self.root, target_bytes=used - 41)
        self.assertEqual(two, {'entries': [first, second], 'reclaimable_bytes': 180})

    def test_apply_rechecks_new_leases_and_preserves_the_callers_keep_set(self):
        blob = self.blob('live', 100)
        key = self.entry(1, [blob])
        approved = set(model_store.prune_store(root=self.root)['entries'])
        with (self.root / (key + '.lease')).open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_SH)
            keep = set()
            result = model_store.prune_store(root=self.root, apply=True, keep=keep, approved_entries=approved)
        self.assertEqual(result['entries'], [])
        self.assertEqual(keep, set())
        self.assertTrue(blob.exists())
        self.assertTrue((self.root / 'entries' / key).exists())

    def test_apply_protects_new_entries_and_their_shared_weights(self):
        shared, old = self.blob('shared', 100), self.blob('old', 40)
        first = self.entry(1, [shared, old])
        approved = set(model_store.prune_store(root=self.root)['entries'])
        second = self.entry(2, [shared])
        result = model_store.prune_store(root=self.root, apply=True, approved_entries=approved)
        self.assertEqual(result, {'entries': [first], 'reclaimable_bytes': 40})
        self.assertTrue((self.root / 'entries' / second).exists())
        self.assertTrue(shared.exists())
        self.assertFalse(old.exists())

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
                self.assertTrue(waiting.wait(2), 'preview must report the held lock')
                cancel.set()
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive(), 'cancel must not wait for the other process to unlock')
            finally:
                cancel.set()
        thread.join(timeout=2)
        self.assertEqual(type(errors[0]).__name__, 'ModelStoreCancelled')
        self.assertTrue(blob.exists())
        self.assertTrue((self.root / 'entries' / key).exists())
