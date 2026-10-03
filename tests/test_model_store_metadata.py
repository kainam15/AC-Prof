"""Model Store inventory reads only bounded, regular entry metadata."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acprof.container.model_files import PLAN_FILENAME, plan_download, seal_plan
from acprof.host import model_store


class ModelStoreMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.root = self.base / 'store'
        self.key = 'a' * 64
        self.path = self.root / 'entries' / self.key / PLAN_FILENAME
        self.plan = plan_download(model_id='example/model', revision='b' * 40, family='nlp',
            backend='transformers_pipeline', policy='full',
            files={'config.json': {'size': 2, 'lfs_sha256': hashlib.sha256(b'{}').hexdigest()}}, read_json=lambda _: {})
        self.plan.update(endpoint='https://huggingface.co', verification='sha256')
        self.plan['files'][0]['sha256'] = hashlib.sha256(b'{}').hexdigest()
        seal_plan(self.plan)
        self.payload = json.dumps(self.plan).encode()

    def publish(self, path=None):
        path = path or self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload)
        return path

    def test_regular_metadata_is_read_at_byte_limit_without_store_lock(self):
        self.publish()
        self.path.write_bytes(self.payload + b' ' * (4 * 1024 * 1024 - len(self.payload)))
        with patch.object(model_store, 'store_lock') as lock:
            self.assertEqual(model_store.read_entry(self.key, self.root), self.plan)
        lock.assert_not_called()

    def test_oversized_metadata_is_rejected_before_json_parsing(self):
        self.publish()
        self.path.write_bytes(self.payload + b' ' * (4 * 1024 * 1024 + 1 - len(self.payload)))
        with patch.object(model_store.json, 'loads', return_value=self.plan) as parse:
            with self.assertRaisesRegex(ValueError, 'metadata.*limit'):
                model_store.read_entry(self.key, self.root)
            parse.assert_not_called()

    def test_metadata_growth_after_size_check_is_still_rejected(self):
        self.publish()
        small_stat = self.path.stat()
        self.path.write_bytes(self.payload + b' ' * (4 * 1024 * 1024 + 1 - len(self.payload)))
        with patch.object(model_store.os, 'fstat', return_value=small_stat):
            with self.assertRaisesRegex(ValueError, 'metadata.*limit'):
                model_store.read_entry(self.key, self.root)

    def test_symlink_metadata_or_entry_directory_cannot_escape_store(self):
        for component in ('metadata', 'entry', 'entries'):
            with self.subTest(component=component), tempfile.TemporaryDirectory(dir=self.base) as directory:
                root = Path(directory) / 'store'
                external = Path(directory) / 'external'
                external.mkdir()
                self.publish(external / self.key / PLAN_FILENAME)
                if component == 'metadata':
                    destination = root / 'entries' / self.key / PLAN_FILENAME
                    destination.parent.mkdir(parents=True)
                    destination.symlink_to(external / self.key / PLAN_FILENAME)
                elif component == 'entry':
                    (root / 'entries').mkdir(parents=True)
                    (root / 'entries' / self.key).symlink_to(external / self.key, target_is_directory=True)
                else:
                    root.mkdir()
                    (root / 'entries').symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'metadata.*path'):
                    model_store.read_entry(self.key, root)

    def test_non_regular_metadata_is_rejected_without_blocking_on_fifo(self):
        for kind in ('directory', 'fifo'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=self.base) as directory:
                root = Path(directory)
                path = root / 'entries' / self.key / PLAN_FILENAME
                path.parent.mkdir(parents=True)
                path.mkdir() if kind == 'directory' else os.mkfifo(path)
                with self.assertRaisesRegex(ValueError, 'regular file'):
                    model_store.read_entry(self.key, root)

    def test_missing_entry_stays_absent_and_explicit_root_symlink_is_supported(self):
        self.assertIsNone(model_store.read_entry(self.key, self.root))
        self.publish()
        alias = self.base / 'configured-store'
        alias.symlink_to(self.root, target_is_directory=True)
        self.assertEqual(model_store.read_entry(self.key, alias), self.plan)
