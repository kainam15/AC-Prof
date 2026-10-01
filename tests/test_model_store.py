import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.container.model_files import plan_download, seal_plan
from acprof.host import model_store
from acprof.network_policy import DownloadPolicyError


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task = SimpleNamespace(model_id="example/test", model_revision="a" * 40,
            task_family="nlp", runtime_backend="transformers_pipeline", model_download_policy="full",
            model_adapter="family-default", model_spec={}, model_resolution={})
        self.content = b"{}"
        self.plan = plan_download(model_id=self.task.model_id, revision=self.task.model_revision,
            family=self.task.task_family, backend=self.task.runtime_backend, policy="full",
            files={"config.json": {"size": 2, "lfs_sha256": hashlib.sha256(self.content).hexdigest()}}, read_json=lambda _: {})
        self.plan["endpoint"] = "https://hf-mirror.com"
        seal_plan(self.plan)

    def snapshot(self, **kwargs):
        repo = self.root / "hf/models--example--test"
        blob = repo / "blobs" / hashlib.sha256(self.content).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(self.content)
        target = repo / "snapshots" / self.task.model_revision
        target.mkdir(parents=True, exist_ok=True)
        path = target / "config.json"
        if not path.is_symlink():
            path.symlink_to(os.path.relpath(blob, target))
        return str(target)

    def prepare(self):
        with patch("huggingface_hub.snapshot_download", side_effect=self.snapshot):
            return model_store.prepare_model(self.task, self.plan, self.root)

    def test_prepared_entry_has_one_payload_and_read_only_mount(self):
        record = self.prepare()
        blob = self.root / "hf/models--example--test/blobs" / hashlib.sha256(self.content).hexdigest()
        entry = self.root / "entries" / record["entry_id"]
        link = entry / "hf/models--example--test/snapshots" / self.task.model_revision / "config.json"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), blob)
        self.assertEqual(len(list(self.root.glob("hf/models--*/blobs/*"))), 1)
        command = model_store.mount_args({"model_store": record}, self.root)
        self.assertIn(f"type=bind,src={self.root},dst=/models,readonly", command)
        self.assertIn("HF_MODULES_CACHE=/tmp/acprof-hf-modules", command)
        # Prune cannot remove an active run's snapshot.
        self.assertEqual(model_store.prune_store(root=self.root, apply=True)["entries"], [])
        self.assertTrue(blob.exists())
        model_store._LEASES.pop(str(entry)).close()

    def test_existing_entry_is_reused_offline_and_cache_savings_are_exact(self):
        first = self.prepare()
        self.assertEqual(model_store.model_sources(self.plan, self.root)[0].estimated_bytes, 0)
        with patch("huggingface_hub.snapshot_download", side_effect=self.snapshot) as download:
            second = model_store.prepare_model(self.task, self.plan, self.root)
        self.assertEqual(first, second)
        self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_unknown_size_or_disk_shortage_stops_before_snapshot(self):
        for free in (0, 1):
            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=free)), patch("huggingface_hub.snapshot_download") as download:
                with self.assertRaises(DownloadPolicyError):
                    model_store.prepare_model(self.task, self.plan, self.root)
                download.assert_not_called()
        self.plan["files"][0]["size"] = None
        with self.assertRaises(DownloadPolicyError):
            model_store.require_space(self.plan, self.root)

    def test_capacity_and_prune_preserve_hash_identity(self):
        with patch.dict(os.environ, {"ACPROF_MODEL_STORE_MAX": "1"}):
            with self.assertRaises(DownloadPolicyError):
                self.prepare()
        record = self.prepare()
        result = model_store.prune_store(root=self.root)
        self.assertEqual(result["reclaimable_bytes"], 2)
        self.assertTrue((self.root / "entries" / record["entry_id"]).exists())
        model_store.prune_store(root=self.root, apply=True)
        self.assertIsNone(model_store.read_entry(record["entry_id"], self.root))

    def test_zero_budget_stops_store_download_even_when_called_directly(self):
        with patch.dict(os.environ, {"ACPROF_MAX_DOWNLOAD": "0"}), patch("huggingface_hub.snapshot_download") as download:
            with self.assertRaises(DownloadPolicyError):
                model_store.prepare_model(self.task, self.plan, self.root)
            download.assert_not_called()

    def test_cache_change_after_preflight_cannot_increase_download(self):
        with patch("huggingface_hub.snapshot_download") as download:
            with self.assertRaisesRegex(DownloadPolicyError, "缓存自预检后已变化"):
                model_store.prepare_model(self.task, self.plan, self.root, planned_download_bytes=0)
            download.assert_not_called()

    def test_hash_mismatch_never_publishes_an_entry(self):
        self.content = b"xx"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepare()
        self.assertIsNone(model_store.read_entry(model_store.entry_key(self.task), self.root))

    def test_missing_or_changed_store_is_fatal_without_network(self):
        record = self.prepare()
        path = self.root / "entries" / record["entry_id"] / "model_download_plan.json"
        value = json.loads(path.read_text())
        value["model_revision"] = "b" * 40
        path.write_text(json.dumps(value))
        with patch("huggingface_hub.snapshot_download") as download:
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                model_store.mount_args({"model_store": record}, self.root)
            download.assert_not_called()

    def test_blank_store_setting_uses_default_and_recorded_path_supports_posthoc(self):
        with patch.dict(os.environ, {"ACPROF_MODEL_STORE": ""}):
            self.assertEqual(model_store.store_root(), Path.home() / ".cache/acprof/model-store")
            record = {**self.prepare(), "host_path": str(self.root)}
            command = model_store.mount_args({"model_store": record})
        self.assertIn(f"type=bind,src={self.root},dst=/models,readonly", command)
        model_store._LEASES.pop(str(self.root / "entries" / record["entry_id"])).close()
