import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container.model_files import plan_download, seal_plan
from acprof.host import model_store
from acprof.network_policy import DownloadPolicyError


class TestModelStore:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.directory = tmp_path
        self.root = Path(str(self.directory))
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
        assert (link.is_symlink())
        assert (link.resolve()) == (blob)
        assert (len(list(self.root.glob("hf/models--*/blobs/*")))) == (1)
        with model_store.acquire_mount({"model_store": record}, self.root) as mount:
            assert (f"type=bind,src={self.root},dst=/models,readonly") in (mount.args)
            assert ("HF_MODULES_CACHE=/tmp/acprof-hf-modules") in (mount.args)
            # Prune cannot remove an active run's snapshot.
            assert (model_store.prune_store(root=self.root, apply=True)["entries"]) == ([])
            assert (blob.exists())

    def test_prepared_entry_metadata_is_fsynced_before_publication(self):
        with patch("os.fsync") as fsync:
            self.prepare()

        assert fsync.call_count >= 4

    def test_prepared_entry_syncs_directory_tree_and_publication_edge(self):
        synced = []
        with patch("acprof.artifacts.sync_directory", side_effect=lambda path: synced.append(Path(path))):
            record = self.prepare()

        entry = self.root / "entries" / record["entry_id"]
        snapshot_name = self.task.model_revision
        assert self.root / "entries" in synced
        assert any(path.name == snapshot_name and path.parent.name == "snapshots" for path in synced)
        assert entry.exists()

    def test_existing_entry_is_reused_offline_and_cache_savings_are_exact(self):
        first = self.prepare()
        assert (model_store.model_sources(self.plan, self.root)[0].estimated_bytes) == (0)
        with patch("huggingface_hub.snapshot_download", side_effect=self.snapshot) as download:
            second = model_store.prepare_model(self.task, self.plan, self.root)
        assert (first) == (second)
        assert (download.call_args.kwargs["local_files_only"])

    def test_storage_failure_retries_official_with_same_commit(self):
        import httpx
        visited = []
        def snapshot(**kwargs):
            visited.append((kwargs["endpoint"], kwargs["revision"]))
            if kwargs["endpoint"] == "https://hf-mirror.com":
                raise httpx.ConnectError("storage unavailable",
                    request=httpx.Request("GET", "https://us.aws.cdn.hf.co/file"))
            return self.snapshot(**kwargs)
        with patch.dict(os.environ, {"HF_DOWNLOAD_MODE": "auto", "HF_ENDPOINT": "https://hf-mirror.com",
                                      "HF_FALLBACK_ENDPOINTS": ""}), patch(
                "huggingface_hub.snapshot_download", side_effect=snapshot):
            record = model_store.prepare_model(self.task, self.plan, self.root)
        assert visited == [("https://hf-mirror.com", "a" * 40), ("https://huggingface.co", "a" * 40)]
        assert record["model_download"]["source"] == "huggingface"
        assert record["model_download"]["files"][0]["sha256"] == hashlib.sha256(self.content).hexdigest()
        assert record["model_download"]["download_provenance"]["hub_endpoint"] == "https://huggingface.co"

    def test_cached_model_detection_makes_zero_network_requests(self):
        from acprof.host.detect import detect_task
        self.plan["requested_revision"] = None
        self.plan["repository_context"] = {"schema_version": 1, "repository_files": ["config.json"],
            "pipeline_tag": "feature-extraction", "library_name": "transformers",
            "config": {"model_type": "bert", "architectures": ["BertModel"]},
            "tags": [], "card_data": {}, "safetensors": None, "transformers_info": {}}
        self.prepare()
        with patch.dict(os.environ, {"ACPROF_MODEL_STORE": str(self.root), "ACPROF_MODEL_SOURCE": "huggingface"}), patch(
                "huggingface_hub.HfApi.model_info", side_effect=AssertionError("cache hit used network")), patch(
                "huggingface_hub.hf_hub_download", side_effect=AssertionError("cache hit downloaded metadata")):
            task = detect_task(self.task.model_id)
        assert task.model_revision == "a" * 40
        assert task.model_source == "huggingface"

    def test_cached_repository_scan_ignores_broken_entry_symlink(self):
        self.plan["repository_context"] = {"schema_version": 1, "repository_files": ["config.json"],
            "pipeline_tag": "feature-extraction", "library_name": "transformers",
            "config": {"model_type": "bert", "architectures": ["BertModel"]},
            "tags": [], "card_data": {}, "safetensors": None, "transformers_info": {}}
        self.prepare()
        entries = self.root / "entries"
        (entries / ("f" * 64)).symlink_to(self.root / "missing-entry")
        with patch("acprof.host.model_store.store_root", return_value=self.root):
            info = model_store.cached_model_info(self.task.model_id, self.task.model_revision,
                                                 source="huggingface")
        assert info is not None
        assert info.sha == self.task.model_revision


    def test_default_revision_does_not_reuse_a_cached_release(self):
        self.plan["requested_revision"] = "release-v1"
        self.prepare()
        with patch("acprof.host.model_store.store_root", return_value=self.root):
            assert model_store.cached_model_info(self.task.model_id, None, source="huggingface") is None
            assert model_store.cached_metadata(self.task.model_id, "config.json", None, source="huggingface") is None
            assert model_store.cached_metadata(self.task.model_id, "config.json", "release-v1",
                                              source="huggingface").read_bytes() == self.content

    def test_incomplete_legacy_context_is_not_a_full_repository_inventory(self):
        self.prepare()
        with patch("acprof.host.model_store.store_root", return_value=self.root):
            assert model_store.cached_model_info(self.task.model_id, self.task.model_revision,
                                                source="huggingface") is None
            assert model_store.cached_metadata(self.task.model_id, "config.json", self.task.model_revision,
                                              source="huggingface").read_bytes() == self.content

    def test_cached_metadata_rechecks_hash_and_enforces_read_limit(self):
        from acprof.container.model_files import ModelFilesError
        record = self.prepare()
        with patch("acprof.host.model_store.store_root", return_value=self.root):
            with pytest.raises(ModelFilesError, match="exceeds"):
                model_store.cached_metadata(self.task.model_id, "config.json", self.task.model_revision,
                                            source="huggingface", max_bytes=1)
            path = self.root / "entries" / record["entry_id"] / "hf/models--example--test/snapshots"
            (path / self.task.model_revision / "config.json").write_bytes(b"[]")
            with pytest.raises(ModelFilesError, match="SHA256"):
                model_store.cached_metadata(self.task.model_id, "config.json", self.task.model_revision,
                                            source="huggingface")

    def test_identical_names_in_distinct_sources_have_distinct_entries(self):
        hf_key = model_store.entry_key(self.task)
        self.task.model_source = "modelscope"
        assert model_store.entry_key(self.task) != hf_key

    def test_unknown_size_or_disk_shortage_stops_before_snapshot(self):
        for free in (0, 1):
            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=free)), patch("huggingface_hub.snapshot_download") as download:
                with pytest.raises(DownloadPolicyError):
                    model_store.prepare_model(self.task, self.plan, self.root)
                download.assert_not_called()
        self.plan["files"][0]["size"] = None
        with pytest.raises(DownloadPolicyError):
            model_store.require_space(self.plan, self.root)

    def test_capacity_and_prune_preserve_hash_identity(self):
        with patch.dict(os.environ, {"ACPROF_MODEL_STORE_MAX": "1"}):
            with pytest.raises(DownloadPolicyError):
                self.prepare()
        record = self.prepare()
        result = model_store.prune_store(root=self.root)
        assert (result["reclaimable_bytes"]) == (2)
        assert ((self.root / "entries" / record["entry_id"]).exists())
        model_store.prune_store(root=self.root, apply=True)
        assert (model_store.read_entry(record["entry_id"], self.root)) is None

    def test_zero_budget_stops_store_download_even_when_called_directly(self):
        with patch.dict(os.environ, {"ACPROF_MAX_DOWNLOAD": "0"}), patch("huggingface_hub.snapshot_download") as download:
            with pytest.raises(DownloadPolicyError):
                model_store.prepare_model(self.task, self.plan, self.root)
            download.assert_not_called()

    def test_cache_change_after_preflight_cannot_increase_download(self):
        with patch("huggingface_hub.snapshot_download") as download:
            with pytest.raises(DownloadPolicyError, match="缓存自预检后已变化"):
                model_store.prepare_model(self.task, self.plan, self.root, planned_download_bytes=0)
            download.assert_not_called()

    def test_hash_mismatch_never_publishes_an_entry(self):
        self.content = b"xx"
        with pytest.raises(ValueError, match="SHA256"):
            self.prepare()
        assert (model_store.read_entry(model_store.entry_key(self.task), self.root)) is None

    def test_missing_or_changed_store_is_fatal_without_network(self):
        record = self.prepare()
        path = self.root / "entries" / record["entry_id"] / "model_download_plan.json"
        value = json.loads(path.read_text())
        value["model_revision"] = "b" * 40
        path.write_text(json.dumps(value))
        with patch("huggingface_hub.snapshot_download") as download:
            with pytest.raises(ValueError, match="hash mismatch"):
                model_store.acquire_mount({"model_store": record}, self.root)
            download.assert_not_called()

    def test_entry_validation_does_not_pin_cache_against_gc(self):
        record = self.prepare()

        plan = model_store.require_entry({"model_store": record}, self.root)

        assert plan is not None
        assert plan["plan_sha256"] == record["plan_sha256"]
        assert model_store.prune_store(root=self.root)["entries"] == [record["entry_id"]]

    def test_explicit_mount_lease_tracks_each_runtime_consumer(self):
        record = self.prepare()
        first = model_store.acquire_mount({"model_store": record}, self.root)
        second = model_store.acquire_mount({"model_store": record}, self.root)
        try:
            assert model_store.prune_store(root=self.root)["entries"] == []
            first.close()
            assert model_store.prune_store(root=self.root)["entries"] == []
            second.close()
            assert model_store.prune_store(root=self.root)["entries"] == [record["entry_id"]]
        finally:
            first.close()
            second.close()

    def test_entry_verification_releases_gc_lease_when_done(self):
        record = self.prepare()

        model_store.verify_entry({"model_store": record}, self.root)

        assert model_store.prune_store(root=self.root)["entries"] == [record["entry_id"]]

    def test_entry_verification_releases_gc_lease_after_failure(self):
        record = self.prepare()
        with patch(
            "acprof.container.download_model.verify_download",
            side_effect=RuntimeError("verification failed"),
        ), pytest.raises(RuntimeError, match="verification failed"):
            model_store.verify_entry({"model_store": record}, self.root)

        assert model_store.prune_store(root=self.root)["entries"] == [record["entry_id"]]

    def test_blank_store_setting_uses_default_and_recorded_path_supports_posthoc(self):
        with patch.dict(os.environ, {"ACPROF_MODEL_STORE": ""}):
            assert (model_store.store_root()) == (Path.home() / ".cache/acprof/model-store")
            record = {**self.prepare(), "host_path": str(self.root)}
            with model_store.acquire_mount({"model_store": record}) as mount:
                command = mount.args
        assert (f"type=bind,src={self.root},dst=/models,readonly") in (command)
