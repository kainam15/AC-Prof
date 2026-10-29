import hashlib
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container.model_files import PLAN_FILENAME, plan_download, seal_plan
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

    def test_prepared_entry_can_be_traversed_without_dac_override(self):
        record = self.prepare()
        entry = self.root / "entries" / record["entry_id"]
        assert stat.S_IMODE(entry.stat().st_mode) & 0o111 == 0o111
        assert (entry / "hf/models--example--test/snapshots" / self.task.model_revision).is_dir()

    def test_mount_repairs_private_legacy_entry_without_relaxing_file_modes(self):
        record = self.prepare()
        entry = self.root / "entries" / record["entry_id"]
        metadata = entry / "model_download_plan.json"
        original_file_mode = stat.S_IMODE(metadata.stat().st_mode)
        entry.chmod(0o700)  # Published by older TemporaryDirectory-based code.
        with model_store.acquire_mount({"model_store": record}, self.root) as mount:
            assert stat.S_IMODE(entry.stat().st_mode) == 0o711
            assert stat.S_IMODE(metadata.stat().st_mode) == original_file_mode
            assert "--mount" in mount.args

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
        download.assert_not_called()

    def incomplete_entry(self):
        record = self.prepare()
        entry = self.root / 'entries' / record['entry_id']
        (entry / PLAN_FILENAME).unlink()
        (entry / 'obsolete').write_bytes(b'previous view')
        return record, entry

    @pytest.mark.parametrize('damage', ['missing_link', 'missing_ref', 'wrong_ref', 'copied_file', 'absolute_link', 'outside_link'])
    def test_preparation_repairs_incomplete_view_without_changing_plan_identity(self, damage):
        from huggingface_hub import try_to_load_from_cache

        original = self.prepare()
        entry = self.root / 'entries' / original['entry_id']
        cache = entry / 'hf/models--example--test'
        link = cache / 'snapshots' / self.task.model_revision / 'config.json'
        ref = cache / 'refs/main'
        blob = link.resolve()
        outside = self.root.parent / (self.root.name + '-outside.json')
        if damage == 'missing_ref':
            ref.unlink()
        elif damage == 'wrong_ref':
            ref.write_text('b' * 40)
        else:
            link.unlink()
            if damage == 'copied_file':
                link.write_bytes(b'xx')
            elif damage == 'absolute_link':
                link.symlink_to(blob)
            elif damage == 'outside_link':
                outside.write_bytes(b'{}')
                link.symlink_to(os.path.relpath(outside, link.parent))
        with patch('huggingface_hub.snapshot_download', side_effect=self.snapshot) as download:
            repaired = model_store.prepare_model(self.task, self.plan, self.root)

        assert repaired == original
        assert model_store.read_entry(original['entry_id'], self.root) == original['model_download']
        assert download.call_args.kwargs['local_files_only'] is True
        assert link.is_symlink()
        assert not link.readlink().is_absolute()
        assert link.resolve() == blob
        assert ref.read_text() == self.task.model_revision
        cached = try_to_load_from_cache(self.task.model_id, 'config.json', cache_dir=str(entry / 'hf'))
        assert isinstance(cached, str)
        assert Path(cached).read_bytes() == self.content
        if damage == 'outside_link':
            assert outside.read_bytes() == b'{}'

    def test_incomplete_view_cannot_be_replaced_while_a_consumer_holds_lease(self):
        record = self.prepare()
        entry = self.root / 'entries' / record['entry_id']
        ref = entry / 'hf/models--example--test/refs/main'
        before = entry.stat().st_ino
        with model_store.acquire_mount({'model_store': record}, self.root):
            ref.write_text('b' * 40)
            with patch('huggingface_hub.snapshot_download', side_effect=self.snapshot) as download:
                with pytest.raises(RuntimeError, match='lease'):
                    model_store.prepare_model(self.task, self.plan, self.root)
            download.assert_not_called()
            assert entry.stat().st_ino == before
            assert ref.read_text() == 'b' * 40
        assert self.prepare() == record
        assert ref.read_text() == self.task.model_revision

    def test_preparation_hashes_actual_view_once_and_rejects_redirected_content(self):
        from acprof.container.download_model import verify_download

        record = self.prepare()
        entry = self.root / 'entries' / record['entry_id']
        snapshot = entry / 'hf/models--example--test/snapshots' / self.task.model_revision
        link = snapshot / 'config.json'
        original_blob = link.resolve()
        wrong_blob = original_blob.with_name('wrong-content')
        wrong_blob.write_bytes(b'xx')
        link.unlink()
        link.symlink_to(os.path.relpath(wrong_blob, link.parent))
        with patch('acprof.container.download_model.verify_download', wraps=verify_download) as verify:
            with pytest.raises(ValueError, match='SHA256'):
                self.prepare()
        assert verify.call_count == 1
        assert Path(verify.call_args.args[0]) == snapshot
        assert original_blob.read_bytes() == self.content
        assert wrong_blob.read_bytes() == b'xx'
        assert model_store.read_entry(record['entry_id'], self.root) == record['model_download']

    def test_entry_verification_rejects_incomplete_view_and_releases_lease(self):
        record = self.prepare()
        ref = self.root / 'entries' / record['entry_id'] / 'hf/models--example--test/refs/main'
        ref.write_text('b' * 40)
        with pytest.raises(RuntimeError, match='view|视图'):
            model_store.verify_entry({'model_store': record}, self.root)
        assert ref.read_text() == 'b' * 40
        assert model_store.prune_store(root=self.root)['entries'] == [record['entry_id']]

    def test_truncated_view_payload_is_rejected_before_preparation(self):
        record = self.prepare()
        entry = self.root / 'entries' / record['entry_id']
        link = entry / 'hf/models--example--test/snapshots' / self.task.model_revision / 'config.json'
        original_blob = link.resolve()
        truncated = original_blob.with_name('truncated-content')
        truncated.write_bytes(b'x')
        link.unlink()
        link.symlink_to(os.path.relpath(truncated, link.parent))
        with patch('huggingface_hub.snapshot_download', side_effect=self.snapshot) as download:
            with pytest.raises(ValueError, match='size mismatch'):
                model_store.prepare_model(self.task, self.plan, self.root)
        download.assert_not_called()
        assert link.resolve() == truncated
        assert truncated.read_bytes() == b'x'
        assert original_blob.read_bytes() == self.content

    def test_preparation_rebuilds_an_unleased_entry_with_missing_plan(self):
        original, entry = self.incomplete_entry()
        snapshot = entry / 'hf/models--example--test/snapshots' / self.task.model_revision
        (snapshot / 'config.json').unlink()
        with patch('huggingface_hub.snapshot_download', side_effect=self.snapshot) as download:
            repaired = model_store.prepare_model(self.task, self.plan, self.root)

        assert repaired['entry_id'] == original['entry_id']
        assert model_store.read_entry(repaired['entry_id'], self.root) == repaired['model_download']
        assert (snapshot / 'config.json').read_bytes() == self.content
        assert not (entry / 'obsolete').exists()
        assert download.call_args.kwargs['local_files_only'] is True
        assert not list(entry.parent.glob('.preparing-*'))
        with model_store.acquire_mount({'model_store': repaired}, self.root) as mount:
            assert mount.pass_fds

    def test_preparation_cannot_rebuild_a_missing_plan_while_lease_is_active(self):
        record = self.prepare()
        entry = self.root / 'entries' / record['entry_id']
        before = entry.stat().st_ino
        with model_store.acquire_mount({'model_store': record}, self.root):
            (entry / PLAN_FILENAME).unlink()
            with patch('huggingface_hub.snapshot_download', side_effect=self.snapshot) as download:
                with pytest.raises(RuntimeError, match='lease'):
                    model_store.prepare_model(self.task, self.plan, self.root)
            download.assert_not_called()
            assert entry.stat().st_ino == before
            assert not (entry / PLAN_FILENAME).exists()
        repaired = self.prepare()
        assert model_store.read_entry(record['entry_id'], self.root) == repaired['model_download']

    @pytest.mark.parametrize('error_type', [OSError, KeyboardInterrupt])
    def test_rebuild_publication_failure_restores_incomplete_entry(self, monkeypatch, error_type):
        _record, entry = self.incomplete_entry()
        before = entry.stat().st_ino
        rename = Path.rename

        def fail_publication(path, target):
            if Path(target) == entry and path.name.startswith('.preparing-') and (path / PLAN_FILENAME).exists():
                raise error_type('publication interrupted')
            return rename(path, target)

        with monkeypatch.context() as scoped:
            scoped.setattr(Path, 'rename', fail_publication)
            with pytest.raises(error_type, match='publication interrupted'):
                self.prepare()
        assert entry.stat().st_ino == before
        assert (entry / 'obsolete').read_bytes() == b'previous view'
        assert not (entry / PLAN_FILENAME).exists()
        assert not list(entry.parent.glob('.preparing-*'))
        repaired = self.prepare()
        assert model_store.read_entry(repaired['entry_id'], self.root) == repaired['model_download']

    def test_rebuild_verification_failure_preserves_incomplete_entry(self):
        _record, entry = self.incomplete_entry()
        before = entry.stat().st_ino
        snapshot = Path(self.snapshot())
        (snapshot / 'config.json').write_bytes(b'xx')
        with patch('huggingface_hub.snapshot_download', return_value=str(snapshot)):
            with pytest.raises(ValueError, match='SHA256'):
                model_store.prepare_model(self.task, self.plan, self.root)
        assert entry.stat().st_ino == before
        assert (entry / 'obsolete').read_bytes() == b'previous view'
        assert not (entry / PLAN_FILENAME).exists()
        assert not list(entry.parent.glob('.preparing-*'))

    def test_rebuild_sync_failure_keeps_previous_view_for_recovery(self, monkeypatch):
        record, entry = self.incomplete_entry()
        sync = model_store.artifact_io.sync_directory

        def fail_parent_sync(path):
            if Path(path) == entry.parent:
                raise OSError('publication sync failed')
            return sync(path)

        with monkeypatch.context() as scoped:
            scoped.setattr(model_store.artifact_io, 'sync_directory', fail_parent_sync)
            with pytest.raises(OSError, match='publication sync failed'):
                self.prepare()
        assert model_store.read_entry(record['entry_id'], self.root) is not None
        previous = list(entry.parent.glob('.preparing-*'))
        assert len(previous) == 1
        assert (previous[0] / 'obsolete').read_bytes() == b'previous view'
        repaired = self.prepare()
        with model_store.acquire_mount({'model_store': repaired}, self.root):
            assert model_store.prune_store(root=self.root, apply=True, target_bytes=0)['entries'] == [previous[0].name]
        assert not previous[0].exists()

    def test_rebuild_recovers_after_process_exit_between_directory_renames(self):
        record, entry = self.incomplete_entry()
        snapshot = self.snapshot()
        script = textwrap.dedent('''
            import json
            import os
            import sys
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch
            from acprof.container.model_files import PLAN_FILENAME
            from acprof.host import model_store

            root = Path(sys.argv[1])
            task = SimpleNamespace(**json.loads(sys.argv[2]))
            plan = json.loads(sys.argv[3])
            destination = root / 'entries' / model_store.entry_key(task)
            rename = Path.rename
            def crash_before_publish(path, target):
                if Path(target) == destination and (path / PLAN_FILENAME).exists():
                    os._exit(23)
                return rename(path, target)

            with patch('huggingface_hub.snapshot_download', return_value=sys.argv[4]), \\
                    patch.object(Path, 'rename', crash_before_publish):
                model_store.prepare_model(task, plan, root)
        ''')
        process = subprocess.run([sys.executable, '-I', '-c', script, str(self.root),
                                  json.dumps(vars(self.task)), json.dumps(self.plan), snapshot],
                                 capture_output=True, text=True, timeout=15, check=False)
        assert process.returncode == 23, process.stderr
        assert not entry.exists()
        staged = list(entry.parent.glob('.preparing-*'))
        assert len(staged) == 2
        assert sum((path / PLAN_FILENAME).exists() for path in staged) == 1
        assert model_store.disk_report(self.root)['models'] == []

        repaired = self.prepare()
        assert model_store.read_entry(record['entry_id'], self.root) == repaired['model_download']
        with model_store.acquire_mount({'model_store': repaired}, self.root):
            result = model_store.prune_store(root=self.root, target_bytes=0, apply=True)
            assert set(result['entries']) == {path.name for path in staged}
            assert (Path(snapshot) / 'config.json').read_bytes() == self.content

    def test_cached_plan_reuse_does_not_require_a_writable_store_lock(self):
        self.task.pipeline_tag = "feature-extraction"
        self.task.model_config = {}
        self.task.library_name = "transformers"
        record = self.prepare()
        expected = model_store.read_entry(record["entry_id"], self.root)
        with patch(
            "acprof.host.model_store.store_lock",
            side_effect=PermissionError("read-only store lock"),
        ), patch(
            "acprof.container.download_model._prepare_repository_plan",
            side_effect=AssertionError("cached planning attempted a download"),
        ):
            assert model_store.plan_model(self.task, self.root) == expected

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
        seal_plan(self.plan)
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
        seal_plan(self.plan)
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
        seal_plan(self.plan)
        self.prepare()
        with patch("acprof.host.model_store.store_root", return_value=self.root):
            assert model_store.cached_model_info(self.task.model_id, None, source="huggingface") is None
            assert model_store.cached_metadata(self.task.model_id, "config.json", None, source="huggingface") is None
            assert model_store.cached_metadata(self.task.model_id, "config.json", "release-v1",
                                              source="huggingface").content == self.content

    def test_incomplete_legacy_context_is_not_a_full_repository_inventory(self):
        self.prepare()
        with patch("acprof.host.model_store.store_root", return_value=self.root):
            assert model_store.cached_model_info(self.task.model_id, self.task.model_revision,
                                                source="huggingface") is None
            assert model_store.cached_metadata(self.task.model_id, "config.json", self.task.model_revision,
                                              source="huggingface").content == self.content

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

    @pytest.mark.parametrize("reader", ["repository", "dependency", "config_fallback", "source"])
    @pytest.mark.parametrize("mutation", ["replace", "unlink"])
    def test_metadata_readers_consume_the_bytes_verified_before_cache_changes(self, reader, mutation):
        from acprof.host import detect

        self.content = b'{"architectures":["BertForMaskedLM"]}'
        self.plan = plan_download(model_id=self.task.model_id, revision=self.task.model_revision,
            family=self.task.task_family, backend=self.task.runtime_backend, policy="full",
            files={"config.json": {"size": len(self.content), "lfs_sha256": hashlib.sha256(self.content).hexdigest()}},
            read_json=lambda _: json.loads(self.content))
        self.plan["endpoint"] = "https://hf-mirror.com"
        seal_plan(self.plan)
        record = self.prepare()
        path = (self.root / "entries" / record["entry_id"] / "hf/models--example--test/snapshots"
                / self.task.model_revision / "config.json")
        cached_metadata = model_store.cached_metadata

        def change_after_verification(*args, **kwargs):
            result = cached_metadata(*args, **kwargs)
            assert result is not None
            if mutation == "replace":
                path.write_bytes(self.content.replace(b"MaskedLM", b"MaskedXX"))
            else:
                path.unlink()
            return result

        info = SimpleNamespace(sha=self.task.model_revision, siblings=[SimpleNamespace(rfilename="config.json")])
        with patch("acprof.host.model_store.store_root", return_value=self.root), patch(
            "acprof.host.model_store.cached_metadata", side_effect=change_after_verification,
        ), patch("acprof.host.model_store.cached_model_info", return_value=info), patch(
            "huggingface_hub.hf_hub_download", side_effect=AssertionError("cache hit must remain offline"),
        ) as download:
            if reader == "repository":
                result = detect._repository_metadata(self.task.model_id, self.task.model_revision, info)
                assert result["metadata_errors"] == ()
                assert result["repository_metadata"]["config.json"] == json.loads(self.content)
            elif reader == "dependency":
                result = detect.dependency_metadata(self.task.model_id, self.task.model_revision)
                assert result["read_json"]("config.json") == json.loads(self.content)
            elif reader == "config_fallback":
                task = detect._detect_from_config(self.task.model_id, requested_revision=self.task.model_revision)
                assert task is not None
                assert task.model_revision == self.task.model_revision
                assert task.pipeline_tag == "fill-mask"
            else:
                assert detect.read_model_source(self.task.model_id, "config.json", self.task.model_revision) == self.content.decode()
        download.assert_not_called()

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
            model_store.require_space(model_store.model_sources(self.plan, self.root), self.root)

    def test_capacity_and_prune_preserve_hash_identity(self):
        with patch.dict(os.environ, {"ACPROF_MODEL_STORE_MAX": "1"}):
            with pytest.raises(DownloadPolicyError):
                self.prepare()
        record = self.prepare()
        entry = self.root / 'entries' / record['entry_id']
        metadata_bytes = len((entry / PLAN_FILENAME).read_bytes()) + len(self.task.model_revision.encode())
        result = model_store.prune_store(root=self.root)
        assert result['reclaimable_bytes'] == metadata_bytes + 2
        assert ((self.root / "entries" / record["entry_id"]).exists())
        model_store.prune_store(root=self.root, apply=True)
        assert (model_store.read_entry(record["entry_id"], self.root)) is None

    def test_zero_budget_stops_store_download_even_when_called_directly(self):
        with patch.dict(os.environ, {"ACPROF_MAX_DOWNLOAD": "0"}), patch("huggingface_hub.snapshot_download") as download:
            with pytest.raises(DownloadPolicyError):
                model_store.prepare_model(self.task, self.plan, self.root)
            download.assert_not_called()

    def test_model_only_budget_stops_store_download_even_without_total_cap(self):
        with patch.dict(os.environ, {"ACPROF_MAX_DOWNLOAD": "", "ACPROF_MODEL_MAX_DOWNLOAD": "0"}), patch(
            "huggingface_hub.snapshot_download"
        ) as download:
            with pytest.raises(DownloadPolicyError, match="ACPROF_MODEL_MAX_DOWNLOAD"):
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
