import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container import download_model
from acprof.container.model_files import ModelFilesError, plan_download, seal_plan, validate_plan
from acprof.hf_endpoints import hf_endpoints
from acprof.host import model_store
from acprof.network_policy import DownloadPolicyError


class TestModelDownload:
    def test_retired_container_downloader_cannot_bypass_host_budget(self):
        with patch.dict(os.environ, {"MODEL_ID": "example/test"}), patch("huggingface_hub.snapshot_download") as download:
            with pytest.raises(SystemExit, match="host Model Store"):
                download_model.main([])
            download.assert_not_called()

    def test_explicit_mirror_defaults_to_auto_fallback(self):
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://mirror.example/"}, clear=True):
            assert (hf_endpoints()) == (["https://mirror.example", "https://huggingface.co"])

    def test_mirror_preferred_exposes_explicit_fallback(self):
        with patch.dict(os.environ, {"HF_DOWNLOAD_MODE": "mirror-preferred", "HF_FALLBACK_ENDPOINTS": "https://mirror.example"}, clear=True):
            assert (hf_endpoints()) == ([
                "https://hf-mirror.com", "https://mirror.example", "https://huggingface.co",
            ])

    def dependency_plan(self):
        def plan(repo, revision):
            result = plan_download(model_id=repo, revision=revision, family="multimodal",
                                   backend="transformers_pipeline", policy="full",
                                   files={"config.json": {"size": 2}}, read_json=lambda name: {})
            result["endpoint"] = "https://hf-mirror.com"
            return seal_plan(result)
        primary = plan("example/audio", "a" * 40)
        primary["dependencies"] = [{"repo_id": "example/base", "revision": "b" * 40,
                                    "allow_patterns": ["*.json"], "download": plan("example/base", "b" * 40)}]
        primary["total_selected_bytes"] = 4
        return seal_plan(primary)

    def task(self, plan):
        return SimpleNamespace(model_id=plan["model_id"], model_revision=plan["model_revision"],
            task_family=plan["task_family"], runtime_backend=plan["backend"], model_adapter="family-default",
            model_download_policy=plan["requested_policy"], model_resolution={},
            model_spec={"dependencies": [{k: v for k, v in dep.items() if k != "download"}
                                         for dep in plan.get("dependencies", [])]})

    def test_dependencies_are_verified_and_have_independent_offline_default_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.dependency_plan()
            calls = []

            def snapshot(**kwargs):
                calls.append(kwargs)
                target = root / "hf" / ("models--" + kwargs["repo_id"].replace("/", "--")) / "snapshots" / kwargs["revision"]
                target.mkdir(parents=True, exist_ok=True)
                (target / "config.json").write_text("{}")
                return str(target)

            with patch("huggingface_hub.snapshot_download", side_effect=snapshot):
                first = model_store.prepare_model(self.task(plan), plan, root)
                second_plan = self.dependency_plan()
                second_plan["dependencies"][0]["revision"] = "c" * 40
                second_plan["dependencies"][0]["download"]["model_revision"] = "c" * 40
                seal_plan(second_plan["dependencies"][0]["download"])
                seal_plan(second_plan)
                second = model_store.prepare_model(self.task(second_plan), second_plan, root)
            assert ([(call["repo_id"], call["revision"]) for call in calls[:2]]) == ([("example/audio", "a" * 40), ("example/base", "b" * 40)])
            for record, revision in ((first, "b" * 40), (second, "c" * 40)):
                view = root / "entries" / record["entry_id"] / "hf/models--example--base"
                assert ((view / "refs/main").read_text()) == (revision)
                assert ((view / "snapshots" / revision / "config.json").is_symlink())
                assert (record["model_download"]["dependencies"][0]["download"]["verification"]) == ("sha256")
                assert (record["model_download"]["total_selected_bytes"]) == (4)
                validate_plan(record["model_download"])

    def test_dependency_identity_mismatch_is_rejected_even_with_a_resealed_parent(self):
        plan = self.dependency_plan()
        plan["dependencies"][0]["revision"] = "c" * 40
        seal_plan(plan)
        with pytest.raises(ModelFilesError, match="dependency"):
            validate_plan(plan)

    def test_unknown_hub_sizes_stop_before_downloading(self):
        plan = self.dependency_plan()
        plan["dependencies"][0]["download"]["files"][0]["size"] = None
        with tempfile.TemporaryDirectory() as directory, patch("huggingface_hub.snapshot_download") as download:
            with pytest.raises(DownloadPolicyError):
                model_store.prepare_model(self.task(plan), plan, Path(directory))
            download.assert_not_called()

    def test_dependency_size_total_must_match_resealed_file_plans(self):
        plan = self.dependency_plan()
        plan["total_selected_bytes"] = 999
        seal_plan(plan)
        with pytest.raises(ModelFilesError, match="dependency.*bytes|total_selected_bytes"):
            validate_plan(plan)

    def test_standard_checkpoint_download_omits_unused_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps({"model_type": "bert"}))
            files = ["config.json", "model.safetensors", "pytorch_model.bin", "flax_model.msgpack"]
            sizes = {name: config.stat().st_size if name == "config.json" else 7 for name in files}
            info = SimpleNamespace(sha="a" * 40, siblings=[
                SimpleNamespace(rfilename=name, size=sizes[name], blob_id=None, lfs=None) for name in files])
            with patch.dict(os.environ, {"TASK_FAMILY": "nlp", "RUNTIME_BACKEND": "transformers_pipeline"}), patch(
                "huggingface_hub.HfApi.model_info", return_value=info,
            ), patch("huggingface_hub.hf_hub_download", return_value=str(config)):
                plan = download_model._prepare_repository_plan("https://hf-mirror.com", "example/bert", "a" * 40,
                    cache_dir=str(root / "hf"), native_types={"bert"})
            assert ({record["path"] for record in plan["files"]}) == ({"config.json", "model.safetensors"})

            def snapshot(**kwargs):
                target = root / "hf/models--example--bert/snapshots" / kwargs["revision"]
                target.mkdir(parents=True)
                (target / "config.json").write_bytes(config.read_bytes())
                (target / "model.safetensors").write_bytes(b"weights")
                return str(target)

            with patch("huggingface_hub.snapshot_download", side_effect=snapshot) as download:
                model_store.prepare_model(self.task(plan), plan, root)
            assert (set(download.call_args.kwargs["allow_patterns"])) == ({"config.json", "model.safetensors"})
            assert (download.call_args.kwargs["revision"]) == ("a" * 40)
