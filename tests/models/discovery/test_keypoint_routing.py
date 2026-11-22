"""Pinned metadata projections and conflicting declarations for single-image inspection."""
import json
from pathlib import Path

import pytest

from acprof.cli.inspect import main
from acprof.hub.repository import MetadataFile
from acprof.models.support import supports_transformers_task
from acprof.testing.model_corpus import _offline_repositories

SUPERPOINT = ("magic-leap-community/superpoint", "734450e9ffe229074f5998494ddc615475cdb20a",
              "superpoint", "SuperPointModel")


@pytest.mark.parametrize(("repo", "revision", "model_type", "head", "info", "options", "eligible"), [
    ("usyd-community/vitpose-base-simple", "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45", "vitpose",
     "VitPoseForPoseEstimation", {"auto_model": "VitPoseForPoseEstimation"}, [], True),
    (*SUPERPOINT, {"auto_model": "AutoModel", "pipeline_tag": "feature-extraction"}, [], True),
    (*SUPERPOINT, {"auto_model": "AutoModel", "pipeline_tag": "feature-extraction"},
     ["--task", "keypoint-detection"], True),
    (*SUPERPOINT, {"auto_model": "AutoModelForImageClassification", "pipeline_tag": "image-classification"},
     [], False),
    (*SUPERPOINT, {"auto_model": "AutoModel", "pipeline_tag": "feature-extraction", "custom_class": "MyExtractor"},
     [], False),
    ("ETH-CVG/lightglue_superpoint", "5f5f626efee99f37dbc9eafea879d24d297aeab8", "lightglue",
     "LightGlueForKeypointMatching", {"auto_model": "AutoModelForKeypointMatching", "pipeline_tag": "keypoint-matching"},
     ["--task", "keypoint-detection"], False),
], ids=["vitpose", "superpoint", "superpoint-explicit", "superpoint-conflict", "custom-class", "lightglue"])
def test_inspect_single_image_keypoints(repo, revision, model_type, head, info, options, eligible,
                                        tmp_path, monkeypatch):
    config = {"architectures": [head], "model_type": model_type}
    hub = {"pipeline_tag": "keypoint-detection", "library_name": "transformers", "config": config,
           "transformers_info": {**info, "processor": "AutoImageProcessor"}}
    case = {"model_id": repo, "revision": revision, "hub": hub,
            "repository_files": ["config.json", "model.safetensors", "preprocessor_config.json"]}
    metadata = {"config.json": MetadataFile(Path("config.json"), json.dumps(config).encode()),
                "preprocessor_config.json": MetadataFile(Path("preprocessor_config.json"), b"{}")}
    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", lambda root: None)
    with _offline_repositories({"models": [case]}, {("huggingface", repo, revision): metadata}):
        code = main([repo, "--model-source", "huggingface", "--revision", revision, *options, "--output-dir", str(tmp_path)])
    saved = json.loads((tmp_path / "model_resolution.json").read_bytes())
    assert code == (0 if eligible else 2)
    assert saved["runtime_validation"]["status"] == "not_run"
    candidates = saved["configuration_options"]["backend_candidates"]
    offers = [item for item in candidates if item["task"] == "keypoint-detection"]
    assert offers
    assert all(item["selectable"] is eligible for item in offers)
    if model_type == "superpoint" and not options:
        if eligible:
            hints = [item for item in saved["provenance"]["observations"] if item.get("kind") == "loader_hint"]
            assert [(item["field"], item["value"]) for item in hints] == [
                ("hub.transformers_info.pipeline_tag", "feature-extraction")]
        else:
            assert saved["status"] == "ambiguous"
            assert saved["conflicts"]
    if eligible:
        loader = {"vitpose": "VitPoseForPoseEstimation", "superpoint": "AutoModelForKeypointDetection"}[model_type]
        assert saved["loader"] == loader
        assert saved["compatibility"]["source"] == "registered_keypoint_loader"
        assert all(item["loader"] == loader and item["compatibility"]["source"] == "registered_keypoint_loader"
                   for item in offers)
        assert saved["compatibility"]["status"] == "supported"
        assert saved["compatibility"]["checkpoint_verified"] is False


@pytest.mark.parametrize("version", ["4.57.6", "5.10.4"])
def test_single_image_keypoints_require_a_registered_handler(version):
    for model_type in ("vitpose", "superpoint"):
        assert supports_transformers_task(version, "keypoint-detection", model_type) is True
    for model_type in ("lightglue", "superglue", "efficientloftr", "sapiens2"):
        assert supports_transformers_task(version, "keypoint-detection", model_type) is False
    assert supports_transformers_task(version, "keypoint-detection", "") is None
