"""Resolve sentence encoders from the selected source's pinned JSON cache."""
import hashlib
import json
from unittest.mock import Mock

import pytest
from huggingface_hub import constants
from huggingface_hub.errors import OfflineModeIsEnabled

from acprof.host.detect import detect_task
from acprof.host.preparation.task_support import TaskSupportError, require_task_support

MODEL = "fixture/offline-encoder"
SHA = "a" * 40
BACKBONE = {"model_type": "mpnet", "architectures": ["MPNetForMaskedLM"]}


def module(index, path, kind):
    return {"idx": index, "path": path, "type": "sentence_transformers.models." + kind}


def write(snapshot, name, value):
    path = snapshot / name
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2).encode()
    path.write_bytes(content)
    return content


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    repo = tmp_path / "hub/models--fixture--offline-encoder"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs/main").write_text(SHA)
    path = repo / "snapshots" / SHA
    write(path, "config.json", BACKBONE)
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "huggingface")
    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(repo.parent))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", Mock(side_effect=OfflineModeIsEnabled("offline")))
    return path


@pytest.mark.parametrize(("encoder", "revision"), [("", SHA), ("0_Transformer", "main")])
def test_cached_chain_selects_encoding_and_pins_module_settings(snapshot, monkeypatch, encoder, revision):
    import huggingface_hub

    if encoder:
        (snapshot / "config.json").unlink()
        write(snapshot, encoder + "/config.json", BACKBONE)
    settings = "1_Pooling/config.json"
    raw = write(snapshot, "modules.json", [module(0, encoder, "Transformer"),
                                           module(1, "1_Pooling", "Pooling"), module(2, "2_Normalize", "Normalize")])
    write(snapshot, settings, {"pooling_mode_mean_tokens": True})
    if revision == "main":
        other = snapshot.parent / ("b" * 40)
        write(other, settings, {"pooling_mode_mean_tokens": False})
        load = huggingface_hub.try_to_load_from_cache

        def change_ref_after_manifest(repo_id, filename, **kwargs):
            cached = load(repo_id, filename, **kwargs)
            if filename == "modules.json":
                (snapshot.parent.parent / "refs/main").write_text(other.name)
            return cached

        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", change_ref_after_manifest)

    task = detect_task(MODEL, revision=revision)
    require_task_support(task)

    assert (task.pipeline_tag, task.runtime_backend, task.model_revision) == ("feature-extraction", "sentence_transformers", SHA)
    assert task.hub_metadata == {}
    assert task.repository_files == ()
    assert task.repository_metadata[settings] == {"pooling_mode_mean_tokens": True}
    components = task.model_resolution["execution"]["components"]
    assert [part["path"] for part in components] == [encoder, "1_Pooling", "2_Normalize"]
    assert [item["interface"] for item in task.task_discovery["tasks"]] == ["feature-extraction"]
    if not encoder:
        assert any(item["scope"] == "component" and item["task"] == "fill-mask"
                   for item in task.model_resolution["evidence"])
    provenance = task.model_resolution["provenance"]
    source = provenance["sources"]["modules.json"]
    assert (source["encoding"], source["sha256"], source["revision"]) == ("raw", hashlib.sha256(raw).hexdigest(), SHA)
    assert any(item["source_id"] == "modules.json" and item["field"] == "" and item["task"] == "feature-extraction"
               for item in provenance["observations"])
    assert task.model_resolution["runtime_validation"]["status"] == "not_run"
    assert task.model_resolution["compatibility"]["status"] == "unknown"


@pytest.mark.parametrize(("filename", "value", "pointer"), [
    ("modules.json", {}, ""),
    ("modules.json", [module(0, "", "Transformer"), module(1, "../elsewhere", "Pooling")], "/1/path"),
    ("modules.json", [module(0, "invalid\0name", "Transformer")], "/0/path"),
    ("1_Pooling/config.json", {"dimension": float("nan")}, "/dimension"),
    ("1_Pooling/config.json", [], ""),
])
def test_invalid_cached_module_metadata_blocks_even_with_a_task(snapshot, filename, value, pointer):
    write(snapshot, "modules.json", [module(0, "", "Transformer"), module(1, "1_Pooling", "Pooling")])
    raw = write(snapshot, filename, value)
    task = detect_task(MODEL, override_tag="fill-mask", revision=SHA)
    with pytest.raises(TaskSupportError) as caught:
        require_task_support(task)
    assert caught.value.failure.reason_code == "metadata_invalid"
    diagnostic, = task.metadata_diagnostics
    assert (diagnostic["file"], diagnostic["json_pointer"], diagnostic["raw_sha256"]) == (
        filename, pointer, hashlib.sha256(raw).hexdigest(),
    )


@pytest.mark.parametrize("preferred", ["cached_empty", "known_absent", "not_cached"])
def test_cached_legacy_transformer_settings_require_evidence_for_higher_priority_files(snapshot, preferred):
    from acprof.models.evidence import content_digest

    write(snapshot, "modules.json", [module(0, "", "Transformer"), module(1, "pool", "Pooling")])
    first = "sentence_bert_config.json"
    second = "sentence_roberta_config.json"
    raw = write(snapshot, second, {"max_seq_length": 32})
    expected = {"config.json": BACKBONE}
    if preferred == "cached_empty":
        write(snapshot, first, {})
        expected[first] = {}
    elif preferred == "known_absent":
        negative = snapshot.parent.parent / ".no_exist" / SHA / first
        negative.parent.mkdir(parents=True)
        negative.touch()
    if preferred != "not_cached":
        expected[second] = {"max_seq_length": 32}

    task = detect_task(MODEL, revision="main")
    require_task_support(task)
    assert task.model_resolution["execution"]["components"][0]["configuration_sha256"] == content_digest(expected)
    assert task.model_resolution["artifact_layout"]["status"] == "unknown"
    if preferred == "not_cached":
        assert second not in task.repository_metadata
    else:
        source = task.metadata_sources[second]
        assert (source["sha256"], source["revision"]) == (hashlib.sha256(raw).hexdigest(), SHA)


def test_invalid_cached_cnn_configuration_reports_the_original_file_and_field(snapshot):
    write(snapshot, "modules.json", [module(0, "", "Transformer"), module(1, "conv", "CNN"),
                                     module(2, "pool", "Pooling")])
    filename = "conv/cnn_config.json"
    raw = write(snapshot, filename, {"kernel_sizes": [3, float("nan")]})
    task = detect_task(MODEL, revision=SHA)
    with pytest.raises(TaskSupportError) as caught:
        require_task_support(task)
    assert caught.value.failure.reason_code == "metadata_invalid"
    diagnostic, = task.metadata_diagnostics
    assert (diagnostic["file"], diagnostic["json_pointer"], diagnostic["raw_sha256"]) == (
        filename, "/kernel_sizes/1", hashlib.sha256(raw).hexdigest(),
    )


@pytest.mark.parametrize("pool_config_cached", [True, False])
def test_mixed_module_references_select_cached_encoder_without_claiming_weights(snapshot, pool_config_cached):
    chain = [module(0, "", "Transformer"), module(1, "pool", "Pooling"), module(2, "projection", "Dense")]
    chain[0]["type"] = "sentence_transformers.base.modules.transformer.Transformer"
    chain[2]["type"] = "sentence_transformers.base.modules.dense.Dense"
    write(snapshot, "modules.json", chain)
    if pool_config_cached:
        write(snapshot, "pool/config.json", {"embedding_dimension": 8})
    write(snapshot, "projection/config.json", {"in_features": 8, "out_features": 4})

    task = detect_task(MODEL, revision=SHA)
    require_task_support(task)
    assert (task.pipeline_tag, task.runtime_backend) == ("feature-extraction", "sentence_transformers")
    assert [item["type"] for item in task.model_resolution["execution"]["components"]] == [m["type"] for m in chain]
    assert task.model_resolution["artifact_layout"]["status"] == "unknown"
    assert task.model_resolution["runtime_validation"]["status"] == "not_run"


@pytest.mark.parametrize("reference", ["sentence_transformers.models.Unknown", "sentence_transformers.vendor.dense.Dense"])
def test_unregistered_module_cannot_fall_back_to_backbone_task(snapshot, reference):
    write(snapshot, "modules.json", [module(0, "", "Transformer"), {"idx": 1, "path": "custom", "type": reference}])
    task = detect_task(MODEL, revision=SHA)
    with pytest.raises(TaskSupportError, match="unregistered module protocol"):
        require_task_support(task)
    assert task.model_resolution["status"] == "needs_configuration"


def test_module_settings_from_a_different_commit_block_an_explicit_task(snapshot, monkeypatch):
    import huggingface_hub

    from acprof.models.files import ModelFilesError

    write(snapshot, "modules.json", [module(0, "", "Transformer"), module(1, "pool", "Pooling")])
    other = snapshot.parent / ("b" * 40)
    write(other, "pool/config.json", {"pooling_mode_mean_tokens": True})
    load = huggingface_hub.try_to_load_from_cache

    def wrong_snapshot(repo_id, filename, **kwargs):
        if filename == "pool/config.json":
            return str(other / filename)
        return load(repo_id, filename, **kwargs)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", wrong_snapshot)
    with pytest.raises(ModelFilesError, match="different model revision"):
        detect_task(MODEL, override_tag="feature-extraction", revision=SHA)


def test_modelscope_uses_its_own_manifest_and_module_settings(snapshot, tmp_path, monkeypatch):
    from acprof.models.download_plan import PLAN_FILENAME, plan_download, seal_plan

    settings = "1_Pooling/config.json"
    write(snapshot, "modules.json", [module(0, "", "Unknown")])
    write(snapshot, settings, {"pooling_mode_mean_tokens": False})
    convolution = "stage/cnn_config.json"
    write(snapshot, convolution, {"in_embedding_dimension": 99})
    entry = tmp_path / "store/entries" / ("c" * 64)
    local = entry / "hf/models--fixture--offline-encoder/snapshots" / SHA
    files = {
        "modules.json": write(local, "modules.json", [module(0, "", "Transformer"), module(1, "stage", "CNN"),
                                                     module(2, "1_Pooling", "Pooling")]),
        convolution: write(local, convolution, {"in_embedding_dimension": 16}),
        settings: write(local, settings, {"pooling_mode_mean_tokens": True}),
    }
    plan = plan_download(model_id=MODEL, revision=SHA, family="nlp", backend="sentence_transformers", policy="full",
                         files={name: {"size": len(raw), "lfs_sha256": hashlib.sha256(raw).hexdigest()}
                                for name, raw in files.items()}, read_json=lambda name: json.loads(files[name]))
    plan.update(source="modelscope", verification="sha256")
    for record in plan["files"]:
        record["sha256"] = hashlib.sha256(files[record["path"]]).hexdigest()
    (entry / PLAN_FILENAME).write_text(json.dumps(seal_plan(plan)))
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "modelscope")
    monkeypatch.setattr("acprof.hub.repository.modelscope_info", Mock(side_effect=ConnectionError("offline")))
    sdk_cache = Mock(side_effect=AssertionError("ModelScope must not read the Hugging Face SDK cache"))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", sdk_cache)

    task = detect_task(MODEL, revision=SHA)
    require_task_support(task)

    assert task.runtime_backend == "sentence_transformers"
    assert task.repository_metadata[settings] == {"pooling_mode_mean_tokens": True}
    assert "config.json" not in task.repository_metadata
    assert task.repository_metadata[convolution] == {"in_embedding_dimension": 16}
    assert task.metadata_sources[convolution]["sha256"] == hashlib.sha256(files[convolution]).hexdigest()
    sdk_cache.assert_not_called()
