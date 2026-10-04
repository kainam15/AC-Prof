"""Explicit source selection and independent artifact identities."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.container.model_files import ModelFilesError
from acprof.model_repository import modelscope_info, modelscope_metadata, modelscope_revision


def test_modelscope_pins_repository_ref_not_per_file_revision(monkeypatch):
    commit = "a" * 40
    completed = SimpleNamespace(returncode=0, stdout=f"{commit}\trefs/heads/master\n")
    with patch("acprof.host.command.run_command", return_value=completed) as git:
        assert modelscope_revision("demo/model", None) == commit
    assert git.call_args.args[0][:3] == ["git", "ls-remote", "--exit-code"]
    assert "clone" not in git.call_args.args[0]
    with patch("acprof.host.command.run_command") as git:
        assert modelscope_revision("demo/model", commit) == commit
        git.assert_not_called()


def test_modelscope_access_check_never_queries_huggingface():
    from acprof.host.automation import check_repository_access
    with patch("acprof.host.model_store.cached_model_info", return_value=None), patch(
            "acprof.model_repository.modelscope_info", return_value=object()) as ms, patch(
            "huggingface_hub.HfApi.auth_check", side_effect=AssertionError("ModelScope queried HF")):
        evidence = check_repository_access("demo/model", source="modelscope")
    ms.assert_called_once_with("demo/model", None)
    assert evidence["source"] == "modelscope"


@pytest.mark.parametrize("returncode, reason", [(2, "revision_not_found"), (128, "network_error")])
def test_modelscope_ref_failure_retains_lookup_reason(returncode, reason):
    from acprof.host.model_errors import model_lookup_error
    with patch("acprof.host.command.run_command", return_value=SimpleNamespace(returncode=returncode, stdout="")):
        with pytest.raises((OSError, ValueError)) as caught:
            modelscope_revision("demo/model", "release")
    assert model_lookup_error("demo/model", caught.value, revision="release").reason_code == reason


def test_modelscope_metadata_requires_known_size_and_hash_before_transfer(monkeypatch):
    api = Mock()
    api.get_repo.return_value = SimpleNamespace(tags=["library:pytorch"], tasks=["text-generation"], license="apache-2.0")
    item = {"Path": "config.json", "Type": "blob", "Size": 20, "Sha256": "b" * 64, "Revision": "c" * 40}
    api.legacy.list_repo_files.return_value = [item]
    monkeypatch.setattr("acprof.model_repository.modelscope_api", lambda: api)
    info = modelscope_info("demo/model", "a" * 40)
    assert info.sha == "a" * 40
    assert info.siblings[0].lfs["sha256"] == "b" * 64
    item.pop("Size")
    with pytest.raises(ModelFilesError, match="size and SHA256"):
        modelscope_info("demo/model", "a" * 40)
    api.download_file.assert_not_called()


def test_modelscope_metadata_size_guard_prevents_weight_download(monkeypatch):
    info = SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(rfilename="model.safetensors", size=10**10)])
    monkeypatch.setattr("acprof.model_repository.modelscope_info", lambda *_: info)
    with patch("acprof.model_repository.modelscope_file") as download:
        with pytest.raises(ModelFilesError, match="exceeds"):
            modelscope_metadata("demo/model", "model.safetensors", "a" * 40, 1024)
    download.assert_not_called()


def test_configured_endpoint_cannot_change_huggingface_source():
    from acprof.container.download_model import _prepare_repository_plan
    info = SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(rfilename="config.json", size=2)])
    with patch("huggingface_hub.HfApi.model_info", return_value=info) as hf, patch(
            "acprof.model_repository.modelscope_info", side_effect=AssertionError("silent source change")), patch(
            "acprof.container.download_model._library_versions", return_value={}):
        plan = _prepare_repository_plan("https://modelscope.cn", "demo/model", "a" * 40,
                                        dependency={"allow_patterns": ["config.json"]})
    hf.assert_called_once()
    assert plan["source"] == "huggingface"


def test_frozen_source_preserves_historical_hf_options_and_records_modelscope(monkeypatch):
    from acprof.host.run_state import run_options
    from acprof.run_args import build_parser
    monkeypatch.delenv("ACPROF_MODEL_SOURCE", raising=False)
    historical = build_parser().parse_args(["--model", "demo/model"])
    del historical.model_source
    baseline = run_options(historical)
    current = build_parser().parse_args(["--model", "demo/model", "--model-source", "huggingface"])
    assert run_options(current) == baseline
    current.model_source = None
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "modelscope")
    assert run_options(current)["model_source"] == "modelscope"


def test_modelscope_preparation_uses_separate_cache_and_verifies_artifact(tmp_path, monkeypatch):
    from acprof.container.model_files import plan_download, seal_plan
    from acprof.host import model_store
    from acprof.model_repository import modelscope_snapshot
    content = b"{}"
    digest = hashlib.sha256(content).hexdigest()
    task = SimpleNamespace(model_id="demo/model", model_revision="a" * 40, model_source="modelscope",
        task_family="nlp", runtime_backend="transformers_pipeline", model_download_policy="full",
        model_adapter="family-default", model_spec={}, model_resolution={})
    plan = plan_download(model_id=task.model_id, revision=task.model_revision, family="nlp",
        backend=task.runtime_backend, policy="full", files={"config.json": {"size": 2, "lfs_sha256": digest}},
        read_json=lambda _: {})
    plan.update(source="modelscope", endpoint="https://modelscope.cn")
    seal_plan(plan)
    def download(model_id, name, revision, root, *, sha256):
        assert sha256 == digest
        target = modelscope_snapshot(root, model_id, revision) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)
    with patch("acprof.model_repository.modelscope_file", side_effect=download) as ms, patch(
            "huggingface_hub.snapshot_download", side_effect=AssertionError("ModelScope used HF")):
        record = model_store.prepare_model(task, plan, tmp_path)
        assert ms.call_count == 1
        model_store.prepare_model(task, plan, tmp_path)
        assert ms.call_count == 1
    assert record["model_download"]["source"] == "modelscope"
    assert record["model_download"]["files"][0]["sha256"] == digest
    assert record["model_download"]["download_provenance"]["final_endpoint_type"] == "unknown"
    assert not (tmp_path / "hf").exists()
    assert model_store.prune_candidates(tmp_path)["reclaimable_bytes"] == len(content)


@pytest.mark.parametrize("error, reason", [("NotExistError", "repository_unavailable"),
    ("AuthenticationError", "access_denied"), ("PermissionDeniedError", "access_denied"),
    ("RateLimitError", "hub_unavailable"), ("ServerError", "hub_unavailable")])
def test_modelscope_lookup_failure_cannot_be_bypassed_by_manual_task(monkeypatch, error, reason):
    from modelscope_hub import errors

    from acprof.host.detect import detect_task
    from acprof.host.model_errors import ModelLookupError
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "modelscope")
    failure = getattr(errors, error)("source failure")
    with patch("acprof.model_repository.modelscope_info", side_effect=failure), patch(
            "acprof.host.model_store.cached_model_info", return_value=None), patch(
            "acprof.host.model_store.cached_metadata", return_value=None):
        with pytest.raises(ModelLookupError) as caught:
            detect_task("demo/model", override_tag="text-generation", override_family="nlp")
    assert caught.value.reason_code == reason
    assert "Hugging Face Token" not in caught.value.hint


@pytest.mark.parametrize("mode", ["mirror-only", "mirror-preferred", "official"])
def test_legacy_tui_configuration_migrates_without_changing_proxy_environment(tmp_path, mode):
    import os

    from acprof.tui.settings import SETTINGS_VERSION, load_settings
    payload = {"version": SETTINGS_VERSION, "ui": {}, "run_defaults": {
        "model": "demo/model", "download_mode": mode, "HTTP_PROXY": "http://retired.example"}}
    path = tmp_path / "tui.json"
    path.write_text(json.dumps(payload))
    before = dict(os.environ)
    result, warning = load_settings(path, tmp_path)
    assert not warning
    assert result.run_defaults.download_mode == "auto"
    assert dict(os.environ) == before
    assert json.loads(path.read_text()) == payload

def test_filtered_cache_preserves_original_hub_evidence(tmp_path, monkeypatch):
    from acprof.host import model_store
    from acprof.host.detect import TaskInfo, detect_task

    content = b'{"model_type":"bert","architectures":["BertForMaskedLM"]}'
    config_path = tmp_path / "config.json"
    config_path.write_bytes(content)
    info = SimpleNamespace(sha="a" * 40, pipeline_tag="fill-mask", library_name="transformers",
        config=json.loads(content), tags=["license:apache-2.0"],
        card_data={"license": "apache-2.0"}, safetensors={"total": 5, "parameters": {"F32": 5}},
        transformers_info={"auto_model": "AutoModelForMaskedLM", "pipeline_tag": "fill-mask"},
        siblings=[SimpleNamespace(rfilename=name, size=len(data), lfs={"sha256": hashlib.sha256(data).hexdigest()})
                  for name, data in [("config.json", content), ("model.safetensors", b"x"), ("pytorch_model.bin", b"y")]])
    task = TaskInfo(model_id="demo/model", model_revision=info.sha, pipeline_tag="feature-extraction",
        task_family="nlp", runtime_backend="transformers_pipeline", library_name="transformers",
        detection_method="manual", requested_revision=None)
    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path))
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "huggingface")
    with patch("huggingface_hub.HfApi.model_info", return_value=info), patch(
            "huggingface_hub.hf_hub_download", return_value=str(config_path)):
        plan = model_store.plan_model(task, tmp_path)
    assert [item["path"] for item in plan["excluded_files"]] == ["pytorch_model.bin"]
    assert plan["repository_context"]["pipeline_tag"] == "fill-mask"
    assert plan["repository_context"]["repository_files"] == [
        "config.json", "model.safetensors", "pytorch_model.bin"]

    def snapshot(**kwargs):
        path = tmp_path / "hf/models--demo--model/snapshots" / info.sha
        path.mkdir(parents=True)
        (path / "config.json").write_bytes(content)
        (path / "model.safetensors").write_bytes(b"x")
        return str(path)
    with patch("huggingface_hub.snapshot_download", side_effect=snapshot):
        model_store.prepare_model(task, plan, tmp_path)
    with patch("huggingface_hub.HfApi.model_info", side_effect=AssertionError("complete cache queried Hub")), patch(
            "huggingface_hub.hf_hub_download", side_effect=AssertionError("complete cache downloaded metadata")):
        cached = detect_task(task.model_id)
    assert cached.pipeline_tag == "fill-mask"
    assert cached.hub_metadata["pipeline_tag"] == "fill-mask"
    assert cached.repository_files == ("config.json", "model.safetensors", "pytorch_model.bin")
    assert cached.parameter_count == 5
    assert cached.parameter_bytes == 20
    assert cached.parameter_dtype_counts == {"FP32": 5}
    assert cached.model_license == "apache-2.0"


@pytest.mark.parametrize("source, default_ref", [("huggingface", "main"), ("modelscope", "master")])
def test_recorded_default_ref_is_distinct_from_pinned_release(source, default_ref, tmp_path, monkeypatch):
    from acprof.container.download_model import verify_download
    from acprof.container.model_files import plan_download
    from acprof.host import model_store

    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path))
    entry = tmp_path / "entries" / ("b" * 64)
    snapshot = entry / "hf/models--demo--model/snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"{}")
    plan = plan_download(model_id="demo/model", revision="a" * 40, family="nlp",
        backend="transformers_pipeline", policy="full", files={"config.json": {"size": 2}}, read_json=lambda _: {})
    plan["source"] = source
    for recorded in (None, default_ref, "release-v1", "a" * 40):
        plan["requested_revision"] = recorded
        verify_download(snapshot, plan)
        (entry / "model_download_plan.json").write_text(json.dumps(plan))
        result = model_store.cached_metadata("demo/model", "config.json", None, source=source)
        assert (result is not None) == (recorded in (None, default_ref))
        assert model_store.cached_metadata("demo/model", "config.json", default_ref, source=source) == result
        assert model_store.cached_metadata("demo/model", "config.json", "a" * 40, source=source) == snapshot / "config.json"
