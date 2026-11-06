"""Invalid fallback snapshots report lookup failures without entering inference."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import OfflineModeIsEnabled

from acprof.cli.main import main
from acprof.host.detect import detect_task
from acprof.host.preparation.model_errors import ModelLookupError

MODEL = "fixture/config-fallback"
SHA = "a" * 40
VALID_CONFIG = {"model_type": "bert", "architectures": ["BertForMaskedLM"]}


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "huggingface")
    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HF_DOWNLOAD_MODE", "mirror-only")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setenv("HF_FALLBACK_ENDPOINTS", "")
    path = tmp_path / "snapshots" / SHA / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")
    state = SimpleNamespace(path=path)

    def download(*, repo_id, filename, revision, dry_run=False, **_kwargs):
        assert (repo_id, filename, revision) == (MODEL, "config.json", SHA)
        if dry_run:
            return SimpleNamespace(commit_hash=SHA, file_size=state.path.stat().st_size)
        return str(state.path)

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", Mock(side_effect=OfflineModeIsEnabled("fixture offline")))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", lambda *_args: None)
    state.host_check = Mock(side_effect=AssertionError("invalid metadata reached hardware preparation"))
    monkeypatch.setattr("acprof.host.preflight.require_collection_host", state.host_check)
    return state


@pytest.mark.parametrize("payload", [
    b"null", b"[]", b"[1]", b"0", b"17", b'"text"', b"true", b"false",
    b'[["architectures", ["BertForMaskedLM"]]]',
], ids=["null", "empty-list", "nonempty-list", "zero", "number", "string", "true", "false", "key-value-pairs"])
@pytest.mark.parametrize("entrypoint", ["detect", "inspect"])
def test_nonobject_config_is_an_actionable_lookup_error(snapshot, tmp_path, capsys, payload, entrypoint):
    snapshot.path.write_bytes(payload)
    if entrypoint == "detect":
        with pytest.raises(ModelLookupError) as caught:
            detect_task(MODEL, revision=SHA)
        assert caught.value.model_id == MODEL
        assert caught.value.revision == SHA
        detail = caught.value.detail
    else:
        output = tmp_path / "inspection"
        assert main(["inspect", MODEL, "--revision", SHA, "--output-dir", str(output), "--probe-interface"]) == 1
        captured = capsys.readouterr()
        detail = captured.err
        assert "[model-lookup][ERROR]" in detail
        assert MODEL in detail
        assert "Traceback" not in detail
        assert not output.exists()
    assert "config_json: ValueError: config.json must contain dict" in detail
    assert "architecture_infer" not in detail
    snapshot.host_check.assert_not_called()


@pytest.mark.parametrize(("payload", "reason"), [
    (b"", "JSONDecodeError"), (b"{", "JSONDecodeError"), (b"\xff", "UnicodeDecodeError"),
])
def test_unreadable_config_stops_before_task_inference(snapshot, payload, reason):
    snapshot.path.write_bytes(payload)

    with pytest.raises(ModelLookupError) as caught:
        detect_task(MODEL, revision=SHA)

    assert f"config_json: {reason}:" in caught.value.detail
    assert "architecture_infer" not in caught.value.detail
    snapshot.host_check.assert_not_called()


@pytest.mark.parametrize(("directory", "commit", "reason"), [
    ("snapshots", "main", "pinned snapshot commit SHA"),
    ("blobs", SHA, "pinned snapshot commit SHA"),
    ("snapshots", "b" * 40, "different model revision"),
])
def test_invalid_snapshot_identity_stops_before_task_inference(snapshot, tmp_path, directory, commit, reason):
    snapshot.path = tmp_path / directory / commit / "config.json"
    snapshot.path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")

    with pytest.raises(ModelLookupError) as caught:
        detect_task(MODEL, revision=SHA)

    assert reason in caught.value.detail
    assert "architecture_infer" not in caught.value.detail


def test_empty_object_retains_the_unresolved_task_diagnostic(snapshot):
    snapshot.path.write_bytes(b"{}")

    with pytest.raises(ModelLookupError) as caught:
        detect_task(MODEL, revision=SHA)

    assert "config.json has no architectures field" in caught.value.detail
    assert "architecture_infer" in caught.value.detail
    assert "must contain dict" not in caught.value.detail


def test_corrected_cached_config_can_recover_on_the_next_lookup(snapshot):
    snapshot.path.write_bytes(b"[]")
    with pytest.raises(ModelLookupError):
        detect_task(MODEL, revision=SHA)
    snapshot.path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")

    task = detect_task(MODEL, revision=SHA)

    assert (task.model_revision, task.pipeline_tag, task.detection_method) == (SHA, "fill-mask", "config_infer")
    assert task.repository_metadata["config.json"] == VALID_CONFIG
    assert task.metadata_errors == ()
