"""Invalid fallback snapshots stop task inference with actionable diagnostics."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import OfflineModeIsEnabled

from acprof.cli.main import main
from acprof.host.detect import detect_task
from acprof.host.preparation.model_errors import ModelLookupError
from tests.support.model_lookup_fixtures import hf_lookup_environment as hf_lookup_environment

MODEL = "fixture/config-fallback"
SHA = "a" * 40
VALID_CONFIG = {"model_type": "bert", "architectures": ["BertForMaskedLM"]}


@pytest.fixture
def snapshot(tmp_path, monkeypatch, hf_lookup_environment):
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
def test_nonobject_config_reports_the_invalid_pinned_file(snapshot, tmp_path, capsys, payload, entrypoint):
    snapshot.path.write_bytes(payload)
    if entrypoint == "detect":
        report = detect_task(MODEL, revision=SHA).model_resolution
    else:
        output = tmp_path / "inspection"
        assert main(["inspect", MODEL, "--revision", SHA, "--output-dir", str(output), "--probe-interface"]) == 2
        report = json.loads((output / "model_resolution.json").read_bytes())
        assert "config.json must contain dict" in capsys.readouterr().err
        assert not (output / "interface_validation.json").exists()

    assert (report["status"], report["model_id"], report["model_revision"]) == ("blocked", MODEL, SHA)
    failure = report["failure"]
    assert (failure["reason_code"], failure["retryability"]) == ("metadata_invalid", "not_retryable")
    digest = hashlib.sha256(payload).hexdigest()
    assert failure["evidence"]["diagnostics"] == [{
        "reason_code": "metadata_invalid", "file": "config.json", "json_pointer": "",
        "raw_sha256": digest, "detail": "config.json must contain dict",
    }]
    assert failure["evidence"]["sources"]["config.json"] == {
        "revision": SHA, "sha256": digest, "bytes": len(payload), "encoding": "raw",
    }
    assert snapshot.path.read_bytes() == payload
    snapshot.host_check.assert_not_called()


@pytest.mark.parametrize(("payload", "reason"), [
    (b"", "JSONDecodeError"), (b"{", "JSONDecodeError"), (b"\xff", "UnicodeDecodeError"),
    (b"9" * 5000, "ValueError"),
])
@pytest.mark.parametrize("override_tag", [None, "fill-mask"], ids=["auto-task", "explicit-task"])
def test_unreadable_config_stops_before_task_inference(snapshot, monkeypatch, payload, reason, override_tag):
    snapshot.path.write_bytes(payload)
    infer = Mock(side_effect=AssertionError("invalid metadata reached architecture inference"))
    discover = Mock(side_effect=AssertionError("invalid metadata reached candidate discovery"))
    monkeypatch.setattr("acprof.host.detect.CATALOG", SimpleNamespace(resolve=infer))
    monkeypatch.setattr("acprof.models.candidates.discover_model_candidates", discover)

    task = detect_task(MODEL, revision=SHA, override_tag=override_tag)

    assert task.model_resolution["status"] == "blocked"
    assert task.model_resolution["candidates"] == []
    assert task.model_resolution["runtime_validation"] == "not_run"
    failure = task.model_resolution["failure"]
    assert failure["reason_code"] == "metadata_invalid"
    assert failure["evidence"]["diagnostics"] == list(task.metadata_diagnostics)
    diagnostic = task.metadata_diagnostics[0]
    assert diagnostic["exception_type"] == reason
    assert diagnostic["file"] == "config.json"
    assert diagnostic["json_pointer"] == ""
    assert diagnostic["raw_sha256"] == hashlib.sha256(payload).hexdigest()
    assert diagnostic["detail"]
    assert task.metadata_sources["config.json"]["sha256"] == diagnostic["raw_sha256"]
    assert task.metadata_sources["config.json"]["revision"] == SHA
    assert "config.json" not in task.repository_metadata
    assert snapshot.path.read_bytes() == payload
    infer.assert_not_called()
    discover.assert_not_called()
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
    failed = detect_task(MODEL, revision=SHA)
    assert failed.model_resolution["failure"]["reason_code"] == "metadata_invalid"
    assert failed.metadata_sources["config.json"]["sha256"] == hashlib.sha256(b"[]").hexdigest()
    snapshot.path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")

    task = detect_task(MODEL, revision=SHA)

    assert (task.model_revision, task.pipeline_tag, task.detection_method) == (SHA, "fill-mask", "config_infer")
    assert task.repository_metadata["config.json"] == VALID_CONFIG
    assert task.metadata_errors == ()
