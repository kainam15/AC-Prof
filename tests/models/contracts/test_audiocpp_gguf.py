import hashlib
from types import SimpleNamespace

import pytest

from acprof.host.detect import detect_task
from acprof.models.audiocpp import ASR, DEFAULT_ASR, REPOSITORY
from acprof.models.download_plan import plan_download, seal_plan, validate_plan
from acprof.models.files import ModelFilesError
from acprof.models.resolution import resolve_model_interface

REVISION = "a" * 40
QWEN = "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-f16.gguf"
TTS = "Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf"
DIGEST = hashlib.sha256(b"GGUF payload").hexdigest()
FILES = [QWEN, TTS, DEFAULT_ASR, "README.md"]


def _detect(monkeypatch, *, files=FILES, **kwargs):
    info = SimpleNamespace(sha=REVISION, library_name="audio.cpp", pipeline_tag="text-to-speech",
                           tags=["text-to-speech", ASR], siblings=[
                               SimpleNamespace(rfilename=path, size=12, lfs={"sha256": DIGEST}) for path in files])
    monkeypatch.setattr("acprof.host.model_store.store.cached_model_info", lambda *args, **kw: info)
    return detect_task(REPOSITORY, **kwargs)


def test_auto_selects_standalone_asr_with_pinned_file_evidence(monkeypatch):
    task = _detect(monkeypatch)
    assert (task.model_variant, task.pipeline_tag, task.runtime_backend) == (DEFAULT_ASR, ASR, "audio_cpp")
    assert task.hub_metadata["pipeline_tag"] == "text-to-speech"
    assert task.model_config["audiocpp_artifact"]["sha256"] == DIGEST
    assert task.quantization_method == "Q8_0"
    assert {item["interface"] for item in task.task_discovery["tasks"]} == {ASR}
    assert set(task.model_resolution["checkpoint_variants"]) == {QWEN, DEFAULT_ASR}
    layout = resolve_model_interface(task)["artifact_layout"]
    assert layout["weights"][0]["files"] == [DEFAULT_ASR]


@pytest.mark.parametrize("options", [{"model_variant": TTS}, {"override_backend": "onnxruntime"},
                                    {"override_tag": "text-to-speech"}, {"model_variant": "fake.gguf"}])
def test_incompatible_file_or_override_blocks_configuration(monkeypatch, options):
    task = _detect(monkeypatch, **options)
    failure = task.model_resolution["failure"]
    assert failure["reason_code"] == "artifact_layout_unsupported"
    if "override_backend" in options:
        assert failure["evidence"]["backend"] == options["override_backend"]


def test_revision_without_supported_packages_blocks_before_configuration(monkeypatch):
    task = _detect(monkeypatch, files=[TTS, "README.md"])
    assert task.model_resolution["status"] == "blocked"
    assert task.model_resolution["failure"]["reason_code"] == "artifact_layout_unsupported"
    assert not any(item["selectable"] for item in task.model_resolution["gguf_candidates"])


def test_manual_choice_and_store_identity(monkeypatch):
    from acprof.host.model_store.store import entry_key

    first = _detect(monkeypatch)
    second = _detect(monkeypatch, model_variant=QWEN)
    assert second.precision_dtype == "F16"
    assert second.quantized is False
    assert first.model_revision == second.model_revision
    assert entry_key(first) != entry_key(second)


def test_plan_selects_exact_file_and_rejects_tampering():
    options = dict(model_id=REPOSITORY, revision=REVISION, family="audio", backend="audio_cpp",
                   adapter="audio-cpp-asr", model_variant=DEFAULT_ASR,
                   files={path: {"lfs_sha256": DIGEST, "size": 12} for path in FILES}, read_json=lambda _: {})
    plan = plan_download(**options)
    assert [item["path"] for item in plan["files"]] == [DEFAULT_ASR]
    assert plan["selected_bytes"] == 12
    assert plan["model_file_sha256"] == DIGEST
    for override in [{"model_variant": TTS}, {"model_variant": "../bad.gguf"}, {"backend": "onnxruntime"},
                     {"policy": "full"}, {"files": {DEFAULT_ASR: {"size": 12}}}]:
        with pytest.raises(ModelFilesError):
            plan_download(**(options | override))
    plan["files"].append({"path": QWEN, "size": 0})
    with pytest.raises(ModelFilesError):
        validate_plan(seal_plan(plan))

def test_corrupt_payload_and_cached_inventory(tmp_path, monkeypatch):
    from acprof.container.download_model import verify_download
    from acprof.host.model_store import store

    plan = plan_download(model_id=REPOSITORY, revision=REVISION, family="audio", backend="audio_cpp",
                         adapter="audio-cpp-asr", model_variant=DEFAULT_ASR,
                         files={path: {"lfs_sha256": DIGEST, "size": 12} for path in FILES}, read_json=lambda _: {})
    target = tmp_path / DEFAULT_ASR
    target.parent.mkdir()
    target.write_bytes(b"bad! payload")
    with pytest.raises(ModelFilesError, match="SHA256 mismatch"):
        verify_download(tmp_path, plan)
    assert "verification" not in plan
    target.write_bytes(b"GGUF payload")
    plan["repository_context"] = {
        "schema_version": 1, "repository_files": FILES, "pipeline_tag": "text-to-speech", "library_name": "audio.cpp",
        "config": {}, "transformers_info": {}, "card_data": {}, "tags": [], "safetensors": None,
    }
    verify_download(tmp_path, seal_plan(plan))
    monkeypatch.setattr(store, "_matching_repository_plans", lambda *args: [(plan, tmp_path)])
    # A cached single-file view retains the identity of unselected alternatives for future planning.
    second = detect_task(REPOSITORY, revision=REVISION, model_variant=QWEN)
    assert second.model_config["audiocpp_artifact"]["path"] == QWEN
    assert second.model_config["audiocpp_artifact"]["sha256"] == DIGEST
    assert not (tmp_path / QWEN).exists()
