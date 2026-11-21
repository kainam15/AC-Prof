"""Backend offers use registered static evidence without selecting or loading a model."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.failures import Failure
from acprof.host.detect import TaskInfo
from acprof.host.preparation.task_selection import select_tasks
from acprof.models.backend_candidates import describe_backend_candidates
from acprof.models.candidates import apply_model_candidate, discover_model_candidates


def _task(**overrides):
    config = {"model_type": "bert", "architectures": ["BertForMaskedLM"]}
    values = {
        "model_id": "fixture/backend-candidates", "pipeline_tag": "fill-mask", "task_family": "nlp",
        "runtime_backend": "transformers_model", "library_name": "transformers",
        "model_revision": "a" * 40, "detection_method": "hub_api",
        "model_config": config, "repository_files": ("config.json", "model.safetensors"),
        "repository_metadata": {"config.json": config},
    }
    values.update(overrides)
    return TaskInfo(**values)


def _native(candidates):
    return [item for item in candidates if item["backend"] in {"transformers_model", "transformers_pipeline"}]


def test_native_offers_are_static_deterministic_and_do_not_mutate_current_selection():
    task = _task()
    original = copy.deepcopy(asdict(task))

    candidates = describe_backend_candidates(task, task="fill-mask")

    assert {item["backend"] for item in _native(candidates)} == {"transformers_model", "transformers_pipeline"}
    for item in _native(candidates):
        assert item["registered"] is True
        assert item["selectable"] is True
        assert item["compatibility"]["status"] == "supported"
        assert item["compatibility"]["evidence_level"] == "static"
        assert item["compatibility"]["checkpoint_verified"] is False
        assert item["runtime_validation"] == {"status": "not_run"}
        assert item["runtime_profile"]
        assert item["engine"] == "torch"
        assert item["handler"] == "acprof.container.handlers.nlp:NLPHandler"
        assert item["loader"]
        assert item["operation"]
        assert item["author_recommendation"]["status"] == "unknown"
    assert candidates == describe_backend_candidates(task, task="fill-mask")
    assert asdict(task) == original
    json.dumps(candidates, allow_nan=False)


def test_registered_routes_and_library_hints_do_not_prove_checkpoint_support():
    config = {"model_type": "unregistered_voice_model", "architectures": ["UnknownVoiceModel"]}
    task = _task(model_config=config, repository_metadata={"config.json": config}, library_name="kokoro")

    candidates = describe_backend_candidates(task, task="fill-mask")

    assert candidates
    assert not any(item["selectable"] for item in candidates)
    assert all(item["author_recommendation"]["status"] == "unknown" for item in candidates)
    assert not any(item["backend"] == "kokoro" for item in candidates)


def test_registered_kokoro_backend_does_not_admit_unknown_speech_architectures():
    config = {"model_type": "unregistered_voice_model", "architectures": ["UnknownVoiceModel"]}
    context = _task(model_config=config, repository_metadata={"config.json": config}, library_name="kokoro",
                    pipeline_tag="text-to-speech", task_family="audio", runtime_backend="unknown")
    candidates = describe_backend_candidates(context)
    assert any(item["backend"] == "kokoro" for item in candidates)
    assert not any(item["selectable"] for item in candidates)


def test_unknown_artifact_listing_is_not_an_available_offer():
    candidates = describe_backend_candidates(_task(repository_files=()), task="fill-mask")

    assert _native(candidates)
    assert all(not item["selectable"] for item in _native(candidates))
    assert all(item["compatibility"]["status"] == "unknown" for item in _native(candidates))


def test_gguf_with_native_config_cannot_be_offered_to_transformers():
    candidates = describe_backend_candidates(
        _task(repository_files=("config.json", "model.gguf")), task="fill-mask",
    )

    assert _native(candidates)
    for item in _native(candidates):
        assert item["selectable"] is False
        assert item["failure"]["reason_code"] == "artifact_layout_unsupported"


@pytest.mark.parametrize("code", ["metadata_invalid", "model_contract_required", "evidence_conflict"])
def test_upstream_failure_survives_candidate_enumeration_without_rediscovery(monkeypatch, code):
    failure = Failure("resolution", code, "upstream evidence is unresolved", evidence={"source": "fixture"})
    task = _task(model_resolution={"status": "blocked", "failure": failure.to_dict()})
    original = copy.deepcopy(asdict(task))
    discovery = Mock(side_effect=AssertionError("upstream failures cannot be bypassed"))
    monkeypatch.setattr("acprof.models.backend_candidates.discover_model_candidates", discovery)

    candidates = describe_backend_candidates(task, task="fill-mask")

    assert candidates
    for item in candidates:
        assert item["selectable"] is False
        assert item["failure"]["reason_code"] == code
        assert item["failure"]["evidence"]["source"] == "fixture"
    discovery.assert_not_called()
    assert asdict(task) == original


def test_explicit_task_can_resolve_task_selection_without_clearing_other_evidence():
    failure = Failure("resolution", "task_selection_required", "choose one interface")
    task = _task(model_resolution={"status": "selection_required", "failure": failure.to_dict()})
    original = copy.deepcopy(asdict(task))

    unspecified = describe_backend_candidates(task)
    selected = describe_backend_candidates(task, task="fill-mask")

    assert unspecified
    assert not any(item["selectable"] for item in unspecified)
    assert all(item["failure"]["reason_code"] == "task_selection_required" for item in unspecified)
    assert all(item["selectable"] for item in _native(selected))
    assert asdict(task) == original


def test_selected_workload_keeps_its_backend_eligible_without_reselecting_groups():
    config = {
        "model_type": "gpt2", "architectures": ["GPT2LMHeadModel"],
        "task_specific_params": {
            "summarization": {"num_beams": 4},
            "translation_en_to_de": {"num_beams": 1},
        },
    }
    task = _task(pipeline_tag="text-generation", model_config=config,
                 repository_metadata={"config.json": config}, hub_metadata={"pipeline_tag": "text-generation"})
    task.model_resolution = apply_model_candidate(task, discover_model_candidates(task, override_tag="text-generation"))
    unselected = describe_backend_candidates(task, task="text-generation")
    assert _native(unselected)
    assert all(item["failure"]["reason_code"] == "task_selection_required" for item in _native(unselected))

    options = SimpleNamespace(task="text-generation", tasks="summarization", backend="", model_spec="")
    selected = select_tasks(task, options, SimpleNamespace(interactive=False))
    original = copy.deepcopy(asdict(selected))
    candidates = describe_backend_candidates(selected, task="text-generation")

    active = next(item for item in candidates if item["backend"] == selected.runtime_backend)
    assert active["selectable"] is True
    assert active["compatibility"]["status"] == "supported"
    other = next(item for item in _native(candidates) if item["backend"] != selected.runtime_backend)
    assert other["selectable"] is False
    assert other["failure"]["reason_code"] == "task_selection_required"
    assert "another backend" in other["failure"]["detail"]
    assert asdict(selected) == original

    selected.selected_tasks[0]["fingerprint"] = "tampered"
    rejected = describe_backend_candidates(selected, task="text-generation")
    active = next(item for item in rejected if item["backend"] == selected.runtime_backend)
    assert active["selectable"] is False
    assert active["failure"]["reason_code"] == "evidence_conflict"


def test_untyped_conflicts_are_not_erased_by_hypothetical_backend_override():
    task = _task(model_resolution={"status": "ambiguous", "conflicts": ["model spec contradicts artifact evidence"]})

    candidates = describe_backend_candidates(task, task="fill-mask")

    assert candidates
    assert all(not item["selectable"] for item in candidates)
    assert all(item["failure"]["reason_code"] == "evidence_conflict" for item in candidates)


def test_invalid_model_spec_is_a_configuration_failure():
    task = _task(model_spec={"schema_version": 1, "format": "gguf", "task": "fill-mask"})

    candidates = describe_backend_candidates(task, task="fill-mask")

    assert candidates
    assert all(not item["selectable"] for item in candidates)
    assert all(item["failure"]["reason_code"] == "model_contract_required" for item in candidates)


def test_explicit_runtime_profile_constraint_is_never_silently_replaced():
    task = _task(runtime_profile_id="unregistered-runtime")

    candidates = describe_backend_candidates(task, task="fill-mask")

    assert candidates
    assert not any(item["selectable"] for item in candidates)
    assert all(item["failure"]["reason_code"] == "runtime_unavailable" for item in _native(candidates))
    assert task.runtime_profile_id == "unregistered-runtime"


def test_dependency_checks_use_only_already_available_source_text(monkeypatch):
    task = _task(repository_files=("config.json", "model.safetensors", "requirements.txt"))
    download = Mock(side_effect=AssertionError("backend discovery cannot fetch source"))
    monkeypatch.setattr("acprof.host.detect.read_model_source", download)
    monkeypatch.setattr("acprof.host.detect._download_metadata", download)

    candidates = describe_backend_candidates(task, task="fill-mask")

    for item in _native(candidates):
        assert item["selectable"] is False
        assert item["failure"]["reason_code"] == "runtime_dependency_unknown"
        assert "requirements.txt" in str(item["failure"]["evidence"])
    assert task.repository_sources == {}
    download.assert_not_called()


def test_existing_source_requirements_are_checked_against_the_registered_runtime_lock():
    task = _task(
        repository_files=("config.json", "model.safetensors", "requirements.txt"),
        repository_sources={"requirements.txt": "acprof-fixture-missing-package==1.0\n"},
    )

    candidates = describe_backend_candidates(task, task="fill-mask")

    for item in _native(candidates):
        assert item["selectable"] is False
        assert item["failure"]["reason_code"] == "runtime_dependency_missing"
        assert "acprof-fixture-missing-package" in item["failure"]["detail"]


def test_current_unregistered_backend_is_visible_but_never_selectable():
    candidates = describe_backend_candidates(_task(runtime_backend="vllm"), task="fill-mask")

    unsupported = next(item for item in candidates if item["backend"] == "vllm")
    assert unsupported["registered"] is False
    assert unsupported["selectable"] is False
    assert unsupported["failure"]["reason_code"] == "backend_unsupported"
    assert unsupported["author_recommendation"]["status"] == "unknown"


def test_candidate_queries_never_reuse_another_backends_dependency_result():
    previous = {"runtime_profile": "previous-profile", "dependencies": [{"status": "present"}]}
    task = _task(runtime_profile_id="unregistered-runtime", model_resolution={
        "status": "candidate", "dependency_preflight": previous,
        "compatibility": {"status": "supported"}, "precision": {"cpu": "FP32"},
    })
    original = copy.deepcopy(asdict(task))

    candidates = describe_backend_candidates(task, task="fill-mask")

    assert candidates
    assert all("dependency_preflight" not in item for item in candidates)
    assert all(not item["selectable"] for item in candidates)
    assert asdict(task) == original


def test_metadata_failure_has_priority_even_for_an_unregistered_selected_backend():
    failure = Failure("resolution", "metadata_invalid", "config.json is invalid")
    task = _task(runtime_backend="vllm", model_resolution={"status": "blocked", "failure": failure.to_dict()})

    candidates = describe_backend_candidates(task, task="fill-mask")

    assert candidates
    assert all(item["failure"]["reason_code"] == "metadata_invalid" for item in candidates)


@pytest.mark.parametrize("filename", ["config.json", "generation_config.json"])
def test_deep_metadata_is_diagnosed_before_copying_the_snapshot(monkeypatch, filename):
    nested = 1
    for _ in range(600):
        nested = {"nested": nested}
    config = {"model_type": "bert", "architectures": ["BertForMaskedLM"]}
    document = {**config, "deep": nested} if filename == "config.json" else {"deep": nested}
    content = json.dumps(document).encode()
    digest = hashlib.sha256(content).hexdigest()
    task = _task(
        model_config=document if filename == "config.json" else config,
        repository_metadata={"config.json": config, filename: document},
        metadata_sources={filename: {"encoding": "raw", "sha256": digest, "bytes": len(content), "revision": "a" * 40}},
    )
    discovery = Mock(side_effect=AssertionError("invalid metadata cannot be rediscovered"))
    monkeypatch.setattr("acprof.models.backend_candidates.discover_model_candidates", discovery)

    candidates = describe_backend_candidates(task, task="fill-mask")

    assert candidates
    for item in candidates:
        assert item["selectable"] is False
        assert item["status"] == "blocked"
        assert item["failure"]["reason_code"] == "metadata_invalid"
        diagnostic = next(value for value in item["failure"]["evidence"]["diagnostics"] if value["file"] == filename)
        assert diagnostic["raw_sha256"] == digest
        assert diagnostic["json_pointer"].startswith("/deep/nested/")
        assert "depth" in diagnostic["detail"]
    assert task.repository_metadata[filename] is document
    assert task.model_resolution == {}
    assert json.dumps(document).encode() == content
    discovery.assert_not_called()
    json.dumps(candidates, allow_nan=False)


def test_backend_discovery_does_not_import_inference_frameworks_in_a_fresh_process():
    script = """
import builtins
real_import = builtins.__import__
blocked = {"torch", "transformers", "diffusers", "sentence_transformers", "onnxruntime", "chronos", "timm"}

def guarded(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("inference framework imported: " + name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded
from acprof.host.detect import TaskInfo
from acprof.models.backend_candidates import describe_backend_candidates
config = {"model_type": "bert", "architectures": ["BertForMaskedLM"]}
task = TaskInfo(
    model_id="fixture/no-import", pipeline_tag="fill-mask", task_family="nlp",
    runtime_backend="transformers_model", library_name="transformers",
    model_revision="a" * 40, detection_method="hub_api", model_config=config,
    repository_files=("config.json", "model.safetensors"), repository_metadata={"config.json": config},
)
offers = describe_backend_candidates(task, task="fill-mask")
assert any(item["selectable"] for item in offers)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_unknown_task_has_no_available_backend_candidates():
    assert describe_backend_candidates(_task(), task="unregistered-task") == []


def _whisperkit(*, files=(), config=None, library="whisperkit", hub_task="automatic-speech-recognition"):
    config = {} if config is None else config
    return _task(model_id="argmaxinc/whisperkit-coreml", model_revision="0f63a7800b00dd0226abd051b906c246e1907482",
                 pipeline_tag="automatic-speech-recognition", task_family="audio", runtime_backend="unknown",
                 library_name=library, repository_files=files, model_config=config,
                 repository_metadata={"config.json": config}, hub_metadata={"pipeline_tag": hub_task})


@pytest.mark.parametrize("artifact", [
    "openai_whisper-tiny/AudioEncoder.mlmodelc/weights/weight.bin",
    "encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", "encoder.mlmodel",
])
def test_only_coreml_artifacts_preserve_loader_failure_across_backend_offers(artifact):
    task = _whisperkit(files=("config.json", "openai_whisper-tiny/config.json", "README.md",
                              "LICENSE-MIT", "added_tokens.json", artifact))
    task.model_resolution = apply_model_candidate(task, discover_model_candidates(task))
    failure = task.model_resolution["failure"]
    assert task.model_resolution["status"] == "blocked"
    assert failure["reason_code"] == "artifact_layout_unsupported"
    assert failure["evidence"]["artifact_format"] == "CoreML"
    assert failure["evidence"]["declared_task"] == "automatic-speech-recognition"
    assert failure["evidence"]["library_name"] == "whisperkit"
    assert failure["evidence"]["files"] == [artifact]
    assert failure["evidence"]["file_count"] == 6
    assert task.model_resolution["missing"] == []
    assert task.model_resolution["provenance"]["status"] == "blocked"
    assert task.model_resolution["conflicts"] == []
    assert task.model_resolution["runtime_validation"] == {"status": "not_run"}
    offers = describe_backend_candidates(task)
    assert offers
    assert all(not offer["selectable"] and offer["failure"] == failure for offer in offers)


def test_coreml_diagnosis_does_not_promote_an_inferred_task_to_a_hub_declaration():
    task = _whisperkit(files=("weights.mlmodel",), hub_task=None)
    result = discover_model_candidates(task)
    assert result["failure"]["reason_code"] == "artifact_layout_unsupported"
    assert not result["failure"]["evidence"].get("declared_task")


def test_complete_native_whisper_checkpoint_among_coreml_exports_is_selectable():
    config = {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}
    task = _whisperkit(config=config, files=("config.json", "model.safetensors", "export/Encoder.mlmodelc/model.mil"))
    task.model_resolution = apply_model_candidate(task, discover_model_candidates(task))
    assert task.model_resolution["status"] == "candidate"
    assert task.library_name == "whisperkit"
    assert task.model_resolution["candidates"][0]["backend_source"] == "native_transformers_checkpoint"
    assert task.runtime_backend == "transformers_pipeline"
    offers = describe_backend_candidates(task)
    assert any(offer["selectable"] and offer["backend"] == task.runtime_backend for offer in offers)


@pytest.mark.parametrize(("files", "config"), [
    ((), {"model_type": "whisper"}),
    (("README.md", "model.safetensors", "config.json"), {}),
    (("export.mlmodelc/model.mil", "other/custom.weights"), {}),
    (("export.mlmodelc/model.mil", "model.safetensors", "config.json"), {}),
    (("export.mlmodelc/model.mil", "config.json", "model.safetensors.index.json"), {"model_type": "whisper"}),
])
def test_unproven_or_mixed_layout_does_not_become_an_automatic_backend(files, config):
    task = _whisperkit(files=files, config=config)
    result = discover_model_candidates(task)
    assert result["selection"]["backend"] == "unknown"
    assert result["status"] == "needs_configuration"
    assert "failure" not in result


@pytest.mark.parametrize("cause", ["unregistered_backend", "invalid_spec", "metadata_error"])
def test_coreml_listing_keeps_explicit_configuration_and_metadata_errors(cause):
    task = _whisperkit(files=("encoder.mlmodel",))
    if cause == "invalid_spec":
        task.model_spec = {"schema_version": 999}
    elif cause == "metadata_error":
        task.metadata_errors = ("invalid config",)
    result = discover_model_candidates(task, override_backend="nonexistent" if cause == "unregistered_backend" else None)
    assert result["status"] == "needs_configuration"
    assert "failure" not in result
    if cause == "invalid_spec":
        assert any("schema_version" in detail for detail in result["missing"])
    elif cause == "unregistered_backend":
        assert any("nonexistent" in detail for detail in result["missing"])
