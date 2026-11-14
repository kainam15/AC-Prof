"""Preset provenance is fixed, visible and independent of runtime validation."""
import hashlib
import json

import pytest

from acprof.host.detect import TaskInfo
from acprof.models.configuration_resolution import (
    build_configuration_draft,
    resolve_inference_configuration,
)
from acprof.models.inference_configuration import freeze_inference_configuration
from acprof.tasks.workloads import parse_task


def model(*, raw=True):
    config = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"], "task_specific_params": {
        "text-generation": {"max_new_tokens": 24, "num_beams": 3},
    }}
    task = TaskInfo("fixture/presets", "text-generation", "nlp", "transformers_model",
                    "transformers", "a" * 40, "fixture", model_config=config,
                    repository_files=("config.json", "model.safetensors"),
                    repository_metadata={"config.json": config}, runtime_profile_id="nlp-cpu")
    if raw:
        content = json.dumps(config).encode()
        task.metadata_sources = {"config.json": {"encoding": "raw", "revision": task.model_revision,
                                                 "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}}
    task.selected_tasks = [parse_task("text-generation", {}, encoder_decoder=False, source="fixture")]
    return task


def test_author_preset_uses_pinned_published_task_parameters_and_keeps_sources():
    task = model()
    draft = build_configuration_draft(task, devices=["off"])
    preset = next(item for item in draft["presets"] if item["id"] == "author")
    updated, candidate = resolve_inference_configuration(task, preset["requests"], devices=["off"])
    generation = updated.selected_tasks[0]["config"]["generation"]
    assert generation["max_new_tokens"] == 24
    assert generation["num_beams"] == 3
    evidence = next(item for item in candidate["decisions"] if item["path"].endswith(".num_beams"))
    source = evidence["source"]["suggested_source"]
    assert source["kind"] == "author_task_preset"
    assert source["evidence"]["file"] == "config.json"
    assert source["evidence"]["json_pointer"] == "/task_specific_params/text-generation/num_beams"
    assert source["evidence"]["revision"] == "a" * 40
    assert source["evidence"]["raw_sha256"] == task.metadata_sources["config.json"]["sha256"]
    assert candidate["validation"]["runtime"] == "not_run"


def test_unverified_model_defaults_cannot_be_presented_as_author_recommendation():
    task = model(raw=False)
    draft = build_configuration_draft(task, devices=["off"])
    assert "author" not in {item["id"] for item in draft["presets"]}
    with pytest.raises(ValueError, match="author"):
        resolve_inference_configuration(task, {"schema_version": 1, "preset": "author"}, devices=["off"])


def test_conflicting_authored_semantic_tasks_require_explicit_choices():
    task = model()
    task.model_config["task_specific_params"]["summarization"] = {"num_beams": 4}
    raw = json.dumps(task.model_config).encode()
    task.metadata_sources["config.json"].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    task.selected_tasks[0]["semantic_tasks"] = [{"task_id": "text-generation"}, {"task_id": "summarization"}]
    draft = build_configuration_draft(task, devices=["off"])
    assert "author" not in {item["id"] for item in draft["presets"]}
    with pytest.raises(ValueError, match="author task presets conflict"):
        resolve_inference_configuration(task, {"schema_version": 1, "preset": "author"}, devices=["off"])


def test_benchmark_is_versioned_and_changes_confirmation_identity():
    task = model()
    _, custom = resolve_inference_configuration(task, {"schema_version": 1}, devices=["off"])
    updated, baseline = resolve_inference_configuration(task, {"schema_version": 1, "preset": "benchmark"}, devices=["off"])
    assert updated.selected_tasks[0]["config"]["generation"] == {"max_new_tokens": 64, "do_sample": False, "num_beams": 1}
    assert baseline["requests"]["preset"] == "benchmark"
    decision = next(item for item in baseline["decisions"] if item["path"].endswith(".num_beams"))
    assert decision["source"]["suggested_source"]["preset_version"] == "acprof-benchmark-v1"
    assert baseline["identity_sha256"] != custom["identity_sha256"]
    with pytest.raises(ValueError, match="stale"):
        freeze_inference_configuration(baseline, custom["identity_sha256"])


@pytest.mark.parametrize("preset", ["author", "benchmark"])
def test_preset_switch_retains_explicit_loading_routing_and_workload(preset):
    task = model()
    task_id = task.selected_tasks[0]["task_id"]
    requests = {
        "schema_version": 1,
        "backend": {"mode": "explicit", "value": "transformers_model"},
        "runtime_profile": {"mode": "explicit", "value": "nlp-cpu"},
        "loading": {"off": {"attention_implementation": {"mode": "explicit", "value": "eager"}}},
        "tasks": {task_id: {"batch_size": {"mode": "explicit", "value": 3}}},
    }
    draft = build_configuration_draft(task, devices=["off"], requests=requests)
    request = next(item["requests"] for item in draft["presets"] if item["id"] == preset)
    for name in ("backend", "runtime_profile"):
        assert request[name] == requests[name]
    assert request["loading"]["off"]["attention_implementation"] == {"mode": "explicit", "value": "eager"}
    assert request["tasks"][task_id]["batch_size"] == {"mode": "explicit", "value": 3}
    updated, candidate = resolve_inference_configuration(task, request, devices=["off"])
    assert candidate["loading"]["off"]["attention_implementation"] == "eager"
    assert updated.selected_tasks[0]["config"]["user_experiment_overrides"]["batch_size"] == 3
