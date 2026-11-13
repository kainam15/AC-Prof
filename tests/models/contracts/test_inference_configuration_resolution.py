"""Confirmed request values survive resolution, freezing and backend changes."""
from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.preparation.task_selection import select_tasks
from acprof.models.candidates import apply_model_candidate, discover_model_candidates
from acprof.models.configuration_resolution import (
    build_configuration_draft,
    requests_from_answers,
    resolve_inference_configuration,
)
from acprof.models.inference_configuration import (
    configuration_identity,
    declared_generation_consumer,
    decode_inference_configuration,
    derive_inference_configuration,
    encode_inference_configuration,
    freeze_inference_configuration,
    validate_configuration_binding,
    validate_frozen_configuration,
)
from acprof.tasks.workloads import merge_generation_parameters


def _model(*, tag="text-generation", config=None, tasks=""):
    config = config or {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}
    task = TaskInfo(
        "fixture/configuration", tag, "nlp", "transformers_model", "transformers",
        "a" * 40, "fixture", model_config=config,
        repository_metadata={"config.json": config},
        repository_files=("config.json", "model.safetensors"),
    )
    task.model_resolution = apply_model_candidate(
        task, discover_model_candidates(task, override_tag=tag, override_backend="transformers_model"),
    )
    args = SimpleNamespace(task=tag, tasks=tasks, backend="transformers_model", model_spec="", workload_spec="")
    task = select_tasks(task, args, SimpleNamespace(interactive=False))
    task.runtime_profile_id = "nlp-cpu"
    return task


def _draft(task, requests=None):
    return build_configuration_draft(
        task, devices=["off"], requests=requests, workload={"batch_size": 1, "input_scales": "8,16"},
    )


def _resolve(task, requests=None):
    requests = _draft(task, requests)["requests"]
    updated, candidate = resolve_inference_configuration(
        task, requests, devices=["off"], workload={"batch_size": 1, "input_scales": "8,16"},
    )
    updated.inference_configuration = freeze_inference_configuration(candidate, candidate["identity_sha256"])
    return updated, candidate


def _task_requests(task, **values):
    return {"schema_version": 1, "tasks": {task.selected_tasks[0]["task_id"]: {
        name: {"mode": "explicit", "value": value} for name, value in values.items()
    }}}


def test_draft_exposes_concrete_defaults_and_inactive_dependencies_without_mutation():
    task = _model()
    original = asdict(copy.deepcopy(task))
    draft = _draft(task)
    fields = {item["parameter_spec"]["name"]: item for item in draft["questions"] + draft["parameter_advanced"]}
    assert fields["dtype"]["request"] == {"mode": "explicit", "value": "FP32"}
    assert fields["dtype"]["options"] == ["FP32", "FP16"]
    assert fields["attention_implementation"]["request"]["value"] == "eager"
    assert fields["max_new_tokens"]["request"]["mode"] == "explicit"
    assert fields["temperature"]["enabled"] is False
    assert fields["temperature"]["request"] == {"mode": "unset"}
    assert fields["temperature"]["dependency_paths"]["do_sample"] == fields["do_sample"]["path"]
    assert asdict(task) == original


def test_defaults_are_materialized_and_workload_fields_do_not_become_generation_kwargs():
    task, _ = _resolve(_model())
    generation = task.selected_tasks[0]["config"]["generation"]
    assert generation["do_sample"] is False
    assert generation["num_beams"] == 1
    assert generation["max_new_tokens"] == 64
    assert not {"temperature", "top_p", "batch_size", "input_scales"} & generation.keys()
    assert task.selected_tasks[0]["config"]["user_experiment_overrides"] == {"batch_size": 1, "input_scales": "8,16"}
    validate_configuration_binding(task.inference_configuration, task)


def test_requested_sampling_values_and_lengths_reach_final_task_snapshot():
    initial = _model()
    task, candidate = _resolve(initial, _task_requests(
        initial, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=23, num_beams=2,
    ))
    actual = task.selected_tasks[0]["config"]["generation"]
    assert {key: actual[key] for key in ("do_sample", "temperature", "top_p", "max_new_tokens", "num_beams")} == {
        "do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 23, "num_beams": 2,
    }
    frozen_task = next(iter(candidate["tasks"].values()))
    assert frozen_task["generation"] == actual
    assert frozen_task["source_task_ids"] == [initial.selected_tasks[0]["task_id"]]
    assert initial.selected_tasks[0]["task_id"] != task.selected_tasks[0]["task_id"]


def test_sampling_defaults_are_materialized_when_sampling_is_enabled():
    initial = _model()
    task, _ = _resolve(initial, _task_requests(initial, do_sample=True))
    assert task.selected_tasks[0]["config"]["generation"]["temperature"] == 1.0
    assert task.selected_tasks[0]["config"]["generation"]["top_p"] == 1.0


def test_explicit_inactive_parameter_is_a_conflict_even_if_equal_to_its_default():
    initial = _model()
    with pytest.raises(ValueError, match="temperature"):
        _resolve(initial, _task_requests(initial, do_sample=False, temperature=1.0))


def test_auto_is_authorized_per_field_and_records_rule_source_and_unrun_runtime():
    initial = _model()
    task, candidate = _resolve(initial, {"schema_version": 1, "preset": "auto"})
    assert candidate["requests"]["backend"] == {"mode": "auto"}
    active = [decision for decision in candidate["decisions"] if decision.get("status") != "inactive"]
    assert all(decision["request"]["mode"] == "auto" and decision["source"] and decision["reason"]
               and decision["rule_version"] == "acprof-inference-v1"
               and decision["runtime_validation"] == "not_run" for decision in active)
    assert task.inference_configuration["validation"] == {"static": "validated", "runtime": "not_run"}


@pytest.mark.parametrize("path", ["backend", "runtime_profile"])
def test_unset_can_be_edited_but_never_authorizes_resolution(path):
    task = _model()
    requests = {"schema_version": 1, path: {"mode": "unset"}}
    draft = _draft(task, requests)
    assert draft["requests"][path] == {"mode": "unset"}
    with pytest.raises(ValueError, match="unset"):
        _resolve(task, requests)


@pytest.mark.parametrize("value", [True, 2, None, ["FP32"]])
def test_malformed_loading_value_is_rejected_before_widget_construction(value):
    with pytest.raises(ValueError, match="dtype"):
        _draft(_model(), {"schema_version": 1, "loading": {"off": {"dtype": {"mode": "explicit", "value": value}}}})


def test_unknown_paths_and_unavailable_devices_fail_before_freezing():
    task = _model()
    with pytest.raises(ValueError, match="unknown or unavailable"):
        _draft(task, _task_requests(task, invented_parameter=3))
    with pytest.raises(ValueError, match="GPU"):
        build_configuration_draft(task, devices=["on"])
    with pytest.raises(ValueError, match="duplicates"):
        build_configuration_draft(task, devices=["off", "cpu"])


def test_form_answer_roundtrip_preserves_modes_and_same_default_explicit_intent():
    task = _model()
    draft = _draft(task)
    answers = {item["path"]: item["request"] for item in draft["questions"] + draft["parameter_advanced"]}
    result = requests_from_answers(draft, answers)
    assert result == draft["requests"]
    key = task.selected_tasks[0]["task_id"]
    result["tasks"][key]["max_new_tokens"] = {"mode": "auto"}
    _, explicit = _resolve(task, draft["requests"])
    _, automatic = _resolve(task, result)
    assert next(iter(explicit["tasks"].values()))["generation"] == next(iter(automatic["tasks"].values()))["generation"]
    assert explicit["identity_sha256"] != automatic["identity_sha256"]


def test_backend_switch_rebinds_execution_and_cannot_reuse_the_old_confirmation():
    initial = _model()
    _, first = _resolve(initial)
    switched, second = _resolve(initial, {"schema_version": 1, "backend": {"mode": "explicit", "value": "transformers_pipeline"}})
    assert switched.runtime_backend == "transformers_pipeline"
    assert switched.runtime_profile_id == initial.runtime_profile_id == "nlp-cpu"
    assert switched.selected_tasks[0]["execution_profile"]["signature"]["backend"] == "transformers_pipeline"
    assert switched.selected_tasks[0]["execution_path"]["backend"] == "transformers_pipeline"
    assert first["identity_sha256"] != second["identity_sha256"]
    with pytest.raises(ValueError, match="stale"):
        freeze_inference_configuration(second, first["identity_sha256"])


def test_static_unsupported_artifacts_or_invalid_metadata_cannot_be_confirmed():
    task = _model()
    task.repository_files = ("config.json", "model.gguf")
    with pytest.raises(ValueError, match="GGUF|artifact|gguf"):
        _draft(task)
    task = _model()
    task.model_config["bad"] = float("inf")
    with pytest.raises(ValueError, match="metadata|finite|JSON"):
        _draft(task)


def test_unregistered_generation_consumers_keep_loading_and_generation_undeclared():
    task = _model(tag="fill-mask", config={"model_type": "bert", "architectures": ["BertForMaskedLM"]})
    task, candidate = _resolve(task)
    assert candidate["loading"] == {"off": {}}
    assert next(iter(candidate["tasks"].values()))["generation"] == {}
    for interface in ("fill-mask", "text-classification", "feature-extraction"):
        assert not declared_generation_consumer(
            family="nlp", backend="transformers_model", adapter="family-default", task=interface,
        )
    assert declared_generation_consumer(
        family="nlp", backend="transformers_model", adapter="family-default", task="text-generation",
    )


def test_frozen_identity_transport_and_task_binding_reject_mutation():
    task, resolved = _resolve(_model())
    frozen = task.inference_configuration
    assert decode_inference_configuration(encode_inference_configuration(frozen)) == frozen
    with pytest.raises(ValueError, match="state"):
        encode_inference_configuration(resolved)
    changed = copy.deepcopy(frozen)
    changed["loading"]["off"]["dtype"] = "FP16"
    with pytest.raises(ValueError, match="identity_sha256"):
        validate_frozen_configuration(changed)
    task.selected_tasks[0]["config"]["generation"]["max_new_tokens"] += 1
    with pytest.raises(ValueError, match="active task"):
        validate_configuration_binding(frozen, task)


def test_profiler_variant_has_separate_identity_and_never_changes_parent_requests():
    task, _ = _resolve(_model())
    original = copy.deepcopy(task.inference_configuration)
    derived = derive_inference_configuration(
        original, loading_overrides={"off": {"attention_implementation": "sdpa"}},
        reason="Fixture alternate implementation", tool="torch-profiler-fixture",
    )
    assert task.inference_configuration == original
    assert derived["identity_sha256"] != original["identity_sha256"]
    assert derived["derivation"]["parent_identity_sha256"] == original["identity_sha256"]
    assert derived["requests"] == original["requests"]
    assert derived["loading"]["off"]["attention_implementation"] == "sdpa"
    with pytest.raises(ValueError, match="undeclared"):
        derive_inference_configuration(original, loading_overrides={"off": {"invented": True}}, reason="x", tool="x")


@pytest.mark.parametrize("bad", ["not base64", "e30=", "x" * 90000, 7])
def test_malformed_environment_configuration_fails_closed(bad):
    with pytest.raises(ValueError, match="configuration"):
        decode_inference_configuration(bad)


def test_confirmation_is_a_copy_and_changes_to_decision_evidence_change_identity():
    task, resolved = _resolve(_model())
    before = copy.deepcopy(task.inference_configuration)
    resolved["decisions"][0]["reason"] = "changed evidence"
    assert task.inference_configuration == before
    assert configuration_identity(resolved) != before["identity_sha256"]


@pytest.mark.parametrize(("total", "new"), [("max_length", "max_new_tokens"), ("min_length", "min_new_tokens")])
def test_conflicting_length_spellings_in_one_explicit_layer_are_not_silently_dropped(total, new):
    with pytest.raises(ValueError, match="conflict"):
        merge_generation_parameters({}, {total: 20, new: 4})
    assert merge_generation_parameters({total: 20}, {new: 4}) == {new: 4}


def test_cpu_and_gpu_auto_choices_are_frozen_separately():
    task = _model()
    task.runtime_profile_id = "nlp-cu128"
    draft = build_configuration_draft(task, devices=["off", "on"],
                                      requests={"schema_version": 1, "preset": "auto"})
    updated, candidate = resolve_inference_configuration(task, draft["requests"], devices=["off", "on"])
    assert candidate["loading"]["off"]["dtype"] == "FP32"
    assert candidate["loading"]["on"]["dtype"] == "FP16"
    assert {item["path"] for item in candidate["decisions"] if item["path"].endswith(".dtype")} == {
        "loading.off.dtype", "loading.on.dtype",
    }
    updated.inference_configuration = freeze_inference_configuration(candidate, candidate["identity_sha256"])
    validate_configuration_binding(updated.inference_configuration, updated)


def test_coalesced_tasks_keep_each_original_request_and_semantic_snapshot():
    config = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"], "task_specific_params": {
        "summarization": {"num_beams": 4}, "translation_en_to_de": {"num_beams": 1},
    }}
    initial = _model(config=config, tasks="summarization,translation_en_to_de")
    original_ids = {task["task_id"] for task in initial.selected_tasks}
    assert len(original_ids) == 2
    requests = {"schema_version": 1, "tasks": {
        identifier: {"num_beams": {"mode": "explicit", "value": 1}} for identifier in original_ids
    }}
    updated, candidate = _resolve(initial, requests)
    assert len(updated.selected_tasks) == 1
    assert set(candidate["requests"]["tasks"]) == original_ids
    task = next(iter(candidate["tasks"].values()))
    assert set(task["source_task_ids"]) == original_ids
    assert len(task["snapshot"]["semantic_tasks"]) == 2
    assert task["generation"]["num_beams"] == 1
    validate_configuration_binding(updated.inference_configuration, updated)


def test_inconsistent_metadata_source_revision_is_not_usable_as_default_evidence():
    task = _model()
    task.metadata_sources = {"config.json": {
        "encoding": "raw", "sha256": "b" * 64, "revision": "c" * 40, "bytes": 30,
    }}
    with pytest.raises(ValueError, match="config.json.*revision"):
        _draft(task)


def test_frozen_code_evidence_keeps_existing_source_scopes_distinct():
    _task, candidate = _resolve(_model())
    identities = candidate["backend"]["source_identity"]
    assert {kind: value["scope"] for kind, value in identities.items()} == {
        "measurement": "measurement-source-v2", "docker_service": "service-source-v2",
        "process_service": "process-service-v1",
    }
    assert all(len(value["sha256"]) == 64 for value in identities.values())
