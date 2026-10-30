"""Task parameter controls are metadata-driven and frozen before model loading."""
import copy
import io
import json
from types import SimpleNamespace

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.preparation.collection_workflow import PreparationWorkflow
from acprof.host.preparation.task_selection import review_selected_task_parameters, select_tasks
from acprof.models.candidates import discover_model_candidates
from acprof.tasks.parameters import (
    apply_parameter_review,
    describe_task_parameters,
    parameter_review,
)
from acprof.tasks.workloads import parse_task, task_fingerprint


def model(*, generation=None):
    info = TaskInfo("fixture/model", "text2text-generation", "nlp",
                    "transformers_model", "transformers", "a" * 40, "hub_api",
                    model_config={"is_encoder_decoder": True, "model_type": "t5",
                                  "architectures": ["T5ForConditionalGeneration"], "task_specific_params": {
                        "translation_en_to_de": {"prefix": "translate English to German: "},
                        "summarization": {"prefix": "summarize: "},
                    }},
                    repository_metadata={"generation_config.json": generation or {
                        "max_new_tokens": 48, "temperature": 0.8, "do_sample": False,
                    }}, hub_metadata={"pipeline_tag": "text2text-generation"})
    info.model_resolution = discover_model_candidates(info)
    info.task_discovery = info.model_resolution["task_discovery"]
    return info


def selected(info):
    return copy.deepcopy(info.task_discovery["tasks"])


def question_answers(review):
    return {entry["path"]: entry["value"]
            for entry in review["questions"] + review["parameter_advanced"]}


def test_capabilities_are_task_scoped_and_reuse_pinned_generation_defaults():
    info = model()
    task = selected(info)[0]
    controls = describe_task_parameters(info, task)
    values = {field["name"]: field for field in controls}
    assert values["max_new_tokens"]["value"] == 48
    assert values["max_new_tokens"]["source"] == "模型生成配置"
    assert values["temperature"]["value"] == 0.8
    assert values["temperature"]["source"] == "模型生成配置"
    assert all(field["support"] == "interface_candidate" for field in controls)
    assert "max_length" not in values


def test_inherited_total_length_remains_total_length():
    info = model(generation={"max_length": 35})
    task = selected(info)[0]
    controls = describe_task_parameters(info, task)
    assert controls[0]["name"] == "max_length"
    assert controls[0]["value"] == 35
    assert "max_new_tokens" not in task["config"]["generation"]


def test_unedited_defaults_do_not_change_identity_or_model_generation():
    info = model()
    tasks = selected(info)
    original = copy.deepcopy(tasks)
    answers = question_answers(parameter_review(info, tasks))
    applied = apply_parameter_review(info, tasks, answers)
    assert applied == original
    assert tasks == original
    assert all(task_fingerprint(t) == t["fingerprint"] for t in applied)


def test_user_changes_one_task_without_affecting_the_other_or_defaults():
    info = model()
    tasks = selected(info)
    review = parameter_review(info, tasks)
    answers = question_answers(review)
    answers["task_0.max_new_tokens"] = 80
    answers["task_0.do_sample"] = True
    answers["task_0.temperature"] = 0.55
    chosen = apply_parameter_review(info, tasks, answers)
    assert chosen[0]["config"]["generation"]["max_new_tokens"] == 80
    assert chosen[0]["config"]["generation"]["temperature"] == 0.55
    assert chosen[0]["config"]["generation"]["do_sample"] is True
    assert chosen[0]["config"]["user_generation_overrides"] == {
        "max_new_tokens": 80, "do_sample": True, "temperature": 0.55,
    }
    assert chosen[0]["fingerprint"] == task_fingerprint(chosen[0])
    assert chosen[0]["fingerprint"] != tasks[0]["fingerprint"]
    assert chosen[1] == tasks[1]
    assert tasks == selected(info)


@pytest.mark.parametrize(("path", "value", "message"), [
    ("task_0.max_new_tokens", 0, "positive integer"),
    ("task_0.max_new_tokens", True, "positive integer"),
    ("task_0.do_sample", "true", "boolean"),
    ("task_0.temperature", -1, "positive"),
    ("task_0.temperature", float("nan"), "finite"),
    ("task_0.top_p", 1.1, "top_p"),
    ("task_0.num_beams", 0, "positive integer"),
])
def test_invalid_parameter_values_are_rejected(path, value, message):
    info = model()
    tasks = selected(info)
    answers = question_answers(parameter_review(info, tasks))
    answers[path] = value
    with pytest.raises(ValueError, match=message):
        apply_parameter_review(info, tasks, answers)


def test_sampling_controls_require_sampling_and_answers_are_atomic():
    info = model()
    tasks = selected(info)
    answers = question_answers(parameter_review(info, tasks))
    answers["task_0.temperature"] = 0.25
    with pytest.raises(ValueError, match="随机采样"):
        apply_parameter_review(info, tasks, answers)
    assert all("user_generation_overrides" not in task["config"] for task in tasks)
    del answers["task_0.num_beams"]
    with pytest.raises(ValueError, match="exactly"):
        apply_parameter_review(info, tasks, answers)


def test_non_generation_models_do_not_get_generation_controls():
    info = model()
    task = parse_task("text-classification", {}, encoder_decoder=False, source="metadata")
    assert describe_task_parameters(info, task) == []
    info.runtime_backend = "custom_pipeline"
    review = parameter_review(info, selected(info))
    assert {field["path"].split(".")[1] for field in review["questions"]} == {"input_scales", "batch_size"}


def test_interactive_parameter_selection_happens_after_task_choice(monkeypatch, capsys):
    info = model()
    info.model_config["task_specific_params"]["summarization"]["num_beams"] = 4
    info.model_resolution = discover_model_candidates(info)
    info.task_discovery = info.model_resolution["task_discovery"]
    assert len(info.task_discovery["profiles"]) == 2
    profile = next(item["profile_id"] for item in info.task_discovery["profiles"]
                   if "translation_en_to_de" in item["semantic_task_ids"])
    opts = SimpleNamespace(task="", tasks="", task_family="", backend="", model_spec="", workload_spec="")
    preselected = select_tasks(copy.deepcopy(info), SimpleNamespace(
        **{**vars(opts), "tasks": profile}), PreparationWorkflow())
    answers = question_answers(parameter_review(preselected, preselected.selected_tasks))
    answers["task_0.max_new_tokens"] = 96
    stdin = (
        json.dumps({"id": 1, "action": "answer", "answers": {"tasks": [profile]}}) + "\n"
        + json.dumps({"id": 2, "action": "answer", "answers": answers}) + "\n"
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    workflow = PreparationWorkflow(interactive=True)
    result = select_tasks(info, opts, workflow)
    result = review_selected_task_parameters(result, opts, workflow)
    assert result.selected_tasks[0]["config"]["generation"]["max_new_tokens"] == 96
    assert result.model_resolution["selected_task"] == result.selected_tasks[0]
    output = capsys.readouterr().out
    assert output.index('"kind": "multi"') < output.index('"workload_parameters"')


def test_cli_keeps_model_defaults_without_parameter_dialog():
    info = model()
    opts = SimpleNamespace(task="", tasks="all", task_family="", backend="", model_spec="", workload_spec="")
    result = select_tasks(info, opts, PreparationWorkflow(interactive=False))
    assert [item["config"]["generation"]["max_new_tokens"] for item in result.selected_tasks] == [64]


def test_unchanged_input_scale_defaults_do_not_create_an_override():
    info = model()
    tasks = selected(info)
    review = parameter_review(info, tasks, batch_size=3, input_scales="64, 128")
    defaults = question_answers(review)
    assert apply_parameter_review(info, tasks, defaults,
                                  batch_size=3, input_scales="64, 128") == tasks


def test_per_task_input_scales_and_batch_size_are_saved_and_used_by_child(tmp_path):
    from dataclasses import replace

    from acprof.application.run import RunOutcome
    from acprof.application.task_runs import TaskCollection
    from acprof.experiments.configuration import RunConfig

    info = model()
    tasks = selected(info)
    review = parameter_review(info, tasks, batch_size=4)
    answers = question_answers(review)
    answers["task_0.input_scales"] = "256, 64,64"
    answers["task_0.batch_size"] = 2
    frozen = apply_parameter_review(info, tasks, answers, batch_size=4)
    assert frozen[0]["config"]["user_experiment_overrides"] == {
        "input_scales": "64,256", "batch_size": 2,
    }
    assert "user_experiment_overrides" not in frozen[1]["config"]
    info.selected_tasks = frozen
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path), batch_size=4)
    collection = TaskCollection(config, info)
    first = collection.child_config(frozen[0])
    second = collection.child_config(frozen[1])
    assert (first.input_scales, first.batch_size) == ("64,256", 2)
    assert (second.input_scales, second.batch_size) == ("", 4)
    seen = []
    assert collection.run(lambda options, task: seen.append(
        (options.input_scales, options.batch_size),
    ) or RunOutcome("ok")).status == "ok"
    assert seen == [("64,256", 2), ("", 4)]
    restored = TaskCollection.restore(replace(config, resume=True, task_run_id=collection.run_id))
    assert restored.task.selected_tasks == frozen


def test_task_snapshot_cannot_override_runtime_or_result_destination(tmp_path):
    from dataclasses import replace

    from acprof.application.task_runs import TaskCollection
    from acprof.experiments.configuration import RunConfig

    info = model()
    info.selected_tasks = selected(info)
    task = info.selected_tasks[0]
    task["config"]["user_experiment_overrides"] = {"output_dir": "/tmp/elsewhere"}
    collection = TaskCollection(replace(RunConfig.smoke("fixture/model"),
                                        output_dir=str(tmp_path)), info)
    with pytest.raises(ValueError, match="invalid task experiment overrides"):
        collection.child_config(task)


@pytest.mark.parametrize(("scale", "message"), [
    ("1,,2", "comma-separated"), ("nan", "finite"), ("0,16", "positive"),
    ("32.5", "integer"), ("1e999", "finite"), ("1,foo", "numbers"),
])
def test_invalid_input_scales_fail_before_preparing_a_model(scale, message):
    info = model()
    tasks = selected(info)
    answers = question_answers(parameter_review(info, tasks))
    answers["task_0.input_scales"] = scale
    with pytest.raises(ValueError, match=message):
        apply_parameter_review(info, tasks, answers)


def test_non_generative_workloads_allow_task_scoped_input_override():
    info = model()
    task = parse_task("text-classification", {}, encoder_decoder=False, source="metadata")
    review = parameter_review(info, [task], batch_size=1)
    answers = question_answers(review)
    assert "task_0.max_new_tokens" not in answers
    answers["task_0.input_scales"] = "32,64"
    applied = apply_parameter_review(info, [task], answers)
    assert applied[0]["config"]["user_experiment_overrides"] == {"input_scales": "32,64"}
    assert applied[0]["fingerprint"] == task_fingerprint(applied[0])


def _selected_inference_profiles():
    from acprof.tasks.groups import group_inference_tasks

    tasks = [
        parse_task("summarization", {"num_beams": 4}, encoder_decoder=True, source="fixture"),
        parse_task("translation_en_to_de", {"num_beams": 1}, encoder_decoder=True, source="fixture"),
    ]
    return group_inference_tasks(tasks, backend="transformers_model")


def test_beam_edit_rebinds_group_id_signature_and_snapshot_without_changing_evidence():
    from acprof.tasks.groups import inference_signature

    info = model()
    original = _selected_inference_profiles()
    index = next(index for index, task in enumerate(original)
                 if task["execution_profile"]["signature"]["generation"]["num_beams"] == 1)
    answers = question_answers(parameter_review(info, original))
    answers[f"task_{index}.num_beams"] = 8
    answers[f"task_{index}.max_new_tokens"] = 96
    changed = apply_parameter_review(info, original, answers)
    modified = next(item for item in changed if item["config"]["generation"]["num_beams"] == 8)

    assert modified["task_id"] != original[index]["task_id"]
    assert modified["execution_profile"]["signature"] == inference_signature(
        modified, backend="transformers_model")
    assert modified["execution_profile"]["signature"]["generation"]["num_beams"] == 8
    assert modified["execution_profile"]["output_length_policy"] == "user_max_new_tokens_96"
    assert modified["fingerprint"] == task_fingerprint(modified)
    assert modified["semantic_tasks"] == original[index]["semantic_tasks"]
    assert original[index]["config"]["generation"]["num_beams"] == 1
    assert len({task["task_id"] for task in changed}) == len(changed)


def test_matching_compute_edits_coalesce_instead_of_repeating_matrix():
    info = model()
    original = _selected_inference_profiles()
    index = next(index for index, task in enumerate(original)
                 if task["config"]["generation"]["num_beams"] == 4)
    answers = question_answers(parameter_review(info, original))
    answers[f"task_{index}.num_beams"] = 1
    merged = apply_parameter_review(info, original, answers)

    assert len(merged) == 1
    assert set(merged[0]["execution_profile"]["semantic_task_ids"]) == {
        "summarization", "translation_en_to_de"}
    assert {task["task_id"] for task in merged[0]["semantic_tasks"]} == {
        "summarization", "translation_en_to_de"}
    assert merged[0]["execution_profile"]["signature"]["generation"]["num_beams"] == 1
    assert merged[0]["config"]["user_generation_overrides"]["num_beams"] == 1
    assert merged[0]["fingerprint"] == task_fingerprint(merged[0])


@pytest.mark.parametrize("conflict", ["output_length", "input_scale", "batch_size"])
def test_duplicate_compute_path_rejects_conflicting_workload_axes(conflict):
    info = model()
    original = _selected_inference_profiles()
    index = next(index for index, task in enumerate(original)
                 if task["config"]["generation"]["num_beams"] == 4)
    answers = question_answers(parameter_review(info, original))
    answers[f"task_{index}.num_beams"] = 1
    if conflict == "output_length":
        answers[f"task_{index}.max_new_tokens"] = 128
    elif conflict == "input_scale":
        answers[f"task_{index}.input_scales"] = "64,128"
    else:
        answers[f"task_{index}.batch_size"] = 2
    with pytest.raises(ValueError, match="相同推理计算路径"):
        apply_parameter_review(info, original, answers)


def test_unchanged_group_controls_preserve_normalized_identity():
    info = model()
    original = _selected_inference_profiles()
    answers = question_answers(parameter_review(info, original))
    assert apply_parameter_review(info, original, answers) == original
    assert all(task_fingerprint(task) == task["fingerprint"] for task in original)
