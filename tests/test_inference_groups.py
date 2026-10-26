"""Inferred compute groups must not multiply runs for prompt-only use cases."""
import copy
import json

from acprof.inference_groups import group_inference_tasks, inference_signature
from acprof.task_workloads import parse_task, task_fingerprint, workload_manifest
from acprof.workloads.nlp import NLPWorkloadGenerator


def example(name, **generation):
    return parse_task(name, generation, encoder_decoder=True, source="fixture")


def test_t5_task_specific_lengths_and_beam_scoring_share_one_execution_group(tmp_path):
    tasks = [
        example("summarization", max_length=200, min_length=30,
                num_beams=4, length_penalty=2.0, early_stopping=True),
        example("translation_en_to_de", max_length=300, num_beams=4, early_stopping=True),
        example("translation_en_to_fr", max_length=300, num_beams=4, early_stopping=True),
        example("translation_en_to_ro", max_length=300, num_beams=4, early_stopping=True),
    ]
    original = copy.deepcopy(tasks)
    groups = group_inference_tasks(tasks, backend="transformers_model")
    assert tasks == original
    assert len(groups) == 1
    profile = groups[0]
    assert set(profile["execution_profile"]["semantic_task_ids"]) == {item["task_id"] for item in tasks}
    assert profile["execution_profile"]["signature"]["generation"]["num_beams"] == 4
    assert profile["config"]["generation"]["max_new_tokens"] == 64
    assert "max_length" not in profile["config"]["generation"]
    assert "length_penalty" not in profile["config"]["generation"]
    assert profile["fingerprint"] == task_fingerprint(profile)

    spec = tmp_path / "workload.json"
    spec.write_text(json.dumps(workload_manifest(profile)))
    workload = NLPWorkloadGenerator("google-t5/t5-small", profile["interface"], 1, str(spec))
    request = workload.generate_for_word_count(20)
    assert request["text"].startswith("summarize: ")
    assert request["params"]["max_new_tokens"] == 64
    assert request["params"]["num_beams"] == 4
    assert request["task_workload"]["task_id"] == profile["task_id"]


def test_decoding_strategy_and_interface_changes_keep_distinct_measurements():
    tasks = [
        example("summarization", num_beams=4),
        example("translation_en_to_de", num_beams=1),
        example("translation_en_to_fr", do_sample=True, top_p=0.9),
        parse_task("text-classification", {}, encoder_decoder=False, source="fixture"),
    ]
    profiles = group_inference_tasks(tasks)
    assert len(profiles) == 4
    assert len({value["task_id"] for value in profiles}) == 4


def test_explicit_defaults_do_not_create_false_compute_group():
    a = example("summarization")
    b = example("translation_en_to_de", num_beams=1, use_cache=True, do_sample=False)
    assert inference_signature(a) == inference_signature(b)
    assert len(group_inference_tasks([a, b])) == 1


def test_semantic_provenance_affects_result_fingerprint_but_not_compute_group_id():
    first = example("translation_en_to_de", num_beams=4)
    second = copy.deepcopy(first)
    second["config"]["prefix"] = "Translate this English text into German: "
    second["fingerprint"] = task_fingerprint(second)
    a = group_inference_tasks([first])[0]
    b = group_inference_tasks([second])[0]
    assert a["task_id"] == b["task_id"]
    assert a["fingerprint"] != b["fingerprint"]
    assert a["execution_profile"]["representative_task_id"] == "translation_en_to_de"


def test_user_generation_override_changes_effective_grouping_without_overwriting_task_evidence():
    a = example("summarization", num_beams=4)
    b = example("translation_en_to_de", num_beams=1)
    original = copy.deepcopy([a, b])
    grouped = group_inference_tasks([a, b], generation_override={"num_beams": 4})
    assert len(grouped) == 1
    assert grouped[0]["execution_profile"]["signature"]["generation"]["num_beams"] == 4
    assert grouped[0]["semantic_tasks"] == original


def test_generation_special_token_configuration_stays_bound_to_selected_exemplar():
    en = example("translation_en_to_de", forced_bos_token_id=9, num_beams=4)
    fr = example("translation_en_to_fr", forced_bos_token_id=11, num_beams=4)
    profile = group_inference_tasks([en, fr])[0]
    assert len(profile["semantic_tasks"]) == 2
    assert profile["config"]["generation"]["forced_bos_token_id"] == 9
