"""Selection decisions follow complete signatures, never model-name heuristics."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from task_selection_fixtures import t5_profiles

from acprof.inference_groups import group_inference_tasks
from acprof.task_presentation import build_task_selection, compare_configurations
from acprof.task_workloads import parse_task


def values(fields):
    return {field.path: (field.present, field.value) for field in fields}


def test_real_t5_metadata_and_effective_ngram_difference():
    root = Path(__file__).parent / "fixtures" / "t5_small_metadata"
    for name, source in json.loads((root / "source.json").read_text())["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == source["sha256"]
    profiles = t5_profiles()
    before = copy.deepcopy(profiles)
    view = build_task_selection(profiles)
    assert profiles == before
    assert len(view.groups) == 2
    assert view.semantic_count == 4
    assert values(view.common)[("generation", "num_beams")] == (True, 4)
    assert ("generation", "no_repeat_ngram_size") not in values(view.common)
    by_kind = {group.purposes[0].kind: group for group in view.groups}
    summary, translation = by_kind["summarization"], by_kind["translation"]
    assert values(summary.differences) == {("generation", "no_repeat_ngram_size"): (True, 3),
                                           ("generation", "length_penalty"): (True, 2.0)}
    assert values(translation.differences) == {("generation", "no_repeat_ngram_size"): (True, 0),
                                               ("generation", "length_penalty"): (True, 1.0)}
    assert translation.purposes[0].directions == (("en", "de"), ("en", "fr"), ("en", "ro"))
    assert translation.representative["task_id"] == "translation_en_to_de"
    assert summary.representative["task_id"] == "summarization"
    assert translation.profile["semantic_tasks"] == before[profiles.index(translation.profile)]["semantic_tasks"]
    assert build_task_selection(list(reversed(profiles))) == view


def test_comparison_preserves_missing_null_bool_number_and_nested_unknown_fields():
    configs = [
        {"unknown": {"list": [True, {"x": 1}], "empty": {}, "flag": False}, "nullable": None},
        {"unknown": {"flag": 0, "empty": {}, "list": [1, {"x": 1}, "extra"]}},
    ]
    result = compare_configurations(configs)
    assert values(result.common) == {
        ("unknown", "empty"): (True, {}), ("unknown", "list", 1, "x"): (True, 1)}
    assert values(result.differences[0]) == {
        ("nullable",): (True, None), ("unknown", "flag"): (True, False),
        ("unknown", "list", 0): (True, True), ("unknown", "list", 2): (False, None)}
    assert values(result.differences[1]) == {
        ("nullable",): (False, None), ("unknown", "flag"): (True, 0),
        ("unknown", "list", 0): (True, 1), ("unknown", "list", 2): (True, "extra")}
    assert [field.path for field in result.differences[0]] == [
        ("nullable",), ("unknown", "flag"), ("unknown", "list", 0), ("unknown", "list", 2)]


@pytest.mark.parametrize(("left", "right"), [
    (False, 0), (True, 1), (1, 1.0), ({}, []), ({"x": 1}, None),
    ([], [None]), ({"a": []}, {"a": {}}), ("1", 1),
])
def test_values_distinct_in_signature_remain_distinct_in_display(left, right):
    compared = compare_configurations([{"value": left}, {"value": right}])
    assert compared.common == ()
    assert compared.differences[0]
    assert compared.differences[1]


@pytest.mark.parametrize("count", [0, 1, 3])
def test_no_candidates_single_and_identical_signatures(count):
    compared = compare_configurations([{"x": {"a": True, "b": [2, 3]}}] * count)
    assert len(compared.common) == (3 if count else 0)
    assert compared.differences == ((),) * count
    assert build_task_selection([]).groups == ()


def test_override_is_compared_after_normalization_without_rewriting_declarations():
    partial = build_task_selection(t5_profiles(generation_override={"no_repeat_ngram_size": 0}))
    assert len(partial.groups) == 2  # Beam scoring still differs.
    view = build_task_selection(t5_profiles(generation_override={"no_repeat_ngram_size": 0, "length_penalty": 1.0}))
    assert len(view.groups) == 1
    assert view.groups[0].differences == ()
    assert [purpose.kind for purpose in view.groups[0].purposes] == ["summarization", "translation"]
    assert values(view.common)[("generation", "no_repeat_ngram_size")] == (True, 0)
    assert view.groups[0].representative["config"]["generation"]["no_repeat_ngram_size"] == 3
    assert view.groups[0].profile["config"]["generation"]["no_repeat_ngram_size"] == 0


def test_new_fields_are_compared_without_parameter_registry_changes():
    profiles = t5_profiles()
    profiles[0]["execution_profile"]["signature"]["future"] = {"nested": ["a", "b"], "温度": 0.7}
    view = build_task_selection(profiles)
    first, second = view.groups
    assert values(first.differences)[("future",)] == (True, {"nested": ["a", "b"], "温度": 0.7})
    assert values(second.differences)[("future",)] == (False, None)


def test_beam_sampling_cache_backend_and_mixed_purposes():
    declarations = [parse_task(name, params, encoder_decoder=encoder, source="fixture")
                    for name, params, encoder in [
                        ("summarization", {"num_beams": 4}, True),
                        ("translation_en_to_fr", {"num_beams": 4}, True),
                        ("text-generation", {"do_sample": True, "top_p": 0.8, "use_cache": False}, False),
                        ("text-classification", {}, False)]]
    profiles = group_inference_tasks(declarations, backend={
        "text2text-generation": "transformers_model", "text-generation": "vllm",
        "text-classification": "transformers_pipeline"})
    view = build_task_selection(profiles)
    assert len(view.groups) == 3
    assert view.common == ()
    assert view.semantic_count == 4
    assert {purpose.kind for group in view.groups for purpose in group.purposes} == {
        "summarization", "translation", "text-generation", "text-classification"}
    assert all(("backend",) in values(group.differences) for group in view.groups)


def test_unknown_semantic_kind_keeps_standard_id_and_input_configuration():
    profiles = t5_profiles()[:1]
    member = profiles[0]["semantic_tasks"][0]
    member.update(task_id="future-task", kind="new-kind")
    member["config"]["input_text"] = "中文 input [literal]"
    profiles[0]["semantic_tasks"] = [member]
    profiles[0]["execution_profile"]["representative_task_id"] = "future-task"
    view = build_task_selection(profiles)
    assert view.groups[0].purposes[0].task_ids == ("future-task",)
    assert view.groups[0].representative["config"]["input_text"] == "中文 input [literal]"


def test_missing_signature_is_rejected_instead_of_comparing_authored_config():
    profiles = t5_profiles()
    del profiles[0]["execution_profile"]["signature"]
    with pytest.raises(KeyError, match="signature"):
        build_task_selection(profiles)
