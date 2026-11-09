"""Task display stays typed, bounded and separate from grouping and execution."""
import copy
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from acprof.tasks.presentation import ConfigField, TaskSelection, build_task_selection
from acprof.tui.i18n import CatalogMessage, translate
from acprof.tui.models.task_choice_view import (
    selection_summary,
    task_context_view,
    task_selection_view,
)
from acprof.tui.models.task_labels import parameter_fact
from tests.support.task_selection_fixtures import t5_profiles


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize(("path", "value", "zh_label", "en_label", "zh_value", "en_value"), [
    (("generation", "no_repeat_ngram_size"), 0, "禁止重复 n-gram", "No-repeat n-gram", "0", "0"),
    (("generation", "num_beams"), 4, "Beam 数量", "Beam count", "4", "4"),
    (("interface",), "text2text-generation", "推理接口", "Inference interface", "Seq2Seq", "Seq2Seq"),
    (("interface",), "text-generation", "推理接口", "Inference interface", "自回归生成", "Autoregressive generation"),
    (("generation", "do_sample"), False, "随机采样", "Random sampling", "未开启", "Off"),
    (("generation", "use_cache"), True, "KV 缓存", "KV cache", "开启", "On"),
])
def test_known_parameter_labels_and_values_are_localized_separately(
    language, path, value, zh_label, en_label, zh_value, en_value,
):
    row = parameter_fact(ConfigField(path, value))

    assert translate(row.label, language) == (zh_label if language == "zh" else en_label)
    assert translate(row.value, language) == (zh_value if language == "zh" else en_value)
    assert json.loads(row.detail) == {"path": list(path), "present": True, "value": value}
    assert row.copyable
    assert row.status is None


@pytest.mark.parametrize(("value", "literal"), [
    (None, "null"), (False, "false"), (True, "true"), (0, "0"), (0.0, "0.0"),
    (1, "1"), (1.0, "1.0"), ("1", '"1"'), ("", '""'),
    ("开启 [bold]", '"开启 [bold]"'), ({}, "{}"), ([], "[]"),
])
def test_unknown_values_keep_json_types_and_literal_content(value, literal):
    row = parameter_fact(ConfigField(("new.flag",), value))

    assert row.value == literal
    assert not isinstance(row.value, CatalogMessage)
    assert translate(row.label, "en") == '["new.flag"]'
    detail = json.loads(row.detail)
    assert detail["path"] == ["new.flag"]
    assert detail["present"] is True
    assert detail["value"] == value
    assert type(detail["value"]) is type(value)


@pytest.mark.parametrize(("path", "expected"), [
    (("开启",), "开启"), (("new.flag",), '["new.flag"]'),
    (("root", 2, "a.b"), 'root[2]["a.b"]'), ((), "$"),
])
def test_unknown_paths_are_not_translated_even_when_they_match_ui_copy(path, expected):
    row = parameter_fact(ConfigField(path, "raw"))

    assert translate(row.label, "zh") == expected
    assert translate(row.label, "en") == expected
    assert json.loads(row.detail)["path"] == list(path)


@pytest.mark.parametrize("language", ["zh", "en"])
def test_missing_is_not_null_and_keeps_presence_evidence(language):
    missing = parameter_fact(ConfigField(("nullable",), None, present=False))
    explicit_null = parameter_fact(ConfigField(("nullable",), None))

    assert translate(missing.value, language) == ("未设置" if language == "zh" else "Not set")
    assert explicit_null.value == "null"
    assert json.loads(missing.detail) == {"path": ["nullable"], "present": False, "value": None}
    assert json.loads(explicit_null.detail) == {"path": ["nullable"], "present": True, "value": None}


def test_nested_parameters_and_long_values_keep_full_copyable_evidence():
    raw = " 中文 [literal] <markup> = value\n" * 700
    value = {"nested": [False, 0, 0.0, None, raw]}
    before = copy.deepcopy(value)
    row = parameter_fact(ConfigField(("future",), value))

    assert row.copyable
    assert len(row.value) > 8192
    assert json.loads(row.value) == before
    assert json.loads(row.detail)["value"] == before
    value["nested"].append("later mutation")
    assert json.loads(row.value) == before
    assert json.loads(row.detail)["value"] == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_invalid_json_numbers_are_rejected_instead_of_silently_rewritten(value):
    with pytest.raises(ValueError, match="Out of range float values"):
        parameter_fact(ConfigField(("future",), value))


def test_t5_common_and_differences_reuse_resolved_data_without_mutation():
    presentation = build_task_selection(t5_profiles())
    before = copy.deepcopy(presentation)
    view = task_selection_view(presentation)

    assert presentation == before
    assert view.semantic_count == 4
    assert [group.group_id for group in view.groups] == [
        "text2text-generation-aeecc7756456", "text2text-generation-d497d576a075",
    ]
    assert [json.loads(row.detail)["path"] for row in view.common.rows] == [
        ["interface"], ["backend"], ["operation"],
    ]
    assert [json.loads(row.detail)["path"] for row in view.common.detail_rows[:-1]] == [
        ["generation", "do_sample"], ["generation", "early_stopping"],
        ["generation", "encoder_no_repeat_ngram_size"], ["generation", "num_beams"],
        ["generation", "num_return_sequences"], ["generation", "repetition_penalty"],
        ["generation", "use_cache"],
    ]
    assert "Shared inference paths" in translate(view.common.detail_rows[-1].value, "en")
    assert view.common.detail_rows[-1].label.msgid == "common.field.note"

    summary, translation = view.groups
    assert [json.loads(row.detail)["value"] for row in summary.differences.rows] == [2.0, 3]
    assert [json.loads(row.detail)["value"] for row in translation.differences.rows] == [1.0, 0]
    for group in view.groups:
        assert len(group.differences.rows) == 2
        assert len(group.differences.detail_rows) == 3
        assert all(row.copyable for row in group.differences.detail_rows)
    assert summary.differences.detail_rows[0].value == "text2text-generation-aeecc7756456"
    assert summary.differences.detail_rows[1].value == "summarization"
    assert json.loads(summary.differences.detail_rows[2].value) == ["summarization"]
    assert translation.differences.detail_rows[1].value == "translation_en_to_de"
    assert json.loads(translation.differences.detail_rows[2].value) == [
        "translation_en_to_de", "translation_en_to_fr", "translation_en_to_ro",
    ]


def test_common_priority_is_bounded_and_preserves_other_field_order():
    fields = (
        ConfigField(("future_z",), 1), ConfigField(("operation",), "forward"),
        ConfigField(("future_a",), 2), ConfigField(("backend",), "custom"),
    )
    source = TaskSelection(fields, (), 0)
    view = task_selection_view(source)

    assert [json.loads(row.detail)["path"] for row in view.common.rows] == [
        ["backend"], ["operation"], ["future_z"],
    ]
    assert json.loads(view.common.detail_rows[0].detail)["path"] == ["future_a"]
    assert source.common is fields


def test_all_extra_differences_remain_before_identity_details():
    source = build_task_selection(t5_profiles())
    fields = tuple(ConfigField(("future", index), index) for index in range(7))
    source = replace(source, groups=(replace(source.groups[0], differences=fields),))
    view = task_selection_view(source)
    section = view.groups[0].differences

    assert [json.loads(row.detail)["path"] for row in section.rows] == [
        ["future", 0], ["future", 1], ["future", 2], ["future", 3],
    ]
    assert [json.loads(row.detail)["path"] for row in section.detail_rows[:3]] == [
        ["future", 4], ["future", 5], ["future", 6],
    ]
    assert [row.label.msgid for row in section.detail_rows[3:]] == [
        "models.tasks.group_id", "models.tasks.representative_label", "models.tasks.semantic_tasks",
    ]


def test_empty_common_and_identical_configuration_use_explanations_not_fake_values():
    empty = task_selection_view(TaskSelection((), (), 0))
    assert len(empty.common.rows) == 1
    assert translate(empty.common.rows[0].value, "en") == "No common parameters"
    assert empty.common.rows[0].status is None
    assert empty.common.rows[0].detail == ""
    merged = task_selection_view(build_task_selection(t5_profiles(
        generation_override={"no_repeat_ngram_size": 0, "length_penalty": 1.0},
    )))
    assert len(merged.groups) == 1
    assert len(merged.groups[0].differences.rows) == 1
    assert translate(merged.groups[0].differences.rows[0].value, "en") == "None (same compute configuration)"
    assert merged.semantic_count == 4


@pytest.mark.parametrize(("indices", "expected_count"), [
    ((), 0), ((-1,), 0), ((0,), 1), ((0, 0, -1), 1), ((0, 1), 2), ((1, 0, 1, -1), 2),
])
@pytest.mark.parametrize("language", ["zh", "en"])
def test_summary_counts_only_distinct_existing_groups_without_certifying_readiness(indices, expected_count, language):
    view = task_selection_view(build_task_selection(t5_profiles()))
    selected = tuple("not-an-option" if index < 0 else view.groups[index].group_id for index in indices)
    summary = selection_summary(view, selected)

    assert summary.level == "neutral"
    assert summary.actions == ()
    expected = (
        ("4 种用途归为 2 组 · 待选择" if language == "zh" else "4 use cases in 2 groups · choose tasks")
        if expected_count == 0 else
        (f"4 种用途归为 2 组 · 已选 {expected_count} 组" if language == "zh"
         else f"4 use cases in 2 groups · selected {expected_count}")
    )
    assert translate(summary.headline, language) == expected


def test_empty_summary_and_frozen_view_do_not_infer_success():
    view = task_selection_view(TaskSelection((), (), 0))
    summary = selection_summary(view, ("unavailable",))

    assert summary.level == "neutral"
    assert translate(summary.headline, "en") == "No collection groups available"
    with pytest.raises(FrozenInstanceError):
        view.semantic_count = 9
    filled = task_selection_view(build_task_selection(t5_profiles()))
    with pytest.raises(FrozenInstanceError):
        filled.groups[0].group_id = "replacement"


def test_model_context_preserves_caller_values_and_hides_long_identity_by_default():
    revision = "0123456789abcdef" * 10
    fields = {"Model": " 开启 [bold] ", "Revision": revision, "Missing": None,
              "Zero": 0, "Flag": False, "Empty": ""}
    before = copy.deepcopy(fields)
    view = task_context_view(fields)

    assert fields == before
    assert view.collapsed
    assert view.title.msgid == "models.tasks.model_context"
    assert [row.label for row in view.rows] == list(fields)
    assert [row.value for row in view.rows[:2]] == [" 开启 [bold] ", revision]
    assert translate(view.rows[2].value, "en") == "Unknown"
    assert [row.value for row in view.rows[3:]] == ["0", "False", ""]
    assert all(row.copyable for row in view.rows)
    assert all(row.status is None for row in view.rows)
    fields["Revision"] = "changed"
    assert view.rows[1].value == revision
