"""Authored tasks are workloads, not every operation supported by an architecture."""
import copy

import pytest
from task_selection_fixtures import t5_task as pinned_t5_task

from acprof.host.detect import TaskInfo, _architecture_metadata
from acprof.model_candidates import discover_declared_tasks


@pytest.fixture
def t5_task():
    task = pinned_t5_task()
    task.pipeline_tag = "translation"
    task.runtime_backend = "transformers_pipeline"
    task.hub_metadata.update(tags=["translation", "summarization"], transformers_info={
        "auto_model": "AutoModelForSeq2SeqLM", "pipeline_tag": "text2text-generation"})
    return task


def test_t5_discovers_four_distinct_workloads_without_architecture_inventions(t5_task):
    original = copy.deepcopy(t5_task)
    found = discover_declared_tasks(t5_task)
    assert found["status"] == "selection_required"
    assert len(found["profiles"]) == 2  # Summary forbids repeated 3-grams; translations do not.
    assert {task_id for profile in found["profiles"] for task_id in profile["semantic_task_ids"]} == {
        "translation_en_to_de", "translation_en_to_fr", "translation_en_to_ro", "summarization"}
    assert {task["task_id"] for task in found["tasks"]} == {
        "translation_en_to_de", "translation_en_to_fr", "translation_en_to_ro", "summarization"}
    assert {task["interface"] for task in found["tasks"]} == {"text2text-generation"}
    german = next(task for task in found["tasks"] if task["task_id"] == "translation_en_to_de")
    assert german["config"]["prefix"] == "translate English to German: "
    assert german["config"]["generation"]["num_beams"] == 4
    assert "config.task_specific_params.translation_en_to_de" in german["evidence"]
    assert t5_task == original


def test_group_uses_real_backend_not_the_original_hub_pipeline(t5_task):
    from types import SimpleNamespace

    from acprof.host.collection_workflow import PreparationWorkflow
    from acprof.host.task_selection import select_tasks

    discovery = discover_declared_tasks(t5_task)
    assert len(discovery["profiles"]) == 2
    assert {profile["signature"]["backend"] for profile in discovery["profiles"]} == {"transformers_model"}
    t5_task.task_discovery = discovery
    options = SimpleNamespace(task="", tasks="all", task_family="", backend="", model_spec="", workload_spec="")
    resolved = select_tasks(t5_task, options, PreparationWorkflow())
    assert resolved.runtime_backend == "transformers_model"
    assert resolved.selected_tasks[0]["execution_profile"]["signature"]["backend"] == resolved.runtime_backend


def test_light_metadata_preserves_explicit_task_and_loader_declarations(t5_task):
    assert _architecture_metadata(t5_task.model_config)["task_specific_params"] == t5_task.model_config["task_specific_params"]
    assert _architecture_metadata(t5_task.model_config)["is_encoder_decoder"] is True


def test_architecture_only_requires_manual_configuration(t5_task):
    t5_task.model_config.pop("task_specific_params")
    t5_task.hub_metadata = {"pipeline_tag": None}
    t5_task.detection_method = "config_infer"
    found = discover_declared_tasks(t5_task)
    assert found["status"] == "needs_configuration"
    assert found["tasks"] == []
    assert found["inferred"]


def test_single_declared_pipeline_is_ready():
    task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline",
                    "transformers", "b" * 40, "hub_api", hub_metadata={"pipeline_tag": "fill-mask"})
    found = discover_declared_tasks(task)
    assert found["status"] == "ready"
    assert [item["task_id"] for item in found["tasks"]] == ["fill-mask"]


def test_generic_tags_merge_evidence_without_duplicate_tasks(t5_task):
    t5_task.hub_metadata["tags"] += ["summarization", "translation_en_to_de", "text2text-generation"]
    found = discover_declared_tasks(t5_task)
    assert len(found["tasks"]) == 4
    german = next(task for task in found["tasks"] if task["task_id"] == "translation_en_to_de")
    assert "hub.tags" in german["evidence"]


def test_incompatible_declared_interface_is_reported(t5_task):
    t5_task.hub_metadata["pipeline_tag"] = "image-classification"
    found = discover_declared_tasks(t5_task)
    assert found["conflicts"]
    assert not any(item["task_id"] == "image-classification" for item in found["tasks"])


@pytest.mark.parametrize("declaration", ["unregistered-task", "translation", "translation_en_to_xx"])
def test_unusable_declaration_never_becomes_selectable(t5_task, declaration):
    t5_task.model_config["task_specific_params"] = {declaration: {}}
    t5_task.hub_metadata = {"pipeline_tag": None}
    found = discover_declared_tasks(t5_task)
    assert found["tasks"] == []
    assert found["unresolved"]


def test_generation_parameters_are_not_silently_discarded(t5_task):
    t5_task.model_config["task_specific_params"] = {"summarization": {"unexpected_option": 1}}
    t5_task.hub_metadata = {"pipeline_tag": None}
    found = discover_declared_tasks(t5_task)
    assert found["tasks"] == []
    assert "unexpected_option" in str(found["unresolved"])


def test_generic_tag_cannot_hide_invalid_explicit_configuration(t5_task):
    t5_task.model_config["task_specific_params"] = {"summarization": {"num_beams": "four"}}
    t5_task.hub_metadata = {"pipeline_tag": "summarization"}
    found = discover_declared_tasks(t5_task)
    assert found["tasks"] == []
    assert found["unresolved"]


def test_aliases_with_conflicting_configuration_require_manual_resolution(t5_task):
    t5_task.model_config["task_specific_params"] = {
        "translation_en_to_de": {"max_length": 100},
        "translation-en-to-de": {"max_length": 200},
    }
    t5_task.hub_metadata = {"pipeline_tag": "translation"}
    found = discover_declared_tasks(t5_task)
    assert found["tasks"] == []
    assert found["conflicts"]


def test_generic_classification_label_does_not_duplicate_sentiment_task():
    task = TaskInfo("fixture/model", "text-classification", "nlp", "transformers_pipeline",
                    "transformers", "b" * 40, "hub_api",
                    hub_metadata={"pipeline_tag": "text-classification", "tags": ["sentiment-analysis"]})
    found = discover_declared_tasks(task)
    assert [item["task_id"] for item in found["tasks"]] == ["sentiment-analysis"]
    assert found["tasks"][0]["evidence"] == ["hub.tags", "hub.pipeline_tag"]
