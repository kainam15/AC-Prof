"""One evidence decision drives executable paths and preparation choices."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.preparation.collection_workflow import PreparationWorkflow
from acprof.host.preparation.task_selection import select_tasks
from acprof.models.candidates import apply_model_candidate, discover_model_candidates
from acprof.models.resolution import require_resolved_candidate


def model(*, tag="unknown", config=None, metadata=None, hub=None, library="transformers"):
    return TaskInfo("fixture/unnamed", tag, "unknown", "unknown", library, "a" * 40,
                    "hub_api", model_config=config or {}, repository_metadata=metadata or {},
                    hub_metadata=hub or {})


def resolve(task, **kwargs):
    task.model_resolution = apply_model_candidate(task, discover_model_candidates(task, **kwargs))
    return task


def options(**kwargs):
    return SimpleNamespace(task="", tasks="", task_family="", backend="", model_spec="", **kwargs)


@pytest.mark.parametrize("identifier", ["sentence-transformers/all-mpnet-base-v2", "fixture/renamed"])
def test_module_chain_suppresses_backbone_task_and_skips_selection(identifier):
    task = model(tag="sentence-similarity", library="sentence-transformers",
                 config={"architectures": ["MPNetForMaskedLM"], "model_type": "mpnet"},
                 metadata={"modules.json": [
                     {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                     {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}]},
                 hub={"pipeline_tag": "sentence-similarity", "tags": ["fill-mask", "feature-extraction"],
                      "transformers_info": {"pipeline_tag": "fill-mask", "auto_model": "AutoModelForMaskedLM"}})
    task.model_id = identifier
    resolve(task)
    assert task.task_discovery["status"] == "ready"
    assert {item["interface"] for item in task.task_discovery["tasks"]} == {"sentence-similarity"}
    workflow = Mock(spec=PreparationWorkflow, interactive=True)
    selected = select_tasks(task, options(), workflow)
    workflow.ask.assert_not_called()
    assert len(selected.selected_tasks) == 1
    assert selected.model_resolution["execution"]["operation"] == "encode"
    assert any(item["scope"] == "component" for item in task.model_resolution["evidence"])


def test_registered_single_head_automatically_materializes_workload():
    task = resolve(model(config={"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}))
    workflow = Mock(spec=PreparationWorkflow, interactive=True)
    selected = select_tasks(task, options(), workflow)
    assert len(selected.selected_tasks) == 1
    assert selected.pipeline_tag == "text-generation"
    assert not selected.model_resolution["missing"]
    workflow.ask.assert_not_called()


def test_explicit_choice_cannot_override_an_incompatible_checkpoint_head():
    task = resolve(model(tag="fill-mask", config={"architectures": ["BertForMaskedLM"]}),
                   override_tag="text-generation")
    with pytest.raises(ValueError, match="conflict|incompatible"):
        require_resolved_candidate(task)


def test_unknown_module_requires_configuration_without_importing_it():
    task = resolve(model(tag="sentence-similarity", library="sentence-transformers",
                         metadata={"modules.json": [
                             {"idx": 0, "path": "", "type": "remote_code.Surprise"}]}))
    with pytest.raises(ValueError, match="module|模块"):
        require_resolved_candidate(task)


def test_causal_lm_uses_one_path_for_translation_summary_and_question_answering():
    task = resolve(model(tag="text-generation", config={"architectures": ["GPT2LMHeadModel"],
                         "task_specific_params": {"summarization": {}, "translation_en_to_de": {},
                                                  "question-answering": {}}}))
    assert task.task_discovery["status"] == "ready"
    assert {item["interface"] for item in task.task_discovery["tasks"]} == {"text-generation"}
    assert len(task.task_discovery["profiles"]) == 1


def test_primary_metadata_conflict_does_not_become_a_task_menu():
    task = resolve(model(tag="text-generation", config={"architectures": ["BertForMaskedLM"]}))
    assert task.task_discovery["status"] == "needs_configuration"
    assert not task.task_discovery["profiles"]


def test_module_chain_without_hub_task_still_selects_encoding():
    task = resolve(model(config={"architectures": ["MPNetForMaskedLM"]},
                         metadata={"modules.json": [
                             {"idx": 0, "path": "backbone", "type": "sentence_transformers.models.Transformer"},
                             {"idx": 1, "path": "pooling", "type": "sentence_transformers.models.Pooling"}]}))
    assert task.pipeline_tag == "feature-extraction"
    assert task.runtime_backend == "sentence_transformers"
    assert task.task_discovery["status"] == "ready"


def test_component_pipeline_keeps_one_path_and_binds_component_identity():
    import copy

    task = model(tag="text-to-image", library="diffusers", metadata={"model_index.json": {
        "_class_name": "FixturePipeline", "text_encoder": ["transformers", "FixtureEncoder"],
        "denoiser": ["diffusers", "FixtureDenoiser"], "scheduler": ["diffusers", "FixtureScheduler"]}})
    first = resolve(copy.deepcopy(task))
    task.repository_metadata["model_index.json"]["denoiser"][1] = "DifferentDenoiser"
    second = resolve(task)
    assert first.task_discovery["status"] == "ready"
    assert len(first.model_resolution["execution"]["components"]) == 3
    assert first.task_discovery["profiles"][0]["signature"] != second.task_discovery["profiles"][0]["signature"]


@pytest.mark.parametrize("tag", ["unknown", "fill-mask"])
def test_separate_text_and_image_heads_require_a_real_path_choice(tag):
    task = resolve(model(tag=tag, config={"architectures": ["BertForMaskedLM", "ViTForImageClassification"]}))
    assert task.task_discovery["status"] == "selection_required"
    assert {item["interface"] for item in task.task_discovery["tasks"]} == {"fill-mask", "image-classification"}
    assert len(task.task_discovery["profiles"]) == 2


def test_selected_profile_rejects_changed_execution_components():
    task = resolve(model(tag="fill-mask", config={"architectures": ["BertForMaskedLM"]}))
    selected = select_tasks(task, options(), PreparationWorkflow())
    selected.model_config["architectures"] = ["RobertaForMaskedLM"]
    with pytest.raises(ValueError, match="identity"):
        select_tasks(selected, options(), PreparationWorkflow())


def test_module_configuration_is_read_at_the_pinned_revision(tmp_path, monkeypatch):
    import json

    from acprof.host import detect
    from acprof.hub.repository import MetadataFile
    snapshot = {"modules.json": [{"idx": 0, "path": "encoder", "type": "sentence_transformers.models.Transformer"},
                                  {"idx": 1, "path": "pooling", "type": "sentence_transformers.models.Pooling"}],
                "encoder/config.json": {"model_type": "bert"},
                "pooling/config.json": {"pooling_mode_mean_tokens": True}}
    reads = []
    def download(model_id, filename, revision):
        reads.append((filename, revision))
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot[filename]))
        return MetadataFile(path, path.read_bytes())
    monkeypatch.setattr(detect, "_download_metadata", download)
    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename=name) for name in snapshot])
    result = detect._repository_metadata("fixture/model", "a" * 40, info)
    assert result["repository_metadata"] == snapshot
    assert {revision for _, revision in reads} == {"a" * 40}
    assert len(reads) == 3


def test_auto_selection_keeps_user_explicit_interface_and_checks_compatibility():
    task = model(config={"architectures": ["BertForMaskedLM", "ViTForImageClassification"]})
    resolve(task, override_tag="image-classification")
    args = options()
    args.task = "image-classification"
    selected = select_tasks(task, args, PreparationWorkflow())
    assert selected.pipeline_tag == "image-classification"
    assert len(selected.selected_tasks) == 1


def test_supplemented_configuration_automatically_selects_the_resolved_path(monkeypatch):
    from tests.execution.operations.test_collection_workflow import answer_with_parameter_defaults

    unknown = resolve(model())
    resolved = resolve(model(tag="fill-mask", config={"architectures": ["BertForMaskedLM"]}))
    workflow = PreparationWorkflow(interactive=True)
    monkeypatch.setattr("acprof.host.detect.detect_task", lambda **kwargs: unknown)
    monkeypatch.setattr("acprof.host.preparation.task_support.require_task_support", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow, "_apply", lambda *args: resolved)
    ask = Mock(side_effect=answer_with_parameter_defaults(
        {"action": "answer", "answers": {"model_spec": "model.json"}}, {"action": "confirm"}))
    monkeypatch.setattr(workflow, "ask", ask)
    args = options(model=unknown.model_id, revision="", batch_size=1, gpus="off")
    result = workflow.resolve(args)
    assert ask.call_count == 3
    assert len(result.selected_tasks) == 1
    assert result.selected_tasks[0]["execution_path"]["operation"] == "__call__"


@pytest.mark.parametrize("types", [["Normalize"], ["Transformer", "Normalize"], ["Pooling", "Transformer"]])
def test_module_chain_requires_connected_input_and_output_protocols(types):
    task = resolve(model(metadata={"modules.json": [
        {"idx": i, "path": str(i), "type": "sentence_transformers.models." + kind}
        for i, kind in enumerate(types)]}))
    assert task.task_discovery["status"] == "needs_configuration"
    assert not task.task_discovery["profiles"]
