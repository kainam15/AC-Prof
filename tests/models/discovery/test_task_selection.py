"""The preparation boundary must wait for semantic task choices before downloads."""
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.preparation.collection_workflow import PreparationInterrupted, PreparationWorkflow
from acprof.host.preparation.task_selection import select_tasks
from acprof.models.candidates import discover_model_candidates


def declared_model():
    task = TaskInfo("fixture/model", "text2text-generation", "nlp", "transformers_model",
                    "transformers", "a" * 40, "hub_api",
                    model_config={"is_encoder_decoder": True, "model_type": "t5",
                                  "architectures": ["T5ForConditionalGeneration"],
                                  "task_specific_params": {
                                      "summarization": {"prefix": "summarize: "},
                                      "translation_en_to_de": {"prefix": "translate English to German: "}}},
                    hub_metadata={"pipeline_tag": "text2text-generation"})
    task.task_discovery = discover_model_candidates(task)["task_discovery"]
    return task


def two_profile_model():
    task = declared_model()
    task.model_config["task_specific_params"]["summarization"]["num_beams"] = 4
    task.task_discovery = discover_model_candidates(task)["task_discovery"]
    assert len(task.task_discovery["profiles"]) == 2
    return task


def options(**kwargs):
    return SimpleNamespace(task="", tasks="", task_family="", backend="", model_spec="", **kwargs)


def test_semantic_tasks_share_one_profile_without_prompt():
    result = select_tasks(declared_model(), options(), PreparationWorkflow())
    assert len(result.selected_tasks) == 1
    assert result.selected_tasks[0]["execution_profile"]["semantic_task_ids"] == [
        "summarization", "translation_en_to_de"]


def test_noninteractive_independent_profiles_require_explicit_selection():
    with pytest.raises(ValueError, match="--tasks"):
        select_tasks(two_profile_model(), options(), PreparationWorkflow())


def test_selecting_all_semantics_runs_one_profile():
    args = options()
    args.tasks = "all"
    result = select_tasks(declared_model(), args, PreparationWorkflow())
    assert len(result.selected_tasks) == 1
    assert len(result.selected_tasks[0]["semantic_tasks"]) == 2
    assert result.selected_tasks[0]["config"]["generation"]["max_new_tokens"] == 64


def test_explicit_semantic_subset_changes_input_provenance_not_matrix_count():
    args = options()
    args.tasks = "translation_en_to_de"
    chosen = select_tasks(declared_model(), args, PreparationWorkflow())
    assert len(chosen.selected_tasks) == 1
    assert chosen.selected_tasks[0]["execution_profile"]["representative_task_id"] == "translation_en_to_de"
    assert chosen.selected_tasks[0]["config"]["prefix"] == "translate English to German: "


def test_interactive_selection_waits_then_runs_only_chosen_profile(monkeypatch, capsys):
    task = two_profile_model()
    profile = next(item["profile_id"] for item in task.task_discovery["profiles"]
                   if "translation_en_to_de" in item["semantic_task_ids"])
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "id": 1, "action": "answer", "answers": {"tasks": [profile]}}) + "\n"))
    result = select_tasks(task, options(), PreparationWorkflow(interactive=True))
    assert [item["task_id"] for item in result.selected_tasks] == [profile]
    assert result.selected_tasks[0]["execution_profile"]["semantic_task_ids"] == ["translation_en_to_de"]
    assert '"kind": "multi"' in capsys.readouterr().out


def test_review_transfers_effective_profiles_without_localized_labels():
    task = two_profile_model()
    workflow = Mock(interactive=True)
    profile_ids = [item["profile_id"] for item in task.task_discovery["profiles"]]
    def answer(*args, **kwargs):
        questions = kwargs["questions"]
        if questions[0].get("kind") == "multi":
            return {"answers": {"tasks": profile_ids}}
        return {"answers": {field["path"]: field["value"] for field in
                            questions + kwargs.get("parameter_advanced", [])}}

    workflow.ask.side_effect = answer
    selected = select_tasks(task, options(), workflow)
    question = workflow.ask.call_args_list[0].kwargs["questions"][0]
    assert "profiles" in question, "The UI needs the complete effective signatures and declarations"
    assert question["profiles"] == selected.selected_tasks
    assert "labels" not in question
    assert "reason" not in question


def test_cancel_does_not_apply_or_cache_a_choice(monkeypatch):
    task = two_profile_model()
    monkeypatch.setattr("sys.stdin", io.StringIO('{"id": 1, "action": "cancel"}\n'))
    with pytest.raises(PreparationInterrupted, match="cancelled"):
        select_tasks(task, options(), PreparationWorkflow(interactive=True))
    assert task.selected_tasks == []


@pytest.mark.parametrize("ids", ["missing", "summarization,summarization", "all,summarization"])
def test_invalid_selection_is_rejected(ids):
    args = options()
    args.tasks = ids
    with pytest.raises(ValueError, match="task"):
        select_tasks(declared_model(), args, PreparationWorkflow())


def test_manual_workload_declaration_resolves_an_undeclared_task(tmp_path):
    task = declared_model()
    task.task_discovery = {"tasks": [], "conflicts": []}
    spec = tmp_path / "manual.json"
    spec.write_text(json.dumps({"schema_version": 1, "task_declaration": {
        "task_id": "sentiment-analysis", "params": {"prefix": "sst2 sentence: "}}}))
    args = options(workload_spec=str(spec))
    result = select_tasks(task, args, PreparationWorkflow())
    assert result.selected_tasks[0]["kind"] == "sentiment-analysis"
    assert result.selected_tasks[0]["config"]["prefix"] == "sst2 sentence: "
    assert result.selected_tasks[0]["evidence"] == ["workload_spec.task_declaration"]
    signature = result.selected_tasks[0]["execution_profile"]["signature"]
    assert signature["execution"]["operation"] == "generate"
    assert signature["execution"]["backend"] == "transformers_model"


def test_reusing_frozen_workload_keeps_manual_semantics(tmp_path):
    from acprof.task_workloads import parse_task, workload_manifest
    task = declared_model()
    manual = parse_task("generative-question-answering", {"prefix": "answer this: "},
                        encoder_decoder=True, source="user")
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(workload_manifest(manual)))
    args = options(workload_spec=str(path))
    args.tasks = "generative-question-answering"
    result = select_tasks(task, args, PreparationWorkflow())
    assert result.selected_tasks[0]["config"]["prefix"] == "answer this: "


def test_workload_override_reclassifies_effective_execution_group(tmp_path):
    task = two_profile_model()
    spec = tmp_path / "beam.json"
    spec.write_text(json.dumps({"schema_version": 1, "params": {"num_beams": 4}}))
    args = options(workload_spec=str(spec))
    result = select_tasks(task, args, PreparationWorkflow())
    assert len(result.selected_tasks) == 1
    group = result.selected_tasks[0]
    assert group["execution_profile"]["signature"]["generation"]["num_beams"] == 4
    assert set(group["execution_profile"]["semantic_task_ids"]) == {
        "summarization", "translation_en_to_de"}


def test_frozen_group_rejects_compute_override(tmp_path):
    from acprof.inference_groups import group_inference_tasks
    from acprof.task_workloads import workload_manifest
    task = declared_model()
    group = group_inference_tasks(task.task_discovery["tasks"], backend=task.runtime_backend)[0]
    spec = tmp_path / "frozen-group.json"
    frozen = workload_manifest(group)
    frozen["params"]["num_beams"] = 4
    spec.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="frozen inference profile"):
        select_tasks(task, options(workload_spec=str(spec)), PreparationWorkflow())


def test_historical_frozen_signature_keeps_its_original_identity(tmp_path):
    from acprof.inference_groups import group_inference_tasks
    from acprof.task_workloads import parse_task, task_fingerprint, workload_manifest
    task = declared_model()
    group = group_inference_tasks([parse_task("summarization", {}, encoder_decoder=True, source="historical")],
                                 backend="transformers_model")[0]
    group["execution_profile"]["signature"]["generation"].pop("repetition_penalty")
    group["config"]["generation"].pop("repetition_penalty")
    group["fingerprint"] = task_fingerprint(group)
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(workload_manifest(group)))
    chosen = select_tasks(task, options(workload_spec=str(path)), PreparationWorkflow())
    assert chosen.selected_tasks == [group]


def test_explicit_summary_without_metadata_uses_semantic_workload():
    task = declared_model()
    task.task_discovery = {"tasks": [], "conflicts": []}
    args = options()
    args.task = "summarization"
    result = select_tasks(task, args, PreparationWorkflow())
    assert result.selected_tasks[0]["config"]["prefix"] == "summarize: "
    assert args.task == result.pipeline_tag == "text2text-generation"


@pytest.mark.parametrize("interactive", [False, True])
def test_application_does_not_prepare_weights_or_collect_before_task_confirmation(tmp_path, monkeypatch, interactive):
    from dataclasses import replace

    from acprof.application.notifications import RunNotifications
    from acprof.application.run import RunApplication
    from acprof.experiment import RunConfig
    from acprof.host.runtimes.selection import RuntimeSelection
    workflow = PreparationWorkflow(interactive=interactive)
    if interactive:
        monkeypatch.setattr('sys.stdin', io.StringIO('{"id": 1, "action": "cancel"}\n'))
    preparation, matrix, notices = Mock(), Mock(), Mock(spec=RunNotifications)
    app = RunApplication(workflow_factory=lambda _: workflow, runtime_preparer=preparation,
                         matrix_runner=matrix, notifications=notices, environment_setup=Mock())
    config = replace(RunConfig.smoke('fixture/model'), output_dir=str(tmp_path))
    with patch('acprof.application.run.require_collection_host'), patch(
            'acprof.host.runtimes.selection.select_runtime',
            return_value=RuntimeSelection('auto', 'docker', 'test')), pytest.raises(
                PreparationInterrupted if interactive else ValueError):
        app.run(config, prepared_task=two_profile_model())
    preparation.assert_not_called()
    matrix.assert_not_called()
    notices.activate.assert_not_called()
    assert list(tmp_path.iterdir()) == []
