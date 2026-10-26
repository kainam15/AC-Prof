"""The preparation boundary must wait for semantic task choices before downloads."""
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.host.collection_workflow import PreparationInterrupted, PreparationWorkflow
from acprof.host.detect import TaskInfo
from acprof.host.task_selection import select_tasks
from acprof.model_candidates import discover_declared_tasks


def declared_model():
    task = TaskInfo("fixture/model", "text2text-generation", "nlp", "transformers_model",
                    "transformers", "a" * 40, "hub_api",
                    model_config={"is_encoder_decoder": True, "model_type": "t5",
                                  "architectures": ["T5ForConditionalGeneration"],
                                  "task_specific_params": {
                                      "summarization": {"prefix": "summarize: "},
                                      "translation_en_to_de": {"prefix": "translate English to German: "}}},
                    hub_metadata={"pipeline_tag": "text2text-generation"})
    task.task_discovery = discover_declared_tasks(task)
    return task


def options(**kwargs):
    return SimpleNamespace(task="", tasks="", task_family="", backend="", model_spec="", **kwargs)


def test_noninteractive_multiple_tasks_require_explicit_selection():
    with pytest.raises(ValueError, match="--tasks"):
        select_tasks(declared_model(), options(), PreparationWorkflow())


def test_selecting_all_preserves_separate_task_ids():
    args = options()
    args.tasks = "all"
    result = select_tasks(declared_model(), args, PreparationWorkflow())
    assert [item["task_id"] for item in result.selected_tasks] == ["summarization", "translation_en_to_de"]


def test_interactive_selection_waits_then_runs_only_chosen_task(monkeypatch, capsys):
    from acprof.task_parameters import parameter_review
    task = declared_model()
    selected = [item for item in task.task_discovery["tasks"] if item["task_id"] == "translation_en_to_de"]
    review = parameter_review(task, selected)
    defaults = {field["path"]: field["value"] for field in
                review["questions"] + review["parameter_advanced"]}
    replies = (json.dumps({"id": 1, "action": "answer",
                           "answers": {"tasks": ["translation_en_to_de"]}}) + "\n"
               + json.dumps({"id": 2, "action": "answer", "answers": defaults}) + "\n")
    monkeypatch.setattr("sys.stdin", io.StringIO(replies))
    result = select_tasks(task, options(), PreparationWorkflow(interactive=True))
    assert [item["task_id"] for item in result.selected_tasks] == ["translation_en_to_de"]
    assert '"kind": "multi"' in capsys.readouterr().out


def test_cancel_does_not_apply_or_cache_a_choice(monkeypatch):
    task = declared_model()
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
        app.run(config, prepared_task=declared_model())
    preparation.assert_not_called()
    matrix.assert_not_called()
    notices.activate.assert_not_called()
    assert list(tmp_path.iterdir()) == []
