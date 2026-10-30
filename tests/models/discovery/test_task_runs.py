"""Independent task runs retain failures and cannot resume a different workload."""
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from acprof.application.notifications import RunNotifications
from acprof.application.run import RunApplication, RunOutcome
from acprof.application.task_runs import TaskCollection
from acprof.experiment import RunConfig
from acprof.host.detect import TaskInfo
from acprof.host.runtimes.selection import RuntimeSelection
from acprof.tasks.groups import group_inference_tasks
from acprof.tasks.workloads import parse_task
from tests.support.task_selection_fixtures import t5_task


@pytest.mark.parametrize("override", [False, True])
def test_real_t5_mixed_selections_execute_once_per_effective_signature(tmp_path, override):
    import json
    from types import SimpleNamespace

    from acprof.host.preparation.collection_workflow import PreparationWorkflow
    from acprof.host.preparation.task_selection import select_tasks
    from acprof.tasks.presentation import build_task_selection

    task = t5_task()
    spec = tmp_path / "override.json"
    spec.write_text(json.dumps({"schema_version": 1, "params": {"no_repeat_ngram_size": 0, "length_penalty": 1.0}}))
    # Include both group and semantic IDs; overlapping choices must not repeat a matrix.
    choices = ",".join([item["profile_id"] for item in task.task_discovery["profiles"]] +
                       [item["task_id"] for item in task.task_discovery["tasks"]])
    args = SimpleNamespace(task="", tasks="all" if override else choices, backend="",
                           workload_spec=str(spec) if override else "")
    selected = select_tasks(task, args, PreparationWorkflow())
    view = build_task_selection(selected.selected_tasks)
    collection = TaskCollection(replace(RunConfig.smoke(task.model_id), output_dir=str(tmp_path)), selected)
    execute = Mock(return_value=RunOutcome("ok"))
    assert collection.run(execute).status == "ok"
    assert execute.call_count == len(view.groups) == (1 if override else 2)
    assert sum(len(group.profile["semantic_tasks"]) for group in view.groups) == 4
    assert {call.args[1].selected_tasks[0]["task_id"] for call in execute.call_args_list} == {
        group.task_id for group in view.groups}


def selected_model():
    task = TaskInfo("fixture/model", "text2text-generation", "nlp", "transformers_model",
                    "transformers", "b" * 40, "hub_api", model_config={"is_encoder_decoder": True})
    task.selected_tasks = group_inference_tasks([
        parse_task("translation_en_to_de", {}, encoder_decoder=True, source="fixture"),
        parse_task("summarization", {"num_beams": 4}, encoder_decoder=True, source="fixture"),
    ], backend=task.runtime_backend)
    return task


def test_task_failure_does_not_skip_the_next_task_and_results_are_separate(tmp_path):
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path))
    collection = TaskCollection(config, selected_model())
    seen = []

    def execute(config, task):
        seen.append((config, task))
        if len(seen) == 1:
            raise RuntimeError("Smoke failed")
        return RunOutcome("ok", str(config.result_csv(tmp_path)), 1)

    outcome = collection.run(execute)
    assert outcome.status == "failed"
    assert len(seen) == 2
    assert seen[0][0].result_dir(tmp_path) != seen[1][0].result_dir(tmp_path)
    assert {part[1].selected_tasks[0]["execution_profile"]["signature"]["generation"]["num_beams"]
            for part in seen} == {1, 4}
    assert collection.manifest["tasks"][0]["status"] == "failed"
    assert collection.manifest["tasks"][1]["status"] == "ok"


def test_related_semantic_tasks_execute_one_matrix_with_frozen_provenance(tmp_path):
    task = selected_model()
    original = [parse_task(name, {}, encoder_decoder=True, source="fixture") for name in
                ("translation_en_to_de", "summarization")]
    task.selected_tasks = group_inference_tasks(original, backend=task.runtime_backend)
    assert len(task.selected_tasks) == 1
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path))
    collection = TaskCollection(config, task)
    executed = []
    result = collection.run(lambda child, resolved: executed.append((child, resolved)) or RunOutcome("ok"))
    assert result.status == "ok"
    assert len(executed) == 1
    assert len(collection.manifest["task_snapshot"]["selected_tasks"][0]["semantic_tasks"]) == 2
    assert executed[0][0].tasks.startswith("text2text-generation-")


def test_resume_reuses_frozen_task_selection_and_workload(tmp_path):
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path))
    first = TaskCollection(config, selected_model())
    recorded = []
    first.run(lambda config, task: recorded.append(config) or RunOutcome("ok"))
    restored = TaskCollection.restore(replace(config, resume=True))
    assert restored.task.model_revision == "b" * 40
    assert restored.task.selected_tasks == first.task.selected_tasks
    replay = []
    restored.run(lambda config, task: replay.append(config) or RunOutcome("ok"))
    assert [config.workload_spec for config in recorded] == [config.workload_spec for config in replay]


def test_resume_restricts_explicit_semantic_subset(tmp_path):
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path),
                     tasks="translation_en_to_de,summarization")
    task = selected_model()
    task.selected_tasks = group_inference_tasks([
        parse_task("translation_en_to_de", {}, encoder_decoder=True, source="fixture"),
        parse_task("summarization", {}, encoder_decoder=True, source="fixture"),
    ], backend=task.runtime_backend)
    collection = TaskCollection(config, task)
    collection.run(lambda config, task: RunOutcome("ok"))
    assert TaskCollection.restore(replace(config, resume=True)).task.selected_tasks == task.selected_tasks
    with pytest.raises(ValueError, match="selection changed"):
        TaskCollection.restore(replace(config, resume=True, tasks="translation_en_to_de"))


def test_resume_accepts_original_mixed_profile_and_semantic_selection(tmp_path):
    task = selected_model()
    translation = next(item for item in task.selected_tasks
                       if item["execution_profile"]["semantic_task_ids"] == ["translation_en_to_de"])
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path),
                     tasks=f"{translation['task_id']},summarization")
    collection = TaskCollection(config, task)
    collection.run(lambda config, task: RunOutcome("ok"))
    restored = TaskCollection.restore(replace(config, resume=True))
    assert restored.task.selected_tasks == task.selected_tasks


def test_resume_rejects_changed_task_options(tmp_path):
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path))
    collection = TaskCollection(config, selected_model())
    collection.run(lambda config, task: RunOutcome("ok"))
    with pytest.raises(ValueError, match="configuration"):
        TaskCollection.restore(replace(config, resume=True, batch_size=2))


def test_resume_after_interrupt_prepares_tasks_that_never_started(tmp_path):
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path))
    first = TaskCollection(config, selected_model())
    with pytest.raises(KeyboardInterrupt):
        first.run(lambda config, task: (_ for _ in ()).throw(KeyboardInterrupt()))
    restored = TaskCollection.restore(replace(config, resume=True))
    seen = []
    result = restored.run(lambda config, task: seen.append(task.selected_tasks[0]['task_id']) or RunOutcome('ok'))
    assert result.status == 'ok'
    assert seen == [item['task_id'] for item in first.task.selected_tasks]


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_task_cancellation_keeps_interrupted_state_and_stops_collection(tmp_path, cleanup_fails):
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path))
    workflow = Mock()
    workflow.resolve.return_value = selected_model()
    application = RunApplication(notifications=Mock(spec=RunNotifications),
                                 workflow_factory=lambda _: workflow, environment_setup=Mock())
    state = Mock()
    if cleanup_fails:
        state.close.side_effect = OSError("state save failed")
    cancelled = KeyboardInterrupt("cancelled")

    def run_collection(collection, execute):
        application.run_state = state
        with patch.object(application, "_execute", side_effect=cancelled) as child:
            with pytest.raises(KeyboardInterrupt) as caught:
                execute(config, collection.task)
            assert caught.value is cancelled
            child.assert_called_once()
        raise caught.value

    with patch("acprof.application.run.require_collection_host"), patch(
        "acprof.host.runtimes.selection.select_runtime",
        return_value=RuntimeSelection("auto", "docker", "fixture"),
    ), patch.object(TaskCollection, "run", run_collection), pytest.raises(KeyboardInterrupt):
        application.run(config)

    state.close.assert_called_once_with("interrupted")
    if cleanup_fails:
        assert any("OSError" in note for note in cancelled.__notes__)


def test_resume_keeps_requested_alias_and_resolved_model_identity(tmp_path):
    config = replace(RunConfig.smoke("model-alias"), output_dir=str(tmp_path))
    collection = TaskCollection(config, selected_model())
    collection.run(lambda config, task: RunOutcome('ok'))
    assert TaskCollection.restore(replace(config, resume=True)).task.model_id == 'fixture/model'


def test_user_length_override_replaces_inherited_generation_length(tmp_path):
    from acprof.artifacts import atomic_write_json, read_json_object
    task = selected_model()
    spec = tmp_path / 'override.json'
    atomic_write_json(spec, {'schema_version': 1, 'params': {'max_length': 20}})
    config = replace(RunConfig.smoke('fixture/model'), output_dir=str(tmp_path), workload_spec=str(spec))
    collection = TaskCollection(config, task)
    child = collection.child_config(task.selected_tasks[0])
    params = read_json_object(child.workload_spec)['params']
    assert params['max_length'] == 20
    assert 'max_new_tokens' not in params


def test_two_collections_never_overwrite_previous_task_results(tmp_path):
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path))
    first = TaskCollection(config, selected_model())
    second = TaskCollection(config, selected_model())
    assert first.run_id != second.run_id


@pytest.mark.parametrize("task_run_id", ["../elsewhere", "x/y", "/tmp/escape"])
def test_task_run_path_cannot_escape_result_root(task_run_id):
    with pytest.raises(ValueError, match="task_run_id"):
        RunConfig(model="fixture/model", tasks="summarization", task_run_id=task_run_id).validate()
