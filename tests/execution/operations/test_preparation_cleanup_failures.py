"""Unconfirmed cleanup must stop the public application before measurement."""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.application import run
from acprof.experiments.configuration import RunConfig
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import PlannedInputScales
from acprof.host.preparation.collection_workflow import PreparationWorkflow
from acprof.host.run_state import ResultDirectoryLock, load_run_state
from acprof.host.runtimes.contracts import RuntimeCleanupError
from acprof.host.runtimes.selection import RuntimeSelection


@pytest.fixture
def prepared_application(tmp_path, monkeypatch):
    task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline",
                    "transformers", "a" * 40, "fixture")
    config = replace(RunConfig.smoke(task.model_id), runtime="docker", output_dir=str(tmp_path),
                     compute_profile_tool="torch", execution_profile_tool="massif")
    workflow = PreparationWorkflow()
    monkeypatch.setattr(workflow, "resolve", Mock(return_value=task))
    ask = Mock(side_effect=AssertionError("cleanup failure must not request a retry or rebuild"))
    monkeypatch.setattr(workflow, "ask", ask)
    runtime = Mock()
    runtime.prepare.return_value = ImageInfo(tag="sha256:" + "b" * 64)
    runtime.validate.return_value = {"status": "ok"}
    monkeypatch.setattr("acprof.host.runtimes.service.execution_runtime", Mock(return_value=runtime))
    monkeypatch.setattr("acprof.host.runtimes.selection.select_runtime", Mock(
        return_value=RuntimeSelection("docker", "docker", "fixture")))
    monkeypatch.setattr("acprof.application.run.require_collection_host", Mock())
    monkeypatch.setattr("acprof.host.images.runtime.configure_runtime_profile", Mock())
    monkeypatch.setattr("acprof.host.metadata.static.collect_static_meta", Mock(return_value=SimpleNamespace()))
    def unchanged_metadata(metadata, _values):
        return metadata

    monkeypatch.setattr("acprof.host.metadata.static.enrich_static_meta", unchanged_metadata)
    monkeypatch.setattr("acprof.host.metadata.static.enrich_static_meta_from_input_plan", unchanged_metadata)
    monkeypatch.setattr("acprof.host.metadata.static.enrich_static_meta_from_compute_plan", unchanged_metadata)
    monkeypatch.setattr("acprof.host.metadata.static.enrich_static_meta_from_execution_plan", unchanged_metadata)
    monkeypatch.setattr("acprof.host.metadata.static.write_static_meta_json", Mock())
    stages = {
        "image": runtime.prepare,
        "input": Mock(return_value=PlannedInputScales([1.0], "fixture", None)),
        "runtime": runtime.validate,
        "compute_profile": Mock(return_value=""),
        "execution_profile": Mock(return_value=""),
    }
    monkeypatch.setattr("acprof.host.input_plan.plan_input_scales", stages["input"])
    monkeypatch.setattr("acprof.host.profiling.compute.collect_compute_profile_plan", stages["compute_profile"])
    monkeypatch.setattr("acprof.host.profiling.execution.collect_execution_profile_plan", stages["execution_profile"])
    matrix_reached = RuntimeError("matrix boundary reached")
    matrix = Mock(side_effect=matrix_reached)
    application = run.RunApplication(
        workflow_factory=lambda _output: workflow, matrix_runner=matrix,
        environment_setup=Mock(), terminal_start=Mock(return_value=None), terminal_stop=Mock(),
    )
    monkeypatch.setattr(application, "_run_host_preflight", Mock(return_value=("v2", {}, {})))
    return SimpleNamespace(application=application, config=config, workflow=workflow, ask=ask,
                           stages=stages, matrix=matrix, matrix_reached=matrix_reached,
                           directory=config.result_dir(tmp_path))


@pytest.mark.parametrize("stage", ["image", "input", "runtime", "compute_profile", "execution_profile"])
@pytest.mark.parametrize("interactive", [False, True])
def test_public_application_stops_on_cleanup_failure(prepared_application, stage, interactive):
    fixture = prepared_application
    fixture.workflow.interactive = interactive
    failure = RuntimeCleanupError("owned runtime termination is unconfirmed")
    failure.run_error = RuntimeError("profiler command failed")
    failure.__cause__ = failure.run_error
    fixture.stages[stage].side_effect = failure

    with pytest.raises(RuntimeCleanupError) as caught:
        fixture.application.run(fixture.config)

    assert caught.value is failure
    assert caught.value.__cause__ is failure.run_error
    fixture.ask.assert_not_called()
    fixture.matrix.assert_not_called()
    stage_names = list(fixture.stages)
    for name in stage_names[stage_names.index(stage) + 1:]:
        fixture.stages[name].assert_not_called()
    state = load_run_state(fixture.directory)
    assert state["status"] == "failed"
    assert "runtime" not in state
    attempt = state["attempts"][-1]
    assert attempt["outcome"] == "failed"
    evidence = json.loads((fixture.directory / attempt["preparation_evidence"]).read_text())
    assert evidence["status"] == "failed"
    assert evidence["last_stage"] == stage
    assert evidence["last_stage_status"] == "failed"
    assert evidence["exception"] == {"type": "RuntimeCleanupError", "message": str(failure)}
    assert fixture.application.run_state is None
    with ResultDirectoryLock(fixture.directory):
        pass


@pytest.mark.parametrize("stage", ["compute_profile", "execution_profile"])
def test_ordinary_optional_profiler_failure_still_reaches_matrix(prepared_application, stage):
    fixture = prepared_application
    fixture.stages[stage].side_effect = RuntimeError("optional profiler unavailable")

    with pytest.raises(RuntimeError, match="matrix boundary reached") as caught:
        fixture.application.run(fixture.config)

    assert caught.value is fixture.matrix_reached
    fixture.matrix.assert_called_once()
    fixture.stages["execution_profile"].assert_called_once()
    assert "runtime" in load_run_state(fixture.directory)


@pytest.mark.parametrize("stage", ["resolution", "preflight", "interface"])
def test_interactive_workflow_does_not_retry_cleanup_failure(stage, monkeypatch):
    workflow = PreparationWorkflow(interactive=True)
    observer, ask = Mock(), Mock()
    workflow.observer = observer
    monkeypatch.setattr(workflow, "ask", ask)
    failure = RuntimeCleanupError("owned process termination is unconfirmed")
    operation = Mock(side_effect=[failure, "retried"])

    with pytest.raises(RuntimeCleanupError) as caught:
        workflow.run(stage, operation)

    assert caught.value is failure
    operation.assert_called_once()
    ask.assert_not_called()
    observer.assert_any_call(stage, "failed", {
        "exception_type": "RuntimeCleanupError", "exception_message": str(failure),
    })
