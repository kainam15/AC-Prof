"""Public collection gates freeze intent before state, downloads and task runs."""
from __future__ import annotations

import copy
import io
import json
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.application import run, task_runs
from acprof.application.notifications import RunNotifications
from acprof.application.outcome import RunOutcome
from acprof.experiments.configuration import RunConfig
from acprof.host.preparation import inference_configuration as preparation
from acprof.host.preparation.collection_workflow import PreparationWorkflow
from acprof.host.preparation.task_selection import activate_task
from acprof.models.inference_configuration import configuration_identity
from acprof.results.layout import ArtifactLayout
from acprof.tasks.workloads import task_fingerprint
from tests.support.inference_configuration_fixture import frozen_task


def request_document(task):
    return {
        "schema_version": 1, "preset": "custom",
        "backend": {"mode": "explicit", "value": task.runtime_backend},
        "runtime_profile": {"mode": "explicit", "value": task.runtime_profile_id},
        "loading": {"off": {"dtype": {"mode": "explicit", "value": "FP32"}}},
        "tasks": {task.selected_tasks[0]["task_id"]: {
            "max_new_tokens": {"mode": "explicit", "value": 7},
        }},
    }


def frozen_for_options(task, config):
    value = copy.deepcopy(task.inference_configuration)
    args = config.to_namespace()
    value["workload"] = run.inference_workload(args, *run.resource_matrix(args))
    value["requests"] = request_document(task)
    value["identity_sha256"] = configuration_identity(value)
    task.inference_configuration = value
    return task


@pytest.fixture
def reconfigured_group_workload(tmp_path):
    from acprof.host.preparation.task_selection import select_tasks
    from acprof.models.candidates import apply_model_candidate, discover_model_candidates
    from acprof.models.configuration_resolution import resolve_inference_configuration
    from acprof.models.inference_configuration import freeze_inference_configuration
    from acprof.tasks.workloads import workload_manifest

    initial = frozen_task()
    initial.selected_tasks, initial.inference_configuration = [], {}
    initial.repository_metadata = {"config.json": initial.model_config}
    initial.repository_files = ("config.json", "model.safetensors")
    initial.model_resolution = apply_model_candidate(initial, discover_model_candidates(
        initial, override_tag=initial.pipeline_tag, override_backend=initial.runtime_backend,
    ))
    selector = SimpleNamespace(task=initial.pipeline_tag, tasks="", backend=initial.runtime_backend,
                               model_spec="", workload_spec="")
    initial = select_tasks(initial, selector, SimpleNamespace(interactive=False))
    original = copy.deepcopy(initial.selected_tasks[0])
    specification = workload_manifest(original)
    path = tmp_path / "original-workload.json"
    path.write_text(json.dumps(specification))
    original_bytes = path.read_bytes()

    # The original file is a valid selected profile before the new review.
    initial.selected_tasks = []
    selector.workload_spec = str(path)
    initial = select_tasks(initial, selector, SimpleNamespace(interactive=False))
    initial.runtime_profile_id = "nlp-cpu"
    config = replace(RunConfig.smoke(initial.model_id), output_dir=str(tmp_path),
                     runtime="docker", workload_spec=str(path))
    args = config.to_namespace()
    workload = run.inference_workload(args, *run.resource_matrix(args))
    updated, candidate = resolve_inference_configuration(
        initial, {"schema_version": 1, "tasks": {original["task_id"]: {
            "num_beams": {"mode": "explicit", "value": 2},
        }}}, devices=["off"], workload=workload,
    )
    updated.inference_configuration = freeze_inference_configuration(candidate, candidate["identity_sha256"])
    return SimpleNamespace(task=updated, config=config, original=original, specification=specification,
                           path=path, original_bytes=original_bytes)


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    from acprof.host.images import runtime
    from acprof.models import configuration_resolution

    template = frozen_task()
    initial = copy.deepcopy(template)
    initial.inference_configuration = {}
    config = replace(RunConfig.smoke(initial.model_id), output_dir=str(tmp_path / "results"), runtime="docker")
    workflow = SimpleNamespace(interactive=False, resolve=Mock(side_effect=lambda args, initial=None: initial or task))
    task = initial
    notices = Mock(spec=RunNotifications)
    state = Mock(side_effect=AssertionError("RunState is downstream of confirmation"))
    download = Mock(side_effect=AssertionError("runtime preparation is downstream of confirmation"))
    application = run.RunApplication(
        notifications=notices, workflow_factory=lambda _: workflow,
        state_factory=state, runtime_preparer=download, environment_setup=Mock(),
    )

    def validate(config):
        args = config.to_namespace()
        args.runtime_selection = {"reason": "fixture"}
        return args, [], False

    monkeypatch.setattr(application, "_validate_execution_options", validate)
    monkeypatch.setattr(run, "resolve_runtime_options", lambda config, args, **kwargs: args)
    events = []

    def configure(selected):
        events.append("platform")
        selected.runtime_profile_id = "nlp-cpu"
        return SimpleNamespace(profile_id="nlp-cpu")

    platform = Mock(side_effect=configure)
    monkeypatch.setattr(runtime, "configure_runtime_profile", platform)
    normalized = request_document(template)
    monkeypatch.setattr(configuration_resolution, "build_configuration_draft",
                        lambda *args, **kwargs: {"requests": kwargs.get("requests") or normalized})

    def resolve(selected, requests, *, workload, devices):
        events.append("resolve_configuration")
        updated = copy.deepcopy(selected)
        candidate = copy.deepcopy(template.inference_configuration)
        candidate.update(state="resolved", requests=requests, workload=copy.deepcopy(workload))
        candidate["identity_sha256"] = configuration_identity(candidate)
        return updated, candidate

    monkeypatch.setattr(configuration_resolution, "resolve_inference_configuration", resolve)
    monkeypatch.setattr(run, "emit_event", Mock())
    return SimpleNamespace(config=config, task=task, application=application, workflow=workflow,
                           notices=notices, state=state, download=download, events=events, platform=platform)


def test_unconfirmed_public_run_saves_review_without_formal_side_effects(boundary, monkeypatch):
    collection = Mock(side_effect=AssertionError("TaskCollection must wait for confirmation"))
    monkeypatch.setattr(task_runs, "TaskCollection", collection)

    outcome = boundary.application.run(boundary.config)

    assert outcome.status == "review_required"
    review = outcome.inference_configuration_review
    assert json.loads(Path(review["candidate_path"]).read_text())["state"] == "resolved"
    assert Path(review["requests_path"]).is_file()
    assert len(review["identity_sha256"]) == 64
    assert boundary.events == ["platform", "resolve_configuration"]
    collection.assert_not_called()
    boundary.state.assert_not_called()
    boundary.download.assert_not_called()
    boundary.notices.activate.assert_not_called()
    boundary.notices.notify_started.assert_not_called()
    boundary.notices.record_termination.assert_not_called()
    boundary.notices.deliver.assert_not_called()
    assert not list(Path(boundary.config.output_dir).rglob("run_state.json"))


def test_exact_confirmation_reaches_task_child_with_same_frozen_identity(boundary, monkeypatch):
    waiting = boundary.application.run(boundary.config)
    review = waiting.inference_configuration_review
    config = replace(boundary.config, inference_config=review["requests_path"],
                     confirm_inference=review["identity_sha256"])
    received = []

    def execute(child_config, child_task, **kwargs):
        received.append((child_config, child_task))
        return RunOutcome("ok", completed_cases=1)

    monkeypatch.setattr(boundary.application, "_execute_task_child", execute)
    outcome = boundary.application.run(config)

    assert outcome.status == "ok"
    assert len(received) == 1
    child_config, child = received[0]
    assert child.runtime_profile_id == "nlp-cpu"
    assert child.inference_configuration["identity_sha256"] == review["identity_sha256"]
    manifest = json.loads(Path(child_config.workload_spec).read_text())
    assert manifest["params"] == child.selected_tasks[0]["config"]["generation"]
    assert child.inference_configuration["workload"]["resources"] == {
        "cpus": [1], "mems": [4], "gpus": ["off"], "runtime": "docker",
        "cpuset_cpus": config.cpuset_cpus, "gpu_device": config.to_namespace().gpu_device,
    }
    boundary.state.assert_not_called()
    boundary.download.assert_not_called()


@pytest.mark.parametrize("collection", [False, True])
def test_changed_resume_requests_fail_before_collection_restore_or_state_repair(boundary, monkeypatch, tmp_path, collection):
    frozen = frozen_for_options(frozen_task(), boundary.config)
    document = {"schema_version": 1, "tasks": {
        frozen.selected_tasks[0]["task_id"]: {"max_new_tokens": {"mode": "explicit", "value": 99}},
    }}
    requests_file = tmp_path / "changed.json"
    requests_file.write_text(json.dumps(document))
    config = replace(boundary.config, resume=True, inference_config=str(requests_file))
    restore = Mock(side_effect=AssertionError("restore must follow request checks"))
    monkeypatch.setattr(task_runs.TaskCollection, "restore", restore)
    if collection:
        monkeypatch.setattr(task_runs.TaskCollection, "read_manifest", Mock(return_value=(
            tmp_path / "collection.json", {"task_snapshot": asdict(frozen)},
        )))
    else:
        layout = ArtifactLayout.for_new_run(config.result_dir(Path.cwd()))
        layout.initialize()
        marker = layout.path("run_state.json")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}")
        monkeypatch.setattr(run, "load_run_state", lambda path: {
            "runtime": {"task": asdict(frozen), "kind": "docker"},
        })

    with pytest.raises(ValueError, match="differs from the frozen request"):
        boundary.application.run(config)

    restore.assert_not_called()
    boundary.workflow.resolve.assert_not_called()
    boundary.state.assert_not_called()
    boundary.download.assert_not_called()


def test_frozen_child_reuses_confirmation_and_validates_task_overrides(boundary, monkeypatch):
    frozen = frozen_for_options(frozen_task(), boundary.config)
    selected = frozen.selected_tasks[0]
    selected["config"]["user_experiment_overrides"] = {"batch_size": 2, "input_scales": "64"}
    selected["fingerprint"] = task_fingerprint(selected)
    frozen.inference_configuration["tasks"][selected["task_id"]]["fingerprint"] = selected["fingerprint"]
    frozen.inference_configuration["identity_sha256"] = configuration_identity(frozen.inference_configuration)
    options = replace(boundary.config, batch_size=2, input_scales="64").to_namespace()
    prepare = Mock(side_effect=AssertionError("frozen tasks must not rerun Auto or confirmation"))
    monkeypatch.setattr(preparation, "prepare_inference_configuration", prepare)

    result = boundary.application._prepare_inference_configuration(
        frozen, options, boundary.workflow, axes=run.resource_matrix(options), task_child=True,
    )

    assert result is frozen
    prepare.assert_not_called()
    boundary.platform.assert_not_called()
    options.repeat += 1
    with pytest.raises(ValueError, match="measurement options differ"):
        boundary.application._prepare_inference_configuration(
            frozen, options, boundary.workflow, axes=run.resource_matrix(options), task_child=True,
        )


def test_conflicting_explicit_backend_request_is_rejected_before_review(boundary, tmp_path):
    document = request_document(frozen_task())
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(document))
    config = replace(boundary.config, backend="transformers_pipeline", inference_config=str(path))
    with pytest.raises(ValueError, match="request conflicts with explicit --backend"):
        boundary.application.run(config)
    boundary.platform.assert_not_called()
    boundary.state.assert_not_called()
    boundary.download.assert_not_called()


@pytest.mark.parametrize("name", ["backend", "runtime_profile"])
@pytest.mark.parametrize("value", [[], 1, None])
def test_invalid_route_request_leaf_is_rejected_before_execution(boundary, tmp_path, name, value):
    path = tmp_path / "requests.json"
    path.write_text(json.dumps({"schema_version": 1, name: value}))
    config = replace(boundary.config, inference_config=str(path), backend="transformers_model")
    if name == "backend":
        with pytest.raises(ValueError, match=rf"{name}: parameter request"):
            boundary.application.run(config)
    else:
        args = config.to_namespace()
        args.runtime_profile = "nlp-cpu"
        with pytest.raises(ValueError, match=rf"{name}: parameter request"):
            boundary.application._prepare_inference_configuration(
                boundary.task, args, boundary.workflow, axes=run.resource_matrix(args),
            )
    boundary.platform.assert_not_called()
    boundary.state.assert_not_called()
    boundary.download.assert_not_called()


def test_task_activation_preserves_profile_and_rejects_frozen_backend_override():
    task = frozen_task()
    child = activate_task(task, task.selected_tasks[0])
    assert child.runtime_profile_id == task.runtime_profile_id == "nlp-cpu"
    assert child.inference_configuration == task.inference_configuration
    with pytest.raises(ValueError, match="task backend conflicts"):
        activate_task(task, task.selected_tasks[0], backend="transformers_pipeline")


def test_collection_identity_uses_frozen_sha_and_not_confirmation_file_path(tmp_path):
    task = frozen_task()
    first = replace(RunConfig.smoke(task.model_id), inference_config="/a/requests.json",
                    confirm_inference="c" * 64, output_dir=str(tmp_path))
    second = replace(first, inference_config="", confirm_inference="", resume=True)
    identity = task_runs.configuration_identity(first, task)
    assert identity == task_runs.configuration_identity(second, task)
    assert identity["inference_configuration_sha256"] == task.inference_configuration["identity_sha256"]
    task.inference_configuration = {}
    legacy = task_runs.configuration_identity(first, task)
    assert "inference_configuration_sha256" not in legacy
    assert "inference_config" not in legacy
    assert "confirm_inference" not in legacy


def test_frozen_child_does_not_reapply_preconfirmation_workload_parameters(tmp_path):
    task = frozen_task()
    specification = {"schema_version": 1, "params": {"max_new_tokens": 99}}
    path = tmp_path / "source.json"
    path.write_text(json.dumps(specification))
    config = replace(RunConfig.smoke(task.model_id), output_dir=str(tmp_path), runtime="docker", workload_spec=str(path))
    task = frozen_for_options(task, config)
    collection = task_runs.TaskCollection(config, task)
    child = collection.child_config(task.selected_tasks[0])
    assert json.loads(Path(child.workload_spec).read_text())["params"]["max_new_tokens"] == 7
    path.write_text(json.dumps({"schema_version": 1, "params": {"max_new_tokens": 100}}))
    with pytest.raises(ValueError, match="changed after inference confirmation"):
        collection.child_config(task.selected_tasks[0])


def test_frozen_child_accepts_source_group_id_after_real_beam_review(reconfigured_group_workload):
    fixture = reconfigured_group_workload
    selected = fixture.task.selected_tasks[0]
    assert selected["task_id"] != fixture.original["task_id"]
    assert fixture.task.inference_configuration["tasks"][selected["task_id"]]["source_task_ids"] == [
        fixture.original["task_id"],
    ]
    child = task_runs.TaskCollection(fixture.config, fixture.task).child_config(selected)
    manifest = json.loads(Path(child.workload_spec).read_text())
    assert child.tasks == manifest["task_workload"]["task_id"] == selected["task_id"]
    assert manifest["params"]["num_beams"] == 2
    assert fixture.path.read_bytes() == fixture.original_bytes


def test_frozen_child_checks_source_digest_before_old_group_id(reconfigured_group_workload):
    fixture = reconfigured_group_workload
    fixture.specification["params"]["num_beams"] = 3
    fixture.path.write_text(json.dumps(fixture.specification))
    collection = task_runs.TaskCollection(fixture.config, fixture.task)
    with pytest.raises(ValueError, match="changed after inference confirmation"):
        collection.child_config(fixture.task.selected_tasks[0])


def test_frozen_child_rejects_another_tasks_source_id(reconfigured_group_workload):
    from acprof.models.evidence import content_digest
    from acprof.models.inference_configuration import validate_configuration_binding
    from acprof.tasks.groups import reidentify_inference_profile
    from acprof.tasks.workloads import workload_manifest

    fixture = reconfigured_group_workload
    other = copy.deepcopy(fixture.original)
    other["config"]["generation"]["num_beams"] = 3
    other = reidentify_inference_profile(other)
    specification = workload_manifest(other)
    fixture.path.write_text(json.dumps(specification))
    fixture.task.selected_tasks.append(other)
    frozen = fixture.task.inference_configuration
    frozen["tasks"][other["task_id"]] = {
        "interface": other["interface"], "fingerprint": other["fingerprint"],
        "generation": copy.deepcopy(other["config"]["generation"]),
        "source_task_ids": [other["task_id"]], "snapshot": copy.deepcopy(other),
    }
    frozen["workload"]["specification_sha256"] = content_digest(specification)
    frozen["identity_sha256"] = configuration_identity(frozen)
    validate_configuration_binding(frozen, fixture.task)
    collection = task_runs.TaskCollection(fixture.config, fixture.task)
    with pytest.raises(ValueError, match="conflicts with task identity"):
        collection.child_config(fixture.task.selected_tasks[0])


def test_legacy_child_still_requires_exact_group_id(reconfigured_group_workload):
    fixture = reconfigured_group_workload
    fixture.task.inference_configuration = {}
    collection = task_runs.TaskCollection(fixture.config, fixture.task)
    with pytest.raises(ValueError, match="conflicts with task identity"):
        collection.child_config(fixture.task.selected_tasks[0])


@pytest.mark.parametrize("purpose", ["inference_confirmation", "model_confirmation", "inference_configuration", ""])
def test_only_inference_confirmation_accepts_edit(monkeypatch, purpose):
    workflow = PreparationWorkflow(interactive=True)
    monkeypatch.setattr(workflow, "emit", Mock())
    monkeypatch.setattr("sys.stdin", io.StringIO('{"id":1,"action":"edit"}\n'))
    if purpose == "inference_confirmation":
        assert workflow.ask("resolution", "review", resolved=True, purpose=purpose)["action"] == "edit"
    else:
        with pytest.raises(ValueError, match="unexpected preparation action"):
            workflow.ask("resolution", "review", resolved=True, purpose=purpose)


def test_automatic_cli_keeps_review_status_without_auditing_missing_results(monkeypatch, tmp_path):
    from acprof.cli import auto, run as run_cli
    from acprof.host import automation, env_utils, run_state

    waiting = RunOutcome("review_required", inference_configuration_review={
        "identity_sha256": "d" * 64, "candidate_path": str(tmp_path / "candidate.json"),
        "requests_path": str(tmp_path / "requests.json"),
    })
    state = SimpleNamespace()
    real_finish = automation.AutomaticRun.finish

    class Automatic:
        def __init__(self, args):
            self.root, self.data = tmp_path, {"status": "running"}
            self.preparation_artifacts, self.collection_path = {}, None
            self.started, self.save = True, Mock()
            self.args = args
            state.instance = self

        def prepare(self):
            return frozen_task()

        def finish(self, error=None):
            assert error is None
            return real_finish(self, error)

    monkeypatch.setattr(automation, "AutomaticRun", Automatic)
    monkeypatch.setattr(env_utils, "bootstrap_project_env", lambda path: None)
    monkeypatch.setattr("acprof.application.download.apply_download_arguments", lambda args: None)
    monkeypatch.setattr(run_state, "MeasurementLock", lambda: nullcontext())
    monkeypatch.setattr(run_state, "ResultDirectoryLock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(run_cli, "main", Mock(return_value=waiting))

    assert auto.main(["fixture/frozen", "--output-dir", str(tmp_path), "--notify", "none"]) == 0
    assert state.instance.data["status"] == "review_required"
    assert state.instance.data["inference_configuration_review"] == waiting.inference_configuration_review
    state.instance.save.assert_called_once()


def test_automatic_review_with_explicit_error_finishes_as_failed(tmp_path):
    from acprof.host.automation import AutomaticRun

    automatic = SimpleNamespace(
        started=True, data={"status": "review_required"}, collection_path=None,
        layout=ArtifactLayout.for_new_run(tmp_path), save=Mock(),
    )

    assert AutomaticRun.finish(automatic, error="confirmation failed") == 2
    assert automatic.data["status"] == "failed"
    assert automatic.data["error"] == "confirmation failed"
    automatic.save.assert_called_once()


def test_confirmed_public_task_reaches_runtime_preparer_with_frozen_configuration(boundary, monkeypatch):
    from acprof.application.errors import CollectionFailed
    from acprof.platform import Environment

    boundary.task.model_resolution.update(loader="fixture", artifact_format="safetensors")
    waiting = boundary.application.run(boundary.config)
    review = waiting.inference_configuration_review
    config = replace(boundary.config, inference_config=review["requests_path"],
                     confirm_inference=review["identity_sha256"])
    states = []

    def create_state(output_dir, *args, **kwargs):
        state = SimpleNamespace(
            complete=False, layout=ArtifactLayout.for_new_run(output_dir), close=Mock(),
            preparation_evidence=lambda: SimpleNamespace(capture=lambda workflow: nullcontext()),
        )
        states.append(state)
        return state

    reached = Mock(side_effect=RuntimeError("runtime boundary reached without downloading"))
    boundary.application.state_factory = create_state
    boundary.application.runtime_preparer = reached
    boundary.application.terminal_start = lambda *args: None
    monkeypatch.setattr(boundary.application, "_run_host_preflight", lambda *args, **kwargs: ("v2", {}, {}))
    monkeypatch.setattr(run, "require_result_environment", lambda *args: None)
    monkeypatch.setattr(run, "require_result_cgroup_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr("acprof.platform.detect_environment", lambda: Environment("native_linux", "Linux", "fixture"))

    with pytest.raises(CollectionFailed):
        boundary.application.run(config)

    reached.assert_called_once()
    task = reached.call_args.kwargs["task_info"]
    assert task.runtime_profile_id == "nlp-cpu"
    assert task.inference_configuration["identity_sha256"] == review["identity_sha256"]
    assert len(task.selected_tasks) == 1
    assert reached.call_args.kwargs["gpu_list"] == ["off"]
    assert len(states) == 1
    states[0].close.assert_called_once()
