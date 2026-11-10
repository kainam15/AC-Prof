"""Public run phases retain ownership across success, resume and task failures."""
import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.application import publication, run
from acprof.application.errors import CollectionFailed
from acprof.application.notifications import RunNotifications
from acprof.application.preparation import PreparedRuntime
from acprof.capabilities import CapabilityReport
from acprof.config import CSV_FIELDS
from acprof.experiments.configuration import RunConfig
from acprof.host import orchestrator, run_state
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import PlannedInputScales
from acprof.host.preparation.collection_workflow import PreparationWorkflow, RebuildEnvironment
from acprof.host.runtimes.selection import RuntimeSelection
from acprof.platform import Environment
from acprof.results.layout import ArtifactLayout
from acprof.tasks.groups import group_inference_tasks
from acprof.tasks.workloads import parse_task
from tests.support.platform_fixtures import native_policy


@pytest.fixture
def phases(tmp_path, monkeypatch, request):
    native_policy(request)
    fixture = SimpleNamespace(
        calls=[], faults={}, states=[], configured=[], resolved=[], selected=[],
        events=[], prepared_axes=[], rebuilds=0,
        task=TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline",
                      "transformers", "a" * 40, "fixture"),
    )
    fixture.config = replace(RunConfig.smoke(fixture.task.model_id), output_dir=str(tmp_path / "results"),
                             runtime="auto", prune_startup_oom=False)

    def visit(stage):
        fixture.calls.append(stage)
        fault = fixture.faults.get(stage)
        if isinstance(fault, list):
            fault = fault.pop(0) if fault else None
        if fault is not None:
            raise fault

    def configure(args):
        visit("configure")
        fixture.configured.append(args)

    def select(requested, **_kwargs):
        visit("select")
        fixture.selected.append(requested)
        return RuntimeSelection(requested, "docker", "fixture selection")

    def workflow_factory(_directory):
        workflow = PreparationWorkflow()

        def resolve(args, *, initial=None):
            visit("resolve")
            fixture.resolved.append(args)
            return initial if initial is not None else fixture.task

        monkeypatch.setattr(workflow, "resolve", resolve)
        return workflow

    def preflight():
        visit("preflight")
        return "v2"

    class ObservedState(run_state.RunState):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            fixture.states.append(self)
            visit("state:open")

        def finish(self, final_csv):
            super().finish(final_csv)
            visit("commit")

        def close(self, outcome="interrupted"):
            super().close(outcome)
            visit("state:close")

    def prepare(args, *, run_state, task_info, cpu_list, mem_list, **_kwargs):
        visit("prepare")
        fixture.prepared_axes.append((args.runtime, tuple(cpu_list), tuple(mem_list)))
        if fixture.rebuilds:
            fixture.rebuilds -= 1
            raise RebuildEnvironment()
        image = ImageInfo("sha256:" + "b" * 64)
        planned = PlannedInputScales([1.0], "fixture")
        run_state.layout.path("static_meta.json").write_text(json.dumps({
            **Environment("native_linux", "Linux", "fixture-kernel").metadata(),
            "cgroup_version": "v2", "runtime_kind": "docker",
        }))
        run_state.bind_runtime(task_info, image, planned, "", "")
        return PreparedRuntime(task_info, image, planned, "", "", CapabilityReport("basic"))

    def case(request, **_kwargs):
        path = ArtifactLayout.discover(request.output_dir).case(
            request.task_info.model_id, request.cpu, request.mem, request.gpu,
        ).csv
        row = dict.fromkeys(CSV_FIELDS, "nan")
        row.update(cpu_cores=str(request.cpu), mem_cap_gb=str(request.mem), gpu_mode=request.gpu,
                   input_scale="1", warmup="0", repeat_idx="0", status="ok", error="",
                   latency_app_s="0.1", throughput_samples_per_s="10",
                   container_cpu_util_avg_pct="0", container_mem_usage_avg_bytes="1024")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow(row)
        return str(path)

    def collect(matrix):
        visit("collect")
        return orchestrator.run_matrix(matrix)

    def reports(layout):
        visit("reports")
        assert run_state.load_run_state(layout.root)["status"] == "complete"
        with pytest.raises(run_state.RunStateError, match="测量锁"):
            with run_state.MeasurementLock():
                pytest.fail("reports must retain the measurement lock")
        with pytest.raises(run_state.RunStateError, match="结果目录锁"):
            with run_state.ResultDirectoryLock(layout.root):
                pytest.fail("reports must retain the result lock")

    def start_log(directory, _command):
        visit("log:start")
        return ("fixture", str(Path(directory) / "partial.log"), str(Path(directory) / "final.log"))

    def stop_log(_log):
        visit("log:stop")
        return True

    notices = Mock(spec=RunNotifications)
    notices.active.return_value = False
    notices.reset.side_effect = lambda: visit("notify:reset")
    notices.notify_started.side_effect = lambda: visit("notify:start")
    notices.record_completion.side_effect = lambda **_: visit("notify:complete")
    notices.record_termination.side_effect = lambda status, _detail: visit("notify:" + status)
    notices.deliver.side_effect = lambda _path: visit("notify:deliver")

    def event(name, *args, **kwargs):
        fixture.events.append((name, kwargs))
        if name == "run_finished":
            visit("finished")

    monkeypatch.setattr(run, "require_collection_host", lambda **_: visit("host"))
    monkeypatch.setattr("acprof.host.runtimes.selection.select_runtime", select)
    monkeypatch.setattr(run, "require_cgroup_prerequisites", preflight)
    monkeypatch.setattr("acprof.monitors.rapl_topology.discover_rapl_topology", lambda: {})
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"fixture": "fixed"})
    monkeypatch.setattr(orchestrator, "run_single_case", case)
    monkeypatch.setattr(run, "publish_run_reports", reports)
    monkeypatch.setattr(publication, "publish_run_reports", reports)
    monkeypatch.setattr(run, "emit_event", event)
    monkeypatch.setattr(publication, "emit_event", event)
    fixture.application = run.RunApplication(
        notifications=notices, workflow_factory=workflow_factory, state_factory=ObservedState,
        runtime_preparer=prepare, matrix_runner=collect, environment_setup=configure,
        terminal_start=start_log, terminal_stop=stop_log,
    )
    return fixture


def assert_released(fixture):
    assert fixture.application.run_state is None
    assert fixture.application.terminal_log is None
    with run_state.MeasurementLock():
        pass
    for state in fixture.states:
        with run_state.ResultDirectoryLock(state.directory):
            pass


def test_public_run_preserves_phase_and_publication_order(phases):
    outcome = phases.application.run(phases.config)
    assert outcome.status == "ok"
    assert outcome.completed_cases == 1
    assert Path(outcome.final_csv).is_file()
    assert phases.calls == [
        "notify:reset", "configure", "host", "select", "resolve", "notify:start", "preflight",
        "state:open", "log:start", "prepare", "collect", "commit", "reports", "notify:complete",
        "log:stop", "state:close", "notify:deliver", "notify:reset", "finished",
    ]
    assert phases.states[0].data["status"] == "complete"
    assert phases.events[-1] == ("run_finished", {"status": "ok"})
    assert phases.config.runtime == "auto"
    assert_released(phases)


@pytest.mark.parametrize(("cpus", "mems", "expected_cpu", "expected_mem"), [
    ("auto", "auto", (1, 2, 4, 8), (2, 4, 8, 16)),
    ("auto", "4", (1, 2, 4, 8), (4,)),
    ("2", "auto", (2,), (2, 4, 8, 16)),
])
def test_runtime_resolution_does_not_rewrite_the_validated_input(phases, cpus, mems, expected_cpu, expected_mem):
    config = replace(phases.config, cpus=cpus, mems=mems)
    phases.faults["collect"] = RuntimeError("stop before synthetic matrix")
    with pytest.raises(RuntimeError, match="stop before synthetic matrix"):
        phases.application.run(config)
    requested = phases.configured[0]
    resolved = phases.resolved[0]
    assert (requested.runtime, requested.cpus, requested.mems) == ("auto", cpus, mems)
    assert resolved is not requested
    assert resolved.runtime == "docker"
    assert phases.prepared_axes == [("docker", expected_cpu, expected_mem)]
    assert phases.states[0].data["options"]["cpus"] == ",".join(map(str, expected_cpu))
    assert phases.states[0].data["options"]["mems"] == ",".join(map(str, expected_mem))
    assert_released(phases)


def test_completed_resume_republishes_without_preparation_or_collection(phases):
    first = phases.application.run(phases.config)
    phases.calls.clear()
    second = phases.application.run(replace(phases.config, resume=True))
    assert second.status == "ok"
    assert second.final_csv == first.final_csv
    assert phases.selected == ["auto", "docker"]
    assert phases.calls == [
        "notify:reset", "configure", "host", "select", "resolve", "notify:start", "preflight",
        "state:open", "reports", "state:close", "notify:deliver", "notify:reset", "finished",
    ]
    assert_released(phases)


@pytest.mark.parametrize(("stage", "exception", "status"), [
    ("prepare", RuntimeError, "failed"), ("collect", RuntimeError, "failed"),
    ("collect", KeyboardInterrupt, "cancelled"),
])
def test_failed_public_run_skips_later_phases_and_releases_ownership(phases, stage, exception, status):
    failure = exception("fixture failure")
    phases.faults[stage] = failure
    with pytest.raises(exception, match="fixture failure") as caught:
        phases.application.run(phases.config)
    assert caught.value is failure
    assert "commit" not in phases.calls
    assert "reports" not in phases.calls
    assert "notify:complete" not in phases.calls
    if stage == "prepare":
        assert "collect" not in phases.calls
    assert phases.calls[-6:] == ["notify:" + status, "log:stop", "state:close", "notify:deliver", "notify:reset", "finished"]
    expected = "interrupted" if status == "cancelled" else "failed"
    assert run_state.load_run_state(phases.states[0].directory)["status"] == expected
    assert phases.events[-1] == ("run_finished", {"status": status})
    assert_released(phases)


def test_preflight_failure_never_creates_run_state_or_starts_preparation(phases):
    phases.faults["preflight"] = RuntimeError("host unavailable")
    with pytest.raises(RuntimeError, match="host unavailable"):
        phases.application.run(phases.config)
    assert not phases.states
    assert not phases.prepared_axes
    assert "log:start" not in phases.calls
    assert "collect" not in phases.calls
    assert not Path(phases.config.output_dir).exists()
    assert phases.events[-1] == ("run_finished", {"status": "failed"})
    assert_released(phases)


def test_report_failure_preserves_committed_result_for_resume(phases):
    phases.faults["reports"] = RuntimeError("report unavailable")
    with pytest.raises(RuntimeError, match="report unavailable"):
        phases.application.run(phases.config)
    state = phases.states[0]
    assert run_state.load_run_state(state.directory)["status"] == "complete"
    before = state.layout.result_csv.read_bytes()
    assert phases.calls[-5:] == ["log:stop", "state:close", "notify:deliver", "notify:reset", "finished"]
    assert_released(phases)
    phases.faults.clear()
    phases.calls.clear()
    assert phases.application.run(replace(phases.config, resume=True)).status == "ok"
    assert state.layout.result_csv.read_bytes() == before
    assert "prepare" not in phases.calls
    assert "collect" not in phases.calls
    assert_released(phases)


def test_terminal_cleanup_failure_still_releases_real_state_locks(phases):
    phases.faults["log:stop"] = OSError("terminal unavailable")
    with pytest.raises(OSError, match="terminal unavailable"):
        phases.application.run(phases.config)
    assert phases.calls.index("log:stop") < phases.calls.index("state:close")
    assert phases.calls.index("state:close") < phases.calls.index("notify:deliver")
    assert run_state.load_run_state(phases.states[0].directory)["status"] == "complete"
    assert phases.events[-1] == ("run_finished", {"status": "failed"})
    assert_released(phases)


def test_rebuild_repeats_only_preparation(phases):
    phases.rebuilds = 1
    assert phases.application.run(phases.config).status == "ok"
    assert phases.calls.count("prepare") == 2
    for stage in ("select", "resolve", "preflight", "collect", "commit"):
        assert phases.calls.count(stage) == 1
    assert_released(phases)


@pytest.mark.parametrize(("runtime", "resume", "recorded", "selected", "expected_request"), [
    ("auto", False, "process", "docker", "auto"),
    ("auto", True, "process", "process", "process"),
    ("docker", True, "process", "docker", "docker"),
    ("process", False, "docker", "process", "process"),
])
def test_runtime_stage_keeps_process_inheritance_and_recorded_resume_policy(
        monkeypatch, runtime, resume, recorded, selected, expected_request):
    from acprof.application.resources import resolve_runtime_options

    config = replace(RunConfig.smoke("fixture/model"), runtime=runtime, resume=resume,
                     cpus="auto", mems="auto", prune_startup_oom=True)
    requested = config.to_namespace()
    requested.marker = object()  # Environment setup may add unrelated execution metadata.
    before = vars(requested).copy()
    selection = Mock(return_value=RuntimeSelection(expected_request, selected, "fixture policy"))
    monkeypatch.setattr("acprof.host.runtimes.selection.select_runtime", selection)
    resolved = resolve_runtime_options(config, requested, recorded_runtime=recorded)
    selection.assert_called_once_with(expected_request, profiling_mode="basic")
    assert vars(requested) == before
    assert resolved.marker is requested.marker
    assert resolved is not requested
    assert resolved.runtime == selected
    assert (resolved.cpus, resolved.mems) == (("auto", "auto") if selected == "process" else ("1,2,4,8", "2,4,8,16"))
    assert resolved.prune_startup_oom is (selected == "docker")


def test_rejected_runtime_stage_leaves_requested_options_unchanged(monkeypatch):
    from acprof.application.resources import resolve_runtime_options

    config = replace(RunConfig.smoke("fixture/model"), runtime="auto", compute_profile_tool="torch")
    requested = config.to_namespace()
    before = vars(requested).copy()
    monkeypatch.setattr("acprof.host.runtimes.selection.select_runtime", lambda *_, **__: RuntimeSelection(
        "auto", "process", "fixture fallback",
    ))
    with pytest.raises(ValueError, match="cannot be downgraded"):
        resolve_runtime_options(config, requested)
    assert vars(requested) == before


@pytest.mark.parametrize("first_fails", [False, True])
def test_task_collection_cleans_each_child_before_starting_the_next(phases, first_fails):
    phases.task = TaskInfo("fixture/model", "text2text-generation", "nlp", "transformers_model",
                           "transformers", "a" * 40, "fixture", model_config={"is_encoder_decoder": True},
                           model_resolution={"loader": "auto_seq2seq_lm", "artifact_format": "transformers"})
    phases.task.selected_tasks = group_inference_tasks([
        parse_task("translation_en_to_de", {}, encoder_decoder=True, source="fixture"),
        parse_task("summarization", {"num_beams": 4}, encoder_decoder=True, source="fixture"),
    ], backend=phases.task.runtime_backend)
    if first_fails:
        phases.faults["prepare"] = [RuntimeError("first task failed"), None]
        with pytest.raises(CollectionFailed, match="Some tasks failed"):
            phases.application.run(phases.config)
    else:
        assert phases.application.run(phases.config).status == "ok"
    assert len(phases.states) == 2
    assert phases.states[0].directory != phases.states[1].directory
    opens = [i for i, value in enumerate(phases.calls) if value == "state:open"]
    closes = [i for i, value in enumerate(phases.calls) if value == "state:close"]
    assert opens[0] < closes[0] < opens[1] < closes[1]
    assert "notify:deliver" in phases.calls[closes[0]:opens[1]]
    manifests = list((Path(phases.config.output_dir) / "fixture--model/metadata/task_runs").glob("*.json"))
    assert len(manifests) == 1
    saved = json.loads(manifests[0].read_text())
    assert [task["status"] for task in saved["tasks"]] == (["failed", "ok"] if first_fails else ["ok", "ok"])
    assert phases.calls.count("finished") == 1
    assert phases.calls.count("collect") == (1 if first_fails else 2)
    assert_released(phases)
