"""Reject malformed recovery state before restoring any disposable evidence."""
import copy
import json
import subprocess
import sys

import pytest

from acprof.host import run_state
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import PlannedInputScales
from acprof.host.measurement import window_recovery


def tree_bytes(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def read_entry(state, data, entry):
    if entry == "load":
        return run_state.load_run_state(state.directory)
    if entry == "constructor":
        return run_state.RunState(state.directory, state.data["options"], resume=True, project_dir=".")
    return run_state.validate_resume(state.directory, data, project_dir=".", repair=entry == "repair")


@pytest.fixture
def recovery_state(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"fixture": "fixed"})
    options = {"cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 1}
    state = run_state.RunState(tmp_path / "experiment", options, resume=False, project_dir=".")
    state.bind_runtime(
        TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline",
                 "transformers", "a" * 40, "manual"),
        ImageInfo("sha256:" + "b" * 64),
        PlannedInputScales([64.0], "manual"), "", "",
    )
    state.prepare_case("unused.csv", 1, 4, "off")
    state.close()
    return state


BAD_FIELDS = [
    (("schema_version",), True, "主实验状态格式"),
    (("schema_version",), 1.0, "主实验状态格式"),
    (("layout_version",), 2.0, "产物布局"),
    (("run_id",), True, "run_id"),
    (("run_id",), "", "run_id"),
    (("run_id",), {}, "run_id"),
    (("status",), [], "status"),
    (("status",), None, "status"),
    (("status",), "unrecognized", "status"),
    (("attempts",), {}, "attempts"),
    (("attempts",), [None], "attempts"),
    (("attempts", 0, "attempt_id"), 42, "attempts"),
    (("attempts", 0, "resume"), "true", "attempts"),
    (("attempts", 0, "pid"), True, "attempts"),
    (("cases",), [], "case"),
    (("cases", ".acprof/work/cases/1c_4g_off/result.csv", "resources"), [1, 4], "resources"),
    (("cases", ".acprof/work/cases/1c_4g_off/result.csv", "resources"), [True, 4, "off"], "resources"),
    (("cases", ".acprof/work/cases/1c_4g_off/result.csv", "case_attempt_id"), [], "case_attempt_id"),
    (("cases", ".acprof/work/cases/1c_4g_off/result.csv", "previous_attempts"), {}, "previous_attempts"),
    (("artifacts",), [], "产物"),
    (("artifacts",), {"metadata/input_scale_plan.json": 42}, "input_scale_plan"),
    (("runtime",), [], "runtime"),
    (("runtime",), {}, "runtime"),
    (("runtime", "task"), [], "runtime.task"),
    (("runtime", "image"), [], "runtime.image"),
    (("runtime", "planned"), [], "runtime.planned"),
    (("runtime", "planned", "scales"), {}, "scales"),
    (("runtime", "planned", "scales"), [True], "scales"),
    (("runtime", "planned", "scales"), [float("inf")], "scales|非有限数值"),
    (("runtime", "kind"), {}, "runtime.kind"),
    (("runtime", "compute_plan"), [], "runtime.compute_plan"),
    pytest.param(("runtime", "task", "unknown"), {}, "runtime.task", id="task-unknown-field"),
    pytest.param(("runtime", "image", "unknown"), {}, "runtime.image", id="image-unknown-field"),
    pytest.param(("runtime", "planned", "unknown"), {}, "runtime.planned", id="planned-unknown-field"),
    pytest.param(("runtime", "planned", "scales"), [10**400], "scales", id="scale-overflow"),
    pytest.param(("runtime", "task", "selected_tasks"), {}, "selected_tasks", id="review-selected-container"),
    pytest.param(("runtime", "task", "selected_tasks"), [None], "selected_tasks", id="review-selected-member"),
    pytest.param(("runtime", "task", "selected_tasks"), [{"execution_profile": []}],
                 "execution_profile", id="review-profile-container"),
    pytest.param(("runtime", "task", "selected_tasks"), [{"execution_profile": {"signature": []}}],
                 "signature", id="review-signature-container"),
    pytest.param(("runtime", "task", "selected_tasks"),
                 [{"execution_profile": {"signature": {"execution": {}}}}],
                 "selected_tasks", id="review-execution-fields"),
    pytest.param(("cases",), {"wrong.csv": {"status": "running"}}, "wrong.csv", id="review-case-route"),
    pytest.param(("cases", ".acprof/work/cases/1c_4g_off/result.csv", "cleanup_process"),
                 True, "cleanup_process", id="review-cleanup-boolean"),
    pytest.param(("cases", ".acprof/work/cases/1c_4g_off/result.csv", "cleanup_process"),
                 [{}], "cleanup_process", id="review-cleanup-list"),
    pytest.param(("cases", ".acprof/work/cases/1c_4g_off/result.csv", "cleanup_process"),
                 {"owner": [], "workload": {}, "status_file": "x"},
                 "cleanup_process", id="review-cleanup-owner"),
    pytest.param(("cases", ".acprof/work/cases/1c_4g_off/result.csv", "cleanup_process"),
                 {"owner": {}, "workload": [1], "status_file": "x"},
                 "cleanup_process", id="review-cleanup-workload"),
    pytest.param(("cases", ".acprof/work/cases/1c_4g_off/result.csv", "cleanup_process"),
                 {"owner": {}, "workload": {"pid": True, "start_ticks": 1}, "status_file": "x"},
                 "cleanup_process", id="review-cleanup-pid"),
    pytest.param(("cases", ".acprof/work/cases/1c_4g_off/result.csv", "cleanup_process"),
                 {"owner": {}, "workload": {"pid": 1, "start_ticks": 1}, "status_file": []},
                 "cleanup_process", id="review-cleanup-status-file"),
]


def assert_malformed_state_rejected(
        recovery_state, monkeypatch, keys, bad, diagnostic, entry):
    state = recovery_state
    data = copy.deepcopy(state.data)
    target = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = bad
    state.path.write_text(json.dumps(data), encoding="utf-8")
    # A late validator would truncate this tail while opening ResultWriter.
    layer = state.layout.layer_csv("summary")
    layer.write_bytes(layer.read_bytes() + b"uncommitted-tail\n")
    before = tree_bytes(state.directory)
    repairs = []
    original = window_recovery.repair_result_projections

    def track_repair(*args, **kwargs):
        repairs.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(window_recovery, "repair_result_projections", track_repair)
    resumed = None
    try:
        with pytest.raises(run_state.RunStateError, match=diagnostic):
            resumed = read_entry(state, data, entry)
    finally:
        if isinstance(resumed, run_state.RunState):
            resumed.close()
    assert repairs == []
    assert tree_bytes(state.directory) == before
    with run_state.MeasurementLock(), run_state.ResultDirectoryLock(state.directory):
        pass


@pytest.mark.parametrize(("keys", "bad", "diagnostic"), BAD_FIELDS)
@pytest.mark.parametrize("entry", ["load", "preview", "repair", "constructor"])
def test_malformed_state_is_rejected_without_recovery_side_effects(
        recovery_state, monkeypatch, keys, bad, diagnostic, entry):
    assert_malformed_state_rejected(recovery_state, monkeypatch, keys, bad, diagnostic, entry)


@pytest.mark.parametrize("entry", ["load", "preview", "constructor"])
def test_missing_attempts_is_rejected_before_preparation_archive(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {})
    state = run_state.RunState(tmp_path / "experiment", {}, resume=False, project_dir=".")
    state.close("failed")
    data = json.loads(state.path.read_text())
    del data["attempts"]
    state.path.write_text(json.dumps(data))
    before = tree_bytes(state.directory)
    resumed = None
    try:
        with pytest.raises(run_state.RunStateError, match="attempts"):
            resumed = read_entry(state, data, entry)
    finally:
        if isinstance(resumed, run_state.RunState):
            resumed.close()
    assert tree_bytes(state.directory) == before


def test_valid_measurement_state_is_readable_and_resumable(recovery_state):
    state = recovery_state
    data = run_state.load_run_state(state.directory)
    before = tree_bytes(state.directory)
    assert run_state.validate_resume(state.directory, data, project_dir=".") == "measurement"
    assert tree_bytes(state.directory) == before
    resumed = run_state.RunState(state.directory, data["options"], resume=True, project_dir=".")
    try:
        assert resumed.run_id == data["run_id"]
        assert len(resumed.data["attempts"]) == 2
    finally:
        resumed.close()


@pytest.mark.parametrize("entry", ["preview", "repair", "constructor"])
def test_execution_identity_is_checked_before_resume_recovery(recovery_state, monkeypatch, entry):
    selected = {
        "task_id": "fixture", "interface": "fill-mask", "kind": "fill-mask", "config": {},
        "fingerprint": "0" * 64,
        "execution_profile": {"signature": {"backend": "transformers_pipeline", "execution": {}}},
    }
    assert_malformed_state_rejected(
        recovery_state, monkeypatch, ("runtime", "task", "selected_tasks"),
        [selected], "selected_tasks", entry,
    )


@pytest.mark.parametrize("selection", [[], [{}], [{"execution_profile": {"signature": {}}}]])
def test_legacy_task_selection_remains_readable(recovery_state, selection):
    state = recovery_state
    state.data["runtime"]["task"]["selected_tasks"] = selection
    state.save()
    before = tree_bytes(state.directory)
    data = run_state.load_run_state(state.directory)
    assert run_state.validate_resume(state.directory, data, project_dir=".") == "measurement"
    assert tree_bytes(state.directory) == before


def test_historical_minimal_process_cleanup_identity_remains_readable(recovery_state):
    state = recovery_state
    record = next(iter(state.data["cases"].values()))
    record["cleanup_process"] = {
        "owner": {"pid": 2147483647, "start_ticks": 1},
        "workload": {"pid": 2147483646, "start_ticks": 1},
        "status_file": str(state.layout.path("logs") / "owned.process.json"),
    }
    state.save()
    before = tree_bytes(state.directory)
    data = run_state.load_run_state(state.directory)
    assert run_state.validate_resume(state.directory, data, project_dir=".") == "measurement"
    assert tree_bytes(state.directory) == before


def bind_current_execution_identity(state):
    from acprof.tasks.groups import group_inference_tasks
    from acprof.tasks.interfaces import execution_path
    from acprof.tasks.workloads import parse_task, task_fingerprint

    task = TaskInfo(**state.data["runtime"]["task"])
    selected = parse_task("fill-mask", {}, encoder_decoder=False, source="fixture")
    selected["execution_path"] = execution_path(task, "fill-mask", backend=task.runtime_backend)
    selected["fingerprint"] = task_fingerprint(selected)
    state.data["runtime"]["task"]["selected_tasks"] = group_inference_tasks(
        [selected], backend=task.runtime_backend)


@pytest.mark.parametrize("repository_files", [("config.json",), ["config.json"]])
def test_current_execution_identity_remains_resumable(recovery_state, repository_files):
    state = recovery_state
    state.data["runtime"]["task"]["repository_files"] = repository_files
    bind_current_execution_identity(state)
    state.save()
    before = tree_bytes(state.directory)
    data = run_state.load_run_state(state.directory)
    for snapshot in (data, state.data):
        assert run_state.validate_resume(state.directory, snapshot, project_dir=".") == "measurement"
    assert tree_bytes(state.directory) == before


@pytest.mark.parametrize("bad", [True, None, [{}]])
@pytest.mark.parametrize("entry", ["load", "preview", "repair", "constructor"])
def test_repository_files_are_validated_before_execution_identity(
        recovery_state, monkeypatch, bad, entry):
    bind_current_execution_identity(recovery_state)
    assert_malformed_state_rejected(
        recovery_state, monkeypatch, ("runtime", "task", "repository_files"),
        bad, "runtime.task.repository_files", entry,
    )


def test_flat_case_routes_keep_the_existing_compatibility_branch(recovery_state, tmp_path):
    directory = tmp_path / "flat"
    directory.mkdir()
    data = copy.deepcopy(recovery_state.data)
    data["layout_version"] = 1
    data["cases"] = {"historical.csv": next(iter(data["cases"].values()))}
    (directory / "run_state.json").write_text(json.dumps(data))
    before = tree_bytes(directory)
    loaded = run_state.load_run_state(directory)
    assert run_state.validate_resume(directory, loaded, project_dir=".") == "measurement"
    assert tree_bytes(directory) == before

def test_typed_state_preserves_open_metadata_and_case_history(recovery_state):
    state = recovery_state
    case_name, record = next(iter(state.data["cases"].items()))
    state.data["created_at"] = {"legacy_timestamp": ["retained"]}
    state.data["future_evidence"] = {"path": "raw/uninterpreted"}
    record["started_at"] = {"legacy_timestamp": "retained"}
    record["sha256"] = {"not_complete": "not_interpreted"}
    record["vendor_evidence"] = {"opaque": [1, {"retained": True}]}
    state.save()
    loaded = run_state.load_run_state(state.directory)
    assert loaded["created_at"] == state.data["created_at"]
    assert loaded["future_evidence"] == state.data["future_evidence"]
    assert loaded["cases"][case_name]["sha256"] == record["sha256"]

    resumed = run_state.RunState(state.directory, state.data["options"], resume=True, project_dir=".")
    try:
        original = resumed.data["cases"][case_name]
        vendor_evidence = original["vendor_evidence"]
        resumed.prepare_case(case_name, 1, 4, "off")
        current = resumed.data["cases"][case_name]
        previous = current["previous_attempts"][-1]
        assert previous["vendor_evidence"] is vendor_evidence
        assert previous["started_at"] == record["started_at"]
        assert "previous_attempts" not in previous
        assert original["previous_attempts"] == []
        detached = resumed.case_record(str(state.directory / case_name))
        detached["status"] = "archived"
        assert current["status"] == "running"
    finally:
        resumed.close()


def test_state_validation_still_runs_with_python_optimization(recovery_state):
    state = recovery_state
    state.data["attempts"] = {}
    state.save()
    before = tree_bytes(state.directory)
    source = (
        "import sys\n"
        "from acprof.host.run_state import RunStateError, load_run_state\n"
        "try:\n"
        "    load_run_state(sys.argv[1])\n"
        "except RunStateError as error:\n"
        "    if 'attempts' not in str(error):\n"
        "        raise\n"
        "    print('malformed attempts rejected')\n"
        "else:\n"
        "    raise SystemExit('validation was bypassed')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", source, str(state.directory)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "malformed attempts rejected"
    assert tree_bytes(state.directory) == before

@pytest.mark.parametrize("entry", ["preview", "repair"])
@pytest.mark.parametrize(("keys", "diagnostic"), [
    ((1,), "字段名"),
    (("attempts", 0, 1), "attempts"),
    (("cases", ".acprof/work/cases/1c_4g_off/result.csv", 1), "cases"),
    (("cases", ".acprof/work/cases/1c_4g_off/result.csv", "previous_attempts", 0, 1), "previous_attempts"),
    (("runtime", 1), "runtime"),
    (("options", 1), "options"),
])
def test_typed_snapshot_field_names_are_validated(recovery_state, monkeypatch, keys, diagnostic, entry):
    record = next(iter(recovery_state.data["cases"].values()))
    previous = record.copy()
    previous.pop("previous_attempts", None)
    record["previous_attempts"] = [previous]
    assert_malformed_state_rejected(
        recovery_state, monkeypatch, keys, "non-string key", diagnostic, entry,
    )


def test_planned_guard_field_names_are_validated(recovery_state):
    before = tree_bytes(recovery_state.directory)
    with pytest.raises(run_state.RunStateError, match="runtime.planned"):
        run_state._validate_planned_snapshot(
            {"scales": [1.0], "source": "manual", 1: "non-string key"}, recovery_state.layout,
        )
    assert tree_bytes(recovery_state.directory) == before
