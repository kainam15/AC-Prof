"""Reject malformed recovery state before restoring any disposable evidence."""
import copy
import json

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
]


@pytest.mark.parametrize(("keys", "bad", "diagnostic"), BAD_FIELDS)
@pytest.mark.parametrize("entry", ["load", "preview", "repair", "constructor"])
def test_malformed_state_is_rejected_without_recovery_side_effects(
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
