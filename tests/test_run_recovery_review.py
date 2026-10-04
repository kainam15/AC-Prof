"""Recovery checks explain incompatibility and retain failed preparation evidence."""
import json

import pytest

from acprof.host import run_state


@pytest.fixture
def failed_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"source_sha256": "original", "python": "3.12"})
    directory = tmp_path / "demo--model"
    state = run_state.RunState(directory, {"model": "demo/model", "repeat": 5},
                               resume=False, project_dir=tmp_path)
    (directory / "metadata/interface_validation.json").write_text('{"status":"cancelled"}')
    (directory / "logs").mkdir()
    (directory / "logs/interface_validation.log").write_text("original interrupted probe\n")
    state.close("failed")
    return directory


def test_resume_explains_each_changed_identity_field(failed_preparation, tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"source_sha256": "changed", "python": "3.12"})
    original = (failed_preparation / ".acprof/run_state.json").read_bytes()
    with pytest.raises(run_state.RunStateError) as caught:
        run_state.RunState(failed_preparation, {"model": "demo/model", "repeat": 7},
                           resume=True, project_dir=tmp_path)
    assert "repeat" in str(caught.value)
    assert "源码" in str(caught.value)
    assert (failed_preparation / ".acprof/run_state.json").read_bytes() == original


def test_retry_preparation_archives_logs_and_state_before_overwriting(failed_preparation, tmp_path):
    original = (failed_preparation / ".acprof/run_state.json").read_bytes()
    resumed = run_state.RunState(failed_preparation, {"model": "demo/model", "repeat": 5},
                                 resume=True, project_dir=tmp_path)
    try:
        backup = failed_preparation / resumed.data["attempts"][-1]["preparation_backup"]
        (failed_preparation / "logs/interface_validation.log").write_text("new probe\n")
        assert (backup / "logs/interface_validation.log").read_text() == "original interrupted probe\n"
        assert (backup / "metadata/interface_validation.json").read_text() == '{"status":"cancelled"}'
        assert (backup / ".acprof/run_state.json").read_bytes() == original
        assert resumed.data["run_id"] == json.loads(original)["run_id"]
        assert len(resumed.data["attempts"]) == 2
    finally:
        resumed.close()


@pytest.mark.parametrize("evidence", ["case-state", "case-file", "result-csv"])
def test_missing_runtime_cannot_reclassify_measurement_as_preparation(failed_preparation, tmp_path, evidence):
    path = failed_preparation / ".acprof/run_state.json"
    if evidence == "case-state":
        data = json.loads(path.read_text())
        data["cases"] = {"case.csv": {"status": "running"}}
        path.write_text(json.dumps(data))
    else:
        artifact = failed_preparation / ("result_all.csv" if evidence == "result-csv" else ".acprof/work/cases/1c_4g_off/result.csv")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("partial measurement\n")
    original = path.read_bytes()
    with pytest.raises(run_state.RunStateError, match="测量"):
        resumed = run_state.RunState(failed_preparation, {"model": "demo/model", "repeat": 5},
                                     resume=True, project_dir=tmp_path)
        resumed.close()
    assert path.read_bytes() == original


def test_preview_is_read_only_and_detects_artifact_changes(failed_preparation, tmp_path):
    path = failed_preparation / ".acprof/run_state.json"
    original = path.read_bytes()
    state = run_state.load_run_state(failed_preparation)
    assert run_state.validate_resume(failed_preparation, state, project_dir=tmp_path) == "preparation"
    assert path.read_bytes() == original
    assert len(state["attempts"]) == 1
    state["artifacts"] = {"metadata/interface_validation.json": "not-the-digest"}
    with pytest.raises(run_state.RunStateError, match="interface_validation.json"):
        run_state.validate_resume(failed_preparation, state, project_dir=tmp_path)


def test_failed_archive_leaves_the_previous_attempt_intact(failed_preparation, tmp_path, monkeypatch):
    original = (failed_preparation / ".acprof/run_state.json").read_bytes()
    def disk_full(*_args, **_kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(run_state.shutil, "copy2", disk_full)
    with pytest.raises(OSError, match="disk full"):
        run_state.RunState(failed_preparation, {"model": "demo/model", "repeat": 5},
                           resume=True, project_dir=tmp_path)
    assert (failed_preparation / ".acprof/run_state.json").read_bytes() == original
    assert (failed_preparation / "logs/interface_validation.log").read_text() == "original interrupted probe\n"
