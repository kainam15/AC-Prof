"""Identity rejection must precede every disposable projection recovery write."""
from __future__ import annotations

import json

import pytest

from acprof.host import run_state
from acprof.result_store import protocol
from acprof.result_store import writer as writer_module
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import ResultValidationError
from acprof.results.layers import read_result_layers
from tests.results.storage.test_incremental_results import receipt
from tests.support.result_store_fixtures import assert_writer_lease_available


def files(root):
    return {path.relative_to(root): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def interrupted_store(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "MeasurementLock",
                        lambda: run_state.ResultDirectoryLock(tmp_path / "measurement"))
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"fixture": "fixed"})
    state = run_state.RunState(tmp_path / "experiment", {}, resume=False, project_dir=".")
    state.close("failed")
    layout = state.layout
    first = receipt(layout.root)
    with ResultWriter(layout.root) as writer:
        def fail_after_sync(*args, **kwargs):
            assert layout.layer_csv("summary").stat().st_size > (
                writer.manifest["layers"]["summary"]["committed_bytes"])
            assert layout.result_store_path("pending.json").is_file()
            raise OSError("fixture failed after synchronized projection writes")
        with monkeypatch.context() as fault:
            fault.setattr(writer_module, "save_commit", fail_after_sync)
            with pytest.raises(OSError, match="after synchronized"):
                writer.append_window(first)
    assert read_result_layers(layout.root)[1] == []
    return layout, state.data, first


@pytest.mark.parametrize("entry", ["readonly", "repair", "constructor", "writer"])
def test_wrong_identity_preserves_pending_and_all_evidence(interrupted_store, entry):
    layout, data, _ = interrupted_store
    wrong = {**data, "run_id": "another-run"}
    if entry == "constructor":
        layout.path("run_state.json").write_text(json.dumps(wrong))
    before = files(layout.root)
    with pytest.raises(ValueError, match="another run") as caught:
        if entry == "writer":
            with ResultWriter(layout.root, expected_run_id=wrong["run_id"]):
                pytest.fail("wrong identity acquired a recovered writer")
        elif entry == "constructor":
            resumed = run_state.RunState(layout.root, {}, resume=True, project_dir=".")
            resumed.close()
        else:
            run_state.validate_resume(layout.root, wrong, project_dir=".", repair=entry == "repair")
    assert files(layout.root) == before
    assert isinstance(caught.value, (ResultValidationError, run_state.RunStateError))
    assert_writer_lease_available(layout)
    with protocol.publication_lock(layout, exclusive=True, blocking=False):
        pass


def test_correct_identity_recovers_real_commit_failure_and_continues(interrupted_store):
    layout, data, first = interrupted_store
    before = files(layout.root)
    assert run_state.validate_resume(layout.root, data, project_dir=".") == "preparation"
    assert files(layout.root) == before
    assert run_state.validate_resume(layout.root, data, project_dir=".", repair=True) == "preparation"
    assert not layout.result_store_path("pending.json").exists()
    assert read_result_layers(layout.root)[1] == []
    original_raw = files(layout.root / "raw")
    with ResultWriter(layout.root, expected_run_id=data["run_id"]) as writer:
        assert writer.append_window(first)
        assert not writer.append_window(first)
        second = receipt(layout.root, 1)
        assert writer.append_window(second)
    assert [row["repeat_idx"] for row in read_result_layers(layout.root)[1]] == ["0", "1"]
    assert all((layout.root / "raw" / path).read_bytes() == raw for path, raw in original_raw.items())
    assert_writer_lease_available(layout)


def test_writer_checks_identity_while_holding_both_locks(interrupted_store, monkeypatch):
    import fcntl

    from acprof.result_store import recovery

    layout, data, _ = interrupted_store
    original = protocol.validate_run_identity
    observed = []

    def checked(manifest, expected_run_id):
        for name in ("writer.lock", "lock"):
            with layout.result_store_path(name).open("rb") as stream:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed.append(expected_run_id)
        original(manifest, expected_run_id)

    monkeypatch.setattr(recovery, "validate_run_identity", checked)
    with ResultWriter(layout.root, expected_run_id=data["run_id"]):
        pass
    assert observed == [data["run_id"]]
