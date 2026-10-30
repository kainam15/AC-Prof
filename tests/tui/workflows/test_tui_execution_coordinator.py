"""Execution ownership, stale callbacks and measurement boundary invariants."""

from unittest.mock import Mock

import pytest

from acprof.tui.commands import PendingLaunch
from acprof.tui.run.execution import ExecutionCoordinator
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.results import RunResult
from acprof.tui.run.stages import Stage


def pending(kind: str = "run") -> PendingLaunch:
    return PendingLaunch(command=("acprof", kind), kind=kind)


def test_coordinator_owns_a_single_attempt_and_rejects_parallel_start():
    owner = ExecutionCoordinator()
    first = owner.begin(pending())
    assert owner.accepts(first)
    assert owner.session.kind == "run"
    with pytest.raises(RuntimeError, match="already owned"):
        owner.begin(pending("probe"))


def test_stale_callbacks_cannot_request_stop_or_publish_result():
    owner = ExecutionCoordinator()
    old = owner.begin(pending())
    result = RunResult(detail="valid")
    assert not owner.accept_result(object(), result)
    assert not owner.request_stop(object())
    assert owner.session.result != result
    assert owner.accept_result(old, result)
    owner.finish("run", 1, None, "failure")
    assert not owner.accepts(old)
    assert not owner.request_stop(old)
    assert not owner.accept_result(old, result)
    next_token = owner.begin(pending())
    assert next_token is not old
    assert not owner.accepts(old)
    assert owner.session.result != result


def test_measurement_window_transition_remains_authoritative():
    owner = ExecutionCoordinator()
    owner.begin(pending())
    assert owner.update_snapshot(ProgressSnapshot(stage=Stage.STARTING)) is False
    measuring = ProgressSnapshot(stage=Stage.MEASURING, measurement_active=True)
    assert owner.update_snapshot(measuring) is True
    assert owner.update_snapshot(measuring) is False
    assert owner.update_snapshot(ProgressSnapshot(stage=Stage.VERIFYING_ARTIFACT)) is True


def test_completion_ends_attempt_and_retains_final_snapshot():
    owner = ExecutionCoordinator()
    token = owner.begin(pending("probe"))
    snap = ProgressSnapshot(stage=Stage.PROBE_COMPLETED, probe_result={"status": "ok"}, probe_summary="summary")
    result = owner.finish("probe", 0, snap, "")
    assert result.final_snapshot.stage == Stage.PROBE_COMPLETED
    assert owner.session.snapshot == result.final_snapshot
    assert owner.session.kind == ""
    assert owner.session.command == ()
    assert owner.session.token is None
    assert not owner.accepts(token)


def test_running_child_blocks_completion_and_new_attempt():
    owner = ExecutionCoordinator()
    owner.begin(pending())
    child = Mock()
    owner.lifecycle.process = child
    with pytest.raises(RuntimeError, match="still owned"):
        owner.finish("run", 1, None, "")
    with pytest.raises(RuntimeError, match="already owned"):
        owner.begin(pending("probe"))


def test_reused_kind_cannot_complete_old_attempt():
    owner = ExecutionCoordinator()
    owner.begin(pending("run"))
    with pytest.raises(RuntimeError, match="out-of-order"):
        owner.finish("probe", 1, None, "")
    assert owner.session.kind == "run"


def test_cancel_status_is_used_by_completion():
    owner = ExecutionCoordinator()
    token = owner.begin(pending("run"))
    assert owner.request_stop(token)
    result = owner.finish("run", 1, None, "")
    assert result.final_snapshot.measurement_status == "cancelled"
    assert not owner.session.stop_requested


def test_close_prohibits_future_process_start():
    owner = ExecutionCoordinator()
    assert owner.close().complete
    with pytest.raises(RuntimeError, match="closing"):
        owner.begin(pending())
    with pytest.raises(RuntimeError, match="closing"):
        owner.lifecycle.start(["unused"], cwd=".", env={})


def test_missing_kind_cannot_start():
    owner = ExecutionCoordinator()
    with pytest.raises(ValueError, match="kind"):
        owner.begin(PendingLaunch(command=(), kind=""))
