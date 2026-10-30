"""Run and task teardown retain diagnostic evidence and cancellation semantics."""
import asyncio
import io
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.application.errors import CollectionFailed
from acprof.application.notifications import RunNotifications
from acprof.application.run import RunApplication, RunOutcome
from acprof.experiments.configuration import RunConfig
from acprof.host.preparation.task_support import TaskSupportError
from acprof.host.run_state import RunStateError


@pytest.fixture
def lifecycle(monkeypatch):
    calls, notices, delivered, closed, finished = [], [], [], [], []
    faults = {}
    executed = False

    def visit(stage):
        calls.append(stage)
        if stage in faults:
            raise faults[stage]

    def stop(_log):
        visit("stop")
        return True

    def close(status):
        closed.append(status)
        visit("close")

    def reset():
        visit("reset" if executed else "start")

    def record(status, detail):
        notices.append((status, detail))
        visit("record")

    def deliver(path):
        delivered.append(path)
        visit("deliver")

    def finish(event, *, status):
        finished.append((event, status))
        visit("finish")

    notifications = Mock(spec=RunNotifications, reset=reset, record_termination=record, deliver=deliver)
    application = RunApplication(notifications=notifications, terminal_stop=stop)

    def execute(*_args, **_kwargs):
        nonlocal executed
        executed = True
        application.run_state = SimpleNamespace(close=close)
        application.terminal_log = ("%1", "partial.log", "final.log")
        visit("execute")
        return RunOutcome("ok")

    monkeypatch.setattr(application, "_execute", execute)
    monkeypatch.setattr("acprof.application.run.emit_event", finish)
    return SimpleNamespace(application=application, faults=faults, calls=calls, notices=notices,
                           delivered=delivered, closed=closed, finished=finished)


def invoke(application, child):
    config = RunConfig.smoke("fixture/model")
    if child:
        return application._execute_task_child(config, None, command="fixture")
    return application.run(config, command="fixture")


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("failure_type", [None, RuntimeError, KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_all_cleanup_errors_remain_visible_and_every_owned_resource_is_attempted(lifecycle, child, failure_type):
    primary = failure_type("workload failed") if failure_type is not None else None
    if primary is not None:
        lifecycle.faults["execute"] = primary
    for stage in ("stop", "close", "deliver", "reset"):
        error = OSError(f"{stage} failed")
        error.add_note(f"{stage} diagnostic context")
        lifecycle.faults[stage] = error
    expected = primary if primary is not None else lifecycle.faults["stop"]

    with pytest.raises(type(expected)) as caught:
        invoke(lifecycle.application, child)

    assert caught.value is expected
    diagnostic = "\n".join((str(caught.value), *caught.value.__notes__))
    for stage in ("stop", "close", "deliver", "reset"):
        assert f"{stage} failed" in diagnostic
        assert f"{stage} diagnostic context" in diagnostic
    assert [call for call in lifecycle.calls if call in {"stop", "close", "deliver", "reset"}] == [
        "stop", "close", "deliver", "reset"]
    assert lifecycle.delivered == [None]
    assert lifecycle.closed == ["interrupted" if failure_type is KeyboardInterrupt else "failed"]
    assert lifecycle.application.run_state is None
    assert lifecycle.application.terminal_log is None
    # Notification construction sees both log and state failures, including their notes.
    assert lifecycle.notices[-1][0] == "failed"
    for text in ("stop failed", "stop diagnostic context", "close failed", "close diagnostic context"):
        assert text in lifecycle.notices[-1][1]
    if not child:
        assert lifecycle.finished == [("run_finished", "failed")]
        assert not lifecycle.application._running
    # Cleared handles cannot be finalized twice after an error.
    lifecycle.faults.clear()
    lifecycle.application._finalize("failed")
    assert lifecycle.calls.count("stop") == lifecycle.calls.count("close") == 1


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("stage", ["stop", "close", "deliver", "reset"])
@pytest.mark.parametrize("cancellation_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_cleanup_cancellation_propagates_over_an_ordinary_workload_failure(lifecycle, child, stage, cancellation_type):
    primary = RuntimeError("workload failed")
    primary.add_note("workload diagnostic context")
    cancellation = cancellation_type("cleanup cancelled")
    lifecycle.faults.update(execute=primary, **{stage: cancellation})

    with pytest.raises(cancellation_type) as caught:
        invoke(lifecycle.application, child)

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert "workload failed" in "\n".join(cancellation.__notes__)
    assert "workload diagnostic context" in "\n".join(cancellation.__notes__)
    assert [call for call in lifecycle.calls if call in {"stop", "close", "deliver", "reset"}] == [
        "stop", "close", "deliver", "reset"]
    assert lifecycle.application.run_state is None
    assert lifecycle.application.terminal_log is None


@pytest.mark.parametrize("cleanup_failed", [False, True])
def test_notification_record_failure_cannot_replace_the_failure_being_recorded(lifecycle, cleanup_failed):
    primary = OSError("state save failed") if cleanup_failed else RuntimeError("workload failed")
    lifecycle.faults["close" if cleanup_failed else "execute"] = primary
    lifecycle.faults["record"] = OSError("notification record failed")

    with pytest.raises(type(primary)) as caught:
        invoke(lifecycle.application, False)

    assert caught.value is primary
    assert "notification record failed" in "\n".join(primary.__notes__)
    assert lifecycle.calls[-3:] == ["deliver", "reset", "finish"]
    assert lifecycle.closed == ["failed"]


@pytest.mark.parametrize("failure_type", [None, RuntimeError, KeyboardInterrupt])
def test_finished_event_failure_keeps_earlier_run_and_finalization_errors(lifecycle, failure_type):
    primary = failure_type("workload failed") if failure_type is not None else OSError("state save failed")
    lifecycle.faults["close"] = OSError("state save failed")
    lifecycle.faults["execute" if failure_type is not None else "close"] = primary
    lifecycle.faults["finish"] = BrokenPipeError("event stream closed")

    with pytest.raises(type(primary)) as caught:
        invoke(lifecycle.application, False)

    assert caught.value is primary
    diagnostic = "\n".join((str(primary), *primary.__notes__))
    assert "state save failed" in diagnostic
    assert "event stream closed" in diagnostic
    assert not lifecycle.application._running


def test_failed_initial_notification_reset_releases_application_for_reuse(lifecycle):
    failure = OSError("notification initialization failed")
    lifecycle.faults["start"] = failure
    with pytest.raises(OSError, match="notification initialization failed") as caught:
        invoke(lifecycle.application, False)
    assert caught.value is failure
    assert lifecycle.calls == ["start", "finish"]
    assert not lifecycle.application._running
    lifecycle.faults.clear()
    assert invoke(lifecycle.application, False).status == "ok"


def test_real_closed_tui_event_pipe_cannot_mask_a_workload_failure(monkeypatch):
    application = RunApplication()
    failure = ValueError("invalid workload")
    monkeypatch.setattr(application, "_execute", Mock(side_effect=failure))
    monkeypatch.setenv("ACPROF_TUI", "1")
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    with io.TextIOWrapper(io.FileIO(write_fd, mode="w"), write_through=True) as output:
        with monkeypatch.context() as scoped:
            scoped.setattr("sys.stdout", output)
            with pytest.raises(ValueError, match="invalid workload") as caught:
                invoke(application, False)
    assert caught.value is failure
    assert "BrokenPipeError" in "\n".join(failure.__notes__)
    assert not application._running


@pytest.mark.parametrize(("failure_type", "exit_code"), [(ValueError, 2), (TaskSupportError, 2),
                                                       (RunStateError, 2), (CollectionFailed, 1)])
def test_cli_failure_translation_keeps_cleanup_notes(monkeypatch, capsys, failure_type, exit_code):
    from acprof.cli import run as run_cli

    failure = failure_type("workload failed")
    failure.add_note("Cleanup failed: state save failed\nlock release diagnostic")
    application = Mock(spec=RunApplication)
    application.run.side_effect = failure
    monkeypatch.setattr(run_cli, "RunApplication", Mock(return_value=application))
    args = run_cli.build_parser().parse_args(["--model", "fixture/model"])
    with pytest.raises(SystemExit) as caught:
        run_cli.main(args=args)
    assert caught.value.code == exit_code
    diagnostic = capsys.readouterr().err
    for text in ("workload failed", "state save failed", "lock release diagnostic"):
        assert text in diagnostic
