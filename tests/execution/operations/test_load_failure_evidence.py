"""Phase evidence must survive later cleanup and publication failures."""
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.cli import load as load_cli
from acprof.host.runtimes.contracts import RuntimeCleanupError
from tests.execution.operations import test_streaming_cli
from tests.execution.operations.test_process_load_replay import replay_environment

endpoint_environment = test_streaming_cli.endpoint_cli


@pytest.fixture(name="invocation", params=["replay", "endpoint"])
def load_invocation(request, tmp_path, monkeypatch):
    replay = None
    if request.param == "replay":
        replay = replay_environment(tmp_path, monkeypatch)
        argv, output, module = replay.argv, replay.output, "acprof.cli.load"
    else:
        base, output, _calls = request.getfixturevalue("endpoint_environment")
        argv = [*base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
                "--warmup", "1", "--requests", "2"]
        module = "acprof.cli.streaming_load"
    run = Mock()
    monkeypatch.setattr(module + ".run_load", run)
    return SimpleNamespace(argv=argv, output=output, run=run, replay=replay, module=module)


def phase_failure(invocation, phase, error_type):
    partial = {"successful": False, "success_rate": 0, "requests": [
        {"request_id": phase + ":0", "status": "cancelled" if error_type is KeyboardInterrupt else "error"},
    ]}
    failure = error_type("fixture " + phase + " interrupted")
    failure.load_report = partial
    warmup = {"successful": True, "success_rate": 1, "requests": [{"request_id": "warmup:0", "status": "ok"}]}
    invocation.run.side_effect = [failure] if phase == "warmup" else [warmup, failure]
    return failure, partial


@pytest.mark.parametrize("phase", ["warmup", "result"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_phase_interruption_keeps_partial_records_in_the_correct_phase(invocation, phase, error_type):
    failure, partial = phase_failure(invocation, phase, error_type)
    with pytest.raises(error_type, match="fixture .* interrupted") as caught:
        load_cli.main(invocation.argv)
    assert caught.value is failure
    report = json.loads((invocation.output / "load.json").read_text())
    assert report[phase] == partial
    assert report["phase_errors"][phase] == {"exception_type": error_type.__name__, "detail": str(failure)}
    assert report["status"] == ("cancelled" if error_type is KeyboardInterrupt else "failed")
    if phase == "warmup":
        assert "result" not in report
        assert invocation.run.call_count == 1
    else:
        assert report["warmup"]["requests"][0]["request_id"] == "warmup:0"


@pytest.mark.parametrize("phase", ["warmup", "result"])
def test_phase_evidence_survives_a_later_finalizer_failure(invocation, monkeypatch, phase):
    failure, partial = phase_failure(invocation, phase, KeyboardInterrupt)
    cleanup = RuntimeCleanupError("fixture cleanup unconfirmed")
    if invocation.replay is not None:
        invocation.replay.stop.side_effect = cleanup
    else:
        @contextmanager
        def failing_lock():
            try:
                yield
            finally:
                raise cleanup
        monkeypatch.setattr(invocation.module + ".MeasurementLock", failing_lock)
    with pytest.raises(RuntimeCleanupError, match="cleanup unconfirmed") as caught:
        load_cli.main(invocation.argv)
    assert caught.value is cleanup
    assert caught.value.__context__ is failure
    report = json.loads((invocation.output / "load.json").read_text())
    assert report[phase] == partial
    assert report["phase_errors"][phase]["detail"] == str(failure)
    assert report["error"] == "RuntimeCleanupError: fixture cleanup unconfirmed"
    assert report["status"] == "failed"
    assert report["successful"] is False
    if phase == "warmup":
        assert "result" not in report


def test_failed_warmup_is_retained_without_starting_measurement(invocation):
    warmup = {"successful": False, "success_rate": 0, "requests": [{"request_id": "warmup:0", "status": "error"}]}
    invocation.run.return_value = warmup
    with pytest.raises(RuntimeError, match="warmup failed"):
        load_cli.main(invocation.argv)
    report = json.loads((invocation.output / "load.json").read_text())
    assert report["warmup"] == warmup
    assert "result" not in report
    assert report["phase_errors"]["warmup"]["exception_type"] == "RuntimeError"
    assert invocation.run.call_count == 1


def test_phase_failure_notes_are_preserved_in_the_report(invocation):
    failure, partial = phase_failure(invocation, "result", RuntimeError)
    failure.add_note("fixture worker cleanup also failed")
    with pytest.raises(RuntimeError, match="fixture result interrupted"):
        load_cli.main(invocation.argv)
    report = json.loads((invocation.output / "load.json").read_text())
    assert report["result"] == partial
    detail = "fixture result interrupted\nfixture worker cleanup also failed"
    assert report["phase_errors"]["result"]["detail"] == detail
    assert report["error"] == "RuntimeError: " + detail


@pytest.mark.parametrize("successful", [False, True])
def test_disabled_warmup_keeps_measured_status_and_exit_code(invocation, successful):
    measured = {"successful": successful, "success_rate": int(successful), "requests": []}
    invocation.run.return_value = measured
    assert load_cli.main([*invocation.argv, "--warmup", "0"]) == (0 if successful else 1)
    report = json.loads((invocation.output / "load.json").read_text())
    assert report["result"] == measured
    assert report["successful"] is successful
    assert report["status"] == "complete"
    assert "warmup" not in report
    assert "phase_errors" not in report
    invocation.run.assert_called_once()


def test_success_keeps_warmup_separate_without_failure_diagnostics(invocation):
    warmup = {"successful": True, "success_rate": 1, "requests": [{"request_id": "warmup:0", "status": "ok"}]}
    measured = {"successful": True, "success_rate": 1, "requests": [{"request_id": "measured:0", "status": "ok"}]}
    invocation.run.side_effect = [warmup, measured]
    assert load_cli.main(invocation.argv) == 0
    report = json.loads((invocation.output / "load.json").read_text())
    assert report["warmup"] == warmup
    assert report["result"] == measured
    assert "phase_errors" not in report
    assert "error" not in report


def fail_report_publication(invocation, monkeypatch, *, attempt, error_type: type[BaseException] = PermissionError):
    real_replace = os.replace
    calls = []
    failure = error_type("fixture report publication denied")

    def replace(source, destination):
        if Path(destination) == invocation.output / "load.json":
            calls.append(Path(destination))
            if len(calls) == attempt:
                raise failure
        real_replace(source, destination)

    monkeypatch.setattr("acprof.artifacts.os.replace", replace)
    return failure, calls


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_publication_failure_keeps_active_error_and_reports_write_failure(invocation, monkeypatch, error_type):
    failure, partial = phase_failure(invocation, "result", error_type)
    publication, writes = fail_report_publication(invocation, monkeypatch, attempt=2)
    with pytest.raises(error_type, match="fixture result interrupted") as caught:
        load_cli.main(invocation.argv)
    assert caught.value is failure
    assert caught.value.load_report == partial
    assert any(str(publication) in note and "load report" in note for note in caught.value.__notes__)
    assert len(writes) == 2
    assert json.loads((invocation.output / "load.json").read_text())["status"] == "running"


def test_cancellation_during_publication_keeps_the_preceding_failure_as_cause(invocation, monkeypatch):
    failure, _partial = phase_failure(invocation, "result", RuntimeError)
    cancellation, _writes = fail_report_publication(invocation, monkeypatch, attempt=2, error_type=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt, match="publication denied") as caught:
        load_cli.main(invocation.argv)
    assert caught.value is cancellation
    assert cancellation.__cause__ is failure
    assert any(str(failure) in note for note in cancellation.__notes__)
    assert json.loads((invocation.output / "load.json").read_text())["status"] == "running"


@pytest.mark.parametrize("attempt", [1, 2])
def test_publication_failure_without_active_error_is_not_success(invocation, monkeypatch, capsys, attempt):
    invocation.run.return_value = {"successful": True, "success_rate": 1, "requests": []}
    failure, writes = fail_report_publication(invocation, monkeypatch, attempt=attempt)
    with pytest.raises(PermissionError, match="publication denied") as caught:
        load_cli.main(invocation.argv)
    assert caught.value is failure
    assert len(writes) == attempt
    stdout = capsys.readouterr().out
    assert "成功率" not in stdout
    assert "success rate" not in stdout
    if attempt == 1:
        invocation.run.assert_not_called()
        assert not (invocation.output / "load.json").exists()
        if invocation.replay is not None:
            invocation.replay.start.assert_not_called()
    else:
        assert json.loads((invocation.output / "load.json").read_text())["status"] == "running"


def test_replay_request_evidence_survives_packet_capture_cleanup_failure(tmp_path, monkeypatch):
    replay = replay_environment(tmp_path, monkeypatch)
    partial = {"successful": False, "success_rate": 0, "requests": [{"request_id": "measured:0", "status": "cancelled"}]}
    failure = KeyboardInterrupt("fixture interrupted")
    setattr(failure, "load_report", partial)
    run = Mock(side_effect=[{"successful": True, "success_rate": 1}, failure])
    monkeypatch.setattr("acprof.cli.load.run_load", run)

    @contextmanager
    def failing_capture(*_args, **_kwargs):
        try:
            yield
        finally:
            raise RuntimeError("fixture packet capture cleanup failed")

    monkeypatch.setattr(load_cli, "capture_packets", failing_capture)
    with pytest.raises(RuntimeError, match="capture cleanup failed"):
        load_cli.main([*replay.argv, "--capture"])
    report = json.loads((replay.output / "load.json").read_text())
    assert report["result"] == partial
    assert report["phase_errors"]["result"]["detail"] == str(failure)
    assert report["status"] == "failed"
    replay.stop.assert_called_once()


def test_replay_cleanup_failure_still_propagates_when_publication_also_fails(tmp_path, monkeypatch):
    replay = replay_environment(tmp_path, monkeypatch)
    operation = KeyboardInterrupt("fixture interrupted")
    setattr(operation, "load_report", {"successful": False, "requests": []})
    monkeypatch.setattr("acprof.cli.load.run_load", Mock(side_effect=operation))
    cleanup = RuntimeCleanupError("fixture cleanup unconfirmed")
    replay.stop.side_effect = cleanup
    publication, _writes = fail_report_publication(replay, monkeypatch, attempt=2)
    with pytest.raises(RuntimeCleanupError, match="cleanup unconfirmed") as caught:
        load_cli.main(replay.argv)
    assert caught.value is cleanup
    assert cleanup.__context__ is operation
    assert any(str(publication) in note for note in cleanup.__notes__)
    assert json.loads((replay.output / "load.json").read_text())["status"] == "running"
