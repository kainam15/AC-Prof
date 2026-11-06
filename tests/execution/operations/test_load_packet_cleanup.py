"""Packet capture must preserve load failures and finish bounded child cleanup."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.cli import load as load_cli
from tests.execution.operations.test_process_load_replay import replay_environment


@pytest.fixture
def packet_capture(tmp_path, monkeypatch):
    process = Mock(returncode=None)
    process.poll.return_value = None
    wait_errors = []

    def wait(*, timeout):
        assert 0 < timeout <= 5
        error = wait_errors.pop(0) if wait_errors else None
        if error is not None:
            raise error
        process.returncode = 0
        return 0

    process.wait.side_effect = wait
    launch = Mock(return_value=process)
    sleep = Mock()
    monkeypatch.setattr(load_cli.subprocess, "Popen", launch)
    monkeypatch.setattr(load_cli.time, "sleep", sleep)
    monkeypatch.setattr("acprof.host.measurement.capture.require_packet_latency_prerequisites", Mock())
    path = tmp_path / "load.pcap"
    path.write_bytes(b"fixture" * 8)
    return SimpleNamespace(process=process, wait_errors=wait_errors, launch=launch, sleep=sleep, path=path)


def fail_cleanup(capture, stage, error):
    if stage == "drain":
        capture.sleep.side_effect = [None, error]
    elif stage == "terminate":
        capture.process.terminate.side_effect = error
    elif stage == "wait":
        capture.wait_errors.append(error)
    else:
        capture.wait_errors.append(subprocess.TimeoutExpired("tcpdump", 5))
        if stage == "kill":
            capture.process.kill.side_effect = error
        else:
            capture.wait_errors.append(error)


@pytest.mark.parametrize("stage", ["drain", "terminate", "wait", "kill", "reap"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_capture_keeps_load_error_and_secondary_cleanup_diagnostics(packet_capture, stage, error_type):
    failure = error_type("fixture load interrupted")
    cleanup = OSError(f"fixture {stage} failed")
    fail_cleanup(packet_capture, stage, cleanup)
    with pytest.raises(error_type, match="fixture load interrupted") as caught:
        with load_cli.capture_packets(packet_capture.path, interface="lo"):
            raise failure
    assert caught.value is failure
    assert str(cleanup) in "\n".join(caught.value.__notes__)
    assert packet_capture.process.wait.called
    assert packet_capture.process.returncode == (None if stage == "reap" else 0)
    assert packet_capture.launch.call_args.kwargs["stdout"].closed


@pytest.mark.parametrize("stage", ["drain", "terminate", "wait", "kill"])
def test_cancellation_during_capture_cleanup_takes_precedence_and_reaps(packet_capture, stage):
    failure = RuntimeError("fixture load failed")
    cancel = KeyboardInterrupt(f"fixture {stage} interrupted")
    fail_cleanup(packet_capture, stage, cancel)
    with pytest.raises(KeyboardInterrupt, match=f"fixture {stage} interrupted") as caught:
        with load_cli.capture_packets(packet_capture.path, interface="lo"):
            raise failure
    assert caught.value is cancel
    assert caught.value.__cause__ is failure
    assert str(failure) in "\n".join(caught.value.__notes__)
    assert packet_capture.process.returncode == 0


@pytest.mark.parametrize("stage", ["terminate", "wait", "kill"])
def test_capture_cleanup_error_without_load_failure_still_fails_after_reaping(packet_capture, stage):
    failure = OSError(f"fixture {stage} failed")
    fail_cleanup(packet_capture, stage, failure)
    with pytest.raises(OSError, match=f"fixture {stage} failed") as caught:
        with load_cli.capture_packets(packet_capture.path, interface="lo"):
            pass
    assert caught.value is failure
    assert packet_capture.process.returncode == 0


@pytest.mark.parametrize("timed_out", [False, True])
def test_capture_only_forces_stop_when_graceful_wait_does_not_finish(packet_capture, timed_out):
    if timed_out:
        packet_capture.wait_errors.append(subprocess.TimeoutExpired("tcpdump", 5))
    with load_cli.capture_packets(packet_capture.path, interface="lo"):
        pass
    assert packet_capture.process.returncode == 0
    assert packet_capture.process.kill.called is timed_out
    assert packet_capture.launch.call_args.kwargs["stdout"].closed


@pytest.mark.parametrize("load_failed", [False, True])
def test_unreaped_capture_keeps_every_cleanup_failure_visible(packet_capture, load_failed):
    termination = OSError("fixture terminate failed")
    packet_capture.process.terminate.side_effect = termination
    packet_capture.process.kill.side_effect = PermissionError("fixture kill denied")
    packet_capture.wait_errors.append(subprocess.TimeoutExpired("tcpdump", 5))
    failure = RuntimeError("fixture load failed") if load_failed else termination

    def run_capture():
        with load_cli.capture_packets(packet_capture.path, interface="lo"):
            if load_failed:
                raise failure

    with pytest.raises(type(failure), match=str(failure)) as caught:
        run_capture()
    assert caught.value is failure
    detail = "\n".join((str(failure), *failure.__notes__))
    assert "fixture terminate failed" in detail
    assert "fixture kill denied" in detail
    assert "tcpdump could not be reaped after kill" in detail
    assert packet_capture.process.returncode is None
    assert packet_capture.process.wait.called
    assert packet_capture.launch.call_args.kwargs["stdout"].closed


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_replay_report_keeps_request_failure_and_capture_cleanup_notes(packet_capture, tmp_path, monkeypatch, error_type):
    replay = replay_environment(tmp_path, monkeypatch)
    status = "cancelled" if error_type is KeyboardInterrupt else "error"
    partial = {"successful": False, "requests": [{"request_id": "result:0", "status": status}]}
    failure = error_type("fixture request interrupted")
    setattr(failure, "load_report", partial)
    monkeypatch.setattr(load_cli, "run_load", Mock(side_effect=failure))
    packet_capture.process.terminate.side_effect = OSError("fixture capture termination failed")
    with pytest.raises(error_type, match="fixture request interrupted") as caught:
        load_cli.main([*replay.argv, "--warmup", "0", "--capture"])
    assert caught.value is failure
    report = json.loads((replay.output / "load.json").read_text())
    assert report["result"] == partial
    assert report["phase_errors"]["result"]["exception_type"] == error_type.__name__
    assert report["status"] == ("cancelled" if error_type is KeyboardInterrupt else "failed")
    assert report["successful"] is False
    assert "fixture request interrupted" in report["error"]
    assert "fixture capture termination failed" in report["error"]
    assert packet_capture.process.returncode == 0
    replay.stop.assert_called_once_with(replay.session, log_prefix="[load]")
    assert replay.events == ["locked", "unlocked"]
    assert all(path.read_bytes() == content for path, content in replay.originals.items())


def test_capture_reaps_real_child_when_terminate_fails(tmp_path, monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    kill, wait = child.kill, child.wait
    try:
        monkeypatch.setattr(load_cli.subprocess, "Popen", Mock(return_value=child))
        monkeypatch.setattr(load_cli.time, "sleep", Mock())
        monkeypatch.setattr("acprof.host.measurement.capture.require_packet_latency_prerequisites", Mock())
        monkeypatch.setattr(child, "terminate", Mock(side_effect=OSError("fixture terminate failed")))
        failure = KeyboardInterrupt("fixture real child cancellation")
        with pytest.raises(KeyboardInterrupt, match="fixture real child cancellation") as caught:
            with load_cli.capture_packets(tmp_path / "load.pcap", interface="lo"):
                raise failure
        assert caught.value is failure
        assert "fixture terminate failed" in "\n".join(failure.__notes__)
        assert child.returncode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)
    finally:
        kill()
        wait(timeout=5)
