"""The public host command boundary preserves failures and releases ownership."""
import io
import os
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.system.command import run_command


@pytest.fixture
def command_children(monkeypatch, primary_error):
    real_popen = subprocess.Popen
    real_killpg = os.killpg
    children = []

    def launch(*args, **kwargs):
        process: subprocess.Popen[str] = real_popen(*args, **kwargs)
        children.append(process)
        if primary_error is not None:
            communicate = process.communicate
            first_call = True

            def interrupted(*call_args, **call_kwargs):
                nonlocal first_call
                if first_call:
                    first_call = False
                    raise primary_error
                return communicate(*call_args, **call_kwargs)

            monkeypatch.setattr(process, "communicate", interrupted)
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    try:
        yield children
    finally:
        # The regression must not leak children even against the broken code.
        for child in children:
            try:
                real_killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()


@pytest.mark.parametrize("primary_error", [
    None, KeyboardInterrupt("fixture cancellation"), OSError("fixture pipe read failure"),
], ids=["timeout", "cancellation", "pipe-read"])
def test_term_failure_preserves_error_and_reaps_real_child(monkeypatch, primary_error, command_children):
    real_killpg = os.killpg

    def refuse_term(pid, sig):
        if sig == signal.SIGTERM:
            raise PermissionError("fixture SIGTERM refusal")
        real_killpg(pid, sig)

    monkeypatch.setattr(os, "killpg", refuse_term)
    error_type = type(primary_error) if primary_error is not None else subprocess.TimeoutExpired
    with pytest.raises(error_type) as caught:
        run_command([sys.executable, "-c", "import time; time.sleep(30)"],
                    input="", timeout=0.8, kill_process_group=True)

    if primary_error is None:
        assert isinstance(caught.value, subprocess.TimeoutExpired)
    else:
        assert caught.value is primary_error
    assert getattr(caught.value, "command_metadata").error_type == type(caught.value).__name__
    assert "fixture SIGTERM refusal" in "\n".join(caught.value.__notes__)
    process, = command_children
    assert process.returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))


@pytest.mark.parametrize("returncode", [0, 7])
def test_completed_group_preserves_input_output_and_returncode(returncode):
    source = "import sys; print(sys.stdin.read()); print('diagnostic', file=sys.stderr); sys.exit(int(sys.argv[1]))"
    result = run_command([sys.executable, "-c", source, str(returncode)],
                         input="fixture input", timeout=2, kill_process_group=True)
    assert (result.stdout, result.stderr, result.returncode) == (
        "fixture input\n", "diagnostic\n", returncode)
    assert result.metadata.returncode == returncode
    assert result.metadata.error_type == ""


@pytest.fixture
def command_process(monkeypatch):
    process = SimpleNamespace(pid=12345, returncode=None,
                              stdin=io.StringIO(), stdout=io.StringIO(), stderr=io.StringIO(),
                              communicate=Mock(), wait=Mock(return_value=0))
    signals = Mock()
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(os, "killpg", signals)
    return process, signals


def test_multiple_cleanup_failures_preserve_primary_and_close_remaining_pipes(command_process):
    process, signals = command_process
    primary = OSError("fixture read failed")
    process.communicate.side_effect = [primary, subprocess.TimeoutExpired("fixture grace", 2)]
    process.wait.side_effect = subprocess.TimeoutExpired("fixture reap", 5)

    def fail_kill(_pid, sig):
        if sig == signal.SIGKILL:
            raise PermissionError("fixture SIGKILL refusal")

    class BrokenClose(io.StringIO):
        def close(self):
            super().close()
            raise OSError("fixture pipe close failure")

    process.stderr = BrokenClose()
    signals.side_effect = fail_kill
    with pytest.raises(OSError, match="fixture read failed") as caught:
        run_command(["fixture-command"], timeout=1, kill_process_group=True)

    assert caught.value is primary
    notes = "\n".join(primary.__notes__)
    for detail in ("fixture SIGKILL refusal", "fixture reap", "fixture pipe close failure"):
        assert detail in notes
    process.wait.assert_called_once()
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))


@pytest.mark.parametrize("returncode", [0, 7], ids=["successful-command", "failed-command"])
def test_cleanup_failure_is_reported_after_command_exit(command_process, returncode):
    process, signals = command_process
    process.returncode = returncode
    process.communicate.return_value = ("fixture stdout", "fixture stderr")
    cleanup_error = PermissionError("fixture group cleanup failed")
    signals.side_effect = cleanup_error
    error_type = subprocess.CalledProcessError if returncode else PermissionError
    with pytest.raises(error_type) as caught:
        run_command(["fixture-command"], check=True, timeout=1, kill_process_group=True)

    if returncode:
        assert isinstance(caught.value, subprocess.CalledProcessError)
        assert (caught.value.returncode, caught.value.stdout, caught.value.stderr) == (
            returncode, "fixture stdout", "fixture stderr")
        assert "fixture group cleanup failed" in "\n".join(caught.value.__notes__)
    else:
        assert caught.value is cleanup_error
    assert getattr(caught.value, "command_metadata").error_type == type(caught.value).__name__
    process.wait.assert_called_once()
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))


def test_cleanup_cancellation_retains_command_failure_and_closes_pipes(command_process):
    process, signals = command_process
    primary = OSError("fixture read failed")
    cancellation = KeyboardInterrupt("fixture cleanup cancellation")
    process.communicate.side_effect = primary
    signals.side_effect = [cancellation, None]
    with pytest.raises(KeyboardInterrupt) as caught:
        run_command(["fixture-command"], timeout=1, kill_process_group=True)

    assert caught.value is cancellation
    assert caught.value.__cause__ is primary
    assert "fixture read failed" in "\n".join(caught.value.__notes__)
    assert getattr(caught.value, "command_metadata").error_type == "KeyboardInterrupt"
    process.wait.assert_called_once()
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))
