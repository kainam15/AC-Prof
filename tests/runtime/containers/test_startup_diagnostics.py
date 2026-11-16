"""Startup failures retain complete logs or explicit evidence of incomplete capture."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from acprof.failures import FAILURE_PREFIX, Failure
from acprof.host.startup_diagnostics import capture_startup_diagnostics


@pytest.mark.parametrize("channel", ["stdout", "stderr"])
def test_typed_record_survives_long_lines_on_both_channels(tmp_path, monkeypatch, channel):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    reported = Failure("load", "runtime_dependency_missing", "Missing dependency", exception_type="ImportError")
    content = "root-cause\n" + "x" * 100000 + "\n" + FAILURE_PREFIX + json.dumps(reported.to_dict()) + "\n"

    def command(args, **kwargs):
        kwargs[channel].write(content)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("acprof.system.command.run_command", command)
    path, failure = capture_startup_diagnostics("fixture", "exited")
    assert content in Path(path).read_text()
    assert failure.reason_code == "runtime_dependency_missing"
    assert failure.evidence["log_capture_complete"] is True


def test_log_timeout_keeps_partial_evidence_and_oom_classification(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    def command(args, **kwargs):
        kwargs["stderr"].write("partial diagnostic before timeout")
        raise subprocess.TimeoutExpired(args, 15)

    monkeypatch.setattr("acprof.system.command.run_command", command)
    path, failure = capture_startup_diagnostics("fixture", "exited", state={"Running": False, "OOMKilled": True})
    assert "partial diagnostic" in Path(path).read_text()
    assert failure.reason_code == "resource_limit"
    assert failure.evidence["log_capture_complete"] is False
    assert "TimeoutExpired" in failure.evidence["log_capture_error"]


def test_unwritable_log_reports_failure_without_hiding_startup_error(tmp_path, monkeypatch):
    destination = tmp_path / "not-a-directory"
    destination.write_text("fixture")
    monkeypatch.setenv("XDG_STATE_HOME", str(destination))
    _, failure = capture_startup_diagnostics("fixture", "startup failed")
    assert failure.detail == "startup failed"
    assert failure.evidence["log_capture_complete"] is False
    assert failure.evidence["log_capture_error"]
