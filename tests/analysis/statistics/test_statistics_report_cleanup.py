"""Report directory descriptors cannot hide read, publication or cancellation errors."""
import asyncio
import fcntl
import os
from concurrent.futures import CancelledError
from contextlib import nullcontext

import pytest

from acprof import artifacts
from acprof.analysis import statistics_report


@pytest.mark.parametrize("stage", ["lock", "read", "publish"])
@pytest.mark.parametrize("failure_type", [OSError, CancelledError, asyncio.CancelledError, KeyboardInterrupt, SystemExit])
def test_report_failure_survives_directory_close(tmp_path, monkeypatch, stage, failure_type):
    existing = tmp_path / "window-statistics-existing.json"
    existing.write_text('{"original": true}\n')
    failure = failure_type(f"report {stage} failed")
    close = os.close
    descriptors = []

    def fail(*_args, **_kwargs):
        raise failure

    def fail_close(descriptor):
        descriptors.append(descriptor)
        close(descriptor)
        raise OSError("report directory close failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "close", fail_close)
        if stage == "lock":
            scoped.setattr(fcntl, "flock", fail)
        else:
            scoped.setattr(statistics_report, "read_json_object" if stage == "read" else "atomic_write_json", fail)
        with pytest.raises(failure_type) as caught:
            statistics_report.save_unique_statistics_report(tmp_path, {"new": True})
    assert caught.value is failure
    assert len(descriptors) == 1
    notes = "\n".join(failure.__notes__)
    assert "report directory close failed" in notes
    assert str(tmp_path) in notes
    assert existing.read_text() == '{"original": true}\n'
    assert list(tmp_path.iterdir()) == [existing]
    # A new descriptor can acquire the same directory after the error.
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("reuse", [False, True])
def test_successful_report_still_exposes_directory_close_failure(tmp_path, monkeypatch, reuse):
    report = {"complete": True}
    if reuse:
        artifacts.atomic_write_json(tmp_path / "window-statistics-existing.json", report)
    close, open_directory = os.close, os.open
    failure = OSError("report directory close failed")
    descriptors, calls = [], []

    def track_open(*args, **kwargs):
        descriptor = open_directory(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def fail_report_close(descriptor):
        close(descriptor)
        if descriptor == descriptors[0]:
            calls.append(descriptor)
            raise failure

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "open", track_open)
        scoped.setattr(os, "close", fail_report_close)
        with pytest.raises(OSError, match="report directory close failed") as caught:
            statistics_report.save_unique_statistics_report(tmp_path, report)
    assert caught.value is failure
    assert len(calls) == 1
    assert str(tmp_path) in "\n".join(failure.__notes__)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert artifacts.read_json_object(files[0]) == report
    assert statistics_report.save_unique_statistics_report(tmp_path, report) == (files[0], True)


@pytest.mark.parametrize("stage", ["read", "publish"])
def test_stats_cli_includes_cleanup_diagnostics(tmp_path, monkeypatch, capsys, stage):
    from acprof.cli import stats
    from acprof.host import run_state

    failure = OSError("statistics operation failed")
    failure.add_note("Cleanup failed for statistics directory lock: close failed")

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(run_state, "offline_analysis_lock", nullcontext)
    monkeypatch.setattr(statistics_report, "build_statistics_report", lambda *_args, **_kwargs: {"groups": []})
    monkeypatch.setattr(statistics_report,
                        "build_statistics_report" if stage == "read" else "save_unique_statistics_report", fail)
    with pytest.raises(SystemExit) as caught:
        stats.main([str(tmp_path / "result.csv"), "--output-dir", str(tmp_path / "reports")])
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert "statistics operation failed" in captured.err
    assert "statistics directory lock: close failed" in captured.err
    assert "ACPROF_STATS" not in captured.out
