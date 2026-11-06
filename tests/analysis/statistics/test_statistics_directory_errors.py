"""Directory enumeration failures cannot masquerade as an empty report history."""
import errno
import fcntl
import io
import json
import os
from pathlib import Path

import pytest

from acprof.analysis.statistics_report import save_unique_statistics_report
from acprof.cli import stats
from acprof.host.run_state import MeasurementLock
from acprof.results.csv import read_result_csv_snapshot
from tests.support.run_recovery_fixtures import RunRecoveryFixture


def _fail_directory_reads(monkeypatch, directory: Path, failure: OSError) -> list[str]:
    attempts = []

    def wrap(function, name):
        def read(path="."):
            if not isinstance(path, int) and Path(os.fsdecode(path)) == directory:
                attempts.append(name)
                raise failure
            return function(path)
        return read

    # pathlib uses listdir on Python 3.12 and scandir on Python 3.13.
    for name in ("listdir", "scandir"):
        monkeypatch.setattr(os, name, wrap(getattr(os, name), name))
    return attempts


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
@pytest.mark.parametrize("identical", [False, True])
def test_directory_scan_failure_aborts_without_publication_and_releases_lock(
    tmp_path, monkeypatch, error_number, identical,
):
    report = {"schema_version": 1, "groups": []}
    existing = tmp_path / "window-statistics-existing.json"
    original = json.dumps(report if identical else {"previous": True}).encode()
    existing.write_bytes(original)
    failure = OSError(error_number, "report directory scan failed", str(tmp_path))

    with monkeypatch.context() as scoped:
        attempts = _fail_directory_reads(scoped, tmp_path, failure)
        with pytest.raises(OSError, match="report directory scan failed") as caught:
            save_unique_statistics_report(tmp_path, report)

    assert caught.value is failure
    assert len(attempts) == 1
    assert list(tmp_path.iterdir()) == [existing]
    assert existing.read_bytes() == original
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(descriptor)
    saved, reused = save_unique_statistics_report(tmp_path, report)
    assert reused is identical
    assert (saved == existing) is identical
    assert json.loads(saved.read_text()) == report


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
def test_stats_cli_exposes_scan_failure_without_success_receipt(tmp_path, monkeypatch, capsys, error_number):
    source = tmp_path / "result.csv"
    source.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s\n"
        "2,8,off,64,0,0,ok,0.01\n2,8,off,64,0,1,ok,0.02\n2,8,off,64,0,2,ok,0.03\n"
    )
    before = source.read_bytes()
    output = tmp_path / "reports"
    failure = OSError(error_number, "report directory scan failed", str(output))
    with monkeypatch.context() as scoped:
        _fail_directory_reads(scoped, output, failure)
        with pytest.raises(SystemExit) as caught:
            stats.main([str(source), "--resamples", "10", "--output-dir", str(output)])

    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert "ACPROF_STATS" not in captured.out
    assert "统计保存失败" in captured.err
    assert "report directory scan failed" in captured.err
    assert list(output.iterdir()) == []
    assert source.read_bytes() == before


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
@pytest.mark.parametrize("resume", [False, True])
def test_automatic_statistics_retains_committed_results_and_can_resume_after_scan_error(
    request, tmp_path, monkeypatch, error_number, resume,
):
    experiment = RunRecoveryFixture()
    experiment.build(request, tmp_path)
    if resume:
        experiment.invoke()
    directory = experiment.directory / "analysis"
    previous = {path: path.read_bytes() for path in directory.iterdir()} if directory.exists() else {}
    failure = OSError(error_number, "report directory scan failed", str(directory))
    output, errors = io.StringIO(), io.StringIO()

    with monkeypatch.context() as scoped:
        _fail_directory_reads(scoped, directory, failure)
        with pytest.raises(SystemExit) as caught:
            experiment.invoke(*(["--resume"] if resume else []), output=output, error_output=errors)

    assert caught.value.code == 1
    assert "自动统计失败" in errors.getvalue()
    assert "report directory scan failed" in errors.getvalue()
    assert "acprof stats" in errors.getvalue()
    assert "统计 JSON 已保存" not in output.getvalue()
    assert "已有相同报告" not in output.getvalue()
    assert {path: path.read_bytes() for path in directory.iterdir()} == previous
    state = json.loads((experiment.directory / ".acprof/run_state.json").read_text())
    assert state["status"] == "complete"
    source = experiment.directory / "result_layers.json"
    snapshot = read_result_csv_snapshot(source)
    assert len(snapshot.rows) == 2
    with MeasurementLock():
        pass

    experiment.calls.clear()
    experiment.invoke("--resume")

    assert experiment.calls == []
    assert read_result_csv_snapshot(source).sha256 == snapshot.sha256
    assert len(list(directory.glob("window-statistics-*.json"))) == 1
    assert all(path.read_bytes() == content for path, content in previous.items())


def test_report_lookup_keeps_name_filter_and_deterministic_reuse(tmp_path):
    report = {"groups": []}
    unrelated = ["other.json", "window-statistics-old.json.bak", ".window-statistics-hidden.json"]
    for name in unrelated:
        (tmp_path / name).write_text(json.dumps(report))
    # The same payload in unrelated files is not a statistics report candidate.
    saved, reused = save_unique_statistics_report(tmp_path, report)
    assert not reused
    saved.unlink()
    for name in ("window-statistics-b.json", "window-statistics-a.json"):
        (tmp_path / name).write_text(json.dumps(report))
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}

    assert save_unique_statistics_report(tmp_path, report) == (tmp_path / "window-statistics-a.json", True)
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before
