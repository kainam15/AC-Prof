"""Statistics deduplication must not hide storage faults or retain its lock."""
import errno
import fcntl
import json
import os

import pytest

from acprof.cli import stats


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
def test_unreadable_report_aborts_without_publication_or_retained_lock(tmp_path, monkeypatch, error_number):
    report = {"schema_version": 1, "groups": []}
    existing = tmp_path / "window-statistics-existing.json"
    original = json.dumps(report).encode("utf-8")
    existing.write_bytes(original)
    failure = OSError(error_number, "simulated report read failure", str(existing))

    def fail_read(*args, **kwargs):
        raise failure

    monkeypatch.setattr(stats, "read_json_object", fail_read)
    with pytest.raises(OSError) as caught:
        stats._save_unique_report(tmp_path, report)

    assert caught.value is failure
    assert list(tmp_path.iterdir()) == [existing]
    assert existing.read_bytes() == original
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        # A failed lookup must not block a later statistics process.
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("error_type", [FileNotFoundError, IsADirectoryError])
def test_unusable_candidate_does_not_prevent_publication(tmp_path, monkeypatch, error_type):
    existing = tmp_path / "window-statistics-existing.json"
    existing.write_text("{}", encoding="utf-8")

    def fail_read(*args, **kwargs):
        raise error_type("candidate vanished or is not a document")

    monkeypatch.setattr(stats, "read_json_object", fail_read)
    path, reused = stats._save_unique_report(tmp_path, {"schema_version": 1, "groups": []})

    assert not reused
    assert path != existing
    assert json.loads(path.read_text(encoding="utf-8"))["groups"] == []
    assert existing.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
def test_cli_reports_storage_fault_instead_of_success_receipt(tmp_path, monkeypatch, capsys, error_number):
    source = tmp_path / "result_all.csv"
    source.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s\n"
        "2,8,off,64,0,0,ok,0.01\n2,8,off,64,0,1,ok,0.02\n2,8,off,64,0,2,ok,0.03\n",
        encoding="utf-8",
    )
    output = tmp_path / "reports"
    output.mkdir()
    existing = output / "window-statistics-existing.json"
    existing.write_text("{}", encoding="utf-8")

    def fail_read(*args, **kwargs):
        raise OSError(error_number, "simulated report read failure", str(existing))

    monkeypatch.setattr(stats, "read_json_object", fail_read)
    with pytest.raises(SystemExit) as caught:
        stats.main([str(source), "--resamples", "30", "--output-dir", str(output)])

    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert "ACPROF_STATS" not in captured.out
    assert "统计保存失败" in captured.err
    assert "simulated report read failure" in captured.err
    assert list(output.iterdir()) == [existing]
