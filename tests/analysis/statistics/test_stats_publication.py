"""Statistics publication preserves competing outputs and reports storage failures."""

import errno
import fcntl
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import pytest

from acprof.analysis import statistics_report
from acprof.cli import stats


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "结果.csv"
    path.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s\n"
        "2,8,off,64,0,0,ok,0.01\n2,8,off,64,0,1,ok,0.02\n2,8,off,64,0,2,ok,0.03\n",
        encoding="utf-8",
    )
    return path


def arguments(source, output, option="--output"):
    return [str(source), "--metric", "latency_app_s", "--resamples", "10", option, str(output)]


def create_competitor(output, kind):
    if kind == "file":
        output.write_bytes(b"other writer's JSON\n")
        output.chmod(0o600)
    elif kind == "directory":
        output.mkdir()
        (output / "keep.txt").write_bytes(b"existing contents\n")
    else:
        target = output.parent / "target.json"
        if kind == "symlink":
            target.write_bytes(b"linked contents\n")
        output.symlink_to(target.name)
    return output.lstat()


def assert_competitor_unchanged(output, kind, original):
    assert output.lstat().st_ino == original.st_ino
    assert output.lstat().st_mode == original.st_mode
    if kind == "file":
        assert output.read_bytes() == b"other writer's JSON\n"
    elif kind == "directory":
        assert (output / "keep.txt").read_bytes() == b"existing contents\n"
    else:
        assert output.readlink() == Path("target.json")
        if kind == "symlink":
            assert output.read_bytes() == b"linked contents\n"
        else:
            assert not output.exists()


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_symlink"])
@pytest.mark.parametrize("timing", ["before", "during"])
def test_explicit_output_preserves_competitors(source, tmp_path, monkeypatch, capsys, kind, timing):
    output = tmp_path / "statistics.json"
    entries = []
    original_summarize = statistics_report.summarize_windows

    def summarize(*args, **kwargs):
        assert timing == "during", "Existing outputs must be rejected before analysis"
        entries.append(create_competitor(output, kind))
        return original_summarize(*args, **kwargs)

    monkeypatch.setattr(statistics_report, "summarize_windows", summarize)
    if timing == "before":
        entries.append(create_competitor(output, kind))
    with pytest.raises(SystemExit) as failure:
        stats.main(arguments(source, output))

    assert failure.value.code == (2 if timing == "before" else 1)
    captured = capsys.readouterr()
    assert not captured.out
    assert ("统计输出必须为新的文件" if timing == "before" else "统计保存失败") in captured.err
    assert "Traceback" not in captured.err
    assert len(entries) == 1
    assert_competitor_unchanged(output, kind, entries[0])
    assert not list(tmp_path.glob(".statistics.json.*.tmp"))


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_symlink"])
@pytest.mark.parametrize("timing", ["before", "during"])
def test_timestamp_output_avoids_competitors(source, tmp_path, monkeypatch, capsys, kind, timing):
    directory = tmp_path / "reports"
    directory.mkdir()
    output = directory / "window-statistics-20261028-123456-123456.json"
    expected = directory / "window-statistics-20261028-123456-123457.json"
    original_dump = json.dump
    original_source = source.read_bytes()
    entries = []

    def dump_then_compete(payload, stream, **kwargs):
        original_dump(payload, stream, **kwargs)
        if not entries:
            # This writer ignores the statistics directory lock and arrives
            # after the availability check, before the atomic publication.
            entries.append(create_competitor(output, kind))

    monkeypatch.setattr(json, "dump", dump_then_compete)
    if timing == "before":
        entries.append(create_competitor(output, kind))
    with patch.object(statistics_report, "datetime") as clock:
        clock.now.return_value = datetime(2026, 10, 28, 12, 34, 56, 123456)
        assert stats.main(arguments(source, directory, "--output-dir")) == 0

    captured = capsys.readouterr()
    assert not captured.err
    assert json.loads(captured.out.removeprefix("ACPROF_STATS ")) == {
        "report_path": str(expected), "reused": False,
    }
    payload = json.loads(expected.read_text(encoding="utf-8"))
    assert payload["groups"][0]["mean"] == pytest.approx(0.02)
    assert payload["result_csv"] == str(source)
    assert source.read_bytes() == original_source
    assert len(entries) == 1
    assert_competitor_unchanged(output, kind, entries[0])
    assert not list(directory.glob(".*.tmp"))


def test_competing_explicit_outputs_publish_one_complete_report(source, tmp_path, monkeypatch):
    output = tmp_path / "statistics.json"
    rendezvous = Barrier(2)
    original_dump = json.dump

    def dump_then_wait(payload, stream, **kwargs):
        original_dump(payload, stream, **kwargs)
        rendezvous.wait(timeout=10)

    monkeypatch.setattr(json, "dump", dump_then_wait)

    from acprof.artifacts import atomic_write_json

    def attempt(seed):
        # Exercise the publication boundary independently of Linux CLI exclusion.
        report = statistics_report.build_statistics_report(source, ["latency_app_s"], resamples=10, seed=seed)
        try:
            atomic_write_json(output, report, overwrite=False)
        except FileExistsError:
            return 1
        return 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(attempt, range(2)))

    assert codes.count(0) == codes.count(1) == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["seed"] == codes.index(0)
    assert payload["groups"][0]["mean"] == pytest.approx(0.02)
    assert not list(tmp_path.glob(".statistics.json.*.tmp"))


@pytest.mark.parametrize("option", ["--output", "--output-dir"])
@pytest.mark.parametrize("phase", ["serialize", "sync", "publish", "unrelated_collision"])
def test_write_failure_does_not_retry_or_claim_success(source, tmp_path, monkeypatch, capsys, option, phase):
    directory = tmp_path / "reports"
    directory.mkdir()
    output = directory if option == "--output-dir" else directory / "statistics.json"
    attempts = 0

    def fail(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        assert attempts == 1, "Storage faults must not be retried as filename collisions"
        if phase == "serialize":
            args[1].write('{"incomplete":')
            raise ValueError("simulated serialization failure")
        error_number = {
            "sync": errno.ENOSPC, "publish": errno.EOPNOTSUPP, "unrelated_collision": errno.EEXIST,
        }[phase]
        raise OSError(error_number, "simulated storage failure")

    if phase == "serialize":
        monkeypatch.setattr(json, "dump", fail)
    else:
        monkeypatch.setattr(os, "fsync" if phase == "sync" else "link", fail)
    with pytest.raises(SystemExit) as failure:
        stats.main(arguments(source, output, option))

    assert failure.value.code == 1
    assert attempts == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "统计保存失败" in captured.err
    assert "Traceback" not in captured.err
    assert list(directory.iterdir()) == []
    if option == "--output-dir":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
