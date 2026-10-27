"""A published latency JSON must keep referring to its own complete CSV."""

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from uuid import UUID

import pandas as pd
import pytest

from acprof.analysis.latency_report import (
    LATENCY_MODEL_DIR,
    LATENCY_MODEL_REPORT,
    LATENCY_MODEL_RESIDUAL_FIELDS,
    LATENCY_MODEL_RESIDUALS,
    LatencyModelReport,
    write_latency_model_report,
)
from acprof.artifacts import sync_directory
from acprof.plotting.latency import plot_latency_model_fit_curves


def calculated_report(label: str) -> LatencyModelReport:
    report = {"report_schema_version": 2, "model_name": label,
              "status": "skipped" if label == "skip" else "ok",
              "residuals_csv": LATENCY_MODEL_RESIDUALS}
    rows = () if label == "skip" else ({"report_schema_version": 2,
        "environment_class": "unknown", "case_id": label},)
    return LatencyModelReport(report, rows)


@pytest.fixture(autouse=True)
def calculated_input(monkeypatch):
    monkeypatch.setattr("acprof.analysis.latency_report.build_latency_model_report",
                        lambda _frame, meta: calculated_report(meta["model_name"]))


@pytest.fixture
def previous_pair(tmp_path) -> tuple[Path, Path, bytes, bytes]:
    directory = tmp_path / LATENCY_MODEL_DIR
    directory.mkdir()
    report_path = directory / LATENCY_MODEL_REPORT
    csv_path = directory / LATENCY_MODEL_RESIDUALS
    previous = calculated_report("previous")
    report_path.write_text(json.dumps(previous.report), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LATENCY_MODEL_RESIDUAL_FIELDS)
        writer.writeheader()
        writer.writerows(previous.residuals)
    return report_path, csv_path, report_path.read_bytes(), csv_path.read_bytes()


@pytest.mark.parametrize("failure_stage", ["json_write", "json_publish", "cancel"])
def test_json_failure_preserves_the_previous_report_and_its_csv(tmp_path, previous_pair, monkeypatch, capsys, failure_stage):
    report_path, csv_path, old_report, old_csv = previous_pair
    failure = KeyboardInterrupt("cancel report") if failure_stage == "cancel" else OSError("report disk failure")

    def fail_write(_payload, stream, **_kwargs):
        stream.write('{"incomplete":')
        raise failure

    replace = os.replace

    def fail_publish(source, target):
        if Path(target).name == LATENCY_MODEL_REPORT:
            raise failure
        return replace(source, target)

    if failure_stage == "json_publish":
        monkeypatch.setattr(os, "replace", fail_publish)
    else:
        monkeypatch.setattr("acprof.analysis.latency_report.json.dump", fail_write)
    with pytest.raises(type(failure)) as error:
        write_latency_model_report(pd.DataFrame(), {"model_name": "new"}, str(tmp_path))
    assert error.value is failure
    assert report_path.read_bytes() == old_report
    assert csv_path.read_bytes() == old_csv
    assert not list(report_path.parent.glob(".*.tmp"))
    assert "[saved]" not in capsys.readouterr().out


def test_process_exit_between_csv_and_json_keeps_the_previous_pair(tmp_path, previous_pair):
    report_path, csv_path, old_report, old_csv = previous_pair
    code = """
import os
import sys
import pandas as pd
from acprof.analysis import latency_report as module
result = module.LatencyModelReport(
    {"report_schema_version": 2, "model_name": "child", "status": "ok",
     "residuals_csv": module.LATENCY_MODEL_RESIDUALS},
    ({"report_schema_version": 2, "environment_class": "unknown", "case_id": "child"},),
)
module.build_latency_model_report = lambda *_args: result
module.json.dump = lambda *_args, **_kwargs: os._exit(73)
module.write_latency_model_report(pd.DataFrame(), {}, sys.argv[1])
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert child.returncode == 73, child.stderr
    assert report_path.read_bytes() == old_report
    assert csv_path.read_bytes() == old_csv


def test_concurrent_writers_publish_matching_json_and_csv(tmp_path, previous_pair, monkeypatch):
    report_path, _csv_path, _old_report, _old_csv = previous_pair
    first_ready, second_committed = Event(), Event()
    dump = json.dump

    def ordered_dump(payload, stream, **kwargs):
        if payload["model_name"] == "first":
            first_ready.set()
            assert second_committed.wait(timeout=10)
        return dump(payload, stream, **kwargs)

    monkeypatch.setattr("acprof.analysis.latency_report.json.dump", ordered_dump)

    def second_writer():
        assert first_ready.wait(timeout=10)
        try:
            write_latency_model_report(pd.DataFrame(), {"model_name": "second"}, str(tmp_path))
        finally:
            second_committed.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(write_latency_model_report, pd.DataFrame(), {"model_name": "first"}, str(tmp_path))
        second = pool.submit(second_writer)
        first.result(timeout=15)
        second.result(timeout=15)
    report = json.loads(report_path.read_bytes())
    assert report["model_name"] == "first"
    with (report_path.parent / report["residuals_csv"]).open(newline="") as stream:
        assert [row["case_id"] for row in csv.DictReader(stream)] == ["first"]


def test_skipped_report_selects_an_empty_csv_without_changing_the_old_one(tmp_path, previous_pair):
    report_path, csv_path, _old_report, old_csv = previous_pair
    write_latency_model_report(pd.DataFrame(), {"model_name": "skip"}, str(tmp_path))
    report = json.loads(report_path.read_bytes())
    assert report["status"] == "skipped"
    with (report_path.parent / report["residuals_csv"]).open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []
    assert csv_path.read_bytes() == old_csv


def test_returned_report_snapshot_survives_a_later_publication(tmp_path, previous_pair):
    first = write_latency_model_report(pd.DataFrame(), {"model_name": "first"}, str(tmp_path))
    second = write_latency_model_report(pd.DataFrame(), {"model_name": "second"}, str(tmp_path))
    assert first is not None
    assert second is not None
    assert first.report["model_name"] == "first"
    assert second.report["model_name"] == "second"
    assert first.residuals_path != second.residuals_path
    with open(first.residuals_path, newline="") as stream:
        assert [row["case_id"] for row in csv.DictReader(stream)] == ["first"]
    assert json.loads(Path(second.report_path).read_bytes()) == second.report


def test_error_after_json_replacement_keeps_its_complete_csv(tmp_path, previous_pair, monkeypatch, capsys):
    report_path, csv_path, old_report, old_csv = previous_pair
    failure = OSError("report directory sync failed")

    def fail_after_commit(directory):
        if report_path.read_bytes() != old_report:
            raise failure
        sync_directory(directory)

    monkeypatch.setattr("acprof.artifacts.sync_directory", fail_after_commit)
    with pytest.raises(OSError, match="report directory sync failed") as error:
        write_latency_model_report(pd.DataFrame(), {"model_name": "new"}, str(tmp_path))
    assert error.value is failure
    report = json.loads(report_path.read_bytes())
    assert report["model_name"] == "new"
    with (report_path.parent / report["residuals_csv"]).open(newline="") as stream:
        assert [row["case_id"] for row in csv.DictReader(stream)] == ["new"]
    assert csv_path.read_bytes() == old_csv
    assert not list(report_path.parent.glob(".*.tmp"))
    assert "[saved]" not in capsys.readouterr().out


def test_first_json_failure_does_not_publish_a_selector(tmp_path, monkeypatch):
    def fail_json(_payload, stream, **_kwargs):
        stream.write("{")
        raise OSError("JSON failed")

    monkeypatch.setattr("acprof.analysis.latency_report.json.dump", fail_json)
    with pytest.raises(OSError, match="JSON failed"):
        write_latency_model_report(pd.DataFrame(), {"model_name": "new"}, str(tmp_path))
    directory = tmp_path / LATENCY_MODEL_DIR
    assert not (directory / LATENCY_MODEL_REPORT).exists()
    assert not list(directory.glob(".*.tmp"))
    complete = list(directory.glob("*.csv"))
    assert len(complete) == 1
    with complete[0].open(newline="") as stream:
        assert [row["case_id"] for row in csv.DictReader(stream)] == ["new"]


@pytest.mark.parametrize("kind", ["file", "dangling-symlink"])
def test_generation_collision_never_overwrites_an_existing_entry(tmp_path, previous_pair, monkeypatch, kind):
    report_path, csv_path, old_report, old_csv = previous_pair
    identity = UUID(int=17)
    candidate = report_path.parent / f"latency_model_residuals-{identity.hex}.csv"
    if kind == "file":
        candidate.write_bytes(b"other writer\n")
    else:
        candidate.symlink_to("missing.csv")
    original = candidate.lstat()
    monkeypatch.setattr("acprof.analysis.latency_report.uuid4", lambda: identity)
    with pytest.raises(FileExistsError):
        write_latency_model_report(pd.DataFrame(), {"model_name": "new"}, str(tmp_path))
    assert candidate.lstat().st_ino == original.st_ino
    if kind == "file":
        assert candidate.read_bytes() == b"other writer\n"
    else:
        assert candidate.is_symlink()
        assert candidate.readlink() == Path("missing.csv")
    assert report_path.read_bytes() == old_report
    assert csv_path.read_bytes() == old_csv
    assert not list(report_path.parent.glob(".*.tmp"))


def test_fit_renderer_rejects_a_csv_from_another_report(tmp_path, previous_pair):
    _report_path, csv_path, _old_report, _old_csv = previous_pair
    with pytest.raises(ValueError, match="report and residual CSV do not match"):
        plot_latency_model_fit_curves(str(csv_path), {"residuals_csv": "another.csv"}, None)
