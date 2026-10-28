"""Comparison conditions use the same parsed result rows as their audit."""
import csv
from collections import Counter
from pathlib import Path

import pytest
from comparison_fixtures import ComparisonFixture

from acprof import result_csv
from acprof.analysis import comparison


@pytest.fixture(name="source")
def csv_source(tmp_path):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    return fixture.left / "result_all.csv"


@pytest.mark.parametrize("relative", [False, True])
def test_comparison_parses_plain_csv_once(source, monkeypatch, relative):
    parse = result_csv.parse_result_stream
    calls = Counter()

    def record_parse(stream, path, expected=None, **kwargs):
        calls[Path(path).resolve()] += 1
        return parse(stream, path, expected, **kwargs)

    monkeypatch.setattr("acprof.result_csv.parse_result_stream", record_parse)
    if relative:
        monkeypatch.chdir(source.parent)
    snapshot = comparison.load_comparison_snapshot(source.name if relative else source)
    assert snapshot["valid"]
    assert snapshot["metric_availability"]["latency_app_s"]
    assert snapshot["measurement_status"] == "complete"
    assert calls == Counter({source: 1})


def test_comparison_keeps_the_audited_rows_when_the_file_is_later_replaced(source, monkeypatch):
    audit = comparison.audit_result

    def replace_after_audit(path, **kwargs):
        report = audit(path, **kwargs)
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        assert fields is not None
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows({**row, "latency_app_s": "nan"} for row in rows)
        return report

    monkeypatch.setattr("acprof.analysis.comparison.audit_result", replace_after_audit)
    snapshot = comparison.load_comparison_snapshot(source)
    assert snapshot["valid"]
    assert snapshot["metric_availability"]["latency_app_s"]
    assert not comparison.load_comparison_snapshot(source)["metric_availability"]["latency_app_s"]


def test_audit_still_verifies_the_shared_result_content(source, monkeypatch):
    audit = comparison.audit_result

    def replace_before_audit(path, **kwargs):
        source.write_bytes(source.read_bytes().replace(b"0.25", b"9.99"))
        report = audit(path, **kwargs)
        assert "changing_snapshot" in {item["code"] for item in report["issues"]}
        return report

    monkeypatch.setattr("acprof.analysis.comparison.audit_result", replace_before_audit)
    assert not comparison.load_comparison_snapshot(source)["valid"]


@pytest.mark.parametrize("content", [None, b"", b"cpu_cores,cpu_cores\n1,2\n", b"\xff"])
def test_invalid_csv_keeps_result_and_metadata_diagnostics(source, content):
    if content is None:
        source.unlink()
    else:
        source.write_bytes(content)
    (source.parent / "static_meta.json").write_text("{broken")
    snapshot = comparison.load_comparison_snapshot(source)
    assert not snapshot["valid"]
    assert snapshot["conditions"]["actual_workload"] is None
    assert any("static_meta.json" in issue for issue in snapshot["issues"])
    assert any("actual workload" in issue for issue in snapshot["issues"])


def test_first_read_error_is_not_hidden_by_a_later_successful_audit(source, monkeypatch):
    parse = result_csv.parse_result_stream
    calls = 0

    def fail_once(stream, path, expected=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("first parse failed")
        return parse(stream, path, expected, **kwargs)

    monkeypatch.setattr("acprof.result_csv.parse_result_stream", fail_once)
    snapshot = comparison.load_comparison_snapshot(source)
    assert calls == 2
    assert not snapshot["valid"]
    assert snapshot["conditions"]["actual_workload"] is None
    assert "actual workload: first parse failed" in snapshot["issues"]


def test_csv_parser_error_keeps_its_existing_exception_boundary(source, monkeypatch):
    def fail_parse(*_args, **_kwargs):
        raise csv.Error("parser failure")

    monkeypatch.setattr("acprof.result_csv.parse_result_stream", fail_parse)
    with pytest.raises(csv.Error, match="parser failure"):
        comparison.load_comparison_snapshot(source)
