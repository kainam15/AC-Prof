"""Comparison conditions use the same parsed result rows as their audit."""
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from acprof.analysis import comparison
from acprof.results import csv as result_csv
from tests.support.comparison_fixtures import ComparisonFixture


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

    monkeypatch.setattr("acprof.results.csv.parse_result_stream", record_parse)
    if relative:
        monkeypatch.chdir(source.parent)
    snapshot = comparison.load_comparison_snapshot(source.name if relative else source)
    assert snapshot["valid"]
    assert snapshot["metric_availability"]["latency_app_s"]
    assert snapshot["measurement_status"] == "complete"
    assert calls == Counter({source: 1})


@pytest.mark.parametrize(("evidence", "valid"), [
    (" N/A ", True),
    ({"schema_version": 1, "task": []}, False),
    ({"schema_version": 1, "input": {"actual_scale": float("inf")}}, False),
    ({"schema_version": 1, "task": "bad\ud800"}, False),
])
def test_unavailable_workload_preserves_other_recorded_evidence(source, evidence, valid):
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, row = reader.fieldnames, next(reader)
    summary = json.loads(row["workload_contract"])
    summary["variants"][0]["contract"] = evidence
    row["workload_contract"] = evidence if isinstance(evidence, str) else json.dumps(summary)
    assert fields is not None
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerow(row)
    snapshot = comparison.load_comparison_snapshot(source)
    assert snapshot["valid"] is valid
    assert snapshot["conditions"]["actual_workload"] is None
    assert snapshot["audit"]["counts"]["formal_ok"] == 1
    assert snapshot["metric_availability"]["latency_app_s"]
    assert snapshot["hardware"]["cpu_model"] is not None


def test_nontext_workload_in_supplied_snapshot_is_reported(source):
    supplied = result_csv.read_result_csv_snapshot(source)
    supplied.rows[0]["workload_contract"] = {}
    snapshot = comparison.load_comparison_snapshot(source, result_snapshot=supplied, verify_result_snapshot=False)
    assert not snapshot["valid"]
    assert any(issue["code"] == "workload_contract" for issue in snapshot["audit"]["issues"])
    assert snapshot["metric_availability"]["latency_app_s"]


def test_comparison_keeps_the_audited_rows_when_the_file_is_later_replaced(source, monkeypatch):
    audit = comparison.audit_result_snapshot

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

    monkeypatch.setattr("acprof.analysis.comparison.audit_result_snapshot", replace_after_audit)
    snapshot = comparison.load_comparison_snapshot(source)
    assert snapshot["valid"]
    assert snapshot["metric_availability"]["latency_app_s"]
    assert not comparison.load_comparison_snapshot(source)["metric_availability"]["latency_app_s"]


def test_audit_still_verifies_the_shared_result_content(source, monkeypatch):
    read = result_csv.read_result_csv_snapshot

    def change_after_parse(path, **kwargs):
        snapshot = read(path, **kwargs)
        source.write_bytes(source.read_bytes().replace(b"0.25", b"9.99"))
        return snapshot

    monkeypatch.setattr("acprof.analysis.audit.read_result_csv_snapshot", change_after_parse)
    snapshot = comparison.load_comparison_snapshot(source)
    assert not snapshot["valid"]
    assert "changing_snapshot" in {item["code"] for item in snapshot["audit"]["issues"]}


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


def test_first_read_error_is_shared_by_comparison_and_audit(source, monkeypatch):
    parse = result_csv.parse_result_stream
    calls = 0

    def fail_once(stream, path, expected=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("first parse failed")
        return parse(stream, path, expected, **kwargs)

    monkeypatch.setattr("acprof.results.csv.parse_result_stream", fail_once)
    snapshot = comparison.load_comparison_snapshot(source)
    assert calls == 1
    assert {item["message"] for item in snapshot["audit"]["issues"] if item["code"] == "invalid_csv"} == {"first parse failed"}
    assert not snapshot["valid"]
    assert snapshot["conditions"]["actual_workload"] is None
    assert "actual workload: first parse failed" in snapshot["issues"]


def test_csv_parser_error_keeps_its_existing_exception_boundary(source, monkeypatch):
    def fail_parse(*_args, **_kwargs):
        raise csv.Error("parser failure")

    monkeypatch.setattr("acprof.results.csv.parse_result_stream", fail_parse)
    with pytest.raises(csv.Error, match="parser failure"):
        comparison.load_comparison_snapshot(source)
