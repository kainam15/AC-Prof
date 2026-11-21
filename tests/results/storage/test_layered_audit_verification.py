"""Audits and window statistics verify private snapshots without reparsing layers."""
import json
from collections import Counter
from pathlib import Path

import pytest

from acprof.analysis import audit, statistics_report
from acprof.results import csv as result_csv
from acprof.results.layout import ArtifactLayout
from tests.analysis.comparison.test_comparison_source_integrity import (
    incremental_copy,
    legacy_snapshot,
)
from tests.support.comparison_fixtures import ComparisonFixture
from tests.support.result_store_fixtures import leave_pending_replacement


@pytest.fixture(params=[1, 2, 3])
def source(tmp_path, request):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    root = fixture.left
    if request.param == 1:
        root = legacy_snapshot(root)
    elif request.param == 3:
        root = incremental_copy(root)
    return root / "result_layers.json"


@pytest.mark.parametrize("read", [audit.audit_result, statistics_report.build_statistics_report])
def test_private_snapshot_parses_each_layer_only_once(source, monkeypatch, read):
    expected = read(source)
    if read is audit.audit_result:
        assert expected["valid"]
    else:
        assert expected["groups"]
    counts = Counter()
    original = result_csv.parse_result_stream

    def parse(stream, path, *args, **kwargs):
        counts[Path(path)] += 1
        return original(stream, path, *args, **kwargs)

    monkeypatch.setattr(result_csv, "parse_result_stream", parse)
    monkeypatch.setattr("acprof.result_store.protocol.parse_result_stream", parse)
    assert read(source) == expected
    manifest = json.loads(source.read_text())
    assert counts == Counter({source.parent / layer["path"]: 1 for layer in manifest["layers"].values()})


@pytest.mark.parametrize("edit", ["row", "fields"])
def test_borrowed_snapshot_still_checks_for_external_mutation(source, edit):
    snapshot = result_csv.read_result_csv_snapshot(source)
    if edit == "row":
        snapshot.rows[0]["latency_app_s"] = "9"
    else:
        snapshot.fields.reverse()
    report = audit.audit_result(source, result_snapshot=snapshot)
    assert not report["valid"]
    assert "changing_snapshot" in {issue["code"] for issue in report["issues"]}


@pytest.mark.parametrize("damage", ["corrupt", "truncate", "missing", "manifest", "pending"])
def test_private_snapshot_rejects_source_changes_after_parsing(source, monkeypatch, damage):
    manifest = json.loads(source.read_text())
    layer = source.parent / manifest["layers"]["summary"]["path"]
    original = audit.read_result_csv_snapshot

    def read_then_change(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        raw = layer.read_bytes()
        if damage == "corrupt":
            layer.write_bytes(b"!" + raw[1:])
        elif damage == "truncate":
            layer.write_bytes(raw[:-1])
        elif damage == "missing":
            layer.unlink()
        elif damage == "manifest":
            source.write_bytes(source.read_bytes() + b"\n")
        elif manifest["schema_version"] == 3:
            leave_pending_replacement(ArtifactLayout.discover(source.parent), monkeypatch)
        else:
            # A replaced generation is visible even when it names the same rows.
            source.write_bytes(source.read_bytes() + b" ")
        return snapshot

    monkeypatch.setattr(audit, "read_result_csv_snapshot", read_then_change)
    report = audit.audit_result(source)
    assert not report["valid"]
    assert "changing_snapshot" in {issue["code"] for issue in report["issues"]}
    assert report["counts"]["rows"] == 1


def test_private_snapshot_retains_legacy_import_validation(source, monkeypatch):
    version = json.loads(source.read_text())["schema_version"]
    original = audit.read_result_csv_snapshot

    def read_then_change(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        (source.parent / "result_all.csv").write_bytes(b"changed imported CSV")
        return snapshot

    monkeypatch.setattr(audit, "read_result_csv_snapshot", read_then_change)
    report = audit.audit_result(source)
    assert report["valid"] is (version != 1)
    if version == 1:
        assert "changing_snapshot" in {issue["code"] for issue in report["issues"]}


def test_statistics_rechecks_source_dependencies_after_audit(source, monkeypatch):
    manifest = json.loads(source.read_bytes())
    original = statistics_report.audit_with_metadata

    def audit_then_damage(*args, **kwargs):
        report = original(*args, **kwargs)
        if manifest["schema_version"] == 1:
            (source.parent / "result_all.csv").write_bytes(b"changed imported CSV")
        elif manifest["schema_version"] == 2:
            layer = source.parent / manifest["layers"]["summary"]["path"]
            raw = layer.read_bytes()
            layer.write_bytes(b"!" + raw[1:])
        else:
            leave_pending_replacement(ArtifactLayout.discover(source.parent), monkeypatch)
        return report

    monkeypatch.setattr(statistics_report, "audit_with_metadata", audit_then_damage)
    with pytest.raises(ValueError, match="统计期间结果 CSV 发生变化"):
        statistics_report.build_statistics_report(source)


def test_statistics_rechecks_quality_in_the_selected_layout(source, monkeypatch):
    quality = ArtifactLayout.from_csv(source).path("quality_checks.json")
    quality.write_text('{"schema_version":1,"checks":[]}')
    summarize = statistics_report.summarize_windows

    def remove_observed_quality(*args, **kwargs):
        report = summarize(*args, **kwargs)
        quality.unlink()
        return report

    monkeypatch.setattr(statistics_report, "summarize_windows", remove_observed_quality)
    with pytest.raises(ValueError, match=r"quality_checks\.json"):
        statistics_report.build_statistics_report(source.parent, resamples=10)
