"""The two reports in one audit --compare command describe the same input bytes."""
import json
from collections import Counter
from pathlib import Path

import pytest

from acprof.analysis.audit import audit_result
from acprof.analysis.comparison import compare_results
from acprof.cli import audit
from acprof.results import csv as result_csv
from tests.analysis.comparison.test_comparison_source_integrity import (
    incremental_copy,
    legacy_snapshot,
)
from tests.support.comparison_fixtures import ComparisonFixture


@pytest.fixture
def experiments(tmp_path):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    return fixture.left, fixture.right


@pytest.mark.parametrize("storage", ["csv", "legacy-directory", "v1", "v2", "v3"])
def test_command_parses_each_table_once_and_keeps_both_reports(experiments, storage, monkeypatch, capsys):
    left, right = experiments
    if storage in {"csv", "legacy-directory"}:
        tables = [root / "result_all.csv" for root in (left, right)]
        if storage == "legacy-directory":
            for root in (left, right):
                (root / "result_layers.json").unlink()
        else:
            left, right = tables
    else:
        if storage == "v1":
            left, right = map(legacy_snapshot, (left, right))
        elif storage == "v3":
            left, right = map(incremental_copy, (left, right))
        tables = [root / layer["path"] for root in (left, right)
                  for layer in json.loads((root / "result_layers.json").read_text())["layers"].values()]

    expected_left, expected_right = tables if storage == "legacy-directory" else (left, right)
    expected = {**audit_result(expected_left), "comparison": compare_results(expected_left, expected_right),
                "accepted": True}
    counts = Counter()
    parse = result_csv.parse_result_stream

    def counted(stream, path, *args, **kwargs):
        counts[Path(path)] += 1
        return parse(stream, path, *args, **kwargs)

    monkeypatch.setattr(result_csv, "parse_result_stream", counted)
    monkeypatch.setattr("acprof.result_store.protocol.parse_result_stream", counted)
    assert audit.main([str(left), "--compare", str(right), "--json", "--require-ok", "--require-complete"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert counts == Counter(tables)


@pytest.mark.parametrize("artifact", ["layer", "metadata", "previously-absent"])
def test_command_rejects_left_changes_during_right_read(experiments, artifact, monkeypatch, capsys):
    left, right = experiments
    target = {"metadata": left / "static_meta.json", "previously-absent": left / "model_resolution.json"}.get(artifact)
    if target is None:
        target = left / json.loads((left / "result_layers.json").read_text())["layers"]["summary"]["path"]
    parse = result_csv.parse_result_stream
    changed = False

    def change_left(stream, path, *args, **kwargs):
        nonlocal changed
        if not changed and Path(path).is_relative_to(right):
            changed = True
            target.write_bytes(target.read_bytes() + b"\n" if target.exists() else b"{}")
        return parse(stream, path, *args, **kwargs)

    monkeypatch.setattr(result_csv, "parse_result_stream", change_left)
    assert audit.main([str(left), "--compare", str(right), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert changed
    assert not report["accepted"]
    assert not report["valid"]
    assert not report["comparison"]["valid"]
    assert "changing_snapshot" in {issue["code"] for issue in report["issues"]}
    assert report["comparison"]["experiments"]["right"]["valid"]


def test_require_comparable_reloads_changed_conditions_on_the_next_invocation(experiments, capsys):
    left, right = experiments
    args = [str(left), "--compare", str(right), "--comparison-purpose", "cross-hardware", "--json"]
    assert audit.main([*args, "--require-comparable"]) == 0
    before = json.loads(capsys.readouterr().out)
    assert before["comparison"]["purpose"] == "cross-hardware"
    assert before["comparison"]["status"] == "compatible"
    plan = right / "input_scale_plan.json"
    changed = json.loads(plan.read_text())
    changed["entries"][0]["payload"] = {"features": [[0, 0]]}
    plan.write_text(json.dumps(changed))
    assert audit.main(args) == 0
    assert json.loads(capsys.readouterr().out)["comparison"]["status"] == "incompatible"
    assert audit.main([*args, "--require-comparable"]) == 1
    assert not json.loads(capsys.readouterr().out)["accepted"]


def test_command_reports_a_failed_csv_read_without_retry(experiments, monkeypatch, capsys):
    left, right = (root / "result_all.csv" for root in experiments)
    parse = result_csv.parse_result_stream
    attempts = 0

    def fail_first(stream, path, *args, **kwargs):
        nonlocal attempts
        if path == left:
            attempts += 1
            if attempts == 1:
                raise ValueError("invalid first observation")
        return parse(stream, path, *args, **kwargs)

    monkeypatch.setattr(result_csv, "parse_result_stream", fail_first)
    assert audit.main([str(left), "--compare", str(right), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["valid"]
    assert not report["comparison"]["valid"]
    assert any(issue["code"] == "invalid_csv" and "invalid first observation" in issue["message"]
               for issue in report["issues"])
    assert attempts == 1


@pytest.mark.parametrize("invalid_side", ["left", "right"])
def test_invalid_layout_yields_comparison_diagnostics_and_preserves_other_side(experiments, invalid_side, capsys):
    left, right = experiments
    invalid, valid_side = (left, "right") if invalid_side == "left" else (right, "left")
    manifest = invalid / "result_manifest.json"
    manifest.write_text("{")
    assert audit.main([str(left), "--compare", str(right), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["accepted"]
    assert not report["comparison"]["valid"]
    evidence = report["comparison"]["experiments"]
    assert not evidence[invalid_side]["valid"]
    assert any(issue.startswith(str(manifest)) for issue in evidence[invalid_side]["issues"])
    assert not report["comparison"]["conditions"]["execution_runtime"][invalid_side].get("kind")
    assert evidence[valid_side]["valid"]
    if invalid_side == "left":
        assert any(issue["code"] == "invalid_layout" for issue in report["issues"])
    else:
        assert report["valid"]
