"""The two reports in one audit --compare command describe the same input bytes."""
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from acprof.analysis.audit import audit_result
from acprof.analysis.comparison import compare_results, compare_snapshots, load_comparison_snapshot
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


@pytest.fixture
def workload_audits(experiments, capsys):
    left, right = (root / "result_all.csv" for root in experiments)
    with left.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, row = reader.fieldnames, next(reader)
    assert fields is not None
    summary = json.loads(row["workload_contract"])

    def run(summary):
        row["workload_contract"] = "" if summary is None else json.dumps(summary)
        with left.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerow(row)
        reports = []
        for extra in ([], ["--compare", str(right), "--require-comparable"]):
            code = audit.main([str(left), "--json", *extra])
            report = json.loads(capsys.readouterr().out)
            assert code == (0 if report["accepted"] else 1)
            reports.append(report)
        return reports

    return summary, run


@pytest.mark.parametrize(("level", "version"), [
    ("summary", True), ("summary", 1.0), ("summary", 2), ("summary", None),
    ("contract", True), ("contract", 1.0), ("contract", "1"), ("contract", 2), ("contract", None),
])
def test_audit_and_strict_comparison_reject_malformed_workload_versions(workload_audits, level, version):
    summary, run = workload_audits
    contract = summary if level == "summary" else summary["variants"][0]["contract"]
    if version is None:
        del contract["schema_version"]
    else:
        contract["schema_version"] = version
    if level == "contract" and version is None:
        # An earlier unknown request must not hide a malformed recorded contract.
        summary["variants"][0]["count"] -= 1
        summary["variants"].insert(0, {"count": 1, "contract": None})

    ordinary, strict = run(summary)
    assert not ordinary["accepted"]
    assert any(issue["code"] == "workload_contract" for issue in ordinary["issues"])
    assert not strict["accepted"]
    assert not strict["comparison"]["experiments"]["left"]["valid"]
    assert strict["comparison"]["experiments"]["right"]["valid"]
    assert strict["comparison"]["conditions"]["actual_workload"]["status"] == "unknown"


@pytest.mark.parametrize("evidence", ["complete", "null", "mixed", "historical_blank"])
def test_audit_keeps_unknown_request_evidence_distinct_from_invalid_versions(workload_audits, evidence):
    summary, run = workload_audits
    if evidence == "null":
        summary["variants"][0]["contract"] = None
    elif evidence == "mixed":
        summary["variants"][0]["count"] -= 1
        summary["variants"].append({"count": 1, "contract": None})

    ordinary, strict = run(None if evidence == "historical_blank" else summary)
    assert ordinary["accepted"]
    assert strict["comparison"]["valid"]
    assert strict["accepted"] is (evidence == "complete")
    actual = strict["comparison"]["conditions"]["actual_workload"]
    assert actual["status"] == ("compatible" if evidence == "complete" else "unknown")
    assert actual["right"]
    if evidence == "historical_blank":
        assert actual["left"] is None
    else:
        case, = actual["left"].values()
        assert case["request_count"] == 3
        assert sum(v["count"] for v in case["variants"]) == 3
        missing = [v for v in case["variants"] if v["contract"] is None]
        if evidence == "mixed":
            assert missing == [{"contract": None, "count": 1, "proportion": "1/3"}]
        elif evidence == "null":
            assert missing == [{"contract": None, "count": 3, "proportion": "1"}]


@pytest.mark.parametrize(("recorded_count", "total", "proportion"), [
    ("nan", None, None), ("4", 7, "3/7"), ("0", 3, "1"),
    ("9.007199254740993e15", 9007199254740996, "1/3002399751580332"),
])
def test_audit_keeps_measured_requests_when_another_window_has_no_summary(
    experiments, capsys, recorded_count, total, proportion,
):
    left, right = (root / "result_all.csv" for root in experiments)
    for source in (left, right):
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, row = reader.fieldnames, next(reader)
        assert fields is not None
        second = {**row, "repeat_idx": "1"}
        if source == left:
            second.update(workload_contract="N/A", repeat_in_window=recorded_count)
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows([row, second])
        state_path = source.parent / "run_state.json"
        state = json.loads(state_path.read_text())
        state["options"]["repeat"] = 2
        state_path.write_text(json.dumps(state))

    code = audit.main([str(left), "--compare", str(right), "--json", "--require-comparable"])
    report = json.loads(capsys.readouterr().out)
    assert report["valid"]
    assert report["counts"]["formal_ok"] == 2
    actual = report["comparison"]["conditions"]["actual_workload"]
    case, = actual["left"].values()
    assert case["request_count"] == total
    measured, = (v for v in case["variants"] if v["contract"] is not None)
    assert measured["count"] == 3
    assert measured["proportion"] == proportion
    if total is None:
        assert case["observed_request_count"] == 3
        assert case["uncounted_windows"] == 1
    assert code == (0 if recorded_count == "0" else 1)


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
    expected = {
        **audit_result(expected_left),
        "comparison": compare_snapshots(load_comparison_snapshot(expected_left), load_comparison_snapshot(expected_right)),
        "accepted": True,
    }
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

    counts.clear()
    assert compare_results(left, right) == expected["comparison"]
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
    with pytest.raises(ValueError, match="result_manifest.json"):
        compare_results(left, right)
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


def change_after_qualification(monkeypatch, update):
    from acprof.analysis import comparison

    original = comparison.compare_snapshots
    applied = False

    def compare(*args, **kwargs):
        nonlocal applied
        verdict = original(*args, **kwargs)
        if not applied:
            applied = True
            update()
        return verdict

    monkeypatch.setattr(comparison, "compare_snapshots", compare)


@pytest.mark.parametrize(("changed_side", "change"), [
    ("left", "promote-legacy"), ("right", "retarget-alias"), ("left", "replace-invalid-quality"),
])
def test_strict_comparison_rechecks_user_sources_after_qualification(
    experiments, monkeypatch, capsys, changed_side, change,
):
    left, right = experiments
    root = left if changed_side == "left" else right
    source = root
    if change == "promote-legacy":
        target = root / "result_layers.json"
        staged = target.rename(root / "staged-manifest.json")
    elif change == "retarget-alias":
        original = root / "result_all.csv"
        source = target = root / "selected.csv"
        target.symlink_to(original.name)
        replacement = root / "replacement.csv"
        replacement.write_bytes(original.read_bytes())
        staged = root / "next-link"
        staged.symlink_to(replacement.name)
    else:
        target = root / "quality_checks.json"
        target.write_text("{")
        staged = root / "quality-next.json"
        staged.write_text(json.dumps({"schema_version": 1, "checks": []}))

    left, right = (source, right) if changed_side == "left" else (left, source)
    args = [str(left), "--compare", str(right), "--json", "--require-comparable"]
    assert audit.main(args) == 0
    before = json.loads(capsys.readouterr().out)
    assert before["comparison"]["status"] == "compatible"

    change_after_qualification(monkeypatch, lambda: staged.replace(target))
    assert audit.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    verdict = report["comparison"]
    assert not report["accepted"]
    assert not verdict["valid"]
    assert verdict["status"] == "incompatible"
    assert not verdict["native_baseline_eligible"]
    assert verdict["metric_comparability"]["latency_app_s"]["status"] == "not comparable"
    assert not verdict["experiments"][changed_side]["valid"]
    assert verdict["experiments"]["right" if changed_side == "left" else "left"]["valid"]
    assert report["valid"] is (changed_side == "right")
    assert verdict["experiments"][changed_side]["issues"]


def test_comparison_reports_all_late_changes_on_both_sides(experiments, monkeypatch, capsys):
    left, right = experiments
    targets = [left / "static_meta.json", left / "hardware_conditions.json", right / "result_all.csv"]

    def change_both_sides():
        for path in targets:
            path.write_bytes(path.read_bytes() + b"\n")

    change_after_qualification(monkeypatch, change_both_sides)
    assert audit.main([str(left), "--compare", str(targets[-1]), "--json", "--require-comparable"]) == 1
    report = json.loads(capsys.readouterr().out)
    sides = report["comparison"]["experiments"]
    for side, paths in (("left", targets[:2]), ("right", targets[2:])):
        assert not sides[side]["valid"]
        for path in paths:
            assert any(str(path) in message for message in sides[side]["issues"])
    assert sum(item["code"] == "changing_snapshot" for item in report["issues"]) == 2


def test_comparison_binds_rows_to_the_initial_target_when_alias_is_restored(experiments, monkeypatch, capsys):
    from acprof.analysis import audit as source_audit

    left, right = experiments
    original = left / "result_all.csv"
    alias = left / "selected.csv"
    alias.symlink_to(original.name)
    replacement = left / "substituted.csv"
    replacement.write_bytes(original.read_bytes() + b"\n")
    read = source_audit.read_result_csv_snapshot

    def read_retargeted(path, *args, **kwargs):
        if path == alias:
            alias.unlink()
            alias.symlink_to(replacement.name)
        return read(path, *args, **kwargs)

    def restore():
        alias.unlink()
        alias.symlink_to(original.name)

    monkeypatch.setattr(source_audit, "read_result_csv_snapshot", read_retargeted)
    change_after_qualification(monkeypatch, restore)
    args = [str(alias), "--compare", str(right), "--json", "--require-comparable"]
    assert audit.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["valid"]
    assert not report["comparison"]["experiments"]["left"]["valid"]
    assert report["comparison"]["experiments"]["right"]["valid"]


def test_compare_results_agrees_with_strict_cli_when_left_changes_during_qualification(
    experiments, monkeypatch, capsys,
):
    left, right = experiments
    target = left / "static_meta.json"
    original = target.read_bytes()

    def mutate():
        target.write_bytes(original + b"\n")

    with monkeypatch.context() as patch:
        change_after_qualification(patch, mutate)
        direct = compare_results(left, right)
    target.write_bytes(original)

    change_after_qualification(monkeypatch, mutate)
    assert audit.main([str(left), "--compare", str(right), "--json", "--require-comparable"]) == 1
    checked = json.loads(capsys.readouterr().out)["comparison"]
    assert checked["status"] == "incompatible"
    assert not checked["valid"]
    assert direct == checked
