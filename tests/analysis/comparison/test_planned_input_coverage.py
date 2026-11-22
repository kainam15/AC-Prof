"""Whole-run input comparisons need materialized payloads for every observed scale."""
import csv

import pytest

from acprof.analysis.comparison import compare_results, load_comparison_snapshot
from acprof.analysis.conditions import PreparedComparison, compare_profiles
from acprof.analysis.independent_comparison import compare_experiments
from tests.support.comparison_fixtures import ComparisonFixture


def materialized_runs(tmp_path, windows, planned, *, change_payload=False, right_extra=()):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    for source in (fixture.left, fixture.right):
        # Keep the run's window plan valid independently of its materialized input entries.
        scales = sorted({window["input_scale"] for window in windows})
        warmup = int(any(window.get("warmup") == 1 for window in windows))
        fixture.change_json(source, "run_state.json", lambda state: (
            state["runtime"]["planned"].update(scales=scales), state["options"].update(warmup=warmup),
        ))
        entry_scales = [*planned, *right_extra] if source == fixture.right else planned
        entries = [{"input_scale": scale, "payload": {"features": [scale]}} for scale in entry_scales]
        if change_payload and source == fixture.right:
            entries[0]["payload"] = {"features": [99]}
        fixture.change_json(source, "input_scale_plan.json", lambda plan: plan.update(entries=entries))
        path = source / "result_all.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, template = reader.fieldnames, next(reader)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for window in windows:
                writer.writerow({**template, **window})
        fixture.refresh(source)
    return fixture.left, fixture.right


@pytest.mark.parametrize(("scale", "planned", "expected"), [
    (2, 2, "compatible"),
    (9007199254740993, 9007199254740992, "unknown"),
], ids=["covered", "identical-plans-without-the-observed-scale"])
def test_independent_comparison_requires_a_materialized_input_for_observed_windows(tmp_path, scale, planned, expected):
    left, right = materialized_runs(tmp_path, [{"input_scale": scale}], [planned])
    compared = compare_results(left, right)
    independent = compare_experiments([left], [right], metrics=["latency_app_s"], resamples=10)

    assert compared["valid"]
    assert compared["status"] == independent["status"] == expected
    condition = compared["conditions"]["planned_inputs"]
    assert condition["status"] == expected
    group, = independent["groups"]
    if expected == "unknown":
        assert condition["unmatched_scales"] == {"left": [str(scale)], "right": [str(scale)]}
        assert group["difference"] is None
    else:
        assert "unmatched_scales" not in condition
        assert group["difference"] == 0


@pytest.mark.parametrize(("different", "status"), [(False, "unknown"), (True, "incompatible")])
def test_mixed_plan_coverage_preserves_the_matched_sibling(tmp_path, different, status):
    sources = materialized_runs(tmp_path, [{"input_scale": 2}, {"input_scale": 4}], [2], change_payload=different)
    compared = compare_results(*sources)
    assert compared["valid"]
    assert compared["status"] == status
    condition = compared["conditions"]["planned_inputs"]
    assert condition["unmatched_scales"] == {"left": ["4"], "right": ["4"]}
    assert condition["status"] == status
    prepared = [PreparedComparison(load_comparison_snapshot(source)) for source in sources]
    for scale, expected in (("2", "incompatible" if different else "compatible"), ("4", "unknown")):
        case = ("1", "4", "off", scale)
        assert compare_profiles(*(source.profile(case=case) for source in prepared))["status"] == expected


def test_materialized_coverage_counts_only_observed_formal_ok_or_warn_windows(tmp_path):
    windows = [
        {"input_scale": 2},
        {"input_scale": 4, "status": " Warn "},
        {"input_scale": 6, "status": "error", "error": "not executed"},
        {"input_scale": 8, "result_origin": "inferred_not_measured"},
    ]
    warmups = [{"input_scale": window["input_scale"], "warmup": 1} for window in windows]
    left, right = materialized_runs(tmp_path, [*warmups, *windows], [2], right_extra=(4,))
    compared = compare_results(left, right)

    assert compared["valid"]
    assert compared["conditions"]["planned_inputs"]["unmatched_scales"] == {"left": ["4"], "right": []}
