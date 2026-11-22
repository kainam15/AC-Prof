"""An exact measured scale selects only its own frozen input entries."""
import csv
import json
from decimal import Decimal

import pytest

from acprof.analysis.comparison import audit_and_compare_results
from acprof.analysis.model import load_analysis
from acprof.plotting.report import report_payload
from tests.support.comparison_fixtures import ComparisonFixture

NEAR = "2.0000000000000001"


@pytest.fixture
def runs(tmp_path):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    return fixture


def set_scales(fixture, scales, *, recorded_matrix=False):
    for latency, directory in enumerate((fixture.left, fixture.right), 1):
        path = directory / "result_all.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, template = reader.fieldnames, next(reader)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for scale in scales:
                summary = json.loads(template["workload_contract"])
                summary["variants"][0]["contract"]["input"]["planned_scale"] = json.loads(scale)
                writer.writerow({**template, "input_scale": scale, "latency_app_s": latency,
                                 "workload_contract": json.dumps(summary)})
        if recorded_matrix:
            fixture.change_json(directory, "run_state.json", lambda state: state["runtime"]["planned"].update(
                scales=[int(scale) for scale in scales]))
        else:
            fixture.change_json(directory, "run_state.json", lambda state: state.pop("runtime"))
        fixture.refresh(directory)


def write_plan(directory, entries):
    # Insert real JSON tokens so a Python float cannot round the fixture while writing it.
    path = directory / "input_scale_plan.json"
    plan = json.loads(path.read_text())
    del plan["entries"]
    raw = ['{"input_scale":' + literal + ',"payload":' + json.dumps({"features": [[value]]}) + '}'
           for literal, value in entries]
    path.write_text(json.dumps(plan)[:-1] + ',"entries":[' + ",".join(raw) + "]}")


def report_pairs(fixture):
    report = report_payload(load_analysis([fixture.left, fixture.right]),
                            statistics_options={"metrics": ["latency_app_s"], "resamples": 10})
    table = report["baseline_comparisons"]
    configs = {(c["run_id"], Decimal(str(json.loads(c["input_case"])["scale"]))): c for c in report["configs"]}
    pairs = {}
    for (run_id, scale), config in configs.items():
        if run_id != "torch":
            continue
        target = configs["onnxruntime", scale]
        pair = table["pairs"][config["config_id"]][target["config_id"]]
        verdict = table["qualifications"][pair["policies"]["same-hardware"]]
        pairs[scale] = (verdict["status"], verdict["reasons"], pair["metrics"].get("latency_app_s", {}).get("delta"))
    return pairs


def test_report_keeps_independent_payloads_at_adjacent_integer_scales(runs):
    first, second = "9007199254740992", "9007199254740993"
    set_scales(runs, [first, second], recorded_matrix=True)
    write_plan(runs.left, [(first, 1), (second, 2)])
    write_plan(runs.right, [(first, 1), (second, 3)])
    assert report_pairs(runs) == {Decimal(first): ("compatible", [], 1.0),
                                  Decimal(second): ("incompatible", ["planned_inputs"], None)}


def test_report_withholds_difference_when_only_a_neighbor_was_planned(runs):
    set_scales(runs, ["2", NEAR])
    for directory in (runs.left, runs.right):
        write_plan(directory, [("2", 1)])
    assert report_pairs(runs) == {Decimal("2"): ("compatible", [], 1.0),
                                  Decimal(NEAR): ("unknown", ["planned_inputs"], None)}


@pytest.mark.parametrize(("right_entries", "status"), [
    ([(NEAR, 1), (json.dumps(NEAR), 2)], "compatible"),
    ([(json.dumps(NEAR), 1), (json.dumps(NEAR), 2)], "incompatible"),
    ([(json.dumps(NEAR), 2), (NEAR, 1)], "incompatible"),
])
def test_report_preserves_scale_json_types_and_selected_entry_order(runs, right_entries, status):
    set_scales(runs, [NEAR])
    write_plan(runs.left, [("2", 8), (NEAR, 1), (json.dumps(NEAR), 2)])
    write_plan(runs.right, [("2", 9), *right_entries])
    assert report_pairs(runs)[Decimal(NEAR)][0] == status


def test_whole_run_audit_distinguishes_adjacent_bare_json_plan_values(runs):
    # The active CSV coordinate is the same on both sides. Only frozen plans differ.
    write_plan(runs.left, [("2", 1), (NEAR, 2)])
    write_plan(runs.right, [("2", 1), ("2.0000000000000002", 2)])
    report = audit_and_compare_results(runs.left, runs.right)["comparison"]
    assert report["valid"]
    assert report["status"] == "incompatible"
    assert report["comparison_reasons"] == ["planned_inputs"]
