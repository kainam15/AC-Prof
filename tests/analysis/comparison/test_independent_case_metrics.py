"""An independent comparison qualifies each requested metric at its own configuration."""
import csv
import json

import pytest

from acprof.analysis.independent_comparison import compare_experiments
from tests.support.comparison_fixtures import ComparisonFixture
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture

MEMORY = "container_mem_usage_avg_bytes"


@pytest.fixture
def compare_memory(tmp_path):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)

    def compare(*, second_right="complete", left_hz=(10, 10), right_hz=(10, 10), different_workload=False):
        for path, values, frequencies in (
            (fixture.left, (1000, 2000), left_hz), (fixture.right, (1200, 2400), right_hz),
        ):
            fixture.change_json(path, "run_state.json", lambda state: state["runtime"]["planned"].update(scales=[2, 4]))
            fixture.change_json(path, "input_scale_plan.json", lambda plan: plan.update(entries=[
                {"input_scale": scale, "payload": {"features": [[1, 2]] * scale, "batch_size": 1}} for scale in (2, 4)]))
            csv_path = path / "result_all.csv"
            with csv_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, template = reader.fieldnames, next(reader)
            assert fields is not None
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                for scale, value, hz in zip((2, 4), values, frequencies, strict=True):
                    quality = second_right if scale == 4 and path == fixture.right else "complete"
                    collector = {"status": "available", "sampling": {"sample_hz": hz}}
                    if quality in {"partial", "unavailable"}:
                        collector["reader_errors"] = {MEMORY: {"status": "unavailable", "detail": "memory.current read failed"}}
                    contract = json.loads(template["workload_contract"])
                    observed = contract["variants"][0]["contract"]
                    observed["input"].update(rows=scale, planned_scale=scale, actual_scale=scale)
                    observed["output"].update(count=scale, shape=[scale, 1])
                    if different_workload and path == fixture.right:
                        observed["input"]["feature_dim"] = 3
                    writer.writerow({**template, "runtime_kind": "docker", "input_scale": scale,
                                     MEMORY: value if quality != "unavailable" else "nan",
                                     "workload_contract": json.dumps(contract),
                                     "resource_metric_quality": json.dumps({"schema_version": 1, "metrics": {},
                                                                           "collectors": {"cgroup": collector}})})
            fixture.refresh(path)
        report = compare_experiments([fixture.left], [fixture.right], metrics=[MEMORY], resamples=10)
        return report, {item["input_scale"]: item for item in report["groups"]}

    return compare


@pytest.mark.parametrize("second_right", ["partial", "unavailable"])
def test_incomplete_cgroup_reader_is_local_to_its_input_scale(compare_memory, second_right):
    report, cases = compare_memory(second_right=second_right)
    assert report["status"] == "compatible"
    clean, affected = cases[2], cases[4]
    assert (clean["comparability"], clean["difference"]) == ("comparable", 200)
    assert (affected["comparability"], affected["difference"]) == ("not comparable", None)
    assert affected["right"]["missing_windows"] == 1
    if second_right == "partial":
        assert any(check["metric_comparability"][MEMORY]["reason"] == "incomplete_metric_quality"
                   for check in report["condition_checks"])


@pytest.mark.parametrize(("right_second_hz", "expected_difference"), [(5, 400), (2, None)])
def test_contract_signatures_are_compared_at_the_same_coordinate(compare_memory, right_second_hz, expected_difference):
    _, cases = compare_memory(left_hz=(10, 5), right_hz=(10, right_second_hz))
    assert cases[2]["difference"] == 200
    assert cases[4]["difference"] == expected_difference
    assert cases[4]["comparability"] == ("comparable" if expected_difference is not None else "not comparable")


def test_global_comparison_conditions_still_block_each_metric_group(compare_memory):
    report, cases = compare_memory(different_workload=True)
    assert report["status"] == "incompatible"
    assert all(item["comparability"] == "not comparable" and item["difference"] is None for item in cases.values())


@pytest.mark.parametrize(("reading", "evidence", "right_mean", "difference"), [
    (24, {"quality": "complete"}, 18, 3),
    (24, {"quality": "partial", "reason": "incomplete_requests"}, 12, None),
    (24, {"quality": "complete", "sampling": {"interval_s": 1}}, None, None),
    ("nan", {"quality": "unavailable"}, 12, -3),
], ids=["matching-legacy", "finite-partial", "mixed-contracts", "missing-window"])
def test_window_quality_within_a_case_keeps_usable_descriptive_values(tmp_path, request, reading, evidence,
                                                                     right_mean, difference):
    runs = IndependentComparisonFixture()
    runs.build(request, tmp_path)
    left = runs.replicate("left", 0, [10, 20])
    right = runs.replicate("right", 0, [12, 24])
    path = right / "result_all.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    assert fields is not None
    rows[1].update(latency_app_s=reading, resource_metric_quality=json.dumps({
        "schema_version": 1, "metrics": {"latency_app_s": evidence}}))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)

    group, = runs.compare([left], [right])["groups"]
    assert (group["left"]["mean"], group["right"]["mean"]) == (15, right_mean)
    assert group["difference"] == difference
    assert group["comparability"] == ("not comparable" if difference is None else "comparable")
