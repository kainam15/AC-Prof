"""The independent report must identify each group after JSON or CSV export."""
import csv
import json

import pytest

from acprof.analysis.export import export_analysis
from acprof.analysis.independent_comparison import compare_experiments
from tests.support.comparison_fixtures import ComparisonFixture


def runs_with_cases(tmp_path, left, right):
    f = ComparisonFixture()
    f.build(tmp_path)
    for directory, cases in ((f.left, left), (f.right, right)):
        path = directory / "result_all.csv"
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            fields, template = reader.fieldnames, next(reader)
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fields)
            writer.writeheader()
            for cpu, memory, scale, latency in cases:
                writer.writerow({**template, "cpu_cores": cpu, "mem_cap_gb": memory,
                                 "input_scale": scale, "latency_app_s": latency})
        f.refresh(directory)
    return [f.left], [f.right]


def compare(paths, purpose="same-hardware"):
    return compare_experiments(*paths, metrics=["latency_app_s"], resamples=10, purpose=purpose)


def test_json_and_csv_keep_sibling_group_means_bound_to_the_exact_scale(tmp_path):
    near = "2.0000000000000001"
    cases = [(1, 4, "2", 1), (1, 4, near, 7)]
    report = compare(runs_with_cases(tmp_path, cases, cases))
    groups = json.loads(json.dumps(report, allow_nan=False))["groups"]
    assert [(g.get("exact_coordinates", {}), g["left"]["mean"]) for g in groups] == [
        ({}, 1), ({"input_scale": near}, 7),
    ]
    assert report["coordinate_identity"] == "recorded_decimal_v1"
    path = tmp_path / "export.csv"
    export_analysis(report, path, format="csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["exact_coordinates.input_scale"] for row in rows] == ["", near]
    assert all(row["analysis.coordinate_identity"] == report["coordinate_identity"] for row in rows)


def test_resource_scaling_exports_both_sides_and_the_run_native_coordinates(tmp_path):
    cpu = "1.0000000000000001"
    memory = "4.0000000000000001"
    scale = "2.0000000000000001"
    paths = runs_with_cases(tmp_path, [(cpu, 4, scale, 1)], [(1, memory, scale, 3)])
    group, = compare(paths, "resource-scaling")["groups"]
    assert group["exact_coordinates"] == {"cpu_cores": cpu, "input_scale": scale}
    for side, residual in (("left", {"cpu_cores": cpu}), ("right", {"mem_cap_gb": memory})):
        coords = group["resource_coordinates"][side]
        assert coords["exact_coordinates"] == residual
        run, = group[side]["runs"]
        assert run["resource_coordinates"] == coords


@pytest.mark.parametrize(("scale", "display"), [("1E+400", None), ("1E-400", 0.0)])
def test_out_of_float_range_scale_retains_a_finite_json_report(tmp_path, scale, display):
    cases = [(1, 4, scale, 1)]
    group, = compare(runs_with_cases(tmp_path, cases, cases))["groups"]
    assert group["input_scale"] == display
    assert group["exact_coordinates"] == {"input_scale": scale}
    json.dumps(group, allow_nan=False)
