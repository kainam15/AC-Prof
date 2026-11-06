"""A named input cannot hide recorded workload parameters in analysis."""
import csv
import json
import re

import pytest

from acprof.analysis.model import load_analysis
from acprof.analysis.resource_comparison import build_baseline_comparisons
from acprof.cli.report import main

PARAMETERS = [("input_scale", "64"), ("batch_size", "2"),
              ("input_units_per_request", "64"), ("task_param", "other")]


def row(**changes):
    return {"cpu_cores": "2", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "32",
            "batch_size": "1", "input_units_per_request": "32", "task_param": "fixture",
            "input_case": "named <input>", "warmup": "0", "repeat_idx": "0",
            "status": "ok", "latency_app_s": "2", **changes}


def source(tmp_path, rows):
    path = tmp_path / "result_all.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize(("field", "value"), PARAMETERS)
@pytest.mark.parametrize("run_id", ["", "recorded-run"])
def test_named_input_parameters_keep_window_means_separate(tmp_path, field, value, run_id):
    path = source(tmp_path, [row(run_id=run_id), row(run_id=run_id, repeat_idx="1", latency_app_s="20", **{field: value})])
    before = path.read_bytes()
    model = load_analysis([path])
    assert len(model.configs) == 2
    assert [config["metrics"]["latency_app_s"]["value"] for config in model.configs] == [2, 20]
    assert len({entry["config_id"] for entry in model.records}) == 2
    assert all(json.loads(config["input_case"])["name"] == "named <input>" for config in model.configs)
    assert path.read_bytes() == before


@pytest.mark.parametrize(("field", "value"), PARAMETERS)
def test_named_input_conflicts_never_become_resource_performance_ratios(tmp_path, field, value):
    path = source(tmp_path, [row(), row(cpu_cores="4", latency_app_s="20", **{field: value})])
    model = load_analysis([path])
    before, after = model.configs
    table = build_baseline_comparisons(model)
    pair = table["pairs"][before["config_id"]][after["config_id"]]
    verdict = table["qualifications"][pair["resource"]]
    assert verdict["status"] == "incompatible"
    assert "input_case" in verdict["reasons"]
    assert pair["metrics"] == {}


def test_equal_named_parameters_keep_numeric_normalization_and_resource_comparison(tmp_path):
    path = source(tmp_path, [row(), row(repeat_idx="1", latency_app_s="4", input_scale="3.2e1", batch_size="1.0"),
                             row(cpu_cores="4", latency_app_s="1")])
    model = load_analysis([path])
    assert len(model.configs) == 2
    before, after = model.configs
    assert before["metrics"]["latency_app_s"]["value"] == 3
    table = build_baseline_comparisons(model)
    pair = table["pairs"][before["config_id"]][after["config_id"]]
    assert table["qualifications"][pair["resource"]]["status"] == "descriptive"
    assert pair["metrics"]["latency_app_s"]["delta"] == -2


@pytest.mark.parametrize("name", ["", " NaN ", " null ", None])
def test_missing_input_names_retain_unnamed_parameter_identity(tmp_path, name):
    path = source(tmp_path, [row(input_case=name), row(input_case=name, input_scale="64", repeat_idx="1")])
    configs = load_analysis([path]).configs
    assert len(configs) == 2
    assert json.loads(configs[0]["input_case"]) == {
        "scale": 32, "type": "unknown", "task_param": "fixture", "batch_size": 1, "input_units": 32,
    }


def test_named_legacy_scales_can_restart_window_indices(tmp_path):
    path = source(tmp_path, [row(), row(input_scale="64", latency_app_s="20")])
    assert len(load_analysis([path]).configs) == 2


def test_cli_html_keeps_named_scales_and_statistics_separate(tmp_path, capsys):
    path = source(tmp_path, [row(), row(input_scale="64", repeat_idx="1", latency_app_s="20")])
    before = path.read_bytes()
    output = tmp_path / "report.html"
    assert main([str(path), "--output", str(output), "--resamples", "20"]) == 0
    lines = capsys.readouterr().out.splitlines()
    record = json.loads(next(line.removeprefix("ACPROF_REPORT ") for line in lines if line.startswith("ACPROF_REPORT ")))
    assert record["config_count"] == 2
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', output.read_text(), re.S)
    assert match is not None
    payload = json.loads(match.group(1))
    assert [config["metrics"]["latency_app_s"]["value"] for config in payload["configs"]] == [2, 20]
    groups = payload["window_statistics"][0]["groups"]
    assert sorted(group["mean"] for group in groups if group["metric"] == "latency_app_s") == [2, 20]
    assert path.read_bytes() == before
