"""Observed package and GPU energy keeps component provenance through analysis."""
import csv
import json

import pytest

from acprof.analysis.model import load_analysis
from acprof.analysis.resource_comparison import build_baseline_comparisons
from tests.analysis.reporting.test_report_statistics import row, write_rows


@pytest.fixture
def weighted_energy_csv(tmp_path):
    # Historical rows without collector evidence still describe their recorded CPU energy.
    windows = {
        1: [("1.5", 10)], 2: [("1.0000000000000001", 10)], 3: [(" 2e0 ", 10)],
        4: [("0", 10)], 5: [("", 10)], 6: [("-1", 10)], 7: [("1", 2), ("1.5", 10)],
        8: [("2.00", 2), ("4", 5)], 9: [(str(2**53 + 1), 1)], 10: [(str(2**53 + 2), 1)],
        11: [(str(2**1023), .25), (str(2**1023), .25)],
    }
    path = tmp_path / "weights.csv"
    write_rows(path, [row(i, 3, cpu_cores=case, repeat_in_window=count, cpu_energy_total_j=energy)
                      for case, values in windows.items() for i, (count, energy) in enumerate(values)])
    return path


def assert_request_energy_summaries(configs):
    expected = {
        # request-weighted mean, total, contributing windows, missing windows
        1: (None, None, 0, 1), 2: (None, None, 0, 1), 3: (10, 20, 1, 0),
        4: (None, None, 0, 1), 5: (None, None, 0, 1), 6: (None, None, 0, 1),
        7: (None, None, 1, 1), 8: (4, 24, 2, 0), 9: (None, None, 0, 1),
        10: (1, 2**53 + 2, 1, 0), 11: (None, float(2**1022), 2, 0),
    }
    assert {config["cpu"] for config in configs} == expected.keys()
    for config in configs:
        weighted, total, n, missing = expected[config["cpu"]]
        metrics = config["metrics"]
        for name, value in (("observed_energy_per_request_j", weighted), ("observed_energy_j", total)):
            summary = metrics[name]
            assert (summary["value"], summary["n"], summary["missing"]) == (value, n, missing), (config["cpu"], name)
        assert metrics["latency_app_s"]["n"] == n + missing
        assert metrics["cpu_energy_total_j"]["n"] == n + missing


def test_energy_aggregates_require_positive_integral_request_counts(weighted_energy_csv):
    model = load_analysis([weighted_energy_csv])
    assert_request_energy_summaries(model.configs)
    for entry in model.raw_rows:
        assert entry["values"]["observed_energy_per_request_j"] == entry["values"]["cpu_energy_total_j"]


@pytest.fixture
def energy_csv(tmp_path):
    cpu, gpu = "cpu_energy_total_j", "gpu_energy_total_j"
    complete = {"quality": "complete"}
    partial = {"quality": "partial", "reason": "incomplete_read"}
    fast = {"quality": "complete", "sampling": {"sample_hz": 10, "policy": "burst"}}
    same = {"quality": "complete", "sampling": {"policy": "burst", "sample_hz": 10}}
    slow = {"quality": "complete", "sampling": {"sample_hz": 5, "policy": "burst"}}

    def row(case, mode, index=0, *, cpu_obs=None, gpu_obs=None, **changes):
        metrics = {key: obs for key, obs in [(cpu, cpu_obs), (gpu, gpu_obs)] if obs is not None}
        return {"cpu_cores": case, "mem_cap_gb": 4, "gpu_mode": mode, "input_scale": 32,
                "repeat_idx": index, "warmup": 0, "status": "ok", "repeat_in_window": 4,
                cpu: 3.5, gpu: 2, "gpu_energy_source": "",
                "resource_metric_quality": json.dumps({"schema_version": 1, "metrics": metrics}) if metrics else "",
                **changes}

    def collectors(gpu_hz, *, failed_reader=False):
        state = {"status": "available", "sampling": {"sample_hz": gpu_hz}}
        if failed_reader:
            state["reader_errors"] = {gpu: {"status": "error", "detail": "incomplete_read"}}
        return json.dumps({"schema_version": 1, "metrics": {}, "collectors": {
            "cpu_energy": {"status": "available"}, "gpu_power": state}})

    # Case numbers are resource coordinates within a single input and source.
    rows = [
        row(1, "off", cpu_obs=partial),
        row(2, "off", cpu_obs=complete, gpu_obs=partial),
        row(3, "on", cpu_obs=partial, gpu_obs=complete),
        row(4, "on", cpu_obs=complete, gpu_obs=partial),
        row(4, "on", 1, cpu_obs=complete, gpu_obs=complete),
        row(5, "on", cpu_obs=fast, gpu_obs=complete),
        row(5, "on", 1, cpu_obs=same, gpu_obs=complete, cpu_energy_total_j=2, repeat_in_window=2),
        row(5, "on", 2, cpu_obs=slow, gpu_obs=complete, status="error"),
        row(6, "on", cpu_obs=fast, gpu_obs=complete),
        row(6, "on", 1, cpu_obs=slow, gpu_obs=complete),
        row(7, "on", gpu_energy_source="nvml_total_energy"),
        row(7, "on", 1, gpu_energy_source="power_integration"),
        row(8, "on", gpu_energy_source="nvml_total_energy"),
        row(9, "on", gpu_energy_total_j=""),
        row(10, "off", cpu_obs=fast), row(11, "off", cpu_obs=slow),
        row(12, "off", cpu_obs=same, cpu_energy_total_j=1.5),
        row(13, "on", resource_metric_quality=collectors(5), gpu_energy_source="power_integration"),
        row(13, "on", 1, resource_metric_quality=collectors(10), gpu_energy_source="power_integration"),
        row(14, "on", resource_metric_quality=collectors(10, failed_reader=True)),
    ]
    path = tmp_path / "energy.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_derived_energy_keeps_component_evidence_and_contracts(energy_csv):
    model = load_analysis([energy_csv])
    configs = {c["cpu"]: c for c in model.configs}
    per_request, total = "observed_energy_per_request_j", "observed_energy_j"
    expected = {
        # per request, total, contributing windows, missing successful windows
        1: (None, None, 0, 1), 2: (3.5, 14, 1, 0), 3: (None, None, 0, 1),
        4: (None, None, 1, 1), 5: (5, 30, 2, 0), 6: (None, None, 2, 0),
        7: (None, None, 2, 0), 8: (5.5, 22, 1, 0), 9: (None, None, 0, 1),
        10: (3.5, 14, 1, 0), 11: (3.5, 14, 1, 0), 12: (1.5, 6, 1, 0),
        13: (None, None, 2, 0), 14: (None, None, 0, 1),
    }
    for case, (weighted, summed, n, missing) in expected.items():
        metrics = configs[case]["metrics"]
        assert metrics[per_request] == {"value": weighted, "aggregation": "request_weighted", "n": n, "missing": missing}
        assert metrics[total] == {"value": summed, "aggregation": "sum", "n": n, "missing": missing}

    first = {r["metric"]: r for r in model.records if r["cpu"] == 1}
    assert configs[1]["metrics"]["cpu_energy_total_j"]["value"] is None
    assert first["cpu_energy_total_j"]["value"] == 3.5
    assert first["cpu_energy_total_j"]["observation"]["reason"] == "incomplete_read"
    assert first[total]["value"] is None
    assert first[total]["observation"]["quality"] == "partial"
    assert configs[8]["metric_observations"][total][0]["quality"] == "unknown"

    comparisons = build_baseline_comparisons(model)["pairs"][configs[10]["config_id"]]
    for name, delta in [(per_request, -2), (total, -8)]:
        assert name not in comparisons[configs[11]["config_id"]]["metrics"]
        assert comparisons[configs[11]["config_id"]]["metric_reasons"][name] == "different_source_scope_window_or_sampling"
        assert comparisons[configs[12]["config_id"]]["metrics"][name]["delta"] == delta


def test_energy_evidence_distinguishes_missing_input_and_unknown_device(tmp_path):
    cpu, gpu = "cpu_energy_total_j", "gpu_energy_total_j"
    per_request, total = "observed_energy_per_request_j", "observed_energy_j"
    evidence = json.dumps({"schema_version": 1, "metrics": {cpu: {"quality": "complete"}, gpu: {"quality": "complete"}}})
    common = {cpu: "3", gpu: "2", "repeat_in_window": "4", "resource_metric_quality": evidence}
    path = tmp_path / "windows.csv"
    write_rows(path, [
        row(0, 1, **{**common, gpu: "", "gpu_mode": "on"}),
        row(0, 1, **{**common, "repeat_in_window": ""}),
        row(0, 1, **{**common, "gpu_mode": "", "device": "unknown"}),
    ])
    model = load_analysis([path])
    records = {(r["device"], r["metric"]): r for r in model.records}
    assert records["CPU", per_request]["value"] == 3
    assert records["CPU", total]["value"] is None
    assert records["CPU", total]["observation"]["reason"] == "missing_window_value"
    for name in (per_request, total):
        gpu_reading = records["GPU", name]
        assert gpu_reading["value"] is None
        assert gpu_reading["observation"]["quality"] == "unavailable"
        assert gpu_reading["observation"]["reason"] == f"{gpu}: missing_window_value"
        unknown = records["unknown", name]
        assert unknown["value"] is None
        assert unknown["observation"]["quality"] == "unknown"
        assert unknown["observation"]["reason"] == "unknown_device"
