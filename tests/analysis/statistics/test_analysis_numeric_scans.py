"""Numeric scans reuse plain CSV values without changing audit semantics."""
import csv
from collections import Counter
from unittest.mock import Mock

import pytest

from acprof.analysis import audit, comparison
from acprof.results.metrics import NUMERIC_FIELDS
from tests.support.comparison_fixtures import ComparisonFixture


@pytest.fixture
def numeric_csv(tmp_path):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    path = fixture.left / "result_all.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, base = reader.fieldnames, next(reader)

    def write(changes):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows({**base, "repeat_idx": str(index), **change} for index, change in enumerate(changes))
        fixture.change_json(fixture.left, "run_state.json", lambda state: state["options"].update(repeat=len(changes)))
        return path

    return write


@pytest.mark.parametrize(("gpu_mode", "energy"), [("off", 2.5), ("on", 6.)])
def test_formula_checks_parse_only_their_numeric_inputs(monkeypatch, gpu_mode, energy):
    row = {**dict.fromkeys(NUMERIC_FIELDS, "nan"), "gpu_mode": gpu_mode,
           "vcpu_energy_eff_j": "2.5", "gpu_energy_eff_j": "3.5",
           "packet_request_wire_bytes_per_request": "40", "packet_response_wire_bytes_per_request": "60",
           "packet_total_wire_bytes_per_request": "100", "packet_tcp_payload_bytes_per_request": "70",
           "workload_contract": "not a numeric field"}
    parse = Mock(wraps=audit.number)
    monkeypatch.setattr(audit, "number", parse)

    expected = audit._formula_checks(row)

    assert expected == {"container_attributed_energy_eff_j": energy,
                        "packet_total_wire_bytes_per_request": 100.,
                        "packet_protocol_overhead_bytes_per_request": 30., "packet_protocol_overhead_ratio": .3}
    assert parse.call_count <= 6


@pytest.mark.parametrize("row_count", [1, 12])
def test_availability_skips_known_missing_text_in_each_source_load(numeric_csv, monkeypatch, row_count):
    path = numeric_csv([{} for _ in range(row_count)])
    parsed = Counter()
    original = comparison.number

    def count_parse(value):
        parsed[value] += 1
        return original(value)

    monkeypatch.setattr(comparison, "number", count_parse)
    for _ in range(2):
        parsed.clear()
        snapshot = comparison.load_comparison_snapshot(path)
        assert snapshot["valid"], snapshot["issues"]
        assert snapshot["metric_availability"]["latency_app_s"]
        assert not snapshot["metric_availability"]["cpu_energy_total_j"]
        assert parsed["nan"] == 0


@pytest.mark.parametrize(("value", "available"), [
    ("nan", False), (" NaN ", False), ("", False), ("none", False), ("null", False),
    ("inf", False), ("-inf", False), ("1e999", False), ("invalid", False),
    ("0", True), ("-0", True), ("-0.125", True), (" 2.5e-3 ", True),
])
def test_availability_preserves_finite_number_rules(numeric_csv, value, available):
    snapshot = comparison.load_comparison_snapshot(numeric_csv([{"latency_app_s": value}]))
    assert snapshot["metric_availability"]["latency_app_s"] is available


def test_availability_retains_formal_filter_and_observes_changed_input(numeric_csv):
    changes = [{"warmup": "1", "latency_app_s": "10"},
               {"status": "warn", "latency_app_s": "20", "error": "warning"},
               {"status": "error", "latency_app_s": "30", "error": "failed"},
               {"latency_app_s": "nan"}]
    assert not comparison.load_comparison_snapshot(numeric_csv(changes))["metric_availability"]["latency_app_s"]
    changes[-1]["latency_app_s"] = "0"
    assert comparison.load_comparison_snapshot(numeric_csv(changes))["metric_availability"]["latency_app_s"]


@pytest.mark.parametrize(("row", "expected"), [
    ({}, {}),
    ({"vcpu_energy_eff_j": "-1", "gpu_mode": "off"}, {}),
    ({"vcpu_energy_eff_j": "0", "gpu_mode": "off"}, {"container_attributed_energy_eff_j": 0.}),
    ({"vcpu_energy_eff_j": "2", "gpu_mode": "on"}, {}),
    ({"vcpu_energy_eff_j": "inf", "gpu_mode": "off"}, {}),
    ({"packet_total_wire_bytes_per_request": "0", "packet_tcp_payload_bytes_per_request": "0"},
     {"packet_protocol_overhead_bytes_per_request": 0.}),
    ({"packet_total_wire_bytes_per_request": "10", "packet_tcp_payload_bytes_per_request": "12"},
     {"packet_protocol_overhead_bytes_per_request": -2., "packet_protocol_overhead_ratio": -.2}),
])
def test_formula_checks_keep_missing_negative_and_zero_rules(row, expected):
    assert audit._formula_checks(row) == expected


def test_availability_keeps_unhashable_and_mutable_numeric_values_uncached():
    class ChangingNumber:
        def __init__(self):
            self.calls = 0

        def __float__(self):
            self.calls += 1
            return float("nan") if self.calls == 1 else 2.

    value = ChangingNumber()
    rows = [{"latency_app_s": item} for item in ([], {}, value, value)]
    assert comparison._metric_availability(rows)["latency_app_s"]
    assert value.calls == 2


def test_availability_stops_after_first_finite_observation():
    class UnexpectedNumber:
        def __float__(self):
            raise AssertionError("later observations must not be parsed after availability is established")

    rows = [{"latency_app_s": "0"}, {"latency_app_s": UnexpectedNumber()}]
    assert comparison._metric_availability(rows)["latency_app_s"]


def test_repeated_invalid_values_keep_every_audit_diagnostic(numeric_csv):
    path = numeric_csv([{"latency_app_s": "inf", "cpu_energy_total_j": "inf"}] * 2)
    report = audit.audit_result(path)
    assert not report["valid"]
    assert {(item["row"], item["field"]) for item in report["issues"] if item["code"] == "invalid_number"} == {
        (2, "latency_app_s"), (2, "cpu_energy_total_j"), (3, "latency_app_s"), (3, "cpu_energy_total_j"),
    }
    assert not comparison.load_comparison_snapshot(path)["metric_availability"]["latency_app_s"]
