"""Prepare numeric columns per source without changing long-form evidence."""
import csv
import math
from collections import Counter

import pytest

from acprof.analysis import model


@pytest.fixture
def source(tmp_path):
    def write(changes, *, name="run", remove=(), reverse=False, spaced=False):
        path = tmp_path / name / "result_all.csv"
        path.parent.mkdir(exist_ok=True)
        rows = [{"cpu_cores": "2", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "32",
                 "warmup": "0", "repeat_idx": str(index), "status": "ok", "environment_class": "unknown",
                 "repeat_in_window": "4", "latency_app_s": "0.25", "cpu_energy_total_j": "2",
                 "gpu_energy_total_j": "3", **change} for index, change in enumerate(changes)]
        fields = [field for field in dict.fromkeys(key for row in rows for key in row) if field not in remove]
        if reverse:
            fields.reverse()
        names = {field: f" {field} " if spaced else field for field in fields}
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(names.values()))
            writer.writeheader()
            writer.writerows({names[field]: row.get(field, "") for field in fields} for row in rows)
        return path
    return write


@pytest.mark.parametrize("row_count", [2, 7])
def test_metric_kind_selection_is_bounded_by_source_count(source, monkeypatch, row_count):
    calls = []

    class ObservedMetric:
        def __init__(self, metric):
            self.metric = metric

        def __getattr__(self, name):
            if name == "kind":
                calls.append(name)
            return getattr(self.metric, name)

    registry = {name: ObservedMetric(metric) for name, metric in model.ANALYSIS_METRICS.items()}
    monkeypatch.setattr(model, "ANALYSIS_METRICS", registry)
    data = model.load_analysis([source([{}] * row_count)])
    assert len(data.raw_rows) == row_count
    assert len(calls) <= len(registry)


@pytest.mark.parametrize("row_count", [1, 7])
def test_projection_does_not_parse_known_missing_strings(source, monkeypatch, row_count):
    calls = Counter()
    original = model.number

    def record(value):
        if isinstance(value, str):
            calls[value] += 1
        return original(value)

    monkeypatch.setattr(model, "number", record)
    data = model.load_analysis([source([{"latency_app_s": "nan"}] * row_count)])
    assert all(row["values"]["latency_app_s"] is None for row in data.raw_rows)
    assert calls["nan"] == 0


@pytest.mark.parametrize("spaced", [False, True])
def test_projection_keeps_registry_order_and_original_values(source, spaced):
    data = model.load_analysis([source([{"latency_app_s": " 0.25 ", "unknown_value": " keep "}],
                                       reverse=True, spaced=spaced)])
    entry = data.raw_rows[0]
    assert list(entry["values"]) == [
        "cpu_cores", "mem_cap_gb", "input_scale", "repeat_idx", "warmup", "repeat_in_window",
        "latency_app_s", "gpu_energy_total_j", "cpu_energy_total_j",
        "observed_energy_per_request_j", "observed_energy_j",
    ]
    assert entry["values"]["latency_app_s"] == .25
    assert entry["row"]["latency_app_s"] == " 0.25 "
    assert entry["row"]["unknown_value"] == " keep "
    assert [record["metric"] for record in data.records] == list(entry["values"])


def test_distinct_sources_and_later_loads_prepare_their_own_columns(source):
    left = source([{}], name="left", remove=("gpu_energy_total_j",))
    right = source([{"throughput_samples_per_s": "20"}], name="right", remove=("latency_app_s", "cpu_energy_total_j"))
    data = model.load_analysis([left, right])
    a, b = (row["values"] for row in data.raw_rows)
    assert "latency_app_s" in a
    assert "latency_app_s" not in b
    assert "throughput_samples_per_s" not in a
    assert b["throughput_samples_per_s"] == 20
    assert a["observed_energy_j"] == 8
    assert b["observed_energy_j"] is None
    source([{"cpu_ipc": "1.5"}], name="left", remove=("latency_app_s", "gpu_energy_total_j"))
    later = model.load_analysis([left]).raw_rows[0]["values"]
    assert "latency_app_s" not in later
    assert later["cpu_ipc"] == 1.5
    assert a["latency_app_s"] == .25
    assert "cpu_ipc" not in a


@pytest.mark.parametrize(("changes", "per_request", "total"), [
    ({}, 2., 8.),
    ({"gpu_mode": "on"}, 5., 20.),
    ({"cpu_energy_total_j": "0"}, 0., 0.),
    ({"cpu_energy_total_j": "-1"}, None, None),
    ({"gpu_mode": "on", "gpu_energy_total_j": "-1"}, None, None),
    ({"gpu_energy_total_j": "-1"}, 2., 8.),
    ({"cpu_energy_total_j": "nan"}, None, None),
    ({"gpu_mode": "on", "gpu_energy_total_j": "nan"}, None, None),
    ({"repeat_in_window": "0"}, 2., None),
    ({"repeat_in_window": "-1"}, 2., None),
    ({"repeat_in_window": "nan"}, 2., None),
    ({"gpu_mode": "on", "cpu_energy_total_j": "1e308", "gpu_energy_total_j": "1e308"}, None, None),
    ({"cpu_energy_total_j": "1e308", "repeat_in_window": "4"}, 1e308, None),
    ({"observed_energy_per_request_j": "999", "observed_energy_j": "999"}, 2., 8.),
])
def test_derived_energy_preserves_existing_arithmetic_and_missing_rules(source, changes, per_request, total):
    values = model.load_analysis([source([changes])]).raw_rows[0]["values"]
    assert values["observed_energy_per_request_j"] == per_request
    assert values["observed_energy_j"] == total


@pytest.mark.parametrize(("value", "expected"), [
    ("nan", None), (" NaN ", None), ("none", None), ("null", None), ("n/a", None), ("", None),
    ("inf", None), ("1e999", None), ("invalid", None), ("-0", -0.), ("-2.5", -2.5), (" 1e-3 ", .001),
])
def test_projection_preserves_numeric_and_missing_tokens(source, value, expected):
    values = model.load_analysis([source([{"latency_app_s": value}])]).raw_rows[0]["values"]
    assert values["latency_app_s"] == expected


def test_projection_preserves_negative_zero_in_observations_and_derived_energy(source):
    values = model.load_analysis([source([{"latency_app_s": "-0", "cpu_energy_total_j": "-0"}])]).raw_rows[0]["values"]
    for field in ("latency_app_s", "cpu_energy_total_j", "observed_energy_per_request_j", "observed_energy_j"):
        assert values[field] == 0
        assert math.copysign(1, values[field]) == -1
