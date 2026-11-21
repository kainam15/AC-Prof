import csv
import json
from collections import Counter

import pytest

from acprof.analysis import comparison
from acprof.config import CSV_FIELDS
from tests.support.comparison_fixtures import ComparisonFixture


@pytest.fixture
def metric_source(tmp_path):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    path = fixture.left / "result_all.csv"
    with path.open(newline="") as stream:
        base = next(csv.DictReader(stream))

    windows = []
    for index, quality in enumerate(("partial", "complete")):
        evidence = {"schema_version": 1, "metrics": {
            "latency_app_s": {"quality": "complete", "sampling": {"interval_s": index + 1}},
            "process_io_read_bytes": {"quality": quality},
            "process_io_write_bytes": {"quality": "partial"},
            "latency_app_p95_s": {"quality": "complete"},
        }}
        windows.append({**base, "repeat_idx": str(index), "process_io_read_bytes": "32",
                        "process_io_write_bytes": "16", "latency_app_p95_s": "0" if index == 0 else "nan",
                        "resource_metric_quality": json.dumps(evidence)})
    windows.extend([
        {**base, "repeat_idx": "2", "status": "warn", "error": "fixture warning"},
        {**base, "repeat_idx": "3", "status": "error", "error": "fixture failure"},
        {**base, "repeat_idx": "4", "result_origin": "inferred_not_measured"},
        {**base, "repeat_idx": "0", "warmup": "1"},
    ])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(windows)
    fixture.change_json(fixture.left, "run_state.json", lambda state: state["options"].update(repeat=5, warmup=1))
    return path


def test_snapshot_parses_each_finite_measured_observation_once(metric_source, monkeypatch):
    calls: Counter[tuple[str, str, str]] = Counter()
    observe = comparison.metric_observation

    def counted(window, name):
        calls[window["warmup"], window["repeat_idx"], name] += 1
        return observe(window, name)

    monkeypatch.setattr(comparison, "metric_observation", counted)
    snapshot = comparison.load_comparison_snapshot(metric_source)
    assert snapshot["valid"], snapshot["issues"]
    common = {"cpu_cores", "mem_cap_gb", "input_scale", "warmup", "repeat_idx",
              "latency_app_s", "process_io_read_bytes", "process_io_write_bytes"}
    expected = Counter(("0", index, name) for index in ("0", "1") for name in common)
    expected["0", "0", "latency_app_p95_s"] = 1
    assert calls == expected


def test_snapshot_metric_evidence_tracks_quality_coverage_and_signatures(metric_source):
    snapshot = comparison.load_comparison_snapshot(metric_source)
    assert snapshot["valid"], snapshot["issues"]
    available, contracts = snapshot["metric_availability"], snapshot["metric_contracts"]
    expected = {
        # available, aggregate quality, finite in every measured window, unique signature
        "latency_app_s": (True, "complete", True, False),
        "process_io_read_bytes": (True, "incomplete", True, True),
        "process_io_write_bytes": (False, "incomplete", True, True),
        "latency_app_p95_s": (True, "complete", False, True),
        "cpu_cores": (True, "unknown", True, True),
        "gpu_temp_c": (False, "unknown", False, False),
    }
    for name, values in expected.items():
        contract = contracts[name]
        assert (available[name], contract["quality"], contract["all_windows"],
                contract["signature"] is not None) == values, name
    assert json.loads(contracts["process_io_read_bytes"]["signature"])["source"] == "psutil_observed_process_tree"
