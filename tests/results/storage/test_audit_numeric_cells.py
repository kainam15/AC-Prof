"""Canonical missing cells avoid normalization without changing audit evidence."""
import csv
import json

import pytest

from acprof.analysis import audit
from acprof.config import CSV_FIELDS
from acprof.platform import Environment
from acprof.results.csv import read_result_csv_snapshot

REPEATED_FIELDS = ("latency_app_s", "latency_p90_s", "throughput_samples_per_s")


def write_rows(root, values, *, environment="native_linux", **changes):
    path = root / "result_all.csv"
    rows = [{**dict.fromkeys(CSV_FIELDS, "nan"), "cpu_cores": "1", "mem_cap_gb": "4",
             "gpu_mode": "off", "input_scale": "1", "repeat_idx": str(index), "warmup": "0",
             "status": "ok", "error": "", "environment_class": environment,
             **dict.fromkeys(REPEATED_FIELDS, value), **changes}
            for index, value in enumerate(values)]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (root / "static_meta.json").write_text(json.dumps(Environment(environment).metadata()))
    return path


@pytest.mark.parametrize(("value", "normalizations"), [
    ("nan", 0), ("", 0), ("none", 0), ("null", 0), ("n/a", 0),
    (" NaN ", 8), ("7.2500", 8), ("not-a-number", 8),
])
def test_canonical_missing_cells_skip_repeated_string_normalization(tmp_path, value, normalizations):
    path = write_rows(tmp_path, [value] * 8)
    snapshot = read_result_csv_snapshot(path)
    normalized = []

    class ObservedText(str):
        def __str__(self):
            return self

        def strip(self, *args):
            normalized.append(self)
            return super().strip(*args)

    for row in snapshot.rows:
        row["latency_app_s"] = ObservedText(value)
    expected = audit.audit_result(path)
    assert audit.audit_result(path, result_snapshot=snapshot) == expected
    assert normalized == [value] * normalizations


@pytest.mark.parametrize(("value", "state"), [
    ("", "missing"), (" nan ", "missing"), ("NaN", "missing"), ("n/a", "missing"),
    (" NULL ", "missing"), ("None", "missing"), ("\t\n", "missing"),
    ("0", "present"), ("-0.0000", "present"), (" 7.2500 ", "present"),
    ("1e-300", "present"), ("1_000.5", "present"),
    ("inf", "invalid"), ("-Infinity", "invalid"), ("1e999", "invalid"),
    ("-nan", "invalid"), ("True", "invalid"), ("7,25", "invalid"),
])
def test_repeated_numeric_tokens_keep_missing_invalid_and_zero_semantics(tmp_path, value, state):
    path = write_rows(tmp_path, [value] * 3)
    report = audit.audit_result(path)
    for field in REPEATED_FIELDS:
        issues = [issue for issue in report["issues"] if issue.get("field") == field]
        if state == "invalid":
            assert [(issue["code"], issue["row"]) for issue in issues] == [
                ("invalid_number", 2), ("invalid_number", 3), ("invalid_number", 4)]
        else:
            assert issues == []
        if state == "missing":
            assert report["missing_metrics"][field] == {"unavailable_unspecified": 3}
        else:
            assert field not in report["missing_metrics"]
    assert report["valid"] == (state != "invalid")


def test_each_audit_reloads_changed_numeric_cells(tmp_path):
    path = write_rows(tmp_path, ["7.2500"] * 4)
    first = audit.audit_result(path)
    second = audit.audit_result(path)
    assert first == second
    write_rows(tmp_path, ["not-a-number"] * 4)
    assert not audit.audit_result(path)["valid"]


def test_missing_tokens_keep_per_row_device_and_status_reasons(tmp_path):
    path = write_rows(tmp_path, ["nan"] * 5)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["gpu_mode"] = "on"
    rows[2]["status"] = "warn"
    rows[3].update(status="error", error="workload failed")
    rows[4]["warmup"] = "1"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = audit.audit_result(path)
    assert report["counts"] == {"rows": 5, "formal_ok": 2, "warmup": 1, "warn": 1, "error": 1}
    assert report["missing_metrics"]["gpu_energy_eff_j"] == {"not_applicable": 1, "unavailable_unspecified": 1}
    assert report["missing_metrics"]["latency_app_s"] == {"unavailable_unspecified": 2}


def test_numeric_cells_keep_wsl_prohibitions_per_row(tmp_path):
    path = write_rows(tmp_path, ["7.2500"] * 4, environment="wsl2", vcpu_energy_eff_j="7.2500")
    report = audit.audit_result(path)
    prohibited = [issue for issue in report["issues"] if issue["code"] == "unsupported_wsl_metric"
                  and issue["field"] == "vcpu_energy_eff_j"]
    assert [issue["row"] for issue in prohibited] == [2, 3, 4, 5]
    assert not report["valid"]
