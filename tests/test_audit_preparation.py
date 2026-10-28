"""Audit preparation stays per source while missing reasons stay per row."""
import csv
import json
from unittest.mock import Mock

import pytest

from acprof.analysis import audit
from acprof.config import CSV_FIELDS
from acprof.platform import Environment


@pytest.fixture
def audit_source(tmp_path):
    def write(rows, metadata, *, fields=CSV_FIELDS):
        path = tmp_path / "result_all.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            for index, changes in enumerate(rows):
                writer.writerow({
                    **dict.fromkeys(CSV_FIELDS, "nan"),
                    "runtime_kind": "docker", "cpu_cores": "1", "mem_cap_gb": "4",
                    "gpu_mode": "off", "input_scale": "1", "repeat_idx": str(index),
                    "warmup": "0", "status": "ok", "error": "", "task_param": "{}",
                    "environment_class": metadata.get("environment_class", "unknown"),
                    **changes,
                })
        (tmp_path / "static_meta.json").write_text(json.dumps(metadata))
        return path

    return write


@pytest.mark.parametrize("row_count", [1, 7])
@pytest.mark.parametrize("environment", ["native_linux", "wsl2", "unknown"])
def test_recorded_identity_is_resolved_once_per_audit(audit_source, monkeypatch, row_count, environment):
    identity = Mock(wraps=audit.recorded_identity)
    monkeypatch.setattr(audit, "recorded_identity", identity)
    path = audit_source([{} for _ in range(row_count)], Environment(environment, "Linux").metadata())

    report = audit.audit_result(path)

    assert report["valid"], report["issues"]
    assert report["environment_class"] == environment
    reason = "unavailable_on_wsl2" if environment == "wsl2" else "unavailable_unspecified"
    assert report["missing_metrics"]["cpu_energy_total_j"] == {reason: row_count}
    assert identity.call_count == 1


@pytest.mark.parametrize("row_count", [2, 11])
def test_missing_counter_allocations_do_not_grow_with_rows(audit_source, monkeypatch, row_count):
    counters = Mock(wraps=audit.Counter)
    monkeypatch.setattr(audit, "Counter", counters)
    path = audit_source([{} for _ in range(row_count)], {})

    report = audit.audit_result(path)

    assert report["valid"], report["issues"]
    assert report["missing_metrics"]
    assert all(sum(reasons.values()) == row_count for reasons in report["missing_metrics"].values())
    assert counters.call_count == len(report["missing_metrics"])


def test_audit_reloads_identity_when_source_metadata_changes(audit_source):
    native = Environment("native_linux", "Linux").metadata()
    wsl = Environment("wsl2", "Linux").metadata()
    conflicting = {**wsl, "comparability_class": "native_linux"}
    for metadata, environment, reason in (
        (native, "native_linux", "unavailable_unspecified"),
        (wsl, "wsl2", "unavailable_on_wsl2"),
        (conflicting, "unknown", "unavailable_unspecified"),
        ({}, "unknown", "unavailable_unspecified"),
        (native, "native_linux", "unavailable_unspecified"),
    ):
        report = audit.audit_result(audit_source([{}], metadata))
        assert report["environment_class"] == environment
        assert report["missing_metrics"]["cpu_energy_total_j"] == {reason: 1}


def test_missing_reasons_keep_row_evidence_and_formal_filter(audit_source):
    path = audit_source([
        {"latency_app_s": "0"},
        {"gpu_mode": "on", "compute_profile_error_torch_profiler_eager": "profiler failed"},
        {"warmup": "1"},
        {"status": "warn", "error": "quality warning"},
        {"status": "error", "error": "measurement failed"},
        {"latency_app_s": "inf"},
    ], {**Environment("native_linux", "Linux").metadata(), "compute_profile_tools": []},
        fields=[name for name in CSV_FIELDS if name != "input_pixels_per_request"])

    report = audit.audit_result(path)

    assert not report["valid"]
    assert [issue["code"] for issue in report["issues"]] == ["invalid_number"]
    assert report["counts"] == {"rows": 6, "formal_ok": 3, "warmup": 1, "warn": 1, "error": 1}
    missing = report["missing_metrics"]
    assert missing["latency_app_s"] == {"unavailable_unspecified": 1}
    assert missing["gpu_energy_eff_j"] == {"not_applicable": 2, "unavailable_unspecified": 1}
    assert missing["input_pixels_per_request"] == {"not_recorded": 3}
    assert missing["model_logical_mflop_per_request_torch_profiler_eager"] == {
        "tool_not_enabled": 2, "profiler_reported_error": 1,
    }
