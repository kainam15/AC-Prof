import json
from subprocess import CompletedProcess

import pytest

from acprof.experiments.configuration import RunConfig
from scripts import check_hardware


def measured_row(gpu_mode="off", **values):
    return {"warmup": "0", "status": "ok", "gpu_mode": gpu_mode, "packet_status": "ok",
            "latency_s": ".1", "latency_app_s": ".12", "cpu_energy_total_j": "2",
            "cpu_instructions_per_request": "2048", "cpu_perf_running_pct": "97.5",
            "gpu_energy_total_j": "3" if gpu_mode == "on" else "nan",
            "resource_metric_quality": json.dumps({"schema_version": 1, "metrics": {}, "collectors": {
                name: {"status": "available"} for name in
                ("cpu_energy", "client_http", "packet_latency", "cpu_instructions", "gpu_power")}}), **values}


def check_rows(tmp_path, monkeypatch, *rows):
    config = RunConfig(model="fixture/model", output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(check_hardware.subprocess, "run", lambda *args, **kwargs: CompletedProcess([], 0))
    warmup = sum(row["warmup"] == "1" for row in rows)
    monkeypatch.setattr(check_hardware, "audit_result", lambda _: {
        "valid": True, "completion": "complete", "issues": [], "quality_status": "unknown",
        "quality_reasons": ["cpu_idle_baseline_unstable"],
        "counts": {"formal_ok": len(rows) - warmup, "rows": len(rows), "warmup": warmup}})
    monkeypatch.setattr(check_hardware, "read_result_csv", lambda _: ([], rows))
    report = check_hardware.run_config(config, tmp_path)
    assert json.loads((tmp_path / "audit.json").read_text()) == report
    return report


def test_complete_metrics_do_not_require_stable_idle_or_gpu_energy_on_cpu(tmp_path, monkeypatch):
    report = check_rows(tmp_path, monkeypatch, measured_row(warmup="1", resource_metric_quality=""),
                        measured_row(), measured_row("on"))
    assert report["successful"]
    assert report["validated_gpu_modes"] == ["off", "on"]
    assert report["quality_reasons"] == ["cpu_idle_baseline_unstable"]
    assert report["required_metrics_missing"] == report["required_metrics_incomplete"] == []


@pytest.mark.parametrize(("field", "changes", "quality", "reason"), [
    ("cpu_instructions_per_request", {"cpu_perf_running_pct": "nan"}, "partial", "unknown_pmu_running_fraction"),
    ("latency_s", {"packet_status": "partial"}, "partial", "incomplete_packet_request_correlation"),
    ("gpu_energy_total_j", {"resource_metric_quality": ""}, "unknown", "no_metric_quality_evidence"),
    ("cpu_energy_total_j", {"resource_metric_quality": "[]"}, "error", "resource_metric_quality must be an object"),
])
def test_finite_but_unverified_required_metric_is_not_hardware_evidence(tmp_path, monkeypatch, field, changes, quality, reason):
    gpu_row = measured_row("on", **changes)
    report = check_rows(tmp_path, monkeypatch, measured_row(), gpu_row)
    assert not report["successful"]
    assert report["validated_gpu_modes"] == ["off"]
    assert report["required_metrics_missing"] == []
    issue = {"row": 3, "field": field, "value": gpu_row[field], "quality": quality, "reason": reason}
    assert issue in report["required_metrics_incomplete"]


def test_absent_gpu_energy_is_missing_not_a_finite_low_quality_observation(tmp_path, monkeypatch):
    report = check_rows(tmp_path, monkeypatch, measured_row("on", gpu_energy_total_j="nan"))
    assert not report["successful"]
    assert report["validated_gpu_modes"] == []
    assert report["required_metrics_missing"] == [{"row": 2, "field": "gpu_energy_total_j"}]
    assert report["required_metrics_incomplete"] == []
