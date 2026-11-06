"""Browser inputs preserve Python's window selection, identity and publication guard."""
import hashlib
import json

import pytest

from acprof.analysis.model import load_analysis
from acprof.plotting.report import report_payload, write_report
from tests.analysis.reporting.test_report_statistics import row, write_rows


def test_inputs_keep_missing_values_and_exclude_non_measurements(tmp_path):
    source = tmp_path / "result_all.csv"
    write_rows(source, [row(0, 1), row(1, "nan"), row(3, 3), row(4, 99, status="error"),
                        row(5, 99, result_origin="inferred_not_measured"), row(0, 99, warmup="1")])
    original = source.read_bytes()
    report = report_payload(load_analysis([source]))["window_statistics"][0]
    inputs = report.get("recalculation")
    assert inputs is not None, "HTML needs Python-selected windows for offline recalculation"
    group = next(group for group in inputs["groups"] if group["metric"] == "latency_app_s")
    assert group["values"] == [1, None, 3]
    assert group["repeat_indices"] == [0, 1, 3]
    assert group["excluded_windows"] == {"error": 1, "inferred_not_measured": 1, "warmup": 1}
    assert sum(group["missing_reasons"].values()) == 1
    assert inputs["status"] == "embedded"
    assert report["engine"] == "python"
    assert source.read_bytes() == original


def test_inputs_preserve_configuration_groups_and_offer_recorded_metrics(tmp_path):
    source = tmp_path / "result_all.csv"
    write_rows(source, [row(i, i + 1, concurrency="1", throughput_samples_per_s="4") for i in range(3)] +
               [row(i, 10, concurrency="2", throughput_samples_per_s="8") for i in range(3, 6)])
    report = report_payload(load_analysis([source]))["window_statistics"][0]
    inputs = report.get("recalculation")
    assert inputs is not None
    groups = [group for group in inputs["groups"] if group["metric"] == "latency_app_s"]
    assert len({group["config_id"] for group in groups}) == 2
    assert [group["values"] for group in groups] == [[1, 2, 3], [10, 10, 10]]
    assert "throughput_samples_per_s" in inputs["metrics"]
    assert "cold_start_s" not in inputs["metrics"]


def test_large_inputs_publish_hash_bound_sidecar_without_overwriting(tmp_path, monkeypatch):
    from acprof.analysis import report_statistics
    # The limit applies to raw windows only; initial Python estimates stay embedded.
    monkeypatch.setattr(report_statistics, "MAX_EMBEDDED_WINDOW_BYTES", 1, raising=False)
    source = tmp_path / "result_all.csv"
    write_rows(source, [row(i, i + 1) for i in range(3)])
    output = write_report(load_analysis([source]), tmp_path / "report.html")
    html = output.read_text()
    raw = html.split('<script id="report-data" type="application/json">')[1].split("</script>")[0]
    report = json.loads(raw)["window_statistics"][0]
    inputs = report.get("recalculation")
    assert inputs is not None
    assert inputs["status"] == "external_required"
    assert "groups" not in inputs
    sidecar = output.parent / inputs["file"]
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == inputs["sha256"]
    decoded = json.loads(sidecar.read_bytes())
    assert decoded["result_sha256"] == report["result_sha256"]
    assert next(g for g in decoded["groups"] if g["metric"] == "latency_app_s")["values"] == [1, 2, 3]
    assert report["groups"][0]["mean"] == 2
    output.unlink()
    before = sidecar.read_bytes()
    with pytest.raises(ValueError, match="new file"):
        write_report(load_analysis([source]), output)
    assert sidecar.read_bytes() == before
    assert not output.exists()
