"""Reject contradictory load summaries without erasing valid partial evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from acprof.analysis.load_report import analyze_load_report, read_load_report
from acprof.analysis.load_sweep import read_load_sweep
from acprof.analysis.streaming import streaming_summary
from acprof.analysis.streaming_capacity import build_capacity
from acprof.cli.main import main
from acprof.plotting.load_report import write_load_report
from tests.analysis.reporting.test_load_dataset_report import saved_dataset_report
from tests.analysis.reporting.test_load_report import (
    experiment,
    request as request_record,
    save_failed_experiment,
)
from tests.support.load_sweep_fixtures import make_sweep

__all__ = ["saved_dataset_report"]


@pytest.fixture(params=[2, 3], name="summary_report")
def valid_summary_report(request, dataset_report) -> dict[str, Any]:
    return copy.deepcopy(experiment() if request.param == 2 else dataset_report)


def save_report(root: Path, payload: dict[str, Any]) -> Path:
    source = root / "load.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    return source


@pytest.mark.parametrize(("path", "value"), [
    (("result", "counts", name), 999)
    for name in ("offered", "sent", "succeeded", "failed", "dropped", "cancelled")
] + [
    (("result", "counts", "succeeded"), True),
    (("result", "counts", "succeeded"), 1.0),
    (("result", "success_rate"), 42.0),
    (("result", "success_rate"), True),
    (("result", "successful_requests_per_s"), -10.0),
    (("result", "successful_requests_per_s"), "0.5"),
    (("result", "sent_requests_per_s"), -1.0),
    (("result", "completed_requests_per_s"), None),
    (("result", "successful"), False),
    (("result", "successful"), 1),
    (("successful",), False),
    (("successful",), 1),
    (("status",), "unknown"),
    (("status",), True),
    pytest.param(("result", "success_rate"), 10 ** 400, id="oversized-integer-rate"),
])
def test_contradictory_summary_cannot_publish_html(summary_report, tmp_path, path, value):
    target = summary_report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    source = save_report(tmp_path, summary_report)
    before = source.read_bytes()
    output = tmp_path / "invalid.html"
    with pytest.raises(ValueError, match="stream load"):
        write_load_report(source, output)
    assert source.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("field", ["counts", "success_rate", "successful_requests_per_s", "successful"])
def test_current_summary_requires_its_core_fields(summary_report, field):
    del summary_report["result"][field]
    with pytest.raises(ValueError, match="stream load"):
        analyze_load_report(summary_report)


@pytest.mark.parametrize("state", ["running", "failed", "cancelled"])
@pytest.mark.parametrize("request_successful", [False, True])
def test_outer_failure_or_cancellation_preserves_completed_requests(
    summary_report, state, request_successful,
):
    summary_report.update(status=state, successful=False)
    summary_report["result"]["successful"] = request_successful
    before = copy.deepcopy(summary_report)
    analyzed = analyze_load_report(summary_report)
    assert analyzed["result"]["successful"] is request_successful
    assert analyzed["experiment"]["successful"] is False
    assert summary_report == before


def test_failed_requests_cannot_claim_success(tmp_path):
    source = save_failed_experiment(tmp_path)
    payload = json.loads(source.read_text())
    payload.update(status="complete", successful=True)
    payload["result"]["successful"] = True
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="success"):
        read_load_report(source)


def test_optional_rates_count_sent_requests_and_completed_streams_separately():
    payload: dict[str, Any] = experiment()
    rows = [
        {**request_record(), "stream_completed_s": 0.3},
        {"status": "error", "sent_s": 0.1, "stream_completed_s": 0.4},
        {"status": "error", "sent_s": 0.2, "http_status": 200},
        {"status": "dropped", "sent_s": None},
    ]
    result = payload["result"]
    result.update(
        requests=rows, successful=False,
        counts={"offered": 4, "sent": 3, "succeeded": 1, "failed": 2, "dropped": 1, "cancelled": 0},
        success_rate=0.25, successful_requests_per_s=0.5,
        sent_requests_per_s=1.5, completed_requests_per_s=1.0,
        capacity=build_capacity(rows, 2.0, 2, payload["identity"]["slo_thresholds_s"]),
        streaming=streaming_summary(rows),
    )
    payload["successful"] = False
    assert analyze_load_report(payload)["result"]["counts"]["offered"] == 4
    result["completed_requests_per_s"] = 1.5
    with pytest.raises(ValueError, match="completed_requests_per_s"):
        analyze_load_report(payload)


def test_declared_request_budget_must_match_retained_records():
    payload: dict[str, Any] = experiment()
    payload["identity"]["load_config"]["requests"] = 2
    payload["result"]["protocol"]["requests"] = 2
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload["identity"], sort_keys=True, allow_nan=False).encode(),
    ).hexdigest()
    with pytest.raises(ValueError, match="request budget"):
        analyze_load_report(payload)


def test_zero_success_rate_is_valid_but_boolean_false_is_not():
    payload: dict[str, Any] = experiment()
    rows = [{"status": "error", "sent_s": 0.0}]
    result = payload["result"]
    result.update(
        requests=rows, successful=False, success_rate=0.0, successful_requests_per_s=0.0,
        counts={"offered": 1, "sent": 1, "succeeded": 0, "failed": 1, "dropped": 0, "cancelled": 0},
        capacity=build_capacity(rows, 2.0, 2, payload["identity"]["slo_thresholds_s"]),
        streaming=streaming_summary(rows),
    )
    payload["successful"] = False
    assert analyze_load_report(payload)["result"]["success_rate"] == 0.0
    result["success_rate"] = False
    with pytest.raises(ValueError, match="success_rate"):
        analyze_load_report(payload)


def test_summary_error_is_reported_by_cli_without_publishing(tmp_path, capsys):
    payload: dict[str, Any] = experiment()
    payload["result"]["success_rate"] = 42
    source = save_report(tmp_path, payload)
    output = tmp_path / "invalid.html"
    before = source.read_bytes()
    with pytest.raises(SystemExit) as caught:
        main(["report", str(source), "--output", str(output)])
    assert caught.value.code == 1
    assert "success_rate" in capsys.readouterr().err
    assert source.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("field", ["success_rate", "successful_requests_per_s"])
def test_bad_summary_point_preserves_other_sweep_points(tmp_path, field):
    root = make_sweep(tmp_path / "sweep")
    source = root / "points/0002/load.json"
    payload = json.loads(source.read_text())
    payload["result"][field] = -1.0
    source.write_text(json.dumps(payload))
    before = source.read_bytes()
    data = read_load_sweep(root).payload
    assert data["valid_points"] == 2
    assert data["points"][1]["evidence_status"] == "invalid"
    assert data["points"][1]["metrics"] == {}
    assert field in " ".join(data["points"][1]["diagnostics"])
    assert data["points"][0]["metrics"]["successful_requests_per_s"] == 1
    assert source.read_bytes() == before


def test_legacy_v1_and_optional_rate_omissions_remain_readable(tmp_path):
    for version in (1, 2):
        payload: dict[str, Any] = experiment(version)
        payload["result"].pop("sent_requests_per_s", None)
        payload["result"].pop("completed_requests_per_s", None)
        assert read_load_report(save_report(tmp_path, payload))["version"] == version
