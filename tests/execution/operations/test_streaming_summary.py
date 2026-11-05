"""Validate successful streaming latency evidence before aggregation."""
from __future__ import annotations

from copy import deepcopy

import pytest

from acprof.analysis.streaming import streaming_summary

INVALID_LATENCIES = [
    pytest.param(-0.1, id="negative"),
    pytest.param(True, id="boolean"),
    pytest.param("0.1", id="text"),
    pytest.param([], id="array"),
    pytest.param({}, id="object"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinity"),
    pytest.param(10 ** 400, id="overflow"),
]


@pytest.mark.parametrize("field", ["ttft_s", "tpot_s", "output_e2e_s", "e2e_s"])
@pytest.mark.parametrize("value", INVALID_LATENCIES)
def test_streaming_summary_rejects_invalid_request_latency(field, value):
    with pytest.raises(ValueError, match=field):
        streaming_summary([{"status": "ok", field: value}])


@pytest.mark.parametrize("field", ["itl_s", "output_event_intervals_s"])
@pytest.mark.parametrize("value", [*INVALID_LATENCIES, pytest.param(None, id="null")])
def test_streaming_summary_rejects_invalid_interval_sample(field, value):
    with pytest.raises(ValueError, match=field):
        streaming_summary([{"status": "ok", field: [value]}])


@pytest.mark.parametrize("field", ["itl_s", "output_event_intervals_s"])
@pytest.mark.parametrize("value", [False, 0, 0.1, "", "0.1", {}])
def test_streaming_summary_rejects_invalid_interval_container(field, value):
    with pytest.raises(ValueError, match=field):
        streaming_summary([{"status": "ok", field: value}])


@pytest.mark.parametrize("intervals", [[0, 0.5], (0, 0.5)])
def test_streaming_summary_preserves_zero_missing_samples_and_input(intervals):
    records = [
        {"status": "ok", "ttft_s": 0, "tpot_s": 0, "e2e_s": 0, "output_e2e_s": 0,
         "itl_s": intervals, "output_event_intervals_s": intervals},
        {"status": "ok", "ttft_s": 0.5, "tpot_s": 0.5, "e2e_s": 0.5, "output_e2e_s": 0.5,
         "itl_s": None, "output_event_intervals_s": None},
        {"status": "ok", "ttft_s": None, "itl_s": [], "output_event_intervals_s": []},
    ]
    before = deepcopy(records)
    summary = streaming_summary(records)

    for field in ("ttft_s", "tpot_s", "e2e_s", "output_e2e_s", "itl_s", "output_event_interval_s"):
        assert summary[field]["mean"] == 0.25
        assert summary[field]["p50"] == 0.25
        assert summary[field]["p90"] == 0.45
        assert summary[field]["sample_count"] == 2
    for field in ("ttft_s", "tpot_s", "e2e_s", "output_e2e_s"):
        assert summary[field]["missing_requests"] == 1
    for field in ("itl_s", "output_event_interval_s"):
        assert summary[field]["requests_with_samples"] == 1
        assert summary[field]["missing_requests"] == 2
    assert records == before


@pytest.mark.parametrize("status", ["error", "cancelled", "dropped"])
def test_streaming_summary_excludes_failed_request_observations(status):
    summary = streaming_summary([
        {"status": "ok", "ttft_s": 0.25, "itl_s": [0.5]},
        {"status": status, "ttft_s": True, "tpot_s": -1, "output_e2e_s": "invalid",
         "e2e_s": {}, "itl_s": False, "output_event_intervals_s": [None]},
    ])
    assert summary["ttft_s"]["mean"] == 0.25
    assert summary["itl_s"]["mean"] == 0.5
    assert summary["excluded_requests"] == 1
    assert summary["errors"] == {status: 1}
