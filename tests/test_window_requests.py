"""Request-loop progress is published even when a partially completed window aborts."""
from __future__ import annotations

import math

import pytest

from acprof.host.window_requests import WindowRequestProgress, execute_requests


def _collect(progress, requester, *, limit=3):
    execute_requests(
        progress,
        should_send=lambda n, total, maximum: n < maximum,
        one_request=requester,
        merge_scale=lambda current, reported, planned: current if reported is None else float(reported),
        scale=64.0,
        group_id="case_r0",
        payload_override={"prompt": "test"},
        request_limit=limit,
    )


def test_request_order_and_summary_are_preserved():
    ids = []
    def request(scale, *, req_id, payload_override):
        ids.append((scale, req_id, payload_override))
        return {
            "latency_app_s": 0.25,
            "request_payload_bytes": 15,
            "output_length": 4,
            "output_token_count": 3,
            "task_param": "preset",
            "effective_input_scale": 60,
            "workload_contract": {"ok": True},
        }
    progress = WindowRequestProgress()
    _collect(progress, request, limit=2)
    assert [item[1] for item in ids] == ["case_r0:0", "case_r0:1"]
    assert progress.count == 2 and progress.elapsed_s == 0.5
    assert progress.latencies == [0.25, 0.25]
    assert progress.payload_sizes == [15.0, 15.0]
    assert progress.output_lengths == [4.0, 4.0]
    assert progress.output_token_counts == [3.0, 3.0]
    assert progress.workload_contracts == [{"ok": True}, {"ok": True}]
    assert progress.executed_task_param == "preset"
    assert progress.effective_input_scale == 60.0
    assert progress.pending_request_id == ""


def test_optional_request_metrics_preserve_float_and_nan_semantics():
    responses = iter([
        {"latency_app_s": 0.1, "request_payload_bytes": "12.5",
         "output_length": None, "output_token_count": "not-a-number"},
        {"latency_app_s": 0.2},
    ])
    progress = WindowRequestProgress()
    _collect(progress, lambda *_args, **_kwargs: next(responses), limit=2)

    assert progress.payload_sizes[0] == 12.5
    assert math.isnan(progress.payload_sizes[1])
    assert all(math.isnan(value) for value in progress.output_lengths)
    assert all(math.isnan(value) for value in progress.output_token_counts)
    assert progress.count == 2


def test_failed_second_request_keeps_partial_window_and_pending_id():
    def request(scale, *, req_id, payload_override):
        if req_id == "case_r0:1":
            raise TimeoutError("request timed out")
        return {"latency_app_s": 0.35, "output_token_count": 2}
    progress = WindowRequestProgress()
    with pytest.raises(TimeoutError, match="timed out"):
        _collect(progress, request)
    assert progress.count == 1
    assert progress.elapsed_s == 0.35
    assert progress.latencies == [0.35]
    assert progress.pending_request_id == "case_r0:1"
    assert progress.effective_input_scale is None


def test_bad_response_retains_original_request_completion_semantics():
    progress = WindowRequestProgress()
    with pytest.raises(KeyError, match="latency_app_s"):
        _collect(progress, lambda *_args, **_kwargs: {})
    # The server returned a packet; the original implementation cleared the
    # pending ID before validating the response, and did not count it.
    assert progress.count == 0
    assert progress.pending_request_id == ""


def test_zero_request_window_does_not_mutate_progress():
    progress = WindowRequestProgress()
    _collect(progress, lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()), limit=0)
    assert progress.count == 0
    assert progress.pending_request_id == ""
