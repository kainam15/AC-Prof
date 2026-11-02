"""Single-point SLO and token-capacity semantics from raw request evidence."""
from __future__ import annotations

import pytest

from acprof.analysis.streaming_capacity import build_capacity


def request(status="ok", *, sent_s=0.0, input_tokens=2, output_tokens=3, **overrides):
    row = {
        "status": status, "sent_s": sent_s,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "input_token_count_source": "server_usage" if input_tokens is not None else "unavailable",
        "output_token_count_source": "server_usage" if output_tokens is not None else "unavailable",
        "input_token_count_scope": "server_prompt_tokens" if input_tokens is not None else "unavailable",
        "token_count_scope": "server_generated_tokens",
        "ttft_s": 0.1, "tpot_s": 0.02, "e2e_s": 0.3,
    }
    row.update(overrides)
    return row


def test_slo_equality_is_satisfied_and_goodput_uses_result_duration():
    capacity = build_capacity(
        [request(), request()], 4, 2,
        {"ttft_s": 0.1, "tpot_s": 0.02, "e2e_s": 0.3},
    )
    assert capacity["slo"]["counts"] == {"satisfied": 2, "violated": 0, "unevaluable": 0}
    assert capacity["goodput_requests_per_s"] == 0.5
    assert capacity["slo"]["per_metric"]["ttft_s"]["satisfied"] == 2


def test_known_violation_wins_over_another_missing_metric():
    capacity = build_capacity(
        [request(ttft_s=0.101, tpot_s=None)], 1, 1,
        {"ttft_s": 0.1, "tpot_s": 0.02, "e2e_s": 0.3},
    )
    assert capacity["slo"]["counts"] == {"satisfied": 0, "violated": 1, "unevaluable": 0}
    assert capacity["slo"]["per_metric"]["tpot_s"]["unevaluable"] == 1
    assert capacity["goodput_requests_per_s"] == 0


def test_missing_configured_metric_without_violation_is_unevaluable():
    capacity = build_capacity(
        [request(tpot_s=None)], 2, 1, {"ttft_s": 0.2, "tpot_s": 0.03},
    )
    assert capacity["slo"]["counts"] == {"satisfied": 0, "violated": 0, "unevaluable": 1}
    assert capacity["slo"]["per_metric"]["ttft_s"]["satisfied"] == 1
    assert capacity["goodput_requests_per_s"] == 0


def test_unconfigured_slo_has_no_request_classification_or_goodput():
    capacity = build_capacity([request()], 2, 1)
    assert capacity["slo"]["configured"] is False
    assert capacity["slo"]["counts"] == {"satisfied": 0, "violated": 0, "unevaluable": 0}
    assert capacity["goodput_requests_per_s"] is None


def test_zero_satisfied_is_valid_and_unsuccessful_requests_are_counted_separately():
    requests = [
        request(ttft_s=2), request(status="error", ttft_s=0),
        request(status="dropped", sent_s=None), request(status="cancelled", sent_s=None),
    ]
    capacity = build_capacity(requests, 2, 4, {"ttft_s": 1})
    assert capacity["counts"] == {
        "offered": 4, "sent": 2, "succeeded": 1, "failed": 1, "dropped": 1, "cancelled": 1,
    }
    assert capacity["slo"]["counts"] == {"satisfied": 0, "violated": 1, "unevaluable": 0}
    assert capacity["goodput_requests_per_s"] == 0
    assert capacity["successful_requests_per_s"] == 0.5


def test_complete_token_throughput_preserves_known_zero_and_same_request_total():
    capacity = build_capacity([request(), request(input_tokens=4, output_tokens=0)], 2, 2)
    throughput = capacity["token_throughput"]
    assert throughput["input"] == {
        "tokens": 6, "tokens_per_s": 3, "observed_tokens": 6, "observed_tokens_per_s": 3,
        "covered_requests": 2, "successful_requests": 2, "coverage_ratio": 1,
    }
    assert throughput["output"]["tokens"] == 3
    assert throughput["output"]["tokens_per_s"] == 1.5
    assert throughput["total"]["tokens"] == 9
    assert throughput["total"]["tokens_per_s"] == 4.5


def test_partial_token_coverage_reports_observed_but_not_complete_values():
    requests = [
        request(),
        request(input_tokens=None, output_tokens=4, token_count_scope="server_generated_tokens"),
        request(input_tokens=0, output_tokens=None,
                input_token_count_source="server_usage", token_count_scope="decoded_output_text"),
    ]
    capacity = build_capacity(requests, 2, 3)
    throughput = capacity["token_throughput"]
    assert throughput["input"]["tokens"] is None
    assert throughput["input"]["tokens_per_s"] is None
    assert throughput["input"]["observed_tokens"] == 2
    assert throughput["input"]["observed_tokens_per_s"] == 1
    assert throughput["input"]["covered_requests"] == 2
    assert throughput["input"]["coverage_ratio"] == pytest.approx(2 / 3)
    assert throughput["output"]["observed_tokens"] == 7
    assert throughput["output"]["covered_requests"] == 2
    assert throughput["total"]["tokens"] is None
    assert throughput["total"]["observed_tokens"] == 5
    assert throughput["total"]["covered_requests"] == 1
    assert throughput["scope_request_counts"] == {
        "input": {"server_prompt_tokens": 2, "unavailable": 1},
        "output": {"server_generated_tokens": 2, "decoded_output_text": 1},
        "total": {
            "server_prompt_tokens+server_generated_tokens": 1,
            "unavailable+server_generated_tokens": 1,
            "server_prompt_tokens+decoded_output_text": 1,
        },
    }
    assert throughput["source_request_counts"]["input"] == {"server_usage": 2, "unavailable": 1}


def test_no_successful_requests_leave_complete_and_observed_tokens_unknown():
    capacity = build_capacity(
        [request(status="error", input_tokens=100, output_tokens=100)], 2, 1, {"e2e_s": 1},
    )
    assert capacity["successful_requests_per_s"] == 0
    assert capacity["goodput_requests_per_s"] == 0
    for component in capacity["token_throughput"].values():
        if not isinstance(component, dict) or "tokens" not in component:
            continue
        assert component["tokens"] is None
        assert component["tokens_per_s"] is None
        assert component["observed_tokens"] is None
        assert component["observed_tokens_per_s"] is None
        assert component["covered_requests"] == 0
        assert component["successful_requests"] == 0
        assert component["coverage_ratio"] is None


def test_failed_request_tokens_never_enter_observed_or_complete_totals():
    capacity = build_capacity([
        request(input_tokens=1, output_tokens=1),
        request(status="error", input_tokens=1000, output_tokens=1000),
    ], 1, 1)
    assert capacity["token_throughput"]["total"]["tokens"] == 2
    assert capacity["token_throughput"]["total"]["observed_tokens"] == 2
