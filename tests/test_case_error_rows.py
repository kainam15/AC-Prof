import pytest

from acprof.host.measurement.error_rows import reconcile_case_error_rows


def _row(scale, *, status="error", repeat_idx=0):
    return {
        "input_scale": str(scale),
        "repeat_idx": str(repeat_idx),
        "warmup": "0",
        "status": status,
        "error": "existing failure" if status == "error" else "",
    }


def test_reconciliation_preserves_committed_success_and_annotates_only_missing_rows():
    result = reconcile_case_error_rows(
        [_row(64), _row(128)],
        [_row(64, status="ok")],
        error="request timeout",
        preserve_existing=True,
        timeout_context={
            "input_scale": 128,
            "request_timeout_s": 30,
            "request_phase": "measurement_repeat",
            "measurement_repeat_idx": 0,
            "request_id": "case_seq128_r0:0",
        },
    )

    assert result.preserved_success_rows == 1
    assert len(result.rows) == 2
    assert len(result.missing_rows) == 1
    assert result.rows[0]["status"] == "ok"
    assert result.rows[0]["error"] == ""
    assert result.missing_rows[0]["error"].startswith(
        "client_request_timeout: planned_request_attempted=true"
    )
    assert "triggering_request_latency_s>30" in result.missing_rows[0]["error"]


def test_reconciliation_rejects_duplicate_committed_measurement():
    with pytest.raises(RuntimeError, match="duplicate measurement"):
        reconcile_case_error_rows(
            [_row(64)],
            [_row(64, status="ok"), _row(64, status="ok")],
            error="case failed",
            preserve_existing=True,
        )


def test_reconciliation_rejects_unplanned_committed_measurement():
    with pytest.raises(RuntimeError, match="unplanned measurement"):
        reconcile_case_error_rows(
            [_row(64)],
            [_row(128, status="ok")],
            error="case failed",
            preserve_existing=True,
        )
