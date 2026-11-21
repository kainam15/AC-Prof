import pytest

from acprof.host.measurement.error_rows import reconcile_case_error_rows


def _row(scale, *, status="error", repeat_idx=0):
    return {
        "cpu_cores": "1", "mem_cap_gb": "4", "gpu_mode": "off",
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


@pytest.mark.parametrize("changes", [
    {"cpu_cores": "2"}, {"mem_cap_gb": "8"}, {"gpu_mode": "on"},
    {"input_scale": "64.00000000000000000000000001"}, {"warmup": "2"},
])
def test_partial_row_must_identify_a_planned_window(changes):
    with pytest.raises(RuntimeError, match="unplanned measurement|invalid measurement key"):
        reconcile_case_error_rows([_row(64)], [_row(64, status="ok") | changes],
                                  error="runtime OOM", preserve_existing=True)


def test_duplicate_planned_windows_are_rejected_before_filling():
    with pytest.raises(RuntimeError, match="duplicate.*planned|plan.*duplicate"):
        reconcile_case_error_rows([_row(8), _row("8.00")], [], error="runtime OOM")


def test_reconciliation_matches_decimal_warmup_and_inherited_process_limits():
    resources = {"runtime_kind": "process", "cpu_cores": "nan", "mem_cap_gb": "nan"}
    formal = _row(8) | resources
    warmup = formal | {"warmup": "1"}
    prior = warmup | {"cpu_cores": "", "mem_cap_gb": "none", "warmup": "1.0", "status": "ok",
                      "repeat_idx": "0.0", "input_scale": "8.000", "latency_app_s": "1.25"}
    result = reconcile_case_error_rows([warmup, formal], [prior], error="timeout", preserve_existing=True)
    assert result.missing_rows == [formal]
    assert {name: result.rows[0][name] for name in prior} == prior


@pytest.mark.parametrize("change", [
    {"input_scale": None}, {"input_scale": "invalid"}, {"input_scale": "8.00000000000000000000001"},
    {"measurement_repeat_idx": 0.9}, {"measurement_repeat_idx": None}, {"request_phase": "unknown"},
])
def test_unresolved_timeout_cannot_claim_request_attempts(change):
    context = {"request_phase": "measurement_repeat", "input_scale": "8", "measurement_repeat_idx": 0} | change
    result = reconcile_case_error_rows([_row(8), _row(8, repeat_idx=1)], [], error="timeout", timeout_context=context)
    assert all("planned_request_attempted=unknown" in row["error"] for row in result.missing_rows)


def test_exact_timeout_sidecar_identifies_only_one_high_precision_window():
    near = "8.000000000000000000000000001"
    plan = [_row(8), _row(near), _row(near, repeat_idx=1)]
    context = {"request_phase": "measurement_repeat", "input_scale": near + "0", "measurement_repeat_idx": "0.0"}
    result = reconcile_case_error_rows(plan, [], error="timeout", timeout_context=context)
    assert ["planned_request_attempted=true" in row["error"] for row in result.rows] == [False, True, False]
    assert all("planned_request_attempted=false" in result.rows[i]["error"] for i in (0, 2))


@pytest.mark.parametrize("field", ["runtime_kind", "runtime_id", "resource_scope"])
def test_explicit_runtime_mismatch_is_not_preserved(field):
    planned = _row(64) | {"runtime_kind": "docker", "runtime_id": "image-1", "resource_scope": "container_cgroup"}
    with pytest.raises(RuntimeError, match=f"conflicting {field}"):
        reconcile_case_error_rows([planned], [planned | {field: "another-runtime"}],
                                  error="runtime OOM", preserve_existing=True)


def test_preserved_window_count_uses_recorded_collection_outcomes():
    plan = [_row(8, repeat_idx=i) for i in range(3)]
    prior = [plan[0] | {"status": "warn"}, plan[1] | {"status": "ok", "result_origin": "inferred_not_measured"},
             plan[2] | {"status": "error"}]
    result = reconcile_case_error_rows(plan, prior, preserve_existing=True, error="runtime OOM")
    assert result.preserved_success_rows == 1
    assert result.missing_rows == []
    assert [row["status"] for row in result.rows] == ["warn", "ok", "error"]


@pytest.mark.parametrize("overrides", [
    {"error_type": "client_not_ready"}, {"measurement_completed": True},
    {"request_phase": "auto_repeat_window_warmup", "input_scale": "999"},
    {"request_id": "case_seq8_w0:0"}, {"request_id": "case_seq8_r1:0"},
    {"request_phase": "auto_repeat_window_warmup", "request_id": "case_seq8_r0:0"},
])
def test_contradictory_sidecar_never_attributes_a_missing_window(overrides):
    context = {"error_type": "client_request_timeout", "input_scale": "8", "request_phase": "measurement_repeat",
               "measurement_repeat_idx": 0} | overrides
    result = reconcile_case_error_rows([_row(8), _row(8, repeat_idx=1)], [], error="timeout", timeout_context=context)
    assert all("planned_request_attempted=unknown" in row["error"] for row in result.rows)


def test_unknown_request_id_shape_does_not_override_explicit_window_fields():
    context = {"error_type": "client_request_timeout", "input_scale": "8", "request_phase": "measurement_warmup",
               "measurement_repeat_idx": 0, "request_id": "opaque-legacy-id"}
    result = reconcile_case_error_rows([_row(8) | {"warmup": "1"}], [], error="timeout", timeout_context=context)
    assert "planned_request_attempted=true" in result.rows[0]["error"]


def test_oom_annotation_requires_the_committed_error_diagnostic():
    with pytest.raises(RuntimeError, match="without an error diagnostic"):
        reconcile_case_error_rows([_row(8)], [_row(8) | {"error": ""}], error="runtime OOM",
                                  preserve_existing=True, annotate_existing_error_rows=True)
