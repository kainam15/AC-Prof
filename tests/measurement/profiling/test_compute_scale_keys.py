"""Compute plans may carry exact historical JSON coordinates outside float precision."""
import math
from decimal import Decimal, InvalidOperation, localcontext

import pytest

from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES
from acprof.host.profiling import compute_plan


def test_bounded_plan_loader_preserves_decimal_tokens_only_on_request(tmp_path):
    path = tmp_path / "compute_profile_plan.json"
    path.write_text('{"profiles":{"gpu":{"ncu":{"entries":[{"input_scale":1.00000000000000001,'
                    '"gpu_executed_mflop_per_request_ncu":2.5},{"input_scale":1e-400,'
                    '"gpu_executed_mflop_per_request_ncu":7}]}}}}', encoding="utf-8-sig")
    default = compute_plan.load_compute_profile_plan(str(path))
    assert type(default["profiles"]["gpu"]["ncu"]["entries"][0]["input_scale"]) is float
    exact = compute_plan.load_compute_profile_plan(str(path), exact_decimals=True)
    for raw, expected in (("1.00000000000000001", 2.5), ("1e-400", 7.0), ("1", None), ("0", None)):
        result = compute_plan.find_compute_profile_entry(exact, "on", Decimal(raw))
        metric = result[compute_plan.NCU_TOTAL_MFLOP_FIELD]
        if expected is None:
            assert math.isnan(metric)
            assert result[compute_plan.NCU_ERROR_FIELD] == f"compute_profile_missing_scale:{raw}"
        else:
            assert type(metric) is float
            assert metric == expected


@pytest.mark.parametrize("tail", [b"1e999", b"NaN", b"Infinity", b"1e-9999999999999999999999", None],
                         ids=["overflow", "nan", "infinity", "decimal-exponent-overflow", "oversized"])
def test_exact_plan_loader_reports_out_of_range_json(tmp_path, tail):
    path = tmp_path / "plan.json"
    if tail is None:
        tail = b'"' + b"x" * MAX_JSON_ARTIFACT_BYTES + b'"'
    path.write_bytes(b'{"profiles":{},"extra":' + tail + b'}')
    assert "compute_profile_plan_invalid" in compute_plan.load_compute_profile_plan(
        str(path), exact_decimals=True,
    )["_load_error"]


def test_lookup_preserves_native_integer_and_display_float_identity():
    field = compute_plan.TORCH_LOGICAL_MFLOP_FIELD
    entries = [{"input_scale": value, field: i, "error": ""} for i, value in enumerate(
        [True, 10 ** 309, 2**53 + 1, float(2**53), 0.1, "0.100000000000000001", "0.1"], 1)]
    plan = {"profiles": {"cpu": {"torch_profiler_eager": {"entries": entries, "error": "profile error"}}}}
    matches: list[tuple[int | float | Decimal, int]] = [
        (2**53 + 1, 3), (float(2**53), 4), (Decimal("0.1"), 5), (Decimal("0.100000000000000001"), 6),
    ]
    for scale, expected in matches:
        result = compute_plan.find_compute_profile_entry(plan, "off", scale)
        assert result[field] == expected
        assert result[compute_plan.TORCH_ERROR_FIELD] == ""
    for missing in (True, 1, None, 10 ** 309, Decimal("sNaN")):
        result = compute_plan.find_compute_profile_entry(plan, "off", missing)
        assert math.isnan(result[field])
        assert result[compute_plan.TORCH_ERROR_FIELD].startswith("compute_profile_missing_scale:")


def test_scale_keys_preserve_historical_finite_values_independently_of_decimal_context():
    with localcontext(prec=2) as ctx:
        ctx.traps[InvalidOperation] = False
        for raw in ("0", "-1.25", "1e-400", "1.00000000000000001", str(2 ** 53 + 1)):
            assert compute_plan.compute_scale_key(raw) == Decimal(raw)
        for invalid in (False, "", "nan", "inf", "1e999", "1e-9999999999999999999999", Decimal("sNaN"), 10 ** 309):
            assert compute_plan.compute_scale_key(invalid) is None
