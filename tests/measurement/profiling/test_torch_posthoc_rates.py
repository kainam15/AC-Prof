"""Posthoc Torch work must include rates from the original row latencies."""
from types import SimpleNamespace

import pytest

from acprof.host.posthoc import backfill

WORK = "model_logical_mflop_per_request_torch_profiler_eager"
APP = "model_logical_mflops_app_torch_profiler_eager"
PACKET = "model_logical_mflops_packet_torch_profiler_eager"
ERROR = "compute_profile_error_torch_profiler_eager"


@pytest.mark.parametrize("missing", [APP, PACKET])
def test_torch_rate_is_required_even_when_logical_work_is_present(missing):
    row = {WORK: "120", APP: "300", PACKET: "240", ERROR: ""}
    row[missing] = "nan"

    assert not backfill.row_tool_complete(row, "torch")


@pytest.mark.parametrize(
    ("work", "app_latency", "packet_latency", "app_rate", "packet_rate"),
    [
        (120, "0.4", "0.5", "300.000000", "240.000000"),
        (120, "0.4", "nan", "300.000000", "300.000000"),
        (120, "nan", "0.5", "nan", "240.000000"),
        (120, "nan", "nan", "nan", "nan"),
        (0, "0.4", "0.5", "0.000000", "0.000000"),
    ],
)
def test_torch_backfill_derives_rates_without_changing_original_latencies(
    work, app_latency, packet_latency, app_rate, packet_rate,
):
    row = {
        "gpu_mode": "on", "input_scale": "8",
        "latency_app_s": app_latency, "latency_s": packet_latency,
        WORK: "nan", ERROR: "compute_profile_disabled",
    }
    before = dict(row)
    context = SimpleNamespace(fieldnames=list(row), rows=[row])
    plan = {"profiles": {"gpu": {"torch_profiler_eager": {
        "tool": "torch_profiler_eager", "error": "",
        "entries": [{"input_scale": 8, WORK: work, "error": ""}],
    }}}}

    fields, rows, updated = backfill.backfill_rows(
        context, tools=("torch",), compute_plan=plan,
    )

    assert updated == {"torch": 1}
    assert APP in fields
    assert PACKET in fields
    assert rows[0][WORK] == f"{work:.6f}"
    assert rows[0][APP] == app_rate
    assert rows[0][PACKET] == packet_rate
    assert rows[0][ERROR] == ""
    assert rows[0]["latency_app_s"] == app_latency
    assert rows[0]["latency_s"] == packet_latency
    assert row == before
