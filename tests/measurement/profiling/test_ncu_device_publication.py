"""Publish NCU measurements only for a compatible physical device."""
import copy
import csv
import json
import math

import pytest

from acprof.cli.backfill_compute import backfill_compute_profile_csv
from acprof.gpu_probe import gpu_probe_report
from acprof.host.posthoc import backfill, context, plans
from acprof.host.profiling import compute_plan
from tests.measurement.profiling.test_posthoc import TestPosthocProfile as _Fixture


@pytest.fixture
def source(tmp_path):
    fixture = _Fixture()
    fixture._write_fixture(tmp_path, include_cpu=False)
    record = context.load_result_context(tmp_path)
    plan = fixture._compute_plan()
    return record, plan


def set_probe(plan, observed, *, expected=None, status=None):
    entry = plan["profiles"]["gpu"]["ncu"]["entries"][0]
    report = gpu_probe_report((), status="unknown")
    report["device"] = {"cuda_uuid": observed, "expected_uuid": expected, "identity_status": status}
    entry["gpu_compute_metadata"] = report
    return entry


@pytest.mark.parametrize(("recorded", "expected", "status", "publish"), [
    ("other-device", "GPU-fixture", "mismatch", False),
    ("other-device", "GPU-other-device", "verified", False),
    ("fixture", "GPU-fixture", "verified", True),
    (None, "GPU-fixture", "unverified", True),
])
def test_posthoc_reuse_and_projection_compare_to_source_uuid(source, recorded, expected, status, publish):
    record, plan = source
    entry = set_probe(plan, recorded, expected=expected, status=status)
    original = copy.deepcopy(entry)
    assert plans.compute_plan_covers_tool(plan, record, "ncu") is publish
    _, rows, _ = backfill.backfill_rows(record, tools=("ncu",), compute_plan=plan)
    row = rows[0]
    if publish:
        assert row[compute_plan.NCU_TOTAL_MFLOP_FIELD] == "100.000000"
        assert not row[compute_plan.NCU_ERROR_FIELD]
    else:
        assert "gpu_identity_mismatch:" in row[compute_plan.NCU_ERROR_FIELD]
        assert all(math.isnan(float(row[name])) for name in context.TOOL_METRIC_FIELDS["ncu"])
    assert entry == original


@pytest.mark.parametrize("force", [False, True])
def test_wrong_device_cannot_replace_successful_ncu_results(source, force):
    record, plan = source
    for name in context.TOOL_METRIC_FIELDS["ncu"]:
        record.rows[0][name] = "777.000000"
    record.rows[0][compute_plan.NCU_ERROR_FIELD] = ""
    set_probe(plan, "GPU-other-device", expected="GPU-fixture", status="mismatch")
    _, rows, updated = backfill.backfill_rows(record, tools=("ncu",), compute_plan=plan, force=force)
    assert updated == {"ncu": 0}
    assert rows[0] == record.rows[0]


def test_recorded_device_mismatch_is_unavailable_outside_posthoc(source):
    _, plan = source
    entry = set_probe(plan, "other", expected="GPU-fixture", status="mismatch")
    entry["error"] = "gpu_identity_mismatch:GPU UUID mismatch: expected GPU-fixture, received other"
    resolved = compute_plan.find_compute_profile_entry(plan, "on", 8)
    assert resolved[compute_plan.NCU_ERROR_FIELD] == entry["error"]
    assert all(math.isnan(resolved[name]) for name in context.COMPUTE_PLAN_METRIC_FIELDS["ncu"])


def test_mismatch_only_affects_ncu_for_the_matching_scale(source):
    record, plan = source
    gpu = plan["profiles"]["gpu"]
    gpu.update(_Fixture()._torch_plan()["profiles"]["gpu"])
    foreign = set_probe(plan, "other", expected="GPU-fixture", status="mismatch")
    healthy = {**foreign, "input_scale": 16, "gpu_compute_metadata": gpu_probe_report(())}
    healthy["gpu_compute_metadata"]["device"] = {"cuda_uuid": "fixture"}
    gpu["ncu"]["entries"].append(healthy)

    affected = compute_plan.find_compute_profile_entry(plan, "on", 8, expected_gpu_uuid=record.gpu_uuid)
    assert math.isnan(affected[compute_plan.NCU_TOTAL_MFLOP_FIELD])
    assert affected[compute_plan.TORCH_LOGICAL_MFLOP_FIELD] == 222
    assert not affected[compute_plan.TORCH_ERROR_FIELD]
    other = compute_plan.find_compute_profile_entry(plan, "on", 16, expected_gpu_uuid=record.gpu_uuid)
    assert other[compute_plan.NCU_TOTAL_MFLOP_FIELD] == 100
    assert not other[compute_plan.NCU_ERROR_FIELD]


def test_historical_summary_keeps_its_mismatch_diagnostic_without_publishing_values(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    plan = tmp_path / "plan.json"
    summary = tmp_path / "summary.csv"
    source.write_text("gpu_mode,input_scale,latency_app_s,latency_s\non,8,1,2\n")
    plan.write_text(json.dumps(_Fixture()._compute_plan()))
    error = "gpu_identity_mismatch:GPU UUID mismatch: expected A, received B"
    with summary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["input_scale", compute_plan.NCU_TOTAL_MFLOP_FIELD, compute_plan.NCU_ERROR_FIELD])
        writer.writerow(["8", "900", error])
    backfill_compute_profile_csv(str(source), str(plan), str(output), ncu_summary=str(summary))
    with output.open() as stream:
        row, = csv.DictReader(stream)
    assert row[compute_plan.NCU_ERROR_FIELD] == error
    assert all(math.isnan(float(row[name])) for name in context.TOOL_METRIC_FIELDS["ncu"])
