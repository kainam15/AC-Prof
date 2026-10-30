"""Large result fields survive downstream validation and postprocessing."""
from __future__ import annotations

import csv
import json
import subprocess
import sys

import pytest

from acprof.application import notifications
from acprof.cli import backfill_compute
from acprof.host import orchestrator
from acprof.host.collection_history import empty_collection_history
from acprof.host.detect import TaskInfo
from acprof.host.measurement import error_rows
from acprof.host.measurement.energy_validation import (
    EnergyProfilingError,
    check_case_cpu_idle_power_stable,
    check_case_gpu_idle_power_stable,
)
from acprof.host.measurement.publication import reconcile_sniff_group_sidecar
from acprof.host.posthoc.context import ResultContext
from acprof.host.posthoc.storage import commit_result_files, create_backup
from acprof.results.csv import read_result_csv
from tests.analysis.comparison.test_large_result_fields import (
    default_csv_limit as default_csv_limit,
    result_file,
)
from tests.measurement.windows.test_measurement_checkpoint import _fixture, _publish, _read

pytestmark = pytest.mark.usefixtures(default_csv_limit.__name__)


@pytest.mark.parametrize("consumer", ["notification-counts", "case-status", "packet-completeness",
                                     "compute-backfill", "ncu-summary"])
def test_result_consumers_accept_large_fields(tmp_path, consumer):
    path, expected = result_file(tmp_path, encoding="utf-8", gpu_executed_mflop_per_request_ncu="12")
    original = path.read_bytes()
    if consumer == "notification-counts":
        assert notifications._result_status_counts(str(path)) == (1, 0)
    elif consumer == "case-status":
        assert orchestrator._case_result_status(str(path)) == "ok"
    elif consumer == "packet-completeness":
        orchestrator._assert_packet_latency_csv_complete(str(path))
    elif consumer == "compute-backfill":
        _, rows, _ = backfill_compute._read_result_csv(str(path))
        assert rows[0]["note"] == expected["note"]
    else:
        entries = backfill_compute._read_ncu_summary(str(path))
        assert entries[0][0] == 1.0
        assert entries[0][1]["gpu_executed_mflop_per_request_ncu"] == "12"
    assert path.read_bytes() == original


def test_ready_failure_diagnostic_remains_complete(tmp_path):
    diagnostic = "ready_failed:" + "诊断\n" * 50000
    path, _ = result_file(tmp_path, encoding="utf-8", status="error", error=diagnostic)
    assert orchestrator._client_ready_failure(str(path)) == diagnostic


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("value", ["20", "nan"])
def test_idle_baseline_validation_keeps_numeric_checks_with_large_fields(tmp_path, device, value):
    path, _ = result_file(tmp_path, encoding="utf-8", gpu_mode="on", **{f"{device}_idle_power_w": value})
    check = check_case_cpu_idle_power_stable if device == "cpu" else check_case_gpu_idle_power_stable
    if value == "nan":
        with pytest.raises(EnergyProfilingError, match="invalid_rows=1"):
            check(str(path))
    else:
        check(str(path))


def test_sidecar_recovery_counts_large_rows_without_changing_csv(tmp_path):
    path, _ = result_file(tmp_path, encoding="utf-8")
    original = path.read_bytes()
    sidecar = tmp_path / "groups.jsonl"
    committed = '{"sniff_group_id":"window"}\n'
    sidecar.write_text(committed + '{"sniff_group_id":"pending"}\n', encoding="utf-8")
    reconcile_sniff_group_sidecar(path, sidecar)
    assert sidecar.read_text() == committed
    assert path.read_bytes() == original


def test_checkpoint_publish_and_recovery_accept_large_committed_fields(tmp_path):
    note = '诊断,"quoted"\r\n' * 16384
    case, _, _, processing = _fixture(tmp_path, capture=False, note=note)
    original = case.csv.read_bytes()
    reference = _publish(case, processing)
    csv.field_size_limit(131072)
    checkpoint = _read(case, reference)
    assert checkpoint is not None
    assert case.csv.read_bytes() == original
    assert read_result_csv(case.csv)[1][0]["note"] == note


def test_packet_postprocessing_preserves_long_fields_in_fresh_process(tmp_path):
    path, expected = result_file(tmp_path, encoding="utf-8", sniff_group_id="window")
    original = path.read_bytes()
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps({"schema_version": 2, "requests": {"window:0": {"latency_s": 0.5}}}))
    output = tmp_path / "merged.csv"
    subprocess.run([sys.executable, "-m", "acprof.packet.merge_packet_latency", str(path), str(packet), str(output)],
                   capture_output=True, text=True, check=True, timeout=15)
    row = read_result_csv(output)[1][0]
    assert row["note"] == expected["note"]
    assert float(row["latency_s"]) == 0.5
    assert path.read_bytes() == original


def test_posthoc_transaction_preserves_long_fields_and_original_backup(tmp_path):
    path, expected = result_file(tmp_path, encoding="utf-8")
    original = path.read_bytes()
    metadata = tmp_path / "static_meta.json"
    history_path = tmp_path / "collection_history.json"
    history = empty_collection_history()
    metadata.write_text('{"fixture": true}')
    history_path.write_text(json.dumps(history))
    context = ResultContext(
        result_dir=tmp_path, result_csv=path, static_meta_path=metadata,
        collection_history_path=history_path, input_scale_plan_path=tmp_path / "input_scale_plan.json",
        fieldnames=list(expected), rows=[expected], csv_encoding="utf-8", static_meta={"fixture": True},
        collection_history=history, collection_history_existed=True, input_scale_plan={},
        task_info=TaskInfo("fixture/model", "text-classification", "text", "transformers_pipeline",
                           "transformers", "a" * 40, "manual"),
        image_tag="fixture", resource_cases=[(1, 4, "off")], scales_by_mode={"off": [1.0]},
    )
    backup = create_backup(context)
    updated = dict(expected, latency_s="0.5")
    csv.field_size_limit(131072)
    commit_result_files(context, fieldnames=list(updated), rows=[updated], static_meta={"fixture": False},
                        collection_history=history, backup_dir=backup)
    assert read_result_csv(path)[1] == [updated]
    assert (backup / path.name).read_bytes() == original
    assert json.loads(metadata.read_bytes()) == {"fixture": False}


def test_case_failure_completion_preserves_long_committed_diagnostics(tmp_path):
    detail = '诊断,"quoted"\n' * 16384
    path, _ = result_file(tmp_path, encoding="utf-8", note="", workload_contract=detail)
    result = error_rows.write_case_error_csv(
        task_info=TaskInfo("fixture/model", "text-classification", "nlp", "transformers_pipeline",
                           "transformers", "a" * 40, "manual"),
        out_csv=str(path), cpu=1, mem=4, gpu="off", warmup=0, repeat=2,
        repeat_in_window=1, input_scales="1", error="fixture case aborted", preserve_existing=True,
    )
    assert result == (1, 1)
    rows = read_result_csv(path)[1]
    assert rows[0]["workload_contract"] == detail
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "error"
    assert "fixture case aborted" in rows[1]["error"]
