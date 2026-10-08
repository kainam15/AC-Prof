"""Exercise host error publication together with packet result merging."""
import csv
import json
from unittest.mock import patch

import pytest

from acprof.artifact_layout import ArtifactLayout, case_sidecar
from acprof.config import CSV_FIELDS
from acprof.host import orchestrator
from acprof.host.detect import TaskInfo
from acprof.packet.merge_packet_latency import main as merge_packet_latency


@pytest.fixture(params=["flat", "v2"])
def partial_case(tmp_path, request):
    if request.param == "v2":
        layout = ArtifactLayout.for_new_run(tmp_path)
        layout.initialize()
        csv_path = layout.case("org/model", 1, 2, "off").csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        csv_path = tmp_path / "result.csv"
    row = dict.fromkeys(CSV_FIELDS, "nan")
    row.update(cpu_cores="1", mem_cap_gb="2", gpu_mode="off", input_scale="8",
               repeat_idx="0", warmup="0", status="ok", error="",
               latency_app_s="0.04", result_origin="formal_measurement")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    sidecar = case_sidecar(csv_path, "sniff_groups")
    sidecar.write_text('{"sniff_group_id": "case_seq8_r0"}\n', encoding="utf-8")
    return csv_path, sidecar


def complete_error_case(csv_path, error):
    return orchestrator._write_case_error_csv(
        task_info=TaskInfo("org/model", "text-generation", "text",
                           "transformers", "transformers", "a" * 40, "manual"),
        out_csv=str(csv_path), cpu=1, mem=2, gpu="off", warmup=0, repeat=3,
        repeat_in_window=1, input_scales="8", error=error, preserve_existing=True,
    )


@pytest.mark.parametrize("error", ["runtime OOM", "client_request_timeout"])
@pytest.mark.parametrize("orphan_tail", ["", '{"sniff_group_id":"uncommitted"}\n'])
def test_partial_failure_preserves_packet_metrics(partial_case, error, orphan_tail):
    csv_path, sidecar = partial_case
    committed = sidecar.read_text(encoding="utf-8")
    sidecar.write_text(committed + orphan_tail, encoding="utf-8")
    assert complete_error_case(csv_path, error) == (1, 2)
    packet = csv_path.with_suffix(".packet.json")
    packet.write_text(json.dumps({
        "schema_version": 2,
        "requests": {"case_seq8_r0:0": {"latency_s": 0.025}},
    }), encoding="utf-8")
    output = csv_path.with_suffix(".merged.csv")
    merge_packet_latency([str(csv_path), str(packet), str(output)])

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert rows[0]["latency_s"] == "0.025000"
    assert rows[0]["latency_app_s"] == "0.04"
    assert [row["status"] for row in rows] == ["ok", "error", "error"]
    assert [row["latency_s"] for row in rows[1:]] == ["nan", "nan"]
    assert [row["error"] for row in rows[1:]] == [error, error]
    assert [json.loads(line)["sniff_group_id"] for line in sidecar.read_text().splitlines()] == [
        "case_seq8_r0", "", "",
    ]
    before = sidecar.read_bytes()
    assert complete_error_case(csv_path, error) == (1, 0)
    assert sidecar.read_bytes() == before


def test_missing_committed_group_blocks_error_publication(partial_case):
    csv_path, sidecar = partial_case
    original = csv_path.read_bytes()
    sidecar.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing committed rows"):
        complete_error_case(csv_path, "runtime OOM")
    assert csv_path.read_bytes() == original


def test_error_csv_failure_can_retry_without_duplicate_groups(partial_case):
    csv_path, sidecar = partial_case
    original = csv_path.read_bytes()
    with patch("csv.DictWriter.writerows", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            complete_error_case(csv_path, "runtime OOM")
    assert csv_path.read_bytes() == original
    assert complete_error_case(csv_path, "runtime OOM") == (1, 2)
    assert [json.loads(line)["sniff_group_id"] for line in sidecar.read_text().splitlines()] == [
        "case_seq8_r0", "", "",
    ]


def test_sidecar_publication_failure_preserves_csv(partial_case, monkeypatch):
    csv_path, sidecar = partial_case
    original_csv, original_groups = csv_path.read_bytes(), sidecar.read_bytes()
    publish = orchestrator.atomic_write

    def fail_sidecar(path, write, **kwargs):
        if str(path) == str(sidecar):
            raise OSError("sidecar disk full")
        publish(path, write, **kwargs)

    monkeypatch.setattr("acprof.host.orchestrator.atomic_write", fail_sidecar)
    with pytest.raises(OSError, match="sidecar disk full"):
        complete_error_case(csv_path, "runtime OOM")
    assert csv_path.read_bytes() == original_csv
    assert sidecar.read_bytes() == original_groups
