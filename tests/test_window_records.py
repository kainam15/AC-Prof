"""Window publication commits last and makes partial writes distinguishable."""
from __future__ import annotations

import csv
import json
import shutil

import pytest

from acprof.config import CSV_FIELDS
from acprof.host.measurement import publication as client_publication
from acprof.telemetry import window_evidence
from acprof.window_records import (
    WindowPublicationContext,
    read_window_record,
    verify_case_window_records,
)


def publication(tmp_path, attempt="a1"):
    return WindowPublicationContext("run", "case", attempt,
                                    str(tmp_path / "windows" / attempt),
                                    str(tmp_path / "telemetry" / attempt))


def publish(tmp_path, context, *, failure=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    evidence = window_evidence(identity=context.identity("case_scale1_r0"), context={"sampling_policy": "burst"},
                               collectors={"resource": {"status": "unavailable", "reason": "test"}})
    with (tmp_path / "requests.jsonl").open("w") as request_stream, (tmp_path / "result.csv").open(
            "w", newline="") as result_stream, (tmp_path / "sniff.jsonl").open("w") as sidecar_stream:
        request = client_publication.append_request_window(
            request_stream, sniff_group_id="case_scale1_r0", input_scale=1.0,
            warmup=0, repeat_idx=0, latencies=[0.1, 0.3],
        )
        writer = csv.DictWriter(result_stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        if failure == "csv":
            class BrokenWriter:
                def writerow(self, row):
                    raise OSError("CSV disk full")
            writer = BrokenWriter()
        client_publication._append_row(
            writer, {"status": "ok", "error": "", "latency_app_s": "0.200000"},
            result_stream, sidecar_stream, "case_scale1_r0", publication=context,
            telemetry_evidence=evidence, request_record=request,
        )
    return context.record_path("case_scale1_r0")


def test_receipt_is_last_commit_after_request_sidecar_and_csv(tmp_path, monkeypatch):
    context = publication(tmp_path)
    original = client_publication.commit_window_record
    observed = []

    def check_commit(*args, **kwargs):
        assert not context.record_path("case_scale1_r0").exists()
        assert json.loads((tmp_path / "requests.jsonl").read_text())["latency_app_s"] == [0.1, 0.3]
        assert json.loads((tmp_path / "sniff.jsonl").read_text())["sniff_group_id"] == "case_scale1_r0"
        with (tmp_path / "result.csv").open() as stream:
            assert list(csv.DictReader(stream))[0]["latency_app_s"] == "0.200000"
        observed.append("commit")
        return original(*args, **kwargs)

    monkeypatch.setattr(client_publication, "commit_window_record", check_commit)
    record_path = publish(tmp_path, context)
    record = read_window_record(record_path)
    assert observed == ["commit"]
    assert record["identity"]["case_attempt_id"] == "a1"
    assert record["row"]["latency_app_s"] == "0.200000"
    assert record["request"]["latency_app_s"] == [0.1, 0.3]
    assert verify_case_window_records(context.records_dir, run_id="run", case_attempt_id="a1") == [record]


@pytest.mark.parametrize("failure", ["sidecar", "csv", "receipt"])
def test_partial_publication_never_appears_as_committed_window(tmp_path, monkeypatch, failure):
    context = publication(tmp_path)
    if failure != "csv":
        def broken(*args, **kwargs):
            raise OSError(f"{failure} disk full")
        monkeypatch.setattr(client_publication,
                            "_append_sniff_group" if failure == "sidecar" else "commit_window_record", broken)
    with pytest.raises(OSError, match="disk full"):
        publish(tmp_path, context, failure=failure)
    assert not context.record_path("case_scale1_r0").exists()
    assert list((tmp_path / "telemetry" / "a1").glob("*.json"))
    assert verify_case_window_records(context.records_dir) == []


def test_tampering_with_raw_evidence_invalidates_receipt(tmp_path):
    context = publication(tmp_path)
    record_path = publish(tmp_path, context)
    record = read_window_record(record_path)
    raw_path = record_path.parent / record["telemetry"]["path"]
    raw_path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_window_record(record_path)


def test_committed_request_row_cannot_be_silently_changed(tmp_path):
    context = publication(tmp_path)
    record_path = publish(tmp_path, context)
    record = json.loads(record_path.read_text())
    record["request"]["latency_app_s"] = [0.01]
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="content hash mismatch"):
        read_window_record(record_path)


def test_case_attempts_keep_same_window_id_separate(tmp_path):
    one = publication(tmp_path, "a1")
    two = publication(tmp_path, "a2")
    path_one = publish(tmp_path, one)
    path_two = publish(tmp_path, two)
    assert path_one != path_two
    assert read_window_record(path_one)["identity"]["case_attempt_id"] == "a1"
    assert read_window_record(path_two)["identity"]["case_attempt_id"] == "a2"
    with pytest.raises(ValueError, match="different case attempt"):
        verify_case_window_records(two.records_dir, case_attempt_id="a1")


def test_receipt_relocates_with_result_root_and_retains_pre_postprocessing_row(tmp_path):
    original = tmp_path / "source"
    context = publication(original)
    path = publish(original, context)
    relative = path.relative_to(original)
    (original / "result.csv").write_text("postprocessed projection\n")
    relocated = tmp_path / "relocated"
    shutil.move(original, relocated)
    record = read_window_record(relocated / relative)
    assert record["row"]["latency_app_s"] == "0.200000"


def test_duplicate_live_publication_does_not_append_another_window(tmp_path):
    context = publication(tmp_path)
    path = publish(tmp_path, context)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="already committed"):
        publish(tmp_path, context)
    assert path.read_bytes() == before
