"""Historical compute backfill accepts narrow CSVs, but never repairs broken records."""
import codecs
import csv
import hashlib

import pytest

from acprof.cli import backfill_compute

NCU = "gpu_executed_mflop_per_request_ncu"
RESULT_HEADER = f"gpu_mode,input_scale,latency_app_s,{NCU}"
SUMMARY_HEADER = f"input_scale,{NCU}"


@pytest.mark.parametrize(("source", "content", "diagnostic"), [
    ("result", "gpu_mode,input_scale,latency_app_s,latency_app_s\non,1,0.5,9\n", "duplicate CSV columns"),
    ("result", f"{RESULT_HEADER}\non,1,0.5\n", "malformed CSV row"),
    ("result", f"{RESULT_HEADER}\non,1,0.5,700,extra\n", "malformed CSV row"),
    ("result", f'{RESULT_HEADER},note\non,1,0.5,700,"unfinished\n', "unexpected end of data"),
    ("summary", f"{SUMMARY_HEADER},{NCU}\n1,90,99\n", "duplicate CSV columns"),
    ("summary", f"{SUMMARY_HEADER}\n1\n", "malformed CSV row"),
    ("summary", f"{SUMMARY_HEADER}\n1,42,90\n", "malformed CSV row"),
    ("summary", f'{SUMMARY_HEADER},note\n1,90,"unfinished\n', "unexpected end of data"),
])
def test_malformed_backfill_csv_never_overwrites_the_source(tmp_path, capsys, source, content, diagnostic):
    result = tmp_path / "result.csv"
    result.write_text(f"{RESULT_HEADER}\non,1,0.5,700\n", encoding="utf-8")
    sidecar = tmp_path / "summary.csv"
    sidecar.write_text(f"{SUMMARY_HEADER}\n1,90\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text('{"profiles": {}}', encoding="utf-8")
    damaged = result if source == "result" else sidecar
    damaged.write_text(content, encoding="utf-8")
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in tmp_path.iterdir()}

    with pytest.raises(SystemExit) as error:
        backfill_compute.main([str(result), str(plan), "--ncu-summary", str(sidecar),
                               "--output", str(result), "--overwrite"])

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(damaged) in message
    assert diagnostic in message
    assert {path: hashlib.sha256(path.read_bytes()).digest() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_backfill_preserves_narrow_result_encoding_and_multiline_values(tmp_path, encoding):
    result = tmp_path / "result.csv"
    result.write_bytes(f'{RESULT_HEADER},note\r\non,1,0.5,700,"诊断,\r\n原文"\r\n'.encode(encoding))
    sidecar = tmp_path / "summary.csv"
    sidecar.write_bytes(f"{SUMMARY_HEADER},extra\r\n1,,\r\n".encode(encoding))
    plan = tmp_path / "plan.json"
    plan.write_text('{"profiles": {}}', encoding="utf-8")

    summary = backfill_compute.backfill_compute_profile_csv(
        str(result), str(plan), str(result), ncu_summary=str(sidecar), overwrite=True,
    )

    assert summary.row_count == 1
    assert result.read_bytes().startswith(codecs.BOM_UTF8) == (encoding == "utf-8-sig")
    with result.open(encoding=encoding, newline="") as stream:
        row, = csv.DictReader(stream)
    assert (row["latency_app_s"], row["note"], row[NCU]) == ("0.5", "诊断,\r\n原文", "nan")


def test_backfill_accepts_header_only_result_and_summary(tmp_path):
    result = tmp_path / "result.csv"
    result.write_text(RESULT_HEADER + "\n", encoding="utf-8")
    sidecar = tmp_path / "summary.csv"
    sidecar.write_text(SUMMARY_HEADER + "\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text('{"profiles": {}}', encoding="utf-8")

    summary = backfill_compute.backfill_compute_profile_csv(
        str(result), str(plan), str(result), ncu_summary=str(sidecar), overwrite=True,
    )

    assert summary.row_count == 0
    with result.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames[:4] == RESULT_HEADER.split(",")
        assert list(reader) == []
