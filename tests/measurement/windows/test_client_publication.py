import csv
import json
from unittest.mock import Mock, patch

import pytest

from acprof.host.measurement import publication as client_publication


def test_result_row_waits_for_sniff_group_durability() -> None:
    writer = Mock(fieldnames=["status", "error"])
    result_stream = Mock()
    sidecar_stream = Mock()

    with patch.object(client_publication.os, "fsync"), patch.object(
        client_publication,
        "_append_sniff_group",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError, match="disk full"):
            client_publication.append_row(
                writer,
                {"status": "ok", "error": ""},
                result_stream,
                sidecar_stream,
                "case_r0",
            )

    writer.writerow.assert_not_called()
    result_stream.flush.assert_not_called()


def test_reconcile_sniff_group_sidecar_discards_uncommitted_tail(tmp_path) -> None:
    result_csv = tmp_path / "result.csv"
    with result_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["status", "error"])
        writer.writerow(["ok", ""])

    sidecar = tmp_path / "result.csv.sniff_groups.jsonl"
    committed = json.dumps({"sniff_group_id": "case_r0"}) + "\n"
    orphan = json.dumps({"sniff_group_id": "case_r1"}) + "\n"
    sidecar.write_text(committed + orphan, encoding="utf-8")

    client_publication.reconcile_sniff_group_sidecar(result_csv, sidecar)

    assert sidecar.read_text(encoding="utf-8") == committed


@pytest.mark.parametrize("blank_position", ["between", "trailing"])
@pytest.mark.parametrize("crash_tail", [False, True])
def test_reconcile_counts_csv_records_across_blank_lines(tmp_path, blank_position, crash_tail) -> None:
    result = tmp_path / "result.csv"
    with result.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["status", "error"])
        writer.writerow(["", ""])
        if blank_position == "between":
            writer.writerow([])
        writer.writerow(["error", 'line one, "quoted"\n\nline two'])
        if blank_position == "trailing":
            writer.writerow([])
    csv_before = result.read_bytes()
    groups = tmp_path / "sniff_groups.jsonl"
    committed = b'{"sniff_group_id":"first"}\n{"sniff_group_id":"second"}\n'
    tail = b'{"sniff_group_id":"orphan"}\n' if crash_tail else b""
    groups.write_bytes(committed + tail)

    client_publication.reconcile_sniff_group_sidecar(result, groups)

    assert groups.read_bytes() == committed
    assert result.read_bytes() == csv_before


def test_reconcile_sniff_group_sidecar_rejects_missing_committed_rows(tmp_path) -> None:
    result_csv = tmp_path / "result.csv"
    with result_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["status", "error"])
        writer.writerow(["ok", ""])
        writer.writerow(["ok", ""])

    sidecar = tmp_path / "result.csv.sniff_groups.jsonl"
    sidecar.write_text(
        json.dumps({"sniff_group_id": "case_r0"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="1 sniff-group rows for 2 committed CSV rows"):
        client_publication.reconcile_sniff_group_sidecar(result_csv, sidecar)


def test_reconcile_sniff_group_sidecar_rejects_nonfinite_committed_row(tmp_path) -> None:
    result_csv = tmp_path / "result.csv"
    with result_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["status", "error"])
        writer.writerow(["ok", ""])

    sidecar = tmp_path / "result.csv.sniff_groups.jsonl"
    sidecar.write_text('{"sniff_group_id":"case_r0","corrupt_metric":NaN}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="committed row 1 is malformed"):
        client_publication.reconcile_sniff_group_sidecar(result_csv, sidecar)
