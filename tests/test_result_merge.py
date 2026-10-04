import csv
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.config import CSV_FIELDS
from acprof.host.orchestrator import merge_all_csvs
from acprof.result_csv import (
    expected_measurements,
    measurement_key,
    read_result_csv,
    read_result_csv_snapshot,
    result_csv_snapshot_unchanged,
)


class TestResultMerge:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.directory = Path(str(temporary))
        self.destination = self.directory / "result_all.csv"
        self.destination.write_bytes(b"previous result\n")

    def source(self, name="case.csv", *, rows=None, extra_fields=()):
        path = self.directory / name
        row = dict.fromkeys(CSV_FIELDS, "nan")
        row.update(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="64",
                   warmup="0", repeat_idx="0", status="ok", error="")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*CSV_FIELDS, *extra_fields])
            writer.writeheader()
            writer.writerows(rows if rows is not None else [row])
        return path

    def assert_previous_result(self):
        assert (self.destination.read_bytes()) == (b"previous result\n")

    def test_fractional_image_scale_matches_exact_plan_without_tolerance(self):
        scale = 30 / 224
        row = dict(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale=str(scale),
                   warmup="0", repeat_idx="0", status="ok", error="")
        path = self.source(rows=[row])
        expected = expected_measurements([1], [4], ["off"], [scale], 0, 1)
        _, rows = read_result_csv(path, expected=expected)
        assert (rows[0]["input_scale"]) == ("0.13392857142857142")
        changed = {**row, "input_scale": "0.133929"}
        assert (measurement_key(changed)) != (measurement_key(row))

    def test_close_scales_do_not_collapse_to_one_measurement(self):
        keys = expected_measurements([1], [4], ["off"], [0.13392851, 0.13392852], 0, 1)
        assert (len(keys)) == (2)

    def test_snapshot_hashes_exact_bytes_and_preserves_csv_newlines(self):
        raw = (b"\xef\xbb\xbfcpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,note\r"
               b"1,4,off,64,0,0,ok,\"line one\nline two\"\r")
        path = self.directory / "snapshot.csv"
        path.write_bytes(raw)
        snapshot = read_result_csv_snapshot(path)
        assert (snapshot.fields[-2:]) == (["status", "note"])
        assert (snapshot.rows[0]["note"]) == ("line one\nline two")
        assert (snapshot.sha256) == (hashlib.sha256(raw).hexdigest())
        assert (result_csv_snapshot_unchanged(snapshot))

    def test_fractional_error_row_preserves_plan_scale(self):
        from acprof.host.detect import TaskInfo
        from acprof.host.input_plan import serialize_input_scales
        from acprof.host.orchestrator import _write_case_error_csv
        scale = 30 / 224
        serialized = serialize_input_scales([scale])
        assert (float(serialized)) == (scale)
        path = self.directory / 'error.csv'
        _write_case_error_csv(task_info=TaskInfo('org/model', 'image-classification', 'cv',
                                                'transformers_pipeline', 'transformers', 'a' * 40, 'manual'),
                              out_csv=str(path), cpu=1, mem=4, gpu='off', warmup=0, repeat=1,
                              repeat_in_window=1, input_scales=serialized, error='injected startup failure')
        fields, rows = read_result_csv(path, expected=expected_measurements([1], [4], ['off'], [scale], 0, 1))
        assert (fields[-2:]) == (['status', 'error'])
        assert (float(rows[0]['input_scale'])) == (scale)

    def test_missing_source_rejected_before_replacing_previous_result(self):
        source = self.source()
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError), match="missing|exist"):
            merge_all_csvs([str(source), str(self.directory / "missing.csv")], str(self.destination))
        self.assert_previous_result()

    def test_duplicate_source_rejected(self):
        source = self.source()
        with pytest.raises((ValueError, RuntimeError), match="duplicate"):
            merge_all_csvs([str(source), str(source)], str(self.destination))
        self.assert_previous_result()

    def test_duplicate_measurement_across_distinct_files_rejected(self):
        first, second = self.source("first.csv"), self.source("second.csv")
        with pytest.raises((ValueError, RuntimeError), match="duplicate"):
            merge_all_csvs([str(first), str(second)], str(self.destination))
        self.assert_previous_result()

    def test_write_failure_preserves_previous_result_and_removes_temporary_file(self):
        source = self.source()
        before = set(self.directory.iterdir())
        with patch("csv.DictWriter.writerows", side_effect=OSError("injected disk failure")):
            with pytest.raises(OSError, match="injected"):
                merge_all_csvs([str(source)], str(self.destination))
        self.assert_previous_result()
        assert (set(self.directory.iterdir())) == (before)

    def test_publish_failure_preserves_previous_result_and_removes_temporary_file(self):
        source = self.source()
        before = set(self.directory.iterdir())
        with patch("os.replace", side_effect=OSError("injected publish failure")):
            with pytest.raises(OSError, match="injected"):
                merge_all_csvs([str(source)], str(self.destination))
        self.assert_previous_result()
        assert (set(self.directory.iterdir())) == (before)

    def test_malformed_row_rejected(self):
        source = self.source()
        with source.open("a") as stream:
            stream.write("truncated,row\n")
        with pytest.raises((ValueError, RuntimeError), match="row|column"):
            merge_all_csvs([str(source)], str(self.destination))
        self.assert_previous_result()

    def test_empty_case_rejected(self):
        source = self.source(rows=[])
        with pytest.raises((ValueError, RuntimeError), match="empty|no.*rows"):
            merge_all_csvs([str(source)], str(self.destination))
        self.assert_previous_result()

    def test_success_preserves_file_permissions_and_unknown_historical_fields(self):
        source = self.source(extra_fields=("legacy_metric",))
        fields, original_rows = read_result_csv(source)
        original_rows[0]["legacy_metric"] = "preserved,quoted\nvalue"
        # Older or external files may use a different column order.
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(reversed(fields)))
            writer.writeheader()
            writer.writerows(original_rows)
        original_source = source.read_bytes()
        self.destination.chmod(0o640)
        merge_all_csvs([str(source)], str(self.destination))
        with self.destination.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            assert ("legacy_metric") in (reader.fieldnames)
            assert (reader.fieldnames[-3:]) == (["legacy_metric", "status", "error"])
        assert (rows) == (original_rows)
        assert (source.read_bytes()) == (original_source)
        assert (len(rows)) == (1)
        assert (rows[0]["input_scale"]) == ("64")
        assert (os.stat(self.destination).st_mode & 0o777) == (0o640)
