import csv
import hashlib
import os
from decimal import Inexact, InvalidOperation, localcontext
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.config import CSV_FIELDS
from acprof.host.orchestrator import merge_all_csvs
from acprof.results.csv import (
    ResultValidationError,
    expected_measurements,
    measurement_key,
    merge_result_csvs,
    read_result_csv,
    read_result_csv_snapshot,
    result_csv_snapshot_unchanged,
)
from acprof.results.measurement_plan import RecordedMeasurementOptions, recorded_scales
from tests.results.storage.test_result_row_assembly import row_store


def recorded_plan(*, repeat=1):
    from acprof.results.measurement_plan import RecordedMeasurementPlan

    options = RecordedMeasurementOptions.parse({"cpus": "2", "mems": "8", "gpus": "off", "warmup": 0, "repeat": repeat})
    return RecordedMeasurementPlan(options, (1,))


@pytest.mark.parametrize("storage", ["csv", 1, 2, 3])
@pytest.mark.parametrize("reader", [read_result_csv, read_result_csv_snapshot])
def test_readers_match_compact_plan_without_enumerating_it(tmp_path, storage, reader):
    if storage == "csv":
        path = tmp_path / "case.csv"
        path.write_text("cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx\n2,8,off,1,0,0\n")
    else:
        row_store(tmp_path, storage, count=1, width=0)
        path = tmp_path / "result_layers.json"
    actual = reader(path, expected=recorded_plan())
    assert actual == reader(path, expected=iter([("2", "8", "off", "1", "0", "0")]))
    with pytest.raises(ResultValidationError, match="measurement plan mismatch: .*missing=999999999999, unexpected=0"):
        reader(path, expected=recorded_plan(repeat=10**12))
    if storage != "csv":
        with pytest.raises(ResultValidationError, match="^layered measurement plan mismatch$"):
            reader(path, expected=iter([]))


def test_merge_validates_compact_plan_before_replacing_the_destination(tmp_path):
    case = tmp_path / "case.csv"
    case.write_text("cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx\n2,8,off,1,0,0\n")
    destination = tmp_path / "merged.csv"
    destination.write_bytes(b"previous result\n")
    with pytest.raises(ResultValidationError, match="missing=999999999999, unexpected=0"):
        merge_result_csvs([str(case)], str(destination), expected=recorded_plan(repeat=10**12))
    assert destination.read_bytes() == b"previous result\n"
    assert merge_result_csvs([str(case)], str(destination), expected=recorded_plan()) == 1
    assert read_result_csv(destination)[1][0]["repeat_idx"] == "0"


def test_changed_layer_manifest_is_reported_before_plan_mismatch(tmp_path, monkeypatch):
    from acprof.results import layers

    row_store(tmp_path, 3, count=1, width=0)
    path = tmp_path / "result_layers.json"
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    read_json = layers.read_json_object

    def change_after_first_read(*args, **kwargs):
        manifest = read_json(*args, **kwargs)
        with path.open("ab") as stream:
            stream.write(b"\n")
        return manifest

    monkeypatch.setattr(layers, "read_json_object", change_after_first_read)
    with pytest.raises(ResultValidationError, match="layered results changed during read"):
        layers.read_result_layers(tmp_path, expected_manifest_sha256=original_sha, expected=recorded_plan(repeat=2))


@pytest.mark.parametrize("cpus", ["1,10", "auto"])
def test_sparse_coverage_agrees_with_explicit_key_set_differences(cpus):
    options = RecordedMeasurementOptions.parse({"cpus": cpus, "mems": "4,20", "gpus": "off,on",
                                                "warmup": 2, "repeat": 11})
    scales = recorded_scales([0.125, 10, 2**53 + 1])
    planned = expected_measurements(options.cpus, options.mems, options.gpus, scales, options.warmup, options.repeat)
    row = dict(cpu_cores=options.cpus[0], mem_cap_gb="4.00", gpu_mode="on", input_scale=".125", warmup=1,
               repeat_idx="-0.0", runtime_kind="process")
    mixed = set(sorted(planned)[::3])
    for overrides in ({"cpu_cores": "2"}, {"mem_cap_gb": "3"}, {"input_scale": "0.12500000000000001"},
                      {"warmup": 1, "repeat_idx": 2}, {"warmup": 0, "repeat_idx": "1.1e1"}):
        mixed.add(measurement_key(row | overrides))
    for actual in (set(), mixed, planned):
        assert options.coverage(scales, actual) == {
            "expected": len(planned), "actual": len(actual),
            "missing": len(planned - actual), "unexpected": len(actual - planned),
        }


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

    def test_csv_plan_identifies_signed_zero_without_changing_source(self):
        row = dict(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="64",
                   warmup="-0.00", repeat_idx="-0E+7", status="ok", error="")
        path = self.source(rows=[row])
        expected = expected_measurements([1], [4], ["off"], [64], 0, 1)
        _, rows = read_result_csv(path, expected=expected)
        assert rows[0]["warmup"] == "-0.00"
        assert rows[0]["repeat_idx"] == "-0E+7"
        path = self.source(rows=[row, {**row, "warmup": "0", "repeat_idx": "0"}])
        with pytest.raises(ValueError, match="duplicate measurement"):
            read_result_csv(path)

    def test_csv_identity_preserves_all_significant_digits_without_rounding(self):
        scales = ["1.00000000000000000000000000001", "1.00000000000000000000000000002",
                  "1E+1000000", "1E-1000030", "1E+1"]
        rows = [dict(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale=scale,
                     warmup="0", repeat_idx="0", status="ok", error="") for scale in scales]
        path = self.source(rows=rows)
        with localcontext(prec=2, capitals=0) as context:
            context.traps[Inexact] = True
            _, loaded = read_result_csv(path)
            assert [measurement_key(row)[3] for row in loaded] == scales
            assert measurement_key({**rows[0], "input_scale": scales[0] + "00"}) == measurement_key(rows[0])
            assert not any(context.flags.values())

    @pytest.mark.parametrize("trapped", [True, False])
    def test_rejecting_malformed_window_identity_leaves_caller_flags_untouched(self, trapped):
        row = dict(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="bad", warmup="0", repeat_idx="0")
        with localcontext() as context:
            context.traps[InvalidOperation] = trapped
            context.clear_flags()
            with pytest.raises(ValueError, match="invalid input_scale"):
                measurement_key(row)
            assert not any(context.flags.values())

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
        from acprof.host.measurement.error_rows import write_case_error_csv
        scale = 30 / 224
        serialized = serialize_input_scales([scale])
        assert (float(serialized)) == (scale)
        path = self.directory / 'error.csv'
        write_case_error_csv(task_info=TaskInfo('org/model', 'image-classification', 'cv',
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
