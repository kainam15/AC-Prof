import csv
import json
import os
import stat
import tempfile
from unittest.mock import patch

import pytest

from acprof.cli.backfill_compute import (
    COMPUTE_PROFILE_FIELDS,
    backfill_compute_profile_csv,
)
from acprof.host.profiling.compute_plan import find_compute_profile_entry

REMOVED_LEGACY_COMPUTE_FIELDS = (
    "compute_profile_tool",
    "model_mflop_per_request",
    "compute_mflops_app",
    "compute_mflops",
    "compute_profile_error",
)
OBSOLETE_RESULT_FIELDS = (
    *REMOVED_LEGACY_COMPUTE_FIELDS,
    "gpu_profile_report_ncu",
)


class TestBackfillComputeProfile:
    @pytest.mark.parametrize("scales", [
        ("1.0000001", "1.0000002"),
        ("1", "1.0000000000000001"),
        ("9007199254740992", "9007199254740993"),
    ], ids=["float-neighbors", "decimal-neighbors", "integer-neighbors"])
    @pytest.mark.parametrize("entry_count", [1, 2])
    def test_ncu_summary_backfill_keeps_distinct_scale_identity(self, tmp_path, scales, entry_count):
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        plan_path = tmp_path / "compute_profile_plan.json"
        summary_path = tmp_path / "ncu_summary.csv"
        field = "gpu_executed_mflop_per_request_ncu"
        self._write_csv(input_csv, ["gpu_mode", "input_scale", "latency_app_s"], [
            {"gpu_mode": "on", "input_scale": scale, "latency_app_s": 1.0}
            for scale in scales
        ])
        self._write_csv(summary_path, ["input_scale", field], [
            {"input_scale": scale, field: value}
            for scale, value in zip(scales[:entry_count], (10.0, 20.0), strict=False)
        ])
        plan_path.write_text(json.dumps({"profiles": {}}))
        before = input_csv.read_bytes()

        backfill_compute_profile_csv(str(input_csv), str(plan_path), str(output_csv), ncu_summary=str(summary_path))

        with output_csv.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert tuple(row["input_scale"] for row in rows) == scales
        assert float(rows[0][field]) == 10.0
        if entry_count == 2:
            assert float(rows[1][field]) == 20.0
        else:
            assert rows[1][field] == "nan"
            assert rows[1]["compute_profile_error_ncu"]
        assert input_csv.read_bytes() == before

    @pytest.mark.parametrize(("first", "alias", "second", "missing"), [
        ("1.00000000000000001", "100000000000000001e-17", "1.00000000000000002", "1.00000000000000003"),
        ("9007199254740995", "90071992547409950e-1", "9007199254740996", "9007199254740997"),
    ], ids=["json-decimals", "json-integers"])
    def test_plan_resolves_original_json_scale_tokens(self, tmp_path, first, alias, second, missing):
        source, plan, output = (tmp_path / name for name in ("input.csv", "plan.json", "output.csv"))
        field = "model_logical_mflop_per_request_torch_profiler_eager"
        self._write_csv(source, ["gpu_mode", "input_scale", "latency_app_s"], [
            {"gpu_mode": "on", "input_scale": scale, "latency_app_s": "1"} for scale in (alias, second, missing)
        ])
        # Keep the JSON numeric lexemes. Building the fixture with Python floats loses the distinction.
        entries = ",".join(
            f'{{"input_scale":{token},"{field}":{value}}}'
            for token, value in ((first, 10), (json.dumps(alias), 99), (second, 20))
        )
        plan.write_text('{"profiles":{"gpu":{"torch_profiler_eager":{"entries":[' + entries + ']}}}}')
        summary = backfill_compute_profile_csv(str(source), str(plan), str(output))

        with output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [row[field] for row in rows] == ["10.000000", "20.000000", "nan"]
        assert [row["input_scale"] for row in rows] == [alias, second, missing]
        assert rows[2]["compute_profile_error_torch_profiler_eager"].startswith("compute_profile_missing_scale:")
        assert summary.diagnostic_count == 1

    def test_invalid_decimal_coordinate_cannot_borrow_zero_observation(self, tmp_path):
        source, plan, output = (tmp_path / name for name in ("input.csv", "plan.json", "out.csv"))
        source.write_text("gpu_mode,input_scale,latency_app_s\noff,1e-9999999999999999999999,1\noff,0,1\n")
        field = "model_logical_mflop_per_request_torch_profiler_eager"
        plan.write_text(json.dumps({"profiles": {"cpu": {"torch_profiler_eager": {
            "entries": [{"input_scale": 0, field: 19}],
        }}}}))

        summary = backfill_compute_profile_csv(str(source), str(plan), str(output))

        with output.open(newline="") as stream:
            invalid, zero = csv.DictReader(stream)
        assert invalid[field] == "nan"
        assert invalid["compute_profile_error_torch_profiler_eager"].startswith("compute_profile_missing_scale:")
        assert zero[field] == "19.000000"
        assert summary.diagnostic_count == 1

    def test_ncu_summary_rejects_equivalent_scale_spellings_before_overwrite(self, tmp_path):
        source, plan, summary, output = (tmp_path / name for name in ("input.csv", "plan.json", "ncu.csv", "out.csv"))
        source.write_text("gpu_mode,input_scale,latency_app_s\non,64,1\n")
        plan.write_text('{"profiles":{}}')
        summary.write_text("input_scale,gpu_hw_mflop_per_request_ncu\n64.000,10\n6.4e1,20\n")
        output.write_bytes(b"previous output")

        with pytest.raises(ValueError, match="NCU summary CSV has duplicate input_scale"):
            backfill_compute_profile_csv(str(source), str(plan), str(output), ncu_summary=str(summary), overwrite=True)
        assert output.read_bytes() == b"previous output"

    def _write_csv(self, path, fieldnames, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_vendor_plan(self, path):
        plan = {
            "profiles": {
                "cpu": {
                    "intel_advisor": {
                        "tool": "intel_advisor",
                        "entries": [
                            {
                                "input_scale": 64.0,
                                "model_mflop_per_request": 200.0,
                                "error": "",
                            }
                        ],
                    }
                },
                "gpu": {
                    "ncu": {
                        "tool": "ncu",
                        "entries": [
                            {
                                "input_scale": 64.0,
                                "gpu_executed_mflop_per_request_ncu": 100.0,
                                "error": "",
                            }
                        ],
                    }
                },
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f)

    def _write_dual_tool_plan(self, path):
        plan = {
            "profiles": {
                "cpu": {
                    "torch_profiler_eager": {
                        "tool": "torch_profiler_eager",
                        "entries": [
                            {
                                "input_scale": 64.0,
                                "model_logical_mflop_per_request_torch_profiler_eager": 200.0,
                                "error": "",
                            }
                        ],
                    }
                },
                "gpu": {
                    "torch_profiler_eager": {
                        "tool": "torch_profiler_eager",
                        "entries": [
                            {
                                "input_scale": 64.0,
                                "model_logical_mflop_per_request_torch_profiler_eager": 300.0,
                                "error": "",
                            }
                        ],
                    },
                    "ncu": {
                        "tool": "ncu",
                        "entries": [
                            {
                                "input_scale": 64.0,
                                "gpu_executed_mflop_per_request_ncu": 100.0,
                                "gpu_executed_tensor_mflop_per_request_ncu": 90.0,
                                "gpu_executed_scalar_mflop_per_request_ncu": 10.0,
                                "gpu_executed_tensor_share_pct_ncu": 90.0,
                                "gpu_kernel_launch_count_per_request_ncu": 5.0,
                                "gpu_kernel_time_sum_ms_per_request_ncu": 1.0,
                                "error": "",
                            }
                        ],
                    },
                },
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f)

    def test_entry_error_does_not_inherit_another_scale_failure(self):
        plan = {
            "profiles": {
                "gpu": {
                    "torch_profiler_eager": {
                        "tool": "torch_profiler_eager",
                        "error": "scale_2_failed",
                        "entries": [
                            {
                                "input_scale": 1,
                                "model_logical_mflop_per_request_torch_profiler_eager": 10,
                                "error": "",
                            },
                            {
                                "input_scale": 2,
                                "model_logical_mflop_per_request_torch_profiler_eager": None,
                                "error": "scale_2_failed",
                            },
                        ],
                    },
                    "ncu": {
                        "tool": "ncu",
                        "error": "",
                        "entries": [
                            {
                                "input_scale": 1,
                                "gpu_executed_mflop_per_request_ncu": 20,
                                "error": "",
                            }
                        ],
                    },
                }
            },
        }

        profile = find_compute_profile_entry(plan, "on", 1)

        assert (profile["compute_profile_error_torch_profiler_eager"]) == ("")
        assert (profile["compute_profile_error_ncu"]) == ("")
        assert (profile["model_logical_mflop_per_request_torch_profiler_eager"]) == (10.0)

    def test_vendor_cpu_is_not_mislabeled_as_torch_profile(self):
        plan = {
            "compute_profile_tool_mode": "vendor",
            "profiles": {
                "cpu": {
                    "intel_advisor": {
                        "tool": "intel_advisor",
                        "entries": [
                            {
                                "input_scale": 64,
                                "model_mflop_per_request": 321,
                                "error": "",
                            }
                        ],
                    }
                }
            },
        }

        profile = find_compute_profile_entry(plan, "off", 64)

        assert ("tool") not in (profile)
        assert ("model_mflop_per_request") not in (profile)
        assert ("error") not in (profile)
        logical = profile[
            "model_logical_mflop_per_request_torch_profiler_eager"
        ]
        assert (logical) != (logical)
        assert (profile["compute_profile_error_torch_profiler_eager"]) == ("")

    def test_backfills_cpu_and_gpu_and_preserves_other_values_and_order(self):
        fields = [
            "row_marker",
            "gpu_mode",
            "input_scale",
            "latency_app_s",
            "latency_s",
            *REMOVED_LEGACY_COMPUTE_FIELDS,
            *COMPUTE_PROFILE_FIELDS,
            "gpu_profile_report_ncu",
            "status",
        ]
        input_rows = [
            {
                "row_marker": "first",
                "gpu_mode": "off",
                "input_scale": "64.0000005",
                "latency_app_s": "0.5",
                "latency_s": "0.25",
                "compute_profile_tool": "old",
                "model_mflop_per_request": "1",
                "compute_mflops_app": "2",
                "compute_mflops": "3",
                "compute_profile_error": "old error",
                "gpu_profile_report_ncu": "legacy/ncu.csv",
                "status": "ok",
            },
            {
                "row_marker": "second",
                "gpu_mode": "on",
                "input_scale": "64",
                "latency_app_s": "0.4",
                "latency_s": "nan",
                "compute_profile_tool": "old",
                "model_mflop_per_request": "1",
                "compute_mflops_app": "2",
                "compute_mflops": "3",
                "compute_profile_error": "old error",
                "gpu_profile_report_ncu": "legacy/ncu.csv",
                "status": "ok",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            output_csv = os.path.join(tmp, "result_all.with_compute.csv")
            self._write_csv(input_csv, fields, input_rows)
            self._write_vendor_plan(plan_path)
            with open(input_csv, "rb") as f:
                original_input = f.read()

            summary = backfill_compute_profile_csv(
                input_csv,
                plan_path,
                output_csv,
            )

            with open(output_csv, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                output_fields = reader.fieldnames
                output_rows = list(reader)
            with open(input_csv, "rb") as f:
                unchanged_input = f.read()

        assert (summary.row_count) == (2)
        assert (summary.diagnostic_count) == (0)
        assert (output_fields) == ([field for field in fields if field not in OBSOLETE_RESULT_FIELDS])
        assert (all(
                field not in row
                for row in output_rows
                for field in OBSOLETE_RESULT_FIELDS
            ))
        assert (unchanged_input) == (original_input)
        assert ([(row["row_marker"], row["status"]) for row in output_rows]) == ([("first", "ok"), ("second", "ok")])
        assert (output_rows[0][
                "model_logical_mflop_per_request_torch_profiler_eager"
            ]) == ("nan")
        assert (output_rows[1]["gpu_executed_mflop_per_request_ncu"]) == ("100.000000")
        assert (output_rows[1]["gpu_executed_mflops_app_ncu"]) == ("250.000000")
        assert (output_rows[1]["gpu_executed_mflops_packet_ncu"]) == ("250.000000")

    def test_dual_tool_plan_backfills_only_explicit_metrics(self):
        fields = [
            "gpu_mode",
            "input_scale",
            "latency_app_s",
            "latency_s",
            "status",
        ]
        rows = [
            {
                "gpu_mode": "off",
                "input_scale": "64",
                "latency_app_s": "0.5",
                "latency_s": "nan",
                "status": "ok",
            },
            {
                "gpu_mode": "on",
                "input_scale": "64",
                "latency_app_s": "0.5",
                "latency_s": "0.25",
                "status": "ok",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            output_csv = os.path.join(tmp, "result_all.with_compute.csv")
            self._write_csv(input_csv, fields, rows)
            self._write_dual_tool_plan(plan_path)

            summary = backfill_compute_profile_csv(
                input_csv,
                plan_path,
                output_csv,
            )
            with open(output_csv, encoding="utf-8", newline="") as f:
                output_rows = list(csv.DictReader(f))

        assert (summary.diagnostic_count) == (0)
        cpu_row, gpu_row = output_rows
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            assert (field) not in (cpu_row)
            assert (field) not in (gpu_row)
        assert (cpu_row["model_logical_mflops_packet_torch_profiler_eager"]) == ("400.000000")
        assert (cpu_row["gpu_executed_mflop_per_request_ncu"]) == ("nan")
        assert (cpu_row["compute_profile_error_ncu"]) == ("")

        assert (gpu_row["model_logical_mflops_packet_torch_profiler_eager"]) == ("1200.000000")
        assert (gpu_row["gpu_executed_mflop_per_request_ncu"]) == ("100.000000")
        assert (gpu_row["gpu_executed_mflops_app_ncu"]) == ("200.000000")
        assert (gpu_row["gpu_executed_mflops_packet_ncu"]) == ("400.000000")

    def test_ncu_summary_overrides_gpu_rows_only_with_numeric_scale_matching(self):
        fields = [
            "row_marker",
            "gpu_mode",
            "input_scale",
            "latency_app_s",
            "latency_s",
        ]
        rows = [
            {
                "row_marker": "cpu",
                "gpu_mode": "off",
                "input_scale": "64",
                "latency_app_s": "0.5",
                "latency_s": "0.25",
            },
            {
                "row_marker": "gpu",
                "gpu_mode": "on",
                "input_scale": "64",
                "latency_app_s": "0.5",
                "latency_s": "0.25",
            },
        ]
        ncu_fields = [
            "input_scale",
            "gpu_hw_mflop_per_request_ncu",
            "gpu_hw_tensor_mflop_per_request_ncu",
            "gpu_hw_scalar_mflop_per_request_ncu",
            "gpu_hw_tensor_share_pct_ncu",
            "ncu_kernel_count",
            "ncu_kernel_time_sum_ms",
            "ncu_report",
        ]
        ncu_rows = [
            {
                "input_scale": "64.0000000",
                "gpu_hw_mflop_per_request_ncu": "120",
                "gpu_hw_tensor_mflop_per_request_ncu": "100",
                "gpu_hw_scalar_mflop_per_request_ncu": "20",
                "gpu_hw_tensor_share_pct_ncu": "83.333333",
                "ncu_kernel_count": "7",
                "ncu_kernel_time_sum_ms": "2.5",
                "ncu_report": "results/model/compute_profiles_tensor/ncu_scale_64.csv",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            ncu_summary = os.path.join(tmp, "gpu_hardware_flops_by_scale.csv")
            output_csv = os.path.join(tmp, "result_all.with_compute.csv")
            self._write_csv(input_csv, fields, rows)
            self._write_dual_tool_plan(plan_path)
            self._write_csv(ncu_summary, ncu_fields, ncu_rows)

            summary = backfill_compute_profile_csv(
                input_csv,
                plan_path,
                output_csv,
                ncu_summary=ncu_summary,
            )
            with open(output_csv, encoding="utf-8", newline="") as f:
                output_rows = list(csv.DictReader(f))

        assert (summary.row_count) == (2)
        assert (summary.diagnostic_count) == (0)
        cpu_row, gpu_row = output_rows
        assert (cpu_row["row_marker"]) == ("cpu")
        assert (cpu_row["gpu_executed_mflop_per_request_ncu"]) == ("nan")
        assert (cpu_row["compute_profile_error_ncu"]) == ("")

        assert (gpu_row["row_marker"]) == ("gpu")
        assert (gpu_row["gpu_executed_mflop_per_request_ncu"]) == ("120.000000")
        assert (gpu_row["gpu_executed_tensor_mflop_per_request_ncu"]) == ("100.000000")
        assert (gpu_row["gpu_executed_scalar_mflop_per_request_ncu"]) == ("20.000000")
        assert (gpu_row["gpu_executed_tensor_share_pct_ncu"]) == ("83.333333")
        assert (gpu_row["gpu_executed_mflops_app_ncu"]) == ("240.000000")
        assert (gpu_row["gpu_executed_mflops_packet_ncu"]) == ("480.000000")
        assert (gpu_row["gpu_kernel_launch_count_per_request_ncu"]) == ("7.000000")
        assert (gpu_row["gpu_kernel_time_sum_ms_per_request_ncu"]) == ("2.500000")
        assert ("gpu_profile_report_ncu") not in (gpu_row)

    def test_missing_scale_writes_nan_metrics_and_diagnostic(self):
        fields = [
            "gpu_mode",
            "input_scale",
            "latency_app_s",
            "latency_s",
            "note",
        ]
        rows = [
            {
                "gpu_mode": "off",
                "input_scale": "65",
                "latency_app_s": "0.5",
                "latency_s": "0.25",
                "note": "keep me",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            output_csv = os.path.join(tmp, "result_all.with_compute.csv")
            self._write_csv(input_csv, fields, rows)
            self._write_dual_tool_plan(plan_path)

            summary = backfill_compute_profile_csv(
                input_csv,
                plan_path,
                output_csv,
            )
            with open(output_csv, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                output_fields = reader.fieldnames or []
                output_row = next(reader)

        assert (summary.diagnostic_count) == (1)
        assert (output_fields) == (fields + list(COMPUTE_PROFILE_FIELDS))
        assert (output_row["note"]) == ("keep me")
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            assert (field) not in (output_row)
        assert (output_row[
                "model_logical_mflop_per_request_torch_profiler_eager"
            ]) == ("nan")
        assert (output_row["compute_profile_error_torch_profiler_eager"]) == ("compute_profile_missing_scale:65")

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_atomic_publish_syncs_output_directory(self, overwrite):
        fields = ["gpu_mode", "input_scale", "latency_app_s"]
        rows = [{"gpu_mode": "off", "input_scale": "64", "latency_app_s": "0.5"}]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            output_csv = os.path.join(tmp, "result_all.with_compute.csv")
            self._write_csv(input_csv, fields, rows)
            self._write_vendor_plan(plan_path)
            if overwrite:
                with open(output_csv, "w", encoding="utf-8") as f:
                    f.write("old output")

            synced_file_types = []
            real_fsync = os.fsync

            def track_fsync(fd):
                synced_file_types.append(stat.S_IFMT(os.fstat(fd).st_mode))
                real_fsync(fd)

            with patch("acprof.cli.backfill_compute.os.fsync", side_effect=track_fsync):
                backfill_compute_profile_csv(
                    input_csv,
                    plan_path,
                    output_csv,
                    overwrite=overwrite,
                )

        assert stat.S_IFREG in synced_file_types
        assert stat.S_IFDIR in synced_file_types

    def test_existing_output_is_not_overwritten_without_explicit_permission(self):
        fields = ["gpu_mode", "input_scale", "latency_app_s"]
        rows = [{"gpu_mode": "off", "input_scale": "64", "latency_app_s": "0.5"}]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            output_csv = os.path.join(tmp, "existing.csv")
            self._write_csv(input_csv, fields, rows)
            self._write_vendor_plan(plan_path)
            with open(output_csv, "w", encoding="utf-8") as f:
                f.write("do not replace")

            with pytest.raises(FileExistsError):
                backfill_compute_profile_csv(
                    input_csv,
                    plan_path,
                    output_csv,
                )
            with open(output_csv, encoding="utf-8") as f:
                output_contents = f.read()

        assert (output_contents) == ("do not replace")

    def test_explicit_overwrite_can_atomically_replace_input_path(self):
        fields = ["gpu_mode", "input_scale", "latency_app_s"]
        rows = [{"gpu_mode": "off", "input_scale": "64", "latency_app_s": "0.5"}]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            self._write_csv(input_csv, fields, rows)
            self._write_dual_tool_plan(plan_path)

            backfill_compute_profile_csv(
                input_csv,
                plan_path,
                input_csv,
                overwrite=True,
            )
            with open(input_csv, encoding="utf-8", newline="") as f:
                output_row = next(csv.DictReader(f))

        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            assert (field) not in (output_row)
        assert (output_row[
                "model_logical_mflop_per_request_torch_profiler_eager"
            ]) == ("200.000000")
        assert (output_row[
                "model_logical_mflops_app_torch_profiler_eager"
            ]) == ("400.000000")
        assert (output_row[
                "model_logical_mflops_packet_torch_profiler_eager"
            ]) == ("400.000000")
