"""审计真实 CSV/JSON 的故障与历史兼容边界。"""
import csv
import json
from pathlib import Path

import pytest

from acprof.analysis.audit import audit_result
from acprof.config import CSV_FIELDS
from acprof.results.metadata import ResultMetadataSnapshot


@pytest.fixture
def recorded_plan():
    return {
        "status": "complete",
        "options": {"cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 1},
        "runtime": {"planned": {"scales": [1]}},
    }


class TestResultAudit:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.temporary = tmp_path
        self.root = Path(str(self.temporary))
        self.path = self.root / "result_all.csv"
    def test_v2_input_plan_symlink_is_reported_as_invalid_without_reading_it(self):
        from acprof.results.layout import ArtifactLayout
        ArtifactLayout.for_new_run(self.root).initialize()
        self.write(self.row())
        (self.root / "static_meta.json").write_text(json.dumps({"input_scale_plan_sha256": "untrusted"}))
        (self.root / "metadata/input_scale_plan.json").symlink_to(self.path)
        report = audit_result(self.path)
        assert not (report["valid"])
        assert ("input_plan_hash") in ({issue["code"] for issue in report["issues"]})

    def row(self, **changes):
        return {**dict.fromkeys(CSV_FIELDS, "nan"), "cpu_cores": "1", "mem_cap_gb": "4",
                "gpu_mode": "off", "input_scale": "64", "repeat_idx": "0", "warmup": "0",
                "status": "ok", "error": "", "task_param": "{}", **changes}

    def write(self, *rows, fields=CSV_FIELDS):
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_zero_is_not_missing_and_gpu_off_is_inapplicable(self):
        self.write(self.row(container_io_read_bytes_per_request="0"))
        before = self.path.read_bytes()
        report = audit_result(self.path)
        assert (report["valid"])
        assert ("container_io_read_bytes_per_request") not in (report["missing_metrics"])
        assert (report["missing_metrics"]["gpu_energy_eff_j"]) == ({"not_applicable": 1})
        assert (self.path.read_bytes()) == (before)
        assert (list(self.root.iterdir())) == ([self.path])

    def test_legacy_directory_and_file_share_the_same_audited_snapshot(self):
        self.write(self.row())
        expected = audit_result(self.path)
        assert expected["valid"] and expected["counts"]["formal_ok"] == 1
        assert audit_result(self.root) == expected
        assert audit_result(self.path, metadata_snapshot=ResultMetadataSnapshot(self.root)) == expected
        assert audit_result(self.root, metadata_snapshot=ResultMetadataSnapshot(self.path)) == expected

    def test_explicit_missing_layer_file_stays_literal_in_a_legacy_directory(self):
        self.write(self.row())
        layer_path = self.root / "result_layers.json"
        report = audit_result(layer_path)
        assert not report["valid"]
        assert report["result_csv"] == str(layer_path)
        assert report["counts"]["rows"] == 0

    def test_explicit_csv_aliases_must_share_the_snapshot_metadata_root(self):
        self.write(self.row())
        sidecar_root = self.root / "export"
        sidecar_root.mkdir()
        (sidecar_root / "run_state.json").write_text(json.dumps({"run_id": "from-export", "status": "complete"}))
        alias = sidecar_root / "alias.csv"
        alias.symlink_to(self.path)
        other_alias = sidecar_root / "copy.csv"
        other_alias.symlink_to(self.path)
        directory_alias = self.root / "export-link"
        directory_alias.symlink_to(sidecar_root, target_is_directory=True)
        metadata = ResultMetadataSnapshot(alias)
        for source in (alias, other_alias, directory_alias / alias.name):
            report = audit_result(source, metadata_snapshot=metadata)
            assert report["valid"] and report["run_id"] == "from-export"
        report = audit_result(self.path, metadata_snapshot=metadata)
        assert not report["valid"]
        assert any(issue["code"] == "invalid_layout" and "different path" in issue["message"]
                   for issue in report["issues"])

    def test_cyclic_layer_symlink_is_reported_without_resolving_the_failed_artifact(self):
        self.write(self.row())
        layer_path = self.root / "result_layers.json"
        layer_path.symlink_to(layer_path.name)
        for source in (self.root, layer_path):
            report = audit_result(source)
            assert not report["valid"]
            assert any(issue["code"] == "invalid_layout" for issue in report["issues"])
            assert report["counts"]["rows"] == 0

    @pytest.mark.parametrize("layer_present", [True, False])
    def test_declared_v2_directory_keeps_layer_errors_when_a_csv_export_exists(self, layer_present):
        from acprof.results.layout import ArtifactLayout

        ArtifactLayout.for_new_run(self.root).initialize()
        self.write(self.row())
        layer_path = self.root / "result_layers.json"
        if layer_present:
            layer_path.write_text("{")
        else:
            layer_path.unlink()
        report = audit_result(self.root)
        assert not report["valid"]
        assert report["result_csv"] == str(layer_path)
        assert report["counts"]["rows"] == 0
        assert audit_result(self.path)["counts"]["formal_ok"] == 1

    def test_csv_snapshot_uses_two_bounded_binary_passes(self, monkeypatch):
        self.write(self.row())
        original_open = Path.open
        original_read_bytes = Path.read_bytes
        modes = []

        def tracked_open(target, *args, **kwargs):
            if target == self.path:
                modes.append(args[0] if args else kwargs.get("mode", "r"))
            return original_open(target, *args, **kwargs)

        def reject_materialized_read(target):
            if target == self.path:
                raise AssertionError("audit must not materialize the result CSV")
            return original_read_bytes(target)

        monkeypatch.setattr(Path, "open", tracked_open)
        monkeypatch.setattr(Path, "read_bytes", reject_materialized_read)
        report = audit_result(self.path)
        assert (report["valid"])
        assert (modes) == (["rb", "rb"])

    def test_csv_change_between_streaming_passes_is_reported(self, monkeypatch):
        self.write(self.row())
        original_open = Path.open
        binary_opens = 0

        def mutate_before_second_pass(target, *args, **kwargs):
            nonlocal binary_opens
            mode = args[0] if args else kwargs.get("mode", "r")
            if target == self.path and mode == "rb":
                binary_opens += 1
                if binary_opens == 2:
                    with original_open(target, "ab") as stream:
                        stream.write(b"\n")
            return original_open(target, *args, **kwargs)

        monkeypatch.setattr(Path, "open", mutate_before_second_pass)
        report = audit_result(self.path)
        assert not (report["valid"])
        assert ("changing_snapshot") in ({issue["code"] for issue in report["issues"]})
        assert (binary_opens) == (2)

    def test_formal_filter_excludes_warmup_warn_and_error(self):
        self.write(self.row(), self.row(warmup="1"), self.row(repeat_idx="1", status="warn", error="idle drift"),
                   self.row(repeat_idx="2", status="error", error="timeout"))
        report = audit_result(self.path)
        assert (report["counts"]) == ({"rows": 4, "formal_ok": 1, "warmup": 1, "warn": 1, "error": 1})
        assert (report["completion"]) == ("unknown")

    def test_duplicate_and_invalid_status_are_reported(self):
        for rows, code in (([self.row(), self.row()], "invalid_csv"),
                           ([self.row(status="finished")], "invalid_status")):
            self.write(*rows)
            report = audit_result(self.path)
            assert not (report["valid"])
            assert (code) in ([issue["code"] for issue in report["issues"]])

    def test_invalid_number_is_not_explained_as_hardware_unavailable(self):
        self.write(self.row(latency_app_s="inf"))
        report = audit_result(self.path)
        assert not (report["valid"])
        assert ("invalid_number") in ([issue["code"] for issue in report["issues"]])

    @pytest.mark.parametrize(
        ("name", "payload"),
        [
            ("static_meta.json", b'{"schema_version": 7, "value": NaN}'),
            ("static_meta.json", b'{"schema_version": 7, "value": 1e999}'),
            ("static_meta.json", b"\xff"),
            ("runtime_validation.json", b'{"devices": []}'),
            ("runtime_validation.json", b'{"devices": {"off": []}}'),
            ("runtime_failures.json", b'{"failures": {}}'),
            ("runtime_failures.json", b'{"failures": [{"reason_code": "inference_failed"}]}'),
            ("model_resolution.json", b'{"failure": []}'),
        ],
    )
    def test_invalid_recorded_metadata_becomes_an_audit_issue(self, name, payload):
        self.write(self.row())
        path = self.root / name
        path.write_bytes(payload)
        report = audit_result(self.path)
        assert not (report["valid"])
        assert ("invalid_metadata") in ({issue["code"] for issue in report["issues"]})
        assert (path.read_bytes()) == (payload)

    def test_oversized_recorded_metadata_becomes_an_audit_issue_without_rewriting(self):
        self.write(self.row())
        path = self.root / "static_meta.json"
        content = b" " * (4 * 1024 * 1024 + 1)
        path.write_bytes(content)
        report = audit_result(self.path)
        assert not (report["valid"])
        assert any(issue["code"] == "invalid_metadata" and "4 MiB" in issue["message"]
                   for issue in report["issues"])
        assert (path.stat().st_size) == (len(content))

    def test_plan_hash_and_row_coverage_are_checked(self):
        self.write(self.row())
        (self.root / "static_meta.json").write_text(json.dumps({"input_scale_plan_sha256": "0" * 64}))
        (self.root / "input_scale_plan.json").write_text('{}')
        (self.root / "run_state.json").write_text(json.dumps({
            "schema_version": 1, "status": "complete",
            "options": {"cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 2},
            "runtime": {"planned": {"scales": [64]}},
        }))
        report = audit_result(self.path)
        assert not (report["valid"])
        codes = {issue["code"] for issue in report["issues"]}
        assert ({"input_plan_hash", "plan_coverage"} <= codes)
        assert (report["coverage"]["missing"]) == (1)

    @pytest.mark.parametrize(("field", "invalid"), [
        ("warmup", -1), ("warmup", False), ("repeat", 0), ("repeat", -1), ("repeat", True),
        ("cpus", "0"), ("mems", "-4"), ("cpus", [1]), ("gpus", "auto"), ("gpus", ""),
        ("cpus", "1,01"), ("mems", "4,4"), ("gpus", "off,off"),
        pytest.param("scales", [], id="empty-scales"),
        pytest.param("scales", [1, 1.0], id="duplicate-scale"),
        pytest.param("scales", [10**23, 1e23], id="duplicate-csv-coordinate"),
        pytest.param("scales", [True], id="boolean-scale"),
        pytest.param("scales", ["1"], id="encoded-scale"),
        pytest.param("scales", "1", id="string-scale-axis"),
        pytest.param("scales", {"1": 1}, id="object-scale-axis"),
        pytest.param("scales", [0], id="zero-scale"),
        pytest.param("scales", [-1], id="negative-scale"),
        pytest.param("scales", [10**310], id="scale-exceeds-finite-float"),
    ])
    def test_invalid_persisted_plan_keeps_coverage_unknown(self, recorded_plan, field, invalid):
        target = recorded_plan["runtime"]["planned"] if field == "scales" else recorded_plan["options"]
        target[field] = invalid
        (self.root / "run_state.json").write_text(json.dumps(recorded_plan))
        self.write(self.row(input_scale="1"))
        report = audit_result(self.path)
        assert not report["valid"]
        assert report["coverage"] is None
        assert report["measurement_status"] == "unknown"
        assert report["counts"]["formal_ok"] == 1
        assert any(issue["code"] == "invalid_run_state" for issue in report["issues"])

    def test_legacy_plan_with_fractional_scales_and_minimal_options(self, recorded_plan):
        recorded_plan["runtime"]["planned"]["scales"] = [0.5, 1]
        recorded_plan["options"]["warmup"] = 1
        (self.root / "run_state.json").write_text(json.dumps(recorded_plan))
        self.write(*(self.row(input_scale=scale, warmup=flag) for scale in ("0.50", "1.0") for flag in ("0", "1")))
        report = audit_result(self.path)
        assert report["valid"]
        assert report["measurement_status"] == "complete"
        assert report["coverage"] == {"expected": 4, "actual": 4, "missing": 0, "unexpected": 0}

    def test_sparse_plan_counts_include_exact_repeat_boundaries_without_expanding(self, recorded_plan, monkeypatch):
        from acprof.results import csv as result_csv

        count = 10**21
        recorded_plan["options"].update(warmup=2, repeat=count)
        (self.root / "run_state.json").write_text(json.dumps(recorded_plan))
        self.write(self.row(input_scale="1", repeat_idx=str(count - 1)),
                   self.row(input_scale="1", repeat_idx="1E21"),
                   self.row(input_scale="1", warmup="1.0", repeat_idx="1"))

        key = result_csv.measurement_key
        calls = 0

        def counted(row):
            nonlocal calls
            calls += 1
            if calls > 32:
                pytest.fail("audit expanded the planned repeat range")
            return key(row)

        monkeypatch.setattr(result_csv, "measurement_key", counted)
        report = audit_result(self.path)
        assert report["coverage"] == {"expected": count + 2, "actual": 3, "missing": count, "unexpected": 1}

    def test_plan_preserves_an_integer_scale_above_float_precision(self, recorded_plan):
        recorded_plan["runtime"]["planned"]["scales"] = [2**53 + 1]
        (self.root / "run_state.json").write_text(json.dumps(recorded_plan))
        self.write(self.row(input_scale=str(2**53)))
        report = audit_result(self.path)
        assert report["coverage"] == {"expected": 1, "actual": 1, "missing": 1, "unexpected": 1}
        self.write(self.row(input_scale=str(2**53 + 1)))
        assert audit_result(self.path)["measurement_status"] == "complete"

    def test_distinct_decimal_scale_coordinates_survive_numeric_axis_validation(self, recorded_plan):
        scales = [1_000_000_000_000_000_128, 1.0000000000000001e18]
        recorded_plan["runtime"]["planned"]["scales"] = scales
        (self.root / "run_state.json").write_text(json.dumps(recorded_plan))
        self.write(*(self.row(input_scale=str(scale)) for scale in scales))
        report = audit_result(self.path)
        assert report["valid"]
        assert report["coverage"] == {"expected": 2, "actual": 2, "missing": 0, "unexpected": 0}

    def test_legacy_process_plan_preserves_inherited_resources(self, recorded_plan):
        recorded_plan["options"].update(cpus="auto", mems="auto")
        (self.root / "run_state.json").write_text(json.dumps(recorded_plan))
        (self.root / "static_meta.json").write_text(json.dumps({"runtime_kind": "process", "runtime_id": "test"}))
        conditions = {"cases": {"inheritedc_inheritedg_off": {
            "resource_plan": {"verified": True, "method": "inherited", "requested": {}},
        }}}
        (self.root / "hardware_conditions.json").write_text(json.dumps(conditions))
        self.write(self.row(input_scale="1", cpu_cores="", mem_cap_gb="", runtime_kind="process", runtime_id="test",
                            resource_scope="observed_process_tree", resource_limit_source="inherited"))
        report = audit_result(self.path)
        assert report["valid"]
        assert report["measurement_status"] == "complete"
        assert report["coverage"] == {"expected": 1, "actual": 1, "missing": 0, "unexpected": 0}

    def test_legacy_run_without_a_plan_keeps_coverage_unknown(self, recorded_plan):
        del recorded_plan["runtime"]
        (self.root / "run_state.json").write_text(json.dumps(recorded_plan))
        self.write(self.row(input_scale="1"))
        report = audit_result(self.path)
        assert report["valid"]
        assert report["run_status"] == "complete"
        assert report["coverage"] is None
        assert report["measurement_status"] == "unknown"

    def test_historical_missing_columns_are_unknown_and_preserved(self):
        self.write(self.row(), fields=[field for field in CSV_FIELDS if field != "input_pixels_per_request"])
        report = audit_result(self.path)
        assert (report["valid"])
        assert (report["missing_metrics"]["input_pixels_per_request"]) == ({"not_recorded": 1})

    def test_negative_effective_energy_is_valid_but_invalid_derived_value_is_not(self):
        self.write(self.row(vcpu_energy_eff_j="-0.1"))
        assert (audit_result(self.path)["valid"])
        self.write(self.row(vcpu_energy_eff_j="2", container_attributed_energy_eff_j="20"))
        report = audit_result(self.path)
        assert not (report["valid"])
        assert ("formula_mismatch") in ([issue["code"] for issue in report["issues"]])

    def test_terminal_oom_timeout_and_unattempted_rows_are_separate_outcomes(self):
        self.write(
            self.row(status="error", error="container_oom_killed"),
            self.row(repeat_idx="1", status="error", error="client_request_timeout"),
            self.row(repeat_idx="2", status="error", error="not_measured_after_timeout: planned_request_attempted=false"),
        )
        (self.root / "run_state.json").write_text(json.dumps({"status": "complete", "outcome": "partial"}))
        report = audit_result(self.path)
        assert (report["valid"])
        assert (report["completion"]) == ("complete")
        assert (report["execution"]["finished"])
        assert not (report["execution"]["succeeded"])
        assert (report["execution"]["row_counts"]) == ({
            "total": 3, "succeeded": 0, "failed": 2, "not_measured": 1, "unfinished": 0,
        })

    def test_actual_workload_counts_must_match_variants_and_completed_requests(self):
        for request_count, variant_count, repeat_count in ((2, 1, "2"), (1, 1, "2")):
            summary = {"schema_version": 1, "request_count": request_count,
                       "variants": [{"count": variant_count, "contract": None}]}
            self.write(self.row(workload_contract=json.dumps(summary), repeat_in_window=repeat_count))
            report = audit_result(self.path)
            assert not (report["valid"])
            assert ("workload_contract") in ({issue["code"] for issue in report["issues"]})

    def test_unknown_per_request_workload_is_not_invented_or_rejected(self):
        summary = {"schema_version": 1, "request_count": 2, "variants": [{"count": 2, "contract": None}]}
        self.write(self.row(workload_contract=json.dumps(summary), repeat_in_window="2"))
        assert (audit_result(self.path)["valid"])

    @pytest.mark.parametrize(("total", "completed", "valid"), [
        (2**53, str(2**53 + 1), False),
        (2**53 + 1, "9.007199254740993e15", True),
    ])
    def test_workload_request_counts_are_compared_without_rounding(self, total, completed, valid):
        summary = {"schema_version": 1, "request_count": total, "variants": [{"count": total, "contract": None}]}
        self.write(self.row(workload_contract=json.dumps(summary), repeat_in_window=completed))
        report = audit_result(self.path)
        assert report["valid"] is valid
        assert any(issue["code"] == "workload_contract" for issue in report["issues"]) is not valid

    def test_failed_preparation_or_interface_validation_makes_audit_invalid(self):
        from acprof.results.layout import ArtifactLayout
        layout = ArtifactLayout.for_new_run(self.root)
        layout.initialize()
        attempt_id = "91b7d2ffaa81470080f32bebf74495d0"
        prep_rel = f"metadata/preparation/{attempt_id}.json"
        layout.path("run_state.json").write_text(json.dumps({
            "schema_version": 1,
            "layout_version": 2,
            "run_id": "3450983d3b3f461b8813bd2527aadd87",
            "status": "failed",
            "attempts": [{"attempt_id": attempt_id, "preparation_evidence": prep_rel, "outcome": "failed"}],
        }))
        prep_path = layout.contained(prep_rel)
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        prep_path.write_text(json.dumps({
            "schema_version": 1,
            "scope": "preparation",
            "attempt_id": attempt_id,
            "status": "failed",
            "last_stage": "image",
            "reason_code": "preparation_failed",
            "exception": {"type": "CollectionFailed", "message": "image pull failed"},
        }))
        report = audit_result(self.root)
        assert not report["valid"]
        assert report["completion"] == "incomplete"
        assert report["measurement_status"] == "incomplete"
        assert any(issue["code"] == "preparation_failed" and issue["severity"] == "error"
                   for issue in report["issues"])
