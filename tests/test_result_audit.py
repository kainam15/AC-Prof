"""审计真实 CSV/JSON 的故障与历史兼容边界。"""
import csv
import json
from pathlib import Path

import pytest

from acprof.analysis.audit import audit_result
from acprof.config import CSV_FIELDS


class TestResultAudit:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.temporary = tmp_path
        self.root = Path(str(self.temporary))
        self.path = self.root / "result_all.csv"
    def test_v2_input_plan_symlink_is_reported_as_invalid_without_reading_it(self):
        from acprof.artifact_layout import ArtifactLayout
        ArtifactLayout.for_new_run(self.root).initialize()
        self.write(self.row())
        (self.root / "static_meta.json").write_text(json.dumps({"input_scale_plan_sha256": "untrusted"}))
        (self.root / "metadata/input_scale_plan.json").symlink_to(self.path)
        report = audit_result(self.root)
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
        report = audit_result(self.root)
        assert (report["valid"])
        assert ("container_io_read_bytes_per_request") not in (report["missing_metrics"])
        assert (report["missing_metrics"]["gpu_energy_eff_j"]) == ({"not_applicable": 1})
        assert (self.path.read_bytes()) == (before)
        assert (list(self.root.iterdir())) == ([self.path])

    def test_formal_filter_excludes_warmup_warn_and_error(self):
        self.write(self.row(), self.row(warmup="1"), self.row(repeat_idx="1", status="warn", error="idle drift"),
                   self.row(repeat_idx="2", status="error", error="timeout"))
        report = audit_result(self.root)
        assert (report["counts"]) == ({"rows": 4, "formal_ok": 1, "warmup": 1, "warn": 1, "error": 1})
        assert (report["completion"]) == ("unknown")

    def test_duplicate_and_invalid_status_are_reported(self):
        for rows, code in (([self.row(), self.row()], "invalid_csv"),
                           ([self.row(status="finished")], "invalid_status")):
            self.write(*rows)
            report = audit_result(self.root)
            assert not (report["valid"])
            assert (code) in ([issue["code"] for issue in report["issues"]])

    def test_invalid_number_is_not_explained_as_hardware_unavailable(self):
        self.write(self.row(latency_app_s="inf"))
        report = audit_result(self.root)
        assert not (report["valid"])
        assert ("invalid_number") in ([issue["code"] for issue in report["issues"]])

    @pytest.mark.parametrize(
        "name,payload",
        (
            ("static_meta.json", b'{"schema_version": 7, "value": NaN}'),
            ("static_meta.json", b'{"schema_version": 7, "value": 1e999}'),
            ("static_meta.json", b"\xff"),
            ("runtime_validation.json", b'{"devices": []}'),
            ("runtime_validation.json", b'{"devices": {"off": []}}'),
            ("runtime_failures.json", b'{"failures": {}}'),
            ("runtime_failures.json", b'{"failures": [{"reason_code": "inference_failed"}]}'),
            ("model_resolution.json", b'{"failure": []}'),
        ),
    )
    def test_invalid_recorded_metadata_becomes_an_audit_issue(self, name, payload):
        self.write(self.row())
        path = self.root / name
        path.write_bytes(payload)
        report = audit_result(self.root)
        assert not (report["valid"])
        assert ("invalid_metadata") in ({issue["code"] for issue in report["issues"]})
        assert (path.read_bytes()) == (payload)

    def test_oversized_recorded_metadata_becomes_an_audit_issue_without_rewriting(self):
        self.write(self.row())
        path = self.root / "static_meta.json"
        content = b" " * (4 * 1024 * 1024 + 1)
        path.write_bytes(content)
        report = audit_result(self.root)
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
        report = audit_result(self.root)
        assert not (report["valid"])
        codes = {issue["code"] for issue in report["issues"]}
        assert ({"input_plan_hash", "plan_coverage"} <= codes)
        assert (report["coverage"]["missing"]) == (1)

    def test_historical_missing_columns_are_unknown_and_preserved(self):
        self.write(self.row(), fields=[field for field in CSV_FIELDS if field != "input_pixels_per_request"])
        report = audit_result(self.root)
        assert (report["valid"])
        assert (report["missing_metrics"]["input_pixels_per_request"]) == ({"not_recorded": 1})

    def test_negative_effective_energy_is_valid_but_invalid_derived_value_is_not(self):
        self.write(self.row(vcpu_energy_eff_j="-0.1"))
        assert (audit_result(self.root)["valid"])
        self.write(self.row(vcpu_energy_eff_j="2", container_attributed_energy_eff_j="20"))
        report = audit_result(self.root)
        assert not (report["valid"])
        assert ("formula_mismatch") in ([issue["code"] for issue in report["issues"]])

    def test_terminal_oom_timeout_and_unattempted_rows_are_separate_outcomes(self):
        self.write(
            self.row(status="error", error="container_oom_killed"),
            self.row(repeat_idx="1", status="error", error="client_request_timeout"),
            self.row(repeat_idx="2", status="error", error="not_measured_after_timeout: planned_request_attempted=false"),
        )
        (self.root / "run_state.json").write_text(json.dumps({"status": "complete", "outcome": "partial"}))
        report = audit_result(self.root)
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
            report = audit_result(self.root)
            assert not (report["valid"])
            assert ("workload_contract") in ({issue["code"] for issue in report["issues"]})

    def test_unknown_per_request_workload_is_not_invented_or_rejected(self):
        summary = {"schema_version": 1, "request_count": 2, "variants": [{"count": 2, "contract": None}]}
        self.write(self.row(workload_contract=json.dumps(summary), repeat_in_window="2"))
        assert (audit_result(self.root)["valid"])
