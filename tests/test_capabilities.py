import importlib
import importlib.util
import json
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES
from acprof.host.profilers import ncu, torch
from acprof.platform import Environment
from acprof.run_args import build_parser


class TestCapability:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        native = patch("acprof.capabilities.detect_environment", return_value=Environment("native_linux"))
        native.start()
        self._request.addfinalizer(partial(native.stop))

    def test_optional_dram_is_verified_without_becoming_a_full_prerequisite(self):
        caps = self.capabilities()
        report = caps.measurement_report('full', gpu_modes=['off'], dram_energy='auto')
        assert ('dram_energy') not in (report.requested)
        rows = [{'status': 'ok', 'gpu_mode': 'off', 'latency_app_s': '0.1',
                 'throughput_samples_per_s': '10', 'container_cpu_util_avg_pct': '1',
                 'container_mem_usage_avg_bytes': '1024', 'latency_s': '.09',
                 'cpu_energy_total_j': '1', 'vcpu_energy_total_j': '.2',
                 'cpu_instructions_per_request': '1', 'dram_energy_status': 'unavailable'}]
        caps.apply_collection_result(report, rows)
        assert (report.to_dict()['full_profile_complete'])
        required = caps.measurement_report('full', gpu_modes=['off'], dram_energy='required')
        caps.apply_collection_result(required, rows)
        assert (caps.missing_required_measurements(required, rows)) == (['dram_energy'])
        assert not (required.to_dict()['full_profile_complete'])
        rows[0].update(dram_window_energy_j='0', dram_energy_per_request_j='0',
                       dram_window_effective_energy_j='0', dram_energy_status='verified')
        caps.apply_collection_result(report, rows)
        assert (report.measurement['dram_energy'].status.value) == ('verified')

    def test_dram_off_and_basic_do_not_claim_measurement_and_required_basic_is_rejected(self):
        caps = self.capabilities()
        for mode, selection in [('full', 'off'), ('basic', 'auto')]:
            report = caps.measurement_report(mode, dram_energy=selection)
            assert (report.measurement['dram_energy'].status.value) == ('not_requested')
        with pytest.raises(ValueError, match='full'):
            caps.measurement_report('basic', dram_energy='required')

    def test_dram_permission_evidence_is_preserved(self):
        caps = self.capabilities()
        report = caps.measurement_report('full')
        caps.apply_collection_result(report, [{'status': 'ok', 'dram_energy_status': 'permission_denied',
                                               'dram_energy_error': 'energy_uj permission denied'}])
        assert (report.measurement['dram_energy'].status.value) == ('permission_denied')

    def capabilities(self):
        assert (importlib.util.find_spec("acprof.capabilities")) is not None, "a shared capability contract is required"
        return importlib.import_module("acprof.capabilities")

    def test_cli_preserves_full_default_and_exposes_basic(self):
        parser = build_parser()
        assert (getattr(parser.parse_args(["--model", "test"]), "profiling_mode", None)) == ("full")
        assert (parser.parse_args(["--model", "test", "--profiling-mode", "basic"]).profiling_mode) == ("basic")

    def test_statuses_survive_json_without_boolean_coercion(self):
        caps = self.capabilities()
        states = ("available", "verified", "unsupported", "permission_denied", "not_requested", "unavailable", "error")
        report = caps.CapabilityReport(profiling_mode="basic")
        for status in states:
            item = caps.Capability(status, source="test")
            report.measurement[status] = item
            with pytest.raises(TypeError):
                bool(item)
        payload = json.loads(json.dumps(report.to_dict()))
        assert ([payload["measurement"][state]["status"] for state in states]) == (list(states))
        with pytest.raises(ValueError):
            caps.Capability(False)

    def test_ncu_permissions_are_distinct_from_unsupported_missing_and_error(self):
        caps = self.capabilities()
        for detail, expected in (
            ("ncu_failed: ERR_NVGPUCTRPERM Permission to access GPU Performance Counters", "permission_denied"),
            ("this GPU is not supported", "unsupported"),
            ("ncu_not_found", "unavailable"),
            ("ncu failed: malformed CSV", "error"),
        ):
            assert (caps.capability_from_error(detail, source="ncu").status.value) == (expected)

    def test_basic_policy_never_claims_packet_or_energy_measurements(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        for name in ("latency", "throughput", "container_cpu", "container_memory"):
            assert (report.measurement[name].status.value) == ("available")
        for name in ("packet_latency", "cpu_energy", "cpu_instructions", "gpu_power", "gpu_flops"):
            assert (report.measurement[name].status.value) == ("not_requested")
        assert (report.to_dict()["profiling_mode"]) == ("basic")
        assert not (report.to_dict()["full_profile_complete"])

    @pytest.mark.parametrize("case", ("nonfinite", "oversized"))
    def test_run_profiler_plan_reader_rejects_invalid_json_artifacts(self, case):
        from acprof.cli import run

        caps = self.capabilities()
        report = caps.measurement_report("full", gpu_modes=["on"], compute_tool="ncu")
        plan = {"profiles": {"gpu": {"ncu": {"entries": [{
            "gpu_executed_mflop_per_request_ncu": 1.0,
            "error": "",
        }]}}}}
        if case == "nonfinite":
            plan["corrupt_metric"] = float("nan")
        else:
            plan["padding"] = "x" * MAX_JSON_ARTIFACT_BYTES

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "compute_profile_plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            if case == "oversized":
                assert path.stat().st_size > MAX_JSON_ARTIFACT_BYTES
            with pytest.raises(ValueError, match="non-finite|4 MiB read limit"):
                run._apply_profiler_plan_file(
                    report,
                    str(path),
                    source="compute_profile_plan",
                )

    def test_preflight_availability_is_not_completed_measurement_evidence(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        payload = report.to_dict()
        assert not (payload["requested_measurements_complete"])
        assert (payload["requested_measurements_available"])
        assert not (payload["collection_finished"])
        assert not (payload["collection_succeeded"])

    def test_failed_and_not_attempted_rows_are_finished_but_not_successful(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        rows = [
            {"status": "error", "error": "container_oom_killed during startup"},
            {"status": "error", "error": "client_request_timeout: planned_request_attempted=true"},
            {"status": "error", "error": "not_measured_after_timeout: planned_request_attempted=false"},
        ]
        caps.apply_collection_result(report, rows)
        payload = report.to_dict()
        assert (payload["collection_finished"])
        assert not (payload["collection_succeeded"])
        assert not (payload["collection_complete"])
        assert not (payload["requested_measurements_complete"])
        assert (payload["row_counts"]) == ({
            "total": 3, "succeeded": 0, "failed": 2, "not_measured": 1, "unfinished": 0,
        })

    def test_running_or_undiagnosed_error_rows_cannot_claim_finished(self):
        caps = self.capabilities()
        for row in ({"status": "running"}, {"status": "error", "error": ""}):
            report = caps.measurement_report("basic", gpu_modes=["off"])
            caps.apply_collection_result(report, [row])
            assert not (report.to_dict()["collection_finished"])
            assert (report.to_dict()["row_counts"]["unfinished"]) == (1)

    def test_old_report_unknown_completion_is_not_invented(self):
        caps = self.capabilities()
        payload = caps.CapabilityReport.from_dict({
            "schema_version": 1, "profiling_mode": "basic", "collection_complete": False,
        }).to_dict()
        assert (payload["collection_finished"]) is None
        assert (payload["collection_succeeded"]) is None
        assert (payload["row_counts"]) is None

    @pytest.mark.parametrize('version', ({}, {'schema_version': 1}, {'schema_version': 2}))
    def test_report_reader_accepts_known_versions_and_legacy_missing_version(self, version):
        caps = self.capabilities()
        restored = caps.CapabilityReport.from_dict({
            **version, "profiling_mode": "basic", "collection_complete": False,
        }).to_dict()
        assert (restored["schema_version"]) == (3)
        assert (restored["collection_complete"]) is (False)
        assert (restored["collection_finished"]) is None
        assert (restored["collection_succeeded"]) is None
        assert (restored["row_counts"]) is None

    @pytest.mark.parametrize('version', (0, 4, 999, True, False, 1.0, '2', None))
    def test_report_reader_rejects_unknown_and_non_integer_versions(self, version):
        caps = self.capabilities()
        with pytest.raises(ValueError, match="schema_version"):
            caps.CapabilityReport.from_dict({"schema_version": version})

    @pytest.mark.parametrize('value', ('false', 'true', 0, 1, [], {}))
    @pytest.mark.parametrize('field', ('collection_complete', 'collection_finished', 'collection_succeeded'))
    def test_report_reader_rejects_non_boolean_collection_states(self, value, field):
        caps = self.capabilities()
        with pytest.raises(ValueError, match=field):
            caps.CapabilityReport.from_dict({"schema_version": 2, field: value})
        with pytest.raises(ValueError, match="collection_complete"):
            caps.CapabilityReport.from_dict({"schema_version": 2, "collection_complete": None})

    @pytest.mark.parametrize('value', (True, False, None))
    def test_report_reader_preserves_nullable_states_and_legacy_success(self, value):
        caps = self.capabilities()
        legacy = caps.CapabilityReport.from_dict({"collection_complete": True}).to_dict()
        assert (legacy["collection_finished"]) is (True)
        assert (legacy["collection_succeeded"]) is (True)
        assert (legacy["row_counts"]) is None
        restored = caps.CapabilityReport.from_dict({
            "schema_version": 2, "collection_finished": value,
            "collection_succeeded": value,
        }).to_dict()
        assert (restored["collection_finished"]) is (value)
        assert (restored["collection_succeeded"]) is (value)

    def test_empty_collection_has_no_finished_or_measurement_evidence(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        caps.apply_collection_result(report, [])
        payload = report.to_dict()
        for field in ("collection_finished", "collection_succeeded", "collection_complete",
                      "requested_measurements_complete"):
            assert (payload[field]) is (False)
        assert (payload["row_counts"]["total"]) == (0)

    def test_collection_status_roundtrip_preserves_terminal_failure(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        caps.apply_collection_result(report, [{"status": "error", "error": "timeout"}])
        payload = json.loads(json.dumps(report.to_dict()))
        restored = caps.CapabilityReport.from_dict(payload).to_dict()
        assert (restored["collection_finished"])
        assert not (restored["collection_succeeded"])
        assert (restored["row_counts"]["failed"]) == (1)

    def test_failed_requested_profiler_cannot_claim_complete_full_profile(self):
        caps = self.capabilities()
        report = caps.measurement_report("full", gpu_modes=["on"], compute_tool="ncu")
        caps.apply_profiler_plan(report, {
            "profiles": {"gpu": {"ncu": {"error": "ERR_NVGPUCTRPERM", "entries": []}}}
        }, source="compute_profile_plan")
        assert (report.measurement["gpu_flops"].status.value) == ("permission_denied")
        assert not (report.to_dict()["requested_measurements_complete"])
        assert not (report.to_dict()["full_profile_complete"])

    def test_validation_evidence_only_verifies_successful_device(self):
        caps = self.capabilities()
        report = caps.CapabilityReport(profiling_mode="full")
        caps.apply_runtime_validation(report, {
            "devices": {"off": {"status": "ok", "validation": {
                "protocol": {"status": "verified"}, "task": {"status": "verified"},
            }}, "on": {"status": "resource_limit"}},
        }, environment_id="env-test")
        assert (report.execution["cpu"].status.value) == ("verified")
        assert (report.execution["cuda"].status.value) == ("unavailable")
        assert (report.execution["cpu"].evidence["environment_id"]) == ("env-test")

    def test_runtime_ok_without_complete_validation_cannot_claim_verified(self):
        caps = self.capabilities()
        for validation in (None, {}, {"protocol": {"status": "verified"}},
                           {"protocol": {"status": "verified"}, "task": {"status": "available"}}):
            report = caps.CapabilityReport(profiling_mode="basic")
            caps.apply_runtime_validation(report, {
                "devices": {"off": {"status": "ok", "validation": validation}},
            })
            assert (report.execution["cpu"].status.value) == ("available")
            assert ("validation") in (report.execution["cpu"].detail)

    def test_collection_verifies_actual_csv_fields_and_preserves_measured_zero(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        rows = [{"status": "ok", "gpu_mode": "off", "latency_app_s": "0.1",
                 "throughput_samples_per_s": "10", "container_cpu_util_avg_pct": "0",
                 "container_mem_usage_avg_bytes": "1024"}]
        caps.apply_collection_result(report, rows)
        assert (report.measurement["container_memory"].status.value) == ("verified")
        assert (report.measurement["container_cpu"].status.value) == ("verified")
        assert (report.to_dict()["requested_measurements_complete"])
        assert (rows[0]["container_cpu_util_avg_pct"]) == ("0")
        assert not (report.to_dict()["full_profile_complete"])

    def test_empty_error_without_finite_profiler_metric_is_not_verified(self):
        caps = self.capabilities()
        report = caps.measurement_report("full", gpu_modes=["on"], compute_tool="ncu")
        caps.apply_profiler_plan(report, {"profiles": {"gpu": {"ncu": {
            "entries": [{"gpu_executed_mflop_per_request_ncu": None, "error": ""}],
        }}}}, source="test")
        assert (report.measurement["gpu_flops"].status.value) == ("unavailable")

    def test_collection_permission_error_keeps_diagnostic_status(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        caps.apply_collection_result(report, [{"status": "error", "error": "PermissionError: cgroup memory.current"}])
        assert (report.measurement["container_memory"].status.value) == ("permission_denied")

    def test_required_actual_measurements_fail_after_successful_preflight(self):
        caps = self.capabilities()
        for mode in ("full", "basic"):
            report = caps.measurement_report(mode, gpu_modes=["off"])
            rows = [{"status": "ok", "latency_app_s": "0.1", "throughput_samples_per_s": "10",
                     "container_cpu_util_avg_pct": "0", "container_mem_usage_avg_bytes": "nan",
                     "latency_s": "0.1", "cpu_energy_total_j": "1", "vcpu_energy_total_j": "0",
                     "cpu_instructions_per_request": "0"}]
            caps.apply_collection_result(report, rows)
            assert (caps.missing_required_measurements(report, rows)) == (["container_memory"])
            assert (report.measurement["container_cpu"].status.value) == ("verified")

    def test_partial_missing_measurements_do_not_hide_behind_finite_rows(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        row = {"status": "ok", "latency_app_s": "0.1", "throughput_samples_per_s": "10",
               "container_cpu_util_avg_pct": "0", "container_mem_usage_avg_bytes": "123"}
        rows = [row, {**row, "container_mem_usage_avg_bytes": ""}]
        caps.apply_collection_result(report, rows)
        assert (caps.missing_required_measurements(report, rows)) == (["container_memory"])

    def test_explicit_oom_rows_do_not_invent_measurement_success_or_block_valid_rows(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        rows = [{"status": "error", "error": "container_oom_killed"}, {
            "status": "ok", "latency_app_s": "0.1", "throughput_samples_per_s": "10",
            "container_cpu_util_avg_pct": "0", "container_mem_usage_avg_bytes": "123"}]
        caps.apply_collection_result(report, rows)
        assert (caps.missing_required_measurements(report, rows)) == ([])
        assert not (report.collection_complete)

    def test_unsupported_profiler_writes_missing_values_without_running_tool(self):
        from acprof.host import compute_profile
        from acprof.host.detect import TaskInfo
        task = TaskInfo("test/onnx", "tabular-regression", "structured", "onnxruntime", "onnxruntime", "main", "unit")
        with tempfile.TemporaryDirectory() as root:
            plan = Path(root) / "input.json"
            plan.write_text(json.dumps({"schema_version": 2, "entries": [{"input_scale": 2, "payload": {"rows": [[1, 2], [3, 4]]}}]}))
            with patch.object(torch, "_profile_torch_entries", side_effect=AssertionError("unsupported Torch must not launch")), patch.object(
                ncu, "_profile_gpu_entries", side_effect=AssertionError("unsupported NCU must not launch")
            ), patch.object(compute_profile, "find_executable", side_effect=AssertionError("unsupported tool must not be discovered")):
                path = compute_profile.collect_compute_profile_plan(
                    task_info=task, image_tag="test", cpu_list=[1], mem_list=[1], gpu_list=["off", "on"],
                    output_dir=root, input_scale_plan_file=str(plan), compute_profile_tool="both",
                    advisor_root=None, ncu_root=None, advisor_repeat=1, ncu_repeat=1, keep_profiles=True,
                )
            payload = json.loads(Path(path).read_text())
        torch_entry = payload["profiles"]["cpu"]["torch_profiler_eager"]["entries"][0]
        ncu_entry = payload["profiles"]["gpu"]["ncu"]["entries"][0]
        assert ("unsupported") in (torch_entry["error"])
        assert (torch_entry["model_logical_mflop_per_request_torch_profiler_eager"]) is None
        assert ("unsupported") in (ncu_entry["error"])
        assert (ncu_entry["gpu_executed_mflop_per_request_ncu"]) is None
