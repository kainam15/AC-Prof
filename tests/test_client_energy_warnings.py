import csv
import io
import json
import math
import os
import tempfile
from contextlib import ExitStack, redirect_stderr
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from client_fixtures import patch_client

from acprof.config import (
    CSV_FIELDS,
    GPU_RUNTIME_STATE_FIELDS,
    STATIC_META_FIELDS,
    STATIC_META_SCHEMA_VERSION,
)
from acprof.host import client
from acprof.host.client import ClientRunner
from acprof.host.client_config import ClientConfig
from acprof.monitors import energy_cpu

REMOVED_LEGACY_COMPUTE_FIELDS = (
    "compute_profile_tool",
    "model_mflop_per_request",
    "compute_mflops_app",
    "compute_mflops",
    "compute_profile_error",
)


class TestEffectiveEnergyWarning:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from platform_fixtures import native_policy
        native_policy(self._request)
        temporary = tmp_path
        self.runner = ClientRunner(ClientConfig(out_csv=os.path.join(str(temporary), "result.csv")))
        gpu_uuid = patch_client(self.runner, "GPU_DEVICE_UUID", "GPU-fixture")
        gpu_uuid.start()
        self._request.addfinalizer(partial(gpu_uuid.stop))
        for name in ("IDLE_SECONDS", "IDLE_COOLDOWN_SECONDS"):
            mocked = patch_client(self.runner, name, 0.0)
            mocked.start()
            self._request.addfinalizer(partial(mocked.stop))

    @pytest.mark.parametrize('header_case', range(3))
    def test_append_respects_existing_column_order_and_rejects_invalid_headers(self, header_case):
        reordered = ["status", "error", *[field for field in CSV_FIELDS
                                          if field not in {"status", "error"}]]
        (fields, valid) = (((reordered, True), (reordered[:-1], False), ([*reordered[:-1], reordered[0]], False)))[header_case]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "result.csv")
            original = dict.fromkeys(fields, "nan")
            original.update(cpu_cores="1", status="ok", error="")
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(original)
            before = path.read_bytes()
            with ExitStack() as stack:
                for name, value in {
                    "OUT_CSV": str(path), "CPU_CORES": "2", "GPU_MODE": "off",
                    "USE_ENERGY": False, "USE_MIPS": False, "IDLE_DEBUG": False,
                    "PROFILING_MODE": "full", "SNIFF_GROUPS_PATH": "",
                    "input_scale_entries": [{"input_scale": 1.0}],
                }.items():
                    stack.enter_context(patch_client(self.runner, name, value))
                ready = stack.enter_context(patch.object(
                    client.requests, "get", side_effect=RuntimeError("offline fixture")))
                if valid:
                    self.runner.main()
                else:
                    with pytest.raises(RuntimeError, match="CSV.*columns"):
                        self.runner.main()
                    ready.assert_not_called()
                    assert (path.read_bytes()) == (before)
                    assert (list(Path(tmp).iterdir())) == ([path])
                    return
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                assert (reader.fieldnames) == (fields)
            assert (rows[0]) == (original)
            assert (len(rows)) == (2)
            assert (rows[1]["cpu_cores"]) == ("2")
            assert (rows[1]["status"]) == ("error")
            assert ("offline fixture") in (rows[1]["error"])

    @pytest.mark.parametrize('invalid', ('[16,32]', '[16,32,32]', '[16,32,128]'))
    def test_frozen_scale_order_changes_execution_without_changing_payloads(self, invalid):
        entries = [{"input_scale": scale, "payload": {"text": f"payload-{scale}"}}
                   for scale in (16, 32, 64)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "input_scale_plan.json")
            path.write_text(json.dumps({"schema_version": 2, "entries": entries}))
            original = path.read_bytes()
            with patch_client(self.runner, "INPUT_SCALE_PLAN_FILE", str(path)), patch.object(self.runner.config, "input_scale_order", "[64,16,32]"):
                loaded = self.runner._load_input_scale_entries()
                assert ([e["input_scale"] for e in loaded]) == ([64, 16, 32])
                assert ([e["payload"]["text"] for e in loaded]) == (["payload-64", "payload-16", "payload-32"])
                assert (path.read_bytes()) == (original)
                with patch.object(self.runner.config, "input_scale_order", invalid):
                    with pytest.raises(ValueError, match="frozen matrix input-scale"):
                        self.runner._load_input_scale_entries()

    @pytest.mark.parametrize('dram_case', range(3))
    def test_full_client_dram_policy_and_separate_window_request_units(self, dram_case):
        (policy, available) = ((('auto', False), ('required', False), ('required', True)))[dram_case]
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            result = energy_cpu._nan_result(cpu_idle_power_w=1.0)
            result.cpu_energy_total_j = 12.0
            result.vcpu_energy_eff_j = 6.0
            if available:
                result.dram = energy_cpu.DRAMEnergyResult(
                    window_energy_j=10.0, window_duration_s=2.0, avg_power_w=5.0,
                    peak_power_w=7.0, idle_power_w=1.0,
                    window_effective_energy_j=8.0, status="verified")
            monitor = Mock()
            monitor.idle_power_w = 1.0
            monitor.idle_trace = {}
            monitor.stop.return_value = (result, "", [])
            path = Path(tmp, "result.csv")
            settings = {
                "OUT_CSV": str(path), "PROFILING_MODE": "full", "DRAM_ENERGY": policy,
                "WARMUP": 0, "REPEAT": 1, "REPEAT_IN_WINDOW": 2,
                "USE_ENERGY": False, "USE_MIPS": False, "GPU_MODE": "off",
                "energy_mod": None, "resource_usage_mod": None,
                "cpu_energy_mod": SimpleNamespace(CPUEnergyMonitor=Mock(return_value=monitor)),
                "input_scale_entries": [{"input_scale": 1.0, "scale_label": "one", "payload": {}}],
            }
            for name, value in settings.items():
                stack.enter_context(patch_client(self.runner, name, value))
            stack.enter_context(patch.object(client.requests, "get",
                return_value=SimpleNamespace(status_code=200, text="ok")))
            stack.enter_context(patch_client(self.runner, "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0}))
            if policy == "required" and not available:
                with pytest.raises(client.EnergyAbort, match="required DRAM"):
                    self.runner.main()
                return
            self.runner.main()
            with path.open() as stream:
                row = next(csv.DictReader(stream))
            assert (row["status"]) == ("ok"), row["error"]
            assert (float(row["container_attributed_energy_eff_j"])) == (3.0)
            assert (row["result_origin"]) == ("formal_measurement")
            if available:
                assert (float(row["dram_window_energy_j"])) == (10.0)
                assert (float(row["dram_energy_per_request_j"])) == (5.0)
                assert (float(row["dram_effective_energy_per_request_j"])) == (4.0)
            else:
                assert (row["dram_energy_status"]) == ("unavailable")
                assert (row["dram_window_energy_j"]) == ("nan")

    @pytest.mark.parametrize('threshold_case', range(2), ids=['(0.05, 2 / 3)', '(0.1, 1 / 3)'])
    def test_latency_metrics_use_the_explicit_slo_threshold(self, threshold_case) -> None:
        latencies = [0.01, 0.06, 0.2, float("nan"), float("inf")]
        (threshold, expected) = tuple(((0.05, 2 / 3), (0.1, 1 / 3)))[threshold_case]
        assert (client._latency_distribution_metrics(
                "latency_app", latencies, slow_latency_threshold_s=threshold,
            )[
                "latency_app_slow_ratio"
            ]) == (expected) or round(abs((client._latency_distribution_metrics(
                "latency_app", latencies, slow_latency_threshold_s=threshold,
            )[
                "latency_app_slow_ratio"
            ]) - (expected)), 7) == 0

    def test_default_idle_diag_path_uses_dedicated_debug_directory(self) -> None:
        with patch_client(self.runner, "IDLE_DIAG_PATH", ""):
            assert (self.runner._idle_diag_path("results/model/result_case.csv")) == (os.path.join(
                    "results",
                    "model",
                    "debug_idle_diag",
                    "result_case.csv.idle_diag.jsonl",
                ))

    def test_matched_control_starts_all_monitors_before_wait_and_applies_baselines(self) -> None:
        events = []

        class FakeEnergyMonitor:
            def __init__(self, name, avg_field):
                self.name = name
                self.avg_field = avg_field
                self.applied = None

            def start(self):
                events.append(f"start:{self.name}")

            def stop(self):
                events.append(f"stop:{self.name}")
                result = SimpleNamespace(**{self.avg_field: 7.0})
                if self.name == "gpu":
                    return result, "GPU", "", [(0.0, 7.0), (1.0, 7.0)]
                return result, "", [SimpleNamespace(timestamp=0.0), SimpleNamespace(timestamp=1.0)]

            def apply_control_baseline(self, result, samples, trace=False):
                events.append(f"apply:{self.name}:{trace}")
                self.applied = (result, samples)

        class FakeResourceMonitor:
            def start(self):
                events.append("start:resource")

            def stop(self):
                events.append("stop:resource")
                return None, "", []

        class FakeMIPSMonitor:
            def start(self):
                events.append("start:mips")

            def stop(self, repeat_in_window, latency_app_s):
                events.append(f"stop:mips:{repeat_in_window}:{latency_app_s}")

        gpu_monitor = FakeEnergyMonitor("gpu", "avg_power_total_w")
        cpu_monitor = FakeEnergyMonitor("cpu", "cpu_avg_power_total_w")

        with patch_client(self.runner, "IDLE_SECONDS", 2.0), patch_client(self.runner, "IDLE_DEBUG", True
        ), patch.object(
            client.time,
            "sleep",
            side_effect=lambda seconds: events.append(f"sleep:{seconds}"),
        ):
            from acprof.host.measurement_window import MonitorGroup, run_matched_control_window
            monitors = MonitorGroup(close=False)
            for name, monitor in (("gpu", gpu_monitor), ("cpu", cpu_monitor),
                                  ("resource", FakeResourceMonitor()), ("mips", FakeMIPSMonitor())):
                monitors.add(name, monitor)
            run_matched_control_window(monitors, idle_seconds=2.0, trace=True)

        assert (events) == ([
                "start:gpu",
                "start:cpu",
                "start:resource",
                "start:mips",
                "sleep:2.0",
                "stop:mips:1:2.0",
                "stop:resource",
                "stop:gpu",
                "stop:cpu",
                "apply:gpu:True",
                "apply:cpu:True",
            ])
        assert (gpu_monitor.applied) is not None
        assert (cpu_monitor.applied) is not None

    def test_csv_schema_uses_gpu_idle_power_w_field(self) -> None:
        assert ("gpu_idle_power_w") in (CSV_FIELDS)
        assert ("gpu_idle_measured_at") in (CSV_FIELDS)
        assert ("gpu_idle_rel_range_so_far") in (CSV_FIELDS)
        assert ("idle_power_w") not in (CSV_FIELDS)
        gpu_idle_index = CSV_FIELDS.index("gpu_idle_power_w")
        assert (CSV_FIELDS[gpu_idle_index + 1]) == ("gpu_idle_measured_at")
        assert (CSV_FIELDS[gpu_idle_index + 2]) == ("gpu_idle_rel_range_so_far")

    def test_schema_v6_includes_cgroup_swap_and_request_shape_metrics(self) -> None:
        assert (STATIC_META_SCHEMA_VERSION) == (7)
        assert ("parameter_bytes") in (STATIC_META_FIELDS)
        assert ("model_cache_bytes") in (STATIC_META_FIELDS)
        assert ("model_weight_bytes") not in (STATIC_META_FIELDS)
        assert ("host_mem_total_bytes") in (STATIC_META_FIELDS)
        assert ("host_swap_total_bytes") in (STATIC_META_FIELDS)
        assert ("host_swap_used_bytes_at_start") in (STATIC_META_FIELDS)
        assert ("host_swap_type") in (STATIC_META_FIELDS)
        assert ("host_vm_swappiness") in (STATIC_META_FIELDS)
        assert ("docker_storage_total_bytes") in (STATIC_META_FIELDS)
        assert ("workload") in (STATIC_META_FIELDS)
        assert ("input_scale_plan_sha256") in (STATIC_META_FIELDS)
        assert ("cgroup_version") in (STATIC_META_FIELDS)
        assert ("cgroup_collection_mode") in (STATIC_META_FIELDS)
        expected = [
            "input_scale",
            "input_units_per_request",
            "input_num_samples",
            "input_pixels_per_request",
            "output_pixels_per_request",
            "request_payload_bytes",
            "packet_request_wire_bytes_per_request",
            "packet_response_wire_bytes_per_request",
            "packet_total_wire_bytes_per_request",
            "packet_tcp_payload_bytes_per_request",
            "packet_protocol_overhead_bytes_per_request",
            "packet_protocol_overhead_ratio",
            "task_param",
            "output_length_avg",
            "output_token_count_avg",
        ]
        start = CSV_FIELDS.index("input_scale")
        assert (CSV_FIELDS[start:start + len(expected)]) == (expected)

    @pytest.mark.parametrize('plan_case', range(1))
    def test_input_scale_plan_preserves_current_audio_metadata(self, plan_case) -> None:
        plans = [
            (
                {
                    "schema_version": 2,
                    "entries": [
                        {
                            "input_scale": 1.0,
                            "payload": {
                                "audio_base64": "UklGRg==",
                                "audio_format": "wav",
                                "sample_rate": 16000,
                                "params": {},
                            },
                            "input_metadata": {
                                "input_num_samples": 16000,
                                "audio_sha256": "abc123",
                            },
                        }
                    ],
                },
                {"input_num_samples": 16000, "audio_sha256": "abc123"},
            ),
        ]

        (plan, expected_metadata) = tuple(plans)[plan_case]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "input_scale_plan.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(plan, f)
            with patch_client(self.runner, "INPUT_SCALE_PLAN_FILE", path):
                entries = self.runner._load_input_scale_entries()

        assert (entries[0]["input_metadata"]) == (expected_metadata)

    def test_one_request_reports_prepared_body_and_response_counts(self) -> None:
        class FakeResponse:
            status_code = 200
            text = ""
            reason = "OK"
            request = SimpleNamespace(body='{"text":"\u2713"}')

            @staticmethod
            def json():
                return {
                    "effective_input_scale": 3,
                    "output_length": 12,
                    "output_token_count": 5,
                }

        payload = {"text": "hello", "params": {"z": 1, "a": 2}}
        with patch.object(client.requests, "post", return_value=FakeResponse()), patch.object(
            client.time,
            "perf_counter",
            side_effect=[10.0, 10.25],
        ), patch_client(self.runner, "_FIRST_PREDICT_APP_S", float("nan")):
            result = self.runner._one_request(3.0, "req", payload_override=payload)
            first_predict_app_s = self.runner.first_predict_app_s

        assert (result["request_payload_bytes"]) == (14.0)
        assert (result["output_length"]) == (12.0)
        assert (result["output_token_count"]) == (5.0)
        assert (result["task_param"]) == ('{"a":2,"z":1}')
        assert (first_predict_app_s) == (0.25)

    def test_task_param_includes_top_level_timeseries_prediction_length(self) -> None:
        payload = {
            "context": [[0.1, 0.2]],
            "prediction_length": 64,
        }
        assert (client._canonical_task_param(payload)) == ('{"prediction_length":64}')

    def test_request_shape_metrics_are_averaged_per_window(self) -> None:
        responses = iter([
            {
                "latency_app_s": 0.1,
                "effective_input_scale": 1.0,
                "request_payload_bytes": 100,
                "output_length": 10,
                "output_token_count": 3,
            },
            {
                "latency_app_s": 0.2,
                "effective_input_scale": 1.0,
                "request_payload_bytes": 104,
                "output_length": 14,
                "output_token_count": 5,
            },
        ])
        payload = {
            "audio_base64": "UklGRg==",
            "audio_format": "wav",
            "sample_rate": 16000,
            "params": {"task": "transcribe", "language": "english"},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = os.path.join(tmp_dir, "result.csv")
            with patch_client(self.runner, "OUT_CSV", out_csv), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1), patch_client(self.runner, "REPEAT_IN_WINDOW", 2
            ), patch_client(self.runner, "BATCH_SIZE", 2), patch_client(self.runner, "CPU_CORES", "2"
            ), patch_client(self.runner, "USE_ENERGY", False), patch_client(self.runner, "USE_MIPS", False
            ), patch_client(self.runner, "energy_mod", None), patch_client(self.runner, "cpu_energy_mod", None
            ), patch_client(self.runner, "resource_usage_mod", None), patch_client(self.runner,
                "input_scale_entries",
                [
                    {
                        "input_scale": 1.0,
                        "scale_label": "dur1s",
                        "payload": payload,
                        "input_metadata": {"input_num_samples": 16000},
                    }
                ],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner, "_one_request", side_effect=lambda *args, **kwargs: next(responses)):
                self.runner.main()

            with open(out_csv, "r", encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f))

        assert (row["input_units_per_request"]) == ("2.000000")
        assert (row["input_num_samples"]) == ("16000.000000")
        assert (row["request_payload_bytes"]) == ("102.000000")
        assert (row["output_length_avg"]) == ("12.000000")
        assert (row["output_token_count_avg"]) == ("4.000000")
        assert (row["latency_app_s_per_input_unit"]) == ("0.075000")
        assert (row["throughput_samples_per_s_per_cpu_core"]) == ("6.666667")
        assert (row["task_param"]) == ('{"language":"english","task":"transcribe"}')

    def test_csv_schema_includes_idle_debug_fields_after_cpu_idle_power(self) -> None:
        assert ("cpu_idle_measured_at") in (CSV_FIELDS)
        assert ("cpu_idle_rel_range_so_far") in (CSV_FIELDS)
        assert ("idle_measured_at") not in (CSV_FIELDS)
        cpu_idle_index = CSV_FIELDS.index("cpu_idle_power_w")
        assert (CSV_FIELDS[cpu_idle_index + 1]) == ("cpu_idle_measured_at")
        assert (CSV_FIELDS[cpu_idle_index + 2]) == ("cpu_idle_rel_range_so_far")

    def test_csv_schema_prefixes_gpu_energy_fields(self) -> None:
        expected = [
            "gpu_energy_iters",
            "gpu_avg_power_total_w",
            "gpu_peak_power_total_w",
            "gpu_energy_total_j",
            "gpu_avg_power_eff_w",
            "gpu_peak_power_eff_w",
            "gpu_energy_eff_j",
        ]
        old_names = [
            "energy_iters",
            "avg_power_total_w",
            "peak_power_total_w",
            "energy_total_j",
            "avg_power_eff_w",
            "peak_power_eff_w",
            "energy_eff_j",
        ]

        for field in expected:
            assert (field) in (CSV_FIELDS)
        for field in old_names:
            assert (field) not in (CSV_FIELDS)

    def test_csv_schema_includes_gpu_runtime_state_fields(self) -> None:
        expected = [
            "gpu_sm_clock_mhz",
            "gpu_memory_clock_mhz",
            "gpu_pstate",
            "gpu_temp_c",
        ]

        assert (GPU_RUNTIME_STATE_FIELDS) == (expected)
        start = CSV_FIELDS.index(expected[0])
        assert (CSV_FIELDS[start:start + len(expected)]) == (expected)
        assert (CSV_FIELDS[start + len(expected)]) == ("gpu_util_avg_pct")

    def test_gpu_runtime_metrics_normalize_pstate(self) -> None:
        metrics = client._gpu_runtime_metrics_from_result(
            SimpleNamespace(
                gpu_sm_clock_mhz=1800.0,
                gpu_memory_clock_mhz=7500.0,
                gpu_pstate="p0",
                gpu_util_avg_pct=88.0,
                gpu_temp_c=67.0,
            ),
        )

        assert (metrics["gpu_sm_clock_mhz"]) == (1800.0)
        assert (metrics["gpu_memory_clock_mhz"]) == (7500.0)
        assert (metrics["gpu_pstate"]) == ("P0")
        assert (metrics["gpu_temp_c"]) == (67.0)

    def test_csv_schema_distinguishes_torch_logical_and_ncu_executed_flops(self) -> None:
        torch_fields = [
            "model_logical_mflop_per_request_torch_profiler_eager",
            "model_logical_mflops_app_torch_profiler_eager",
            "model_logical_mflops_packet_torch_profiler_eager",
            "compute_profile_error_torch_profiler_eager",
        ]
        ncu_fields = [
            "gpu_executed_mflop_per_request_ncu",
            "gpu_executed_tensor_mflop_per_request_ncu",
            "gpu_executed_scalar_mflop_per_request_ncu",
            "gpu_executed_tensor_share_pct_ncu",
            "gpu_executed_mflops_app_ncu",
            "gpu_executed_mflops_packet_ncu",
            "gpu_kernel_launch_count_per_request_ncu",
            "gpu_kernel_time_sum_ms_per_request_ncu",
            "compute_profile_error_ncu",
        ]

        for field in [*torch_fields, *ncu_fields]:
            assert (field) in (CSV_FIELDS)
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            assert (field) not in (CSV_FIELDS)
        assert ("gpu_profile_report_ncu") not in (CSV_FIELDS)
        assert (CSV_FIELDS.index(torch_fields[0])) < (CSV_FIELDS.index(ncu_fields[0]))

    def test_csv_schema_includes_cpu_memory_behavior_metrics(self) -> None:
        fields = [
            "cpu_cache_references_per_request",
            "cpu_cache_misses_per_request",
            "cpu_cache_miss_rate_pct",
            "cpu_dtlb_loads_per_request",
            "cpu_dtlb_load_misses_per_request",
            "cpu_dtlb_load_miss_rate_pct",
        ]

        for field in fields:
            assert (field) in (CSV_FIELDS)
        assert (CSV_FIELDS[
                CSV_FIELDS.index("cpu_perf_running_pct") + 1:
                CSV_FIELDS.index("cpu_perf_running_pct") + 1 + len(fields)
            ]) == (fields)

    def test_csv_schema_includes_cgroup_pressure_metrics(self) -> None:
        fields = [
            "container_cpu_nr_periods_delta",
            "container_cpu_nr_throttled_delta",
            "container_cpu_throttled_period_ratio_pct",
            "container_cpu_throttled_time_s_per_request",
            "container_cpu_pressure_some_stall_pct",
            "container_cpu_pressure_full_stall_pct",
        ]
        start = CSV_FIELDS.index("container_cpu_util_peak_pct") + 1
        assert (CSV_FIELDS[start:start + len(fields)]) == (fields)

    def test_csv_schema_includes_container_swap_and_io_metrics(self) -> None:
        fields = [
            "container_mem_peak_cgroup_bytes",
            "container_mem_anon_bytes_end",
            "container_mem_file_bytes_end",
            "container_mem_slab_bytes_end",
            "container_mem_pgfault_delta",
            "container_mem_pgmajfault_delta",
            "container_mem_workingset_refault_delta",
            "container_mem_high_events_delta",
            "container_mem_max_events_delta",
            "container_mem_oom_events_delta",
            "container_mem_oom_kill_events_delta",
            "container_mem_pressure_some_stall_pct",
            "container_mem_pressure_full_stall_pct",
            "container_swap_limit_bytes",
            "container_swap_usage_avg_bytes",
            "container_swap_usage_peak_bytes",
            "container_io_read_bytes_per_request",
            "container_io_write_bytes_per_request",
            "container_io_read_ops_per_request",
            "container_io_write_ops_per_request",
            "container_io_pressure_some_stall_pct",
            "container_io_pressure_full_stall_pct",
            "container_pids_current_end",
            "container_pids_peak_cgroup",
            "container_pids_max_events_delta",
        ]

        start = CSV_FIELDS.index("container_mem_util_peak_pct") + 1
        assert (CSV_FIELDS[start:start + len(fields)]) == (fields)

    def test_csv_schema_includes_massif_and_nsys_execution_metrics(self) -> None:
        massif_fields = [
            "cpu_heap_peak_bytes_massif",
            "cpu_heap_extra_peak_bytes_massif",
            "cpu_stack_peak_bytes_massif",
            "cpu_heap_peak_total_bytes_massif",
            "cpu_heap_peak_at_ms_massif",
            "compute_profile_error_massif",
        ]
        nsys_fields = [
            "host_inference_wall_time_ms_per_request_nsys",
            "cuda_api_time_sum_ms_per_request_nsys",
            "cuda_api_call_count_per_request_nsys",
            "gpu_kernel_time_sum_ms_per_request_nsys",
            "gpu_kernel_launch_count_per_request_nsys",
            "gpu_memcpy_time_sum_ms_per_request_nsys",
            "gpu_memcpy_count_per_request_nsys",
            "gpu_memcpy_bytes_per_request_nsys",
            "compute_profile_error_nsys",
        ]

        for field in [*massif_fields, *nsys_fields]:
            assert (field) in (CSV_FIELDS)
        assert (CSV_FIELDS[
                CSV_FIELDS.index("compute_profile_error_ncu") + 1:
                CSV_FIELDS.index("gpu_idle_power_w")
            ]) == ([*massif_fields, *nsys_fields])

    def test_execution_profile_metrics_are_formatted_independently(self) -> None:
        metrics = client._execution_profile_row_metrics({
            "cpu_heap_peak_bytes_massif": 4096,
            "compute_profile_error_massif": "",
            "host_inference_wall_time_ms_per_request_nsys": 2.5,
            "cuda_api_time_sum_ms_per_request_nsys": 1.25,
            "compute_profile_error_nsys": "nsys_stats_failed",
        })

        assert (metrics["cpu_heap_peak_bytes_massif"]) == ("4096.000000")
        assert (metrics["host_inference_wall_time_ms_per_request_nsys"]) == ("2.500000")
        assert (metrics["cuda_api_time_sum_ms_per_request_nsys"]) == ("1.250000")
        assert (metrics["compute_profile_error_massif"]) == ("")
        assert (metrics["compute_profile_error_nsys"]) == ("nsys_stats_failed")
        assert (metrics["gpu_kernel_time_sum_ms_per_request_nsys"]) == ("nan")

    def test_auto_repeat_window_prepares_each_scale_with_warmup_only(self) -> None:
        request_ids = []

        def fake_one_request(scale_value, req_id, payload_override=None):
            request_ids.append(req_id)
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with patch_client(self.runner, "CASE_NAME", "case"
        ), patch_client(self.runner, "REPEAT_IN_WINDOW", 0
        ), patch_client(self.runner, "REPEAT_WINDOW_SECONDS", 0.05, create=True
        ), patch_client(self.runner, "AUTO_WARMUP_REQUESTS", 2, create=True
        ), patch_client(self.runner,
            "_one_request",
            side_effect=fake_one_request,
        ):
            repeat_counts = [
                self.runner._prepare_repeat_window(1.0, "seq1", {}),
                self.runner._prepare_repeat_window(2.0, "seq2", {}),
            ]

        assert (repeat_counts) == ([1, 1])
        assert (request_ids) == ([
                "case_seq1_auto_warmup0",
                "case_seq1_auto_warmup1",
                "case_seq2_auto_warmup0",
                "case_seq2_auto_warmup1",
            ])

    def test_auto_repeat_window_continues_until_target_duration_when_requests_get_faster(self) -> None:
        measurement_req_ids = []

        def fake_one_request(scale_value, req_id, payload_override=None):
            if "_auto_warmup" not in req_id:
                measurement_req_ids.append(req_id)
            latency = 0.2
            return {
                "latency_app_s": latency,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 0
            ), patch_client(self.runner, "REPEAT_WINDOW_SECONDS", 1.0, create=True
            ), patch_client(self.runner, "AUTO_WARMUP_REQUESTS", 0, create=True
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                side_effect=fake_one_request,
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["repeat_in_window"]) == ("5")
        assert (measurement_req_ids) == ([
                "case_seq1_r0:0",
                "case_seq1_r0:1",
                "case_seq1_r0:2",
                "case_seq1_r0:3",
                "case_seq1_r0:4",
            ])

    def test_latency_app_distribution_fields_are_written_per_window(self) -> None:
        latencies = iter([0.01, 0.02, 0.10, 0.20, 0.30])

        def fake_one_request(scale_value, req_id, payload_override=None):
            return {
                "latency_app_s": next(latencies),
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            Path(tmp_dir, "static_meta.json").write_text(json.dumps({
                "schema_version": 7,
                "latency_slo": {"threshold_s": 0.2, "source": "task:fill-mask"},
            }), encoding="utf-8")
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 5
            ), patch.dict(
                os.environ, {"SLOW_LATENCY_THRESHOLD_S": "0.001"}
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                side_effect=fake_one_request,
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

            assert (os.path.isfile(f"{out_csv}.requests.jsonl"))
            with open(f"{out_csv}.requests.jsonl", encoding="utf-8") as requests_file:
                request_windows = [json.loads(line) for line in requests_file]

        assert (request_windows[0]["latency_app_s"]) == ([0.01, 0.02, 0.10, 0.20, 0.30])
        assert (request_windows[0]["sniff_group_id"]) == ("case_seq1_r0")
        assert (request_windows[0]["status"]) == ("ok")
        assert (rows[0]["latency_app_s"]) == ("0.126000")
        assert (rows[0]["latency_app_request_count"]) == ("5.000000")
        assert (rows[0]["latency_app_p50_s"]) == ("0.100000")
        assert (rows[0]["latency_app_p90_s"]) == ("0.300000")
        assert (rows[0]["latency_app_p95_s"]) == ("0.300000")
        assert (rows[0]["latency_app_std_s"]) == ("0.110562")
        assert (rows[0]["latency_app_cv"]) == ("0.877478")
        assert (rows[0]["latency_app_iqr_s"]) == ("0.180000")
        assert (rows[0]["latency_app_max_s"]) == ("0.300000")
        assert (rows[0]["latency_app_slow_ratio"]) == ("0.200000")
        assert (rows[0]["latency_app_tail_ratio"]) == ("3.000000")

    def test_efficiency_metrics_are_derived_without_new_measurements(self) -> None:
        metrics = client._derived_efficiency_metrics(
            gpu_mode="on",
            batch_size=2,
            latency_app_s=0.5,
            output_token_count_avg=10.0,
            gpu_energy_eff_j=0.3,
            vcpu_energy_eff_j=0.2,
            input_units_per_request=8.0,
        )

        assert (metrics["container_attributed_energy_eff_j"]) == (0.5) or round(abs((metrics["container_attributed_energy_eff_j"]) - (0.5)), 7) == 0
        assert (metrics["container_attributed_samples_per_j"]) == (4.0) or round(abs((metrics["container_attributed_samples_per_j"]) - (4.0)), 7) == 0
        assert (metrics["container_attributed_edp_app_js"]) == (0.25) or round(abs((metrics["container_attributed_edp_app_js"]) - (0.25)), 7) == 0
        assert (metrics["output_tokens_per_s_app"]) == (20.0) or round(abs((metrics["output_tokens_per_s_app"]) - (20.0)), 7) == 0
        assert (metrics["container_attributed_j_per_output_token"]) == (0.05) or round(abs((metrics["container_attributed_j_per_output_token"]) - (0.05)), 7) == 0
        assert (metrics["container_attributed_j_per_input_unit"]) == (0.0625) or round(abs((metrics["container_attributed_j_per_input_unit"]) - (0.0625)), 7) == 0

        missing_gpu = client._derived_efficiency_metrics(
            gpu_mode="on",
            batch_size=1,
            latency_app_s=0.5,
            output_token_count_avg=float("nan"),
            gpu_energy_eff_j=float("nan"),
            vcpu_energy_eff_j=0.2,
        )
        assert (math.isnan(missing_gpu["container_attributed_energy_eff_j"]))

    def test_cold_start_row_metrics_include_phases_and_first_predict(self) -> None:
        with patch_client(self.runner,
            "COLD_START_STARTED_AT",
            "2026-08-23T10:00:00.000+08:00",
        ), patch_client(self.runner,
            "COLD_START_READY_AT",
            "2026-08-23T10:00:01.000+08:00",
        ), patch_client(self.runner,
            "COLD_START_CONTAINER_LAUNCH_S",
            "0.1",
        ), patch_client(self.runner,
            "COLD_START_SERVER_SETUP_S",
            "0.2",
        ), patch_client(self.runner,
            "COLD_START_CUDA_INIT_S",
            "0.05",
        ), patch_client(self.runner,
            "COLD_START_MODEL_LOAD_S",
            "0.55",
        ), patch_client(self.runner,
            "COLD_START_READY_WAIT_S",
            "0.1",
        ), patch_client(self.runner,
            "COLD_START_S",
            "1.0",
        ), patch_client(self.runner,
            "_FIRST_PREDICT_APP_S",
            0.25,
        ):
            metrics = self.runner._cold_start_row_metrics()

        assert (metrics["cold_start_container_launch_s"]) == ("0.1")
        assert (metrics["cold_start_cuda_init_s"]) == ("0.05")
        assert (metrics["cold_start_model_load_s"]) == ("0.55")
        assert (metrics["cold_start_first_predict_app_s"]) == ("0.250000")
        assert (metrics["cold_start_s"]) == ("1.0")

    def test_manual_repeat_in_window_skips_auto_warmup(self) -> None:
        def fake_one_request(scale_value, req_id, payload_override=None):
            assert ("_auto_warmup") not in (req_id)
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 20
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                side_effect=fake_one_request,
            ) as one_request:
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["repeat_in_window"]) == ("20")
        assert (one_request.call_count) == (20)

    def test_gpu_energy_uses_one_matched_control_baseline_per_workload(self) -> None:
        sleep_calls = []

        class FakeGpuMonitor:
            apply_control_calls = 0

            def __init__(self, *args, **kwargs):
                self.idle_power_w = float("nan")

            def apply_control_baseline(self, result, samples, trace=False):
                type(self).apply_control_calls += 1
                self.idle_power_w = 10.0
                return 10.0

            def start(self):
                pass

            def stop(self):
                return (
                    SimpleNamespace(
                        idle_power_w=self.idle_power_w,
                        energy_iters=10,
                        avg_power_total_w=12.0,
                        peak_power_total_w=13.0,
                        energy_total_j=1.0,
                        avg_power_eff_w=2.0,
                        peak_power_eff_w=3.0,
                        energy_eff_j=0.2,
                    ),
                    "Test GPU",
                    "",
                    [],
                )

            def close(self):
                pass

        def fake_one_request(scale_value, req_id, payload_override=None):
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 2
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", True
            ), patch_client(self.runner,
                "energy_mod",
                SimpleNamespace(GPUEnergyMonitor=FakeGpuMonitor),
            ), patch_client(self.runner,
                "IDLE_COOLDOWN_SECONDS",
                2.5,
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.time,
                "sleep",
                side_effect=lambda seconds: sleep_calls.append(seconds),
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                side_effect=fake_one_request,
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    rows = list(reader)

        assert (FakeGpuMonitor.apply_control_calls) == (2)
        assert (sleep_calls) == ([2.5, 2.5])
        assert ("gpu_idle_power_w") in (fieldnames)
        assert ("idle_power_w") not in (fieldnames)
        assert ("gpu_energy_iters") in (fieldnames)
        assert ("energy_iters") not in (fieldnames)
        assert ("gpu_energy_eff_j") in (fieldnames)
        assert ("energy_eff_j") not in (fieldnames)
        assert (rows[0]["gpu_idle_power_w"]) == ("10.000000")
        assert (rows[0]["gpu_energy_eff_j"]) == ("0.200000")
        assert (rows[0]["gpu_avg_power_total_w"]) == ("12.000000")
        assert ("gpu_power_w") not in (rows[0])

    def test_idle_cooldown_applies_before_cpu_idle_without_gpu(self) -> None:
        sleep_calls = []

        class FakeCPUMonitor:
            apply_control_calls = 0

            def __init__(self, **kwargs):
                self.idle_power_w = float("nan")

            def apply_control_baseline(self, result, samples, trace=False):
                type(self).apply_control_calls += 1
                self.idle_power_w = 5.0
                return self.idle_power_w

            def start(self):
                pass

            def stop(self):
                return SimpleNamespace(
                    cpu_energy_iters=2,
                    cpu_idle_power_w=self.idle_power_w,
                    cpu_avg_power_total_w=6.0,
                    cpu_peak_power_total_w=7.0,
                    cpu_energy_total_j=1.0,
                    cpu_avg_power_eff_w=1.0,
                    cpu_peak_power_eff_w=2.0,
                    cpu_energy_eff_j=0.5,
                    vcpu_cpu_share=0.5,
                    vcpu_cpu_time_s=0.1,
                    vcpu_avg_power_total_w=3.0,
                    vcpu_peak_power_total_w=4.0,
                    vcpu_energy_total_j=0.3,
                    vcpu_avg_power_eff_w=0.4,
                    vcpu_peak_power_eff_w=0.5,
                    vcpu_energy_eff_j=0.2,
                ), "", []

            def close(self):
                pass

        def fake_one_request(scale_value, req_id, payload_override=None):
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 2
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner,
                "IDLE_COOLDOWN_SECONDS",
                2.5,
                create=True,
            ), patch_client(self.runner,
                "energy_mod",
                None,
            ), patch_client(self.runner,
                "cpu_energy_mod",
                SimpleNamespace(CPUEnergyMonitor=lambda **kwargs: FakeCPUMonitor(**kwargs)),
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.time,
                "sleep",
                side_effect=lambda seconds: sleep_calls.append(seconds),
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                side_effect=fake_one_request,
            ):
                self.runner.main()

        assert (FakeCPUMonitor.apply_control_calls) == (2)
        assert (sleep_calls) == ([2.5, 2.5])

    def test_client_entrypoint_prints_friendly_energy_abort_without_traceback(self) -> None:
        stderr = io.StringIO()
        with patch_client(self.runner,
            "main",
            side_effect=client.EnergyAbort("gpu_idle_power_w failed"),
        ), pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            self.runner.run_cli()

        assert (raised.value.code) == (1)
        assert ("[energy][ERROR] gpu_idle_power_w failed") in (stderr.getvalue())
        assert ("Traceback") not in (stderr.getvalue())

    def test_one_request_converts_http_timeout_to_case_abort(self) -> None:
        with patch_client(self.runner,
            "REQUEST_TIMEOUT_SECONDS",
            0.25,
        ), patch.object(
            client.requests,
            "post",
            side_effect=client.requests.exceptions.ReadTimeout("slow inference"),
        ), pytest.raises(client.RequestTimeoutAbort) as raised:
            self.runner._one_request(
                10.0,
                req_id="case_dur10_auto_warmup0",
                payload_override={},
            )

        message = str(raised.value)
        assert ("inactivity timeout: 0.25s") in (message)
        assert ("input_scale=10") in (message)
        assert ("req_id=case_dur10_auto_warmup0") in (message)
        assert (raised.value.input_scale) == (10.0)
        assert (raised.value.request_id) == ("case_dur10_auto_warmup0")
        assert (raised.value.timeout_s) == (0.25)

    def test_measurement_timeout_escapes_row_level_error_handling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch_client(self.runner,
            "OUT_CSV",
            f"{tmp_dir}/result.csv",
        ), patch_client(self.runner,
            "CASE_NAME",
            "case",
        ), patch_client(self.runner,
            "WARMUP",
            0,
        ), patch_client(self.runner,
            "REPEAT",
            1,
        ), patch_client(self.runner,
            "REPEAT_IN_WINDOW",
            2,
        ), patch_client(self.runner,
            "USE_ENERGY",
            False,
        ), patch_client(self.runner,
            "USE_MIPS",
            False,
        ), patch_client(self.runner,
            "energy_mod",
            None,
        ), patch_client(self.runner,
            "cpu_energy_mod",
            None,
        ), patch_client(self.runner,
            "resource_usage_mod",
            None,
        ), patch_client(self.runner,
            "input_scale_entries",
            [{"input_scale": 10.0, "scale_label": "dur10", "payload": {}}],
        ), patch.object(
            client.requests,
            "get",
            return_value=SimpleNamespace(status_code=200, text="ok"),
        ), patch_client(self.runner,
            "_one_request",
            side_effect=[{"latency_app_s": 0.25, "effective_input_scale": 10.0}, client.RequestTimeoutAbort("slow inference")],
        ):
            with pytest.raises(client.RequestTimeoutAbort):
                self.runner.main()
            with open(f"{tmp_dir}/result.csv.requests.jsonl", encoding="utf-8") as f:
                window = json.loads(f.readline())
            assert (window["status"]) == ("error")
            assert (window["latency_app_s"]) == ([0.25])
            assert (window["failed_request_id"]) == ("case_dur10_r0:1")

    def test_client_entrypoint_uses_dedicated_timeout_exit_code(self) -> None:
        stderr = io.StringIO()
        with patch_client(self.runner,
            "main",
            side_effect=client.RequestTimeoutAbort("slow inference"),
        ), pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            self.runner.run_cli()

        assert (raised.value.code) == (client.CLIENT_REQUEST_TIMEOUT_EXIT_CODE)
        assert ("[case][ERROR] slow inference") in (stderr.getvalue())
        assert ("Traceback") not in (stderr.getvalue())

    def test_client_entrypoint_persists_structured_timeout_context(self) -> None:
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sidecar_path = os.path.join(tmp_dir, "client-error.json")
            exc = client.RequestTimeoutAbort(
                "slow inference",
                input_scale=30.0,
                request_id="case_dur30_auto_warmup0",
                timeout_s=300.0,
            )
            with patch_client(self.runner,
                "CLIENT_ERROR_PATH",
                sidecar_path,
            ), patch_client(self.runner,
                "main",
                side_effect=exc,
            ), pytest.raises(SystemExit), redirect_stderr(stderr):
                self.runner.run_cli()

            with open(sidecar_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

        assert (payload["error_type"]) == ("client_request_timeout")
        assert (payload["input_scale"]) == (30.0)
        assert (payload["request_timeout_s"]) == (300.0)
        assert (payload["request_phase"]) == ("auto_repeat_window_warmup")
        assert (payload["request_index_in_window"]) == (0)

    def test_sniff_group_id_is_hidden_from_csv_but_kept_for_packet_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    fieldnames = reader.fieldnames or []

            with open(f"{out_csv}.sniff_groups.jsonl", "r", encoding="utf-8") as f:
                sidecar_rows = [json.loads(line) for line in f if line.strip()]

        assert ("sniff_group_id") not in (fieldnames)
        assert ("sniff_group_id") not in (rows[0])
        assert (sidecar_rows) == ([{"sniff_group_id": "case_seq1_r0"}])

    def test_negative_effective_metrics_are_reported_per_field(self) -> None:
        warnings = client._eff_negative_warnings(
            avg_power_eff_w=-0.1,
            peak_power_eff_w=0.0,
            energy_eff_j=-0.001,
        )

        assert (warnings) == (["gpu_avg_power_eff_w<0", "gpu_energy_eff_j<0"])

    def test_cpu_vcpu_negative_effective_metrics_keep_full_field_names(self) -> None:
        warnings = client._named_negative_warnings({
            "cpu_energy_eff_j": -0.001,
            "vcpu_avg_power_eff_w": -0.1,
            "vcpu_energy_total_j": 1.0,
        })

        assert (warnings) == (["cpu_energy_eff_j<0", "vcpu_avg_power_eff_w<0"])

    def test_cpu_monitor_unavailable_keeps_successful_row_ok(self) -> None:
        class FakeUnavailableCPUMonitor:
            def apply_control_baseline(self, result, samples, trace=False):
                return float("nan")

            def start(self):
                return None

            def stop(self):
                return energy_cpu._nan_result(), "RAPL unavailable", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                SimpleNamespace(CPUEnergyMonitor=lambda **kwargs: FakeUnavailableCPUMonitor()),
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["status"]) == ("ok")
        assert (rows[0]["error"]) == ("")
        assert (rows[0]["cpu_energy_total_j"]) == ("nan")

    def test_idle_debug_writes_csv_fields_and_diagnostic_jsonl(self) -> None:
        class FakeCPUMonitor:
            idle_values = iter([5.0, 5.5])
            trace_intervals = []

            def __init__(self, **kwargs):
                self.idle_power_w = float("nan")
                self.idle_trace = {}

            def apply_control_baseline(self, result, samples, trace=False):
                type(self).trace_intervals.append(trace)
                self.idle_power_w = next(type(self).idle_values)
                self.idle_trace = {
                    "idle_trace_schema": "cpu_rapl_idle_v1",
                    "actual_idle_duration_s": 3.0,
                    "rapl_trace": {
                        "interval_s": 0.1,
                        "power_windows": [{"t0_s": 0.0, "t1_s": 0.1, "power_w": 6.0}],
                    },
                    "idle_proc_cpu_top": [{"pid": 123, "comm": "python", "cpu_time_ms": 10.0}],
                    "idle_container_cpu_delta_s": 0.01,
                }
                return self.idle_power_w

            def start(self):
                return None

            def stop(self):
                return SimpleNamespace(
                    cpu_energy_iters=2,
                    cpu_idle_power_w=self.idle_power_w,
                    cpu_avg_power_total_w=6.0,
                    cpu_peak_power_total_w=7.0,
                    cpu_energy_total_j=1.0,
                    cpu_avg_power_eff_w=1.0,
                    cpu_peak_power_eff_w=2.0,
                    cpu_energy_eff_j=0.5,
                    vcpu_cpu_share=0.5,
                    vcpu_cpu_time_s=0.1,
                    vcpu_avg_power_total_w=3.0,
                    vcpu_peak_power_total_w=4.0,
                    vcpu_energy_total_j=0.3,
                    vcpu_avg_power_eff_w=0.4,
                    vcpu_peak_power_eff_w=0.5,
                    vcpu_energy_eff_j=0.2,
                ), "", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            diag_path = os.path.join(
                tmp_dir,
                "debug_idle_diag",
                "result.csv.idle_diag.jsonl",
            )
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "CPU_CORES", "1"
            ), patch_client(self.runner, "MEM_CAP_GB", "2"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 2
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "IDLE_DEBUG", True, create=True
            ), patch_client(self.runner, "IDLE_DIAG_PATH", "", create=True
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                SimpleNamespace(CPUEnergyMonitor=lambda **kwargs: FakeCPUMonitor(**kwargs)),
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ), patch_client(self.runner,
                "_collect_idle_debug_snapshot",
                return_value={
                    "snapshot_scope": "after_idle",
                    "loadavg": [0.1, 0.2, 0.3],
                    "top_cpu_processes": [{"pid": 123, "comm": "python", "cpu_pct": 4.5}],
                    "docker_containers": [{"name": "case"}],
                    "docker_stats": [{"name": "case", "cpu_perc": "0.1%"}],
                },
                create=True,
            ), patch_client(self.runner,
                "_now_iso",
                side_effect=[
                    "2026-05-02T10:00:00+08:00",
                    "2026-05-02T10:00:01+08:00",
                ],
                create=True,
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                with open(diag_path, "r", encoding="utf-8") as f:
                    diag_rows = [json.loads(line) for line in f if line.strip()]

        assert (rows[0]["cpu_idle_measured_at"]) == ("2026-05-02T10:00:00+08:00")
        assert (rows[0]["gpu_idle_measured_at"]) == ("nan")
        assert (rows[0]["gpu_idle_rel_range_so_far"]) == ("nan")
        assert (rows[0]["cpu_idle_rel_range_so_far"]) == ("0.000000")
        assert (rows[1]["cpu_idle_measured_at"]) == ("2026-05-02T10:00:01+08:00")
        assert (rows[1]["gpu_idle_measured_at"]) == ("nan")
        assert (rows[1]["gpu_idle_rel_range_so_far"]) == ("nan")
        assert (rows[1]["cpu_idle_rel_range_so_far"]) == ("0.095238")
        assert (FakeCPUMonitor.trace_intervals) == ([True, True])
        assert (len(diag_rows)) == (2)
        assert (diag_rows[1]["sniff_group_id"]) == ("case_seq1_r1")
        assert (diag_rows[1]["cpu_idle_measured_at"]) == ("2026-05-02T10:00:01+08:00")
        assert (diag_rows[1]["idle_trace_schema"]) == ("cpu_rapl_idle_v1")
        assert (diag_rows[1]["actual_idle_duration_s"]) == (3.0)
        assert (diag_rows[1]["rapl_trace"]["interval_s"]) == (0.1)
        assert (diag_rows[1]["idle_proc_cpu_top"][0]["comm"]) == ("python")
        assert (diag_rows[1]["idle_container_cpu_delta_s"]) == (0.01)
        assert (diag_rows[1]["cpu_idle_valid_count"]) == (2)
        assert (diag_rows[1]["cpu_idle_mean_w"]) == (5.25) or round(abs((diag_rows[1]["cpu_idle_mean_w"]) - (5.25)), 7) == 0
        assert (diag_rows[1]["snapshot_scope"]) == ("after_idle")
        assert (diag_rows[1]["loadavg"]) == ([0.1, 0.2, 0.3])
        assert (diag_rows[1]["top_cpu_processes"][0]["comm"]) == ("python")
        assert (diag_rows[1]["docker_containers"][0]["name"]) == ("case")
        assert (diag_rows[1]["docker_stats"][0]["cpu_perc"]) == ("0.1%")

    def test_idle_debug_writes_gpu_idle_fields_and_diagnostics(self) -> None:
        class FakeGpuMonitor:
            idle_values = iter([10.0, 11.0])
            trace_args = []

            def __init__(self, **kwargs):
                self.idle_power_w = float("nan")
                self.idle_trace = {}

            def apply_control_baseline(self, result, samples, trace=False):
                type(self).trace_args.append(trace)
                self.idle_power_w = next(type(self).idle_values)
                self.idle_trace = {
                    "gpu_idle_trace_schema": "nvml_gpu_idle_v1",
                    "gpu_idle_sample_count": 2,
                    "gpu_idle_power_samples": [{"t_s": 0.0, "power_w": self.idle_power_w}],
                }
                return self.idle_power_w

            def start(self):
                return None

            def stop(self):
                return SimpleNamespace(
                    energy_iters=2,
                    idle_power_w=self.idle_power_w,
                    avg_power_total_w=self.idle_power_w + 1.0,
                    peak_power_total_w=self.idle_power_w + 2.0,
                    energy_total_j=1.0,
                    avg_power_eff_w=1.0,
                    peak_power_eff_w=2.0,
                    energy_eff_j=0.5,
                ), "Fake GPU", "", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            diag_path = f"{out_csv}.idle_diag.jsonl"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 2
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", True
            ), patch_client(self.runner, "IDLE_DEBUG", True, create=True
            ), patch_client(self.runner, "IDLE_DIAG_PATH", diag_path, create=True
            ), patch_client(self.runner,
                "energy_mod",
                SimpleNamespace(GPUEnergyMonitor=lambda **kwargs: FakeGpuMonitor(**kwargs)),
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ), patch_client(self.runner,
                "_collect_gpu_idle_debug_snapshot",
                return_value={
                    "gpu_snapshot_scope": "after_gpu_idle",
                    "nvidia_smi_gpu": {"pstate": "P0", "clocks_sm_mhz": 1200.0},
                    "nvidia_smi_pmon": [{"pid": 123, "type": "G", "command": "Xorg"}],
                },
                create=True,
            ), patch_client(self.runner,
                "_now_iso",
                side_effect=[
                    "2026-05-02T10:00:00+08:00",
                    "2026-05-02T10:00:01+08:00",
                ],
                create=True,
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                with open(diag_path, "r", encoding="utf-8") as f:
                    diag_rows = [json.loads(line) for line in f if line.strip()]

        assert (FakeGpuMonitor.trace_args) == ([True, True])
        assert (rows[0]["gpu_idle_measured_at"]) == ("2026-05-02T10:00:00+08:00")
        assert (rows[0]["gpu_idle_rel_range_so_far"]) == ("0.000000")
        assert (rows[1]["gpu_idle_measured_at"]) == ("2026-05-02T10:00:01+08:00")
        assert (rows[1]["gpu_idle_rel_range_so_far"]) == ("0.095238")
        assert (rows[1]["cpu_idle_measured_at"]) == ("nan")
        assert (rows[1]["cpu_idle_rel_range_so_far"]) == ("nan")
        assert (diag_rows[1]["gpu_idle_trace_schema"]) == ("nvml_gpu_idle_v1")
        assert (diag_rows[1]["cpu_idle_measured_at"]) == ("nan")
        assert (diag_rows[1]["gpu_idle_sample_count"]) == (2)
        assert (diag_rows[1]["gpu_idle_valid_count"]) == (2)
        assert (diag_rows[1]["gpu_idle_mean_w"]) == (10.5)
        assert (diag_rows[1]["gpu_snapshot_scope"]) == ("after_gpu_idle")
        assert (diag_rows[1]["nvidia_smi_gpu"]["pstate"]) == ("P0")
        assert (diag_rows[1]["nvidia_smi_pmon"][0]["command"]) == ("Xorg")

    def test_idle_debug_disabled_fills_nan_and_does_not_write_diag_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            diag_path = f"{out_csv}.idle_diag.jsonl"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "IDLE_DEBUG", False, create=True
            ), patch_client(self.runner, "IDLE_DIAG_PATH", diag_path, create=True
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["cpu_idle_measured_at"]) == ("nan")
        assert (rows[0]["gpu_idle_measured_at"]) == ("nan")
        assert (rows[0]["gpu_idle_rel_range_so_far"]) == ("nan")
        assert (rows[0]["cpu_idle_rel_range_so_far"]) == ("nan")
        assert not (os.path.exists(diag_path))

    def test_resource_usage_metrics_are_written_to_successful_row(self) -> None:
        class FakeResourceUsageMonitor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def start(self):
                return None

            def stop(self):
                return SimpleNamespace(
                    resource_usage_iters=2,
                    container_cpu_util_avg_pct=25.0,
                    container_cpu_util_peak_pct=50.0,
                    container_cpu_nr_periods_delta=10.0,
                    container_cpu_nr_throttled_delta=2.0,
                    container_cpu_throttled_period_ratio_pct=20.0,
                    container_cpu_throttled_time_s=0.4,
                    container_cpu_pressure_some_stall_pct=12.5,
                    container_cpu_pressure_full_stall_pct=1.5,
                    cpu_freq_avg_hz=3_000_000_000.0,
                    cpu_freq_peak_hz=3_200_000_000.0,
                    container_mem_usage_avg_bytes=1024.0,
                    container_mem_usage_peak_bytes=2048.0,
                    container_mem_util_avg_pct=1.0,
                    container_mem_util_peak_pct=2.0,
                    container_mem_peak_cgroup_bytes=4096.0,
                    container_mem_anon_bytes_end=1536.0,
                    container_mem_file_bytes_end=512.0,
                    container_mem_slab_bytes_end=256.0,
                    container_mem_pgfault_delta=30.0,
                    container_mem_pgmajfault_delta=2.0,
                    container_mem_workingset_refault_delta=4.0,
                    container_mem_high_events_delta=3.0,
                    container_mem_max_events_delta=2.0,
                    container_mem_oom_events_delta=1.0,
                    container_mem_oom_kill_events_delta=0.0,
                    container_mem_pressure_some_stall_pct=4.5,
                    container_mem_pressure_full_stall_pct=0.5,
                    container_swap_limit_bytes=4096.0,
                    container_swap_usage_avg_bytes=128.0,
                    container_swap_usage_peak_bytes=256.0,
                    container_io_read_bytes=1024.0,
                    container_io_write_bytes=2048.0,
                    container_io_read_ops=6.0,
                    container_io_write_ops=8.0,
                    container_io_pressure_some_stall_pct=2.5,
                    container_io_pressure_full_stall_pct=0.25,
                    container_pids_current_end=7.0,
                    container_pids_peak_cgroup=9.0,
                    container_pids_max_events_delta=1.0,
                    gpu_util_avg_pct=30.0,
                    gpu_util_peak_pct=40.0,
                    gpu_sm_clock_mhz=1500.0,
                    gpu_memory_clock_mhz=7000.0,
                    gpu_pstate="P2",
                    gpu_temp_c=61.0,
                    gpu_mem_used_avg_bytes=4096.0,
                    gpu_mem_used_peak_bytes=8192.0,
                    gpu_mem_util_avg_pct=3.0,
                    gpu_mem_util_peak_pct=4.0,
                    gpu_mem_total_bytes=100000.0,
                ), "", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CPU_CORES", "1"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 2
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                SimpleNamespace(ResourceUsageMonitor=lambda **kwargs: FakeResourceUsageMonitor(**kwargs)),
                create=True,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["status"]) == ("ok")
        assert (rows[0]["resource_usage_iters"]) == ("2.000000")
        assert (rows[0]["container_cpu_util_avg_pct"]) == ("25.000000")
        assert (rows[0]["container_cpu_nr_periods_delta"]) == ("10.000000")
        assert (rows[0]["container_cpu_nr_throttled_delta"]) == ("2.000000")
        assert (rows[0]["container_cpu_throttled_period_ratio_pct"]) == ("20.000000")
        assert (rows[0]["container_cpu_throttled_time_s_per_request"]) == ("0.200000")
        assert (rows[0]["container_cpu_pressure_some_stall_pct"]) == ("12.500000")
        assert (rows[0]["cpu_freq_avg_hz"]) == ("3000000000.000000")
        assert (rows[0]["cpu_freq_peak_hz"]) == ("3200000000.000000")
        assert (rows[0]["cpu_cycles_est_app"]) == ("375000000.000000")
        assert (rows[0]["cpu_cycles_est_packet"]) == ("nan")
        assert (rows[0]["gpu_sm_clock_mhz"]) == ("1500.000000")
        assert (rows[0]["gpu_memory_clock_mhz"]) == ("7000.000000")
        assert (rows[0]["gpu_pstate"]) == ("P2")
        assert (rows[0]["gpu_temp_c"]) == ("61.000000")
        assert (rows[0]["gpu_util_avg_pct"]) == ("30.000000")
        assert (rows[0]["container_swap_limit_bytes"]) == ("4096.000000")
        assert (rows[0]["container_mem_high_events_delta"]) == ("3.000000")
        assert (rows[0]["container_mem_peak_cgroup_bytes"]) == ("4096.000000")
        assert (rows[0]["container_mem_pgfault_delta"]) == ("30.000000")
        assert (rows[0]["container_mem_pgmajfault_delta"]) == ("2.000000")
        assert (rows[0]["container_mem_oom_events_delta"]) == ("1.000000")
        assert (rows[0]["container_mem_pressure_full_stall_pct"]) == ("0.500000")
        assert (rows[0]["container_swap_usage_avg_bytes"]) == ("128.000000")
        assert (rows[0]["container_swap_usage_peak_bytes"]) == ("256.000000")
        assert (rows[0]["container_io_read_bytes_per_request"]) == ("512.000000")
        assert (rows[0]["container_io_write_bytes_per_request"]) == ("1024.000000")
        assert (rows[0]["container_io_read_ops_per_request"]) == ("3.000000")
        assert (rows[0]["container_io_write_ops_per_request"]) == ("4.000000")
        assert (rows[0]["container_io_pressure_full_stall_pct"]) == ("0.250000")
        assert (rows[0]["container_pids_current_end"]) == ("7.000000")
        assert (rows[0]["container_pids_peak_cgroup"]) == ("9.000000")
        assert (rows[0]["container_pids_max_events_delta"]) == ("1.000000")
        assert ("gpu_power_w") not in (rows[0])
        assert ("gpu_util_percent") not in (rows[0])
        assert ("gpu_mem_total_bytes") not in (rows[0])

    def test_mips_metrics_are_written_to_successful_row(self) -> None:
        class FakeMIPSMonitor:
            def start(self):
                return None

            def stop(self, repeat_in_window: int, latency_app_s: float):
                return SimpleNamespace(
                    instructions_total=1_000_000.0,
                    instructions_per_request=500_000.0,
                    perf_elapsed_s=0.25,
                    cpu_mips_app=1.0,
                    cycles_per_request=250_000.0,
                    ref_cycles_per_request=200_000.0,
                    ipc=2.0,
                    running_pct=80.0,
                    cache_references_per_request=10_000.0,
                    cache_misses_per_request=500.0,
                    cache_miss_rate_pct=5.0,
                    dtlb_loads_per_request=2_000.0,
                    dtlb_load_misses_per_request=20.0,
                    dtlb_load_miss_rate_pct=1.0,
                )

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 2
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "USE_MIPS", True, create=True
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "perf_mips_mod",
                SimpleNamespace(PerfMIPSMonitor=lambda container_name: FakeMIPSMonitor()),
                create=True,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.25, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["status"]) == ("ok")
        assert (rows[0]["cpu_instructions_per_request"]) == ("500000.000000")
        assert (rows[0]["cpu_mips_app"]) == ("1.000000")
        assert (rows[0]["cpu_mips_packet"]) == ("nan")
        assert (rows[0]["cpu_cycles_per_request"]) == ("250000.000000")
        assert (rows[0]["cpu_ref_cycles_per_request"]) == ("200000.000000")
        assert (rows[0]["cpu_ipc"]) == ("2.000000")
        assert (rows[0]["cpu_perf_running_pct"]) == ("80.000000")
        assert (rows[0]["cpu_perf_elapsed_s"]) == ("0.250000")
        assert (rows[0]["cpu_cache_references_per_request"]) == ("10000.000000")
        assert (rows[0]["cpu_cache_misses_per_request"]) == ("500.000000")
        assert (rows[0]["cpu_cache_miss_rate_pct"]) == ("5.000000")
        assert (rows[0]["cpu_dtlb_loads_per_request"]) == ("2000.000000")
        assert (rows[0]["cpu_dtlb_load_misses_per_request"]) == ("20.000000")
        assert (rows[0]["cpu_dtlb_load_miss_rate_pct"]) == ("1.000000")

    def test_resource_usage_unavailable_keeps_successful_row_ok(self) -> None:
        class FakeUnavailableResourceUsageMonitor:
            def start(self):
                return None

            def stop(self):
                return SimpleNamespace(
                    resource_usage_iters=0,
                    container_cpu_util_avg_pct=float("nan"),
                    container_cpu_util_peak_pct=float("nan"),
                    cpu_freq_avg_hz=float("nan"),
                    cpu_freq_peak_hz=float("nan"),
                    container_mem_usage_avg_bytes=float("nan"),
                    container_mem_usage_peak_bytes=float("nan"),
                    container_mem_util_avg_pct=float("nan"),
                    container_mem_util_peak_pct=float("nan"),
                    gpu_util_avg_pct=float("nan"),
                    gpu_util_peak_pct=float("nan"),
                    gpu_mem_used_avg_bytes=float("nan"),
                    gpu_mem_used_peak_bytes=float("nan"),
                    gpu_mem_util_avg_pct=float("nan"),
                    gpu_mem_util_peak_pct=float("nan"),
                    gpu_mem_total_bytes=float("nan"),
                ), "resource usage unavailable", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                SimpleNamespace(ResourceUsageMonitor=lambda **kwargs: FakeUnavailableResourceUsageMonitor()),
                create=True,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["status"]) == ("ok")
        assert (rows[0]["error"]) == ("")
        assert (rows[0]["container_cpu_util_avg_pct"]) == ("nan")
        assert (rows[0]["gpu_util_avg_pct"]) == ("nan")
        assert (rows[0]["gpu_sm_clock_mhz"]) == ("nan")
        assert (rows[0]["gpu_memory_clock_mhz"]) == ("nan")
        assert (rows[0]["gpu_pstate"]) == ("nan")
        assert (rows[0]["gpu_temp_c"]) == ("nan")
        assert ("gpu_power_w") not in (rows[0])
        assert ("gpu_util_percent") not in (rows[0])

    def test_flat_compute_plan_is_rejected_without_emitting_generic_columns(
        self,
    ) -> None:
        plan = {
            "profiles": {
                "cpu": {
                    "tool": "intel_advisor",
                    "entries": [
                        {
                            "input_scale": 1.0,
                            "model_mflop_per_request": 200.0,
                            "error": "",
                        }
                    ],
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            plan_path = f"{tmp_dir}/compute_profile_plan.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f)

            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "GPU_MODE", "off"
            ), patch_client(self.runner, "COMPUTE_PROFILE_PLAN_FILE", plan_path, create=True
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["status"]) == ("ok")
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            assert (field) not in (rows[0])
        assert (rows[0]["model_logical_mflop_per_request_torch_profiler_eager"]) == ("nan")
        assert ("unsupported_profile_layout:cpu") in (rows[0]["compute_profile_error_torch_profiler_eager"])

    def test_compute_plan_writes_independent_torch_and_ncu_metrics(self) -> None:
        plan = {
            "profiles": {
                "gpu": {
                    "torch_profiler_eager": {
                        "tool": "torch_profiler_eager",
                        "entries": [
                            {
                                "input_scale": 1.0,
                                "model_logical_mflop_per_request_torch_profiler_eager": 200.0,
                                "error": "",
                            }
                        ],
                    },
                    "ncu": {
                        "tool": "ncu",
                        "entries": [
                            {
                                "input_scale": 1.0,
                                "gpu_executed_mflop_per_request_ncu": 100.0,
                                "gpu_executed_tensor_mflop_per_request_ncu": 80.0,
                                "gpu_executed_scalar_mflop_per_request_ncu": 20.0,
                                "gpu_executed_tensor_share_pct_ncu": 80.0,
                                "gpu_kernel_launch_count_per_request_ncu": 12.0,
                                "gpu_kernel_time_sum_ms_per_request_ncu": 1.5,
                                "error": "",
                            }
                        ],
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            plan_path = f"{tmp_dir}/compute_profile_plan.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f)

            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "GPU_MODE", "on"
            ), patch_client(self.runner, "COMPUTE_PROFILE_PLAN_FILE", plan_path, create=True
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner, "cpu_energy_mod", None
            ), patch_client(self.runner, "resource_usage_mod", None
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    row = next(csv.DictReader(f))

        assert (row["status"]) == ("ok")
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            assert (field) not in (row)
        assert (row["model_logical_mflop_per_request_torch_profiler_eager"]) == ("200.000000")
        assert (row["model_logical_mflops_app_torch_profiler_eager"]) == ("400.000000")
        assert (row["model_logical_mflops_packet_torch_profiler_eager"]) == ("400.000000")
        assert (row["gpu_executed_mflop_per_request_ncu"]) == ("100.000000")
        assert (row["gpu_executed_mflops_app_ncu"]) == ("200.000000")
        assert (row["gpu_executed_mflops_packet_ncu"]) == ("200.000000")
        assert ("gpu_profile_report_ncu") not in (row)
        assert (row["compute_profile_error_torch_profiler_eager"]) == ("")
        assert (row["compute_profile_error_ncu"]) == ("")

    def test_compute_profile_failures_are_isolated(self) -> None:
        profile = {
            "model_logical_mflop_per_request_torch_profiler_eager": float("nan"),
            "compute_profile_error_torch_profiler_eager": "torch_failed",
            "gpu_executed_mflop_per_request_ncu": 100.0,
            "gpu_executed_tensor_mflop_per_request_ncu": 90.0,
            "gpu_executed_scalar_mflop_per_request_ncu": 10.0,
            "gpu_executed_tensor_share_pct_ncu": 90.0,
            "gpu_kernel_launch_count_per_request_ncu": 4.0,
            "gpu_kernel_time_sum_ms_per_request_ncu": 0.75,
            "compute_profile_error_ncu": "",
        }

        row_metrics = client._compute_profile_row_metrics(profile, 0.5)

        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            assert (field) not in (row_metrics)
        assert (row_metrics["compute_profile_error_torch_profiler_eager"]) == ("torch_failed")
        assert (row_metrics["gpu_executed_mflop_per_request_ncu"]) == ("100.000000")
        assert (row_metrics["gpu_executed_mflops_app_ncu"]) == ("200.000000")
        assert (row_metrics["compute_profile_error_ncu"]) == ("")

    def test_missing_compute_profile_keeps_successful_row_ok_with_nan_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            missing_plan_path = f"{tmp_dir}/missing_compute_profile_plan.json"

            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "GPU_MODE", "off"
            ), patch_client(self.runner, "COMPUTE_PROFILE_PLAN_FILE", missing_plan_path, create=True
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["status"]) == ("ok")
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            assert (field) not in (rows[0])
        assert (rows[0]["model_logical_mflop_per_request_torch_profiler_eager"]) == ("nan")
        assert ("compute_profile_plan_not_found") in (rows[0]["compute_profile_error_torch_profiler_eager"])
