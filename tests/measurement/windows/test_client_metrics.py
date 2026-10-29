import json
import os
import tempfile
from functools import partial
from types import SimpleNamespace

import pytest

from acprof.config import (
    CSV_FIELDS,
    GPU_RUNTIME_STATE_FIELDS,
    STATIC_META_FIELDS,
    STATIC_META_SCHEMA_VERSION,
)
from acprof.host.client import ClientRunner
from acprof.host.measurement import metrics as client_metrics
from acprof.host.measurement.config import ClientConfig
from tests.support.client_fixtures import patch_client

REMOVED_LEGACY_COMPUTE_FIELDS = (
    "compute_profile_tool",
    "model_mflop_per_request",
    "compute_mflops_app",
    "compute_mflops",
    "compute_profile_error",
)


class TestClientMetrics:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from tests.support.platform_fixtures import native_policy
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

    def test_input_scale_plan_preserves_current_audio_metadata(self) -> None:
        plan = {
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
        }
        expected_metadata = {"input_num_samples": 16000, "audio_sha256": "abc123"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "input_scale_plan.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(plan, f)
            with patch_client(self.runner, "INPUT_SCALE_PLAN_FILE", path):
                entries = self.runner._load_input_scale_entries()

        assert (entries[0]["input_metadata"]) == (expected_metadata)

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
        metrics = client_metrics._gpu_runtime_metrics_from_result(
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
        metrics = client_metrics._execution_profile_row_metrics({
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
