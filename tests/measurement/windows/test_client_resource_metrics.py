import csv
import json
import os
import tempfile
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host import client
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


class TestClientResourceMetrics:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from tests.support.platform_fixtures import native_policy
        native_policy(self._request)
        self.runner = ClientRunner(ClientConfig(out_csv=os.path.join(str(tmp_path), "result.csv")))
        gpu_uuid = patch_client(self.runner, "GPU_DEVICE_UUID", "GPU-fixture")
        gpu_uuid.start()
        self._request.addfinalizer(partial(gpu_uuid.stop))
        for name in ("IDLE_SECONDS", "IDLE_COOLDOWN_SECONDS"):
            mocked = patch_client(self.runner, name, 0.0)
            mocked.start()
            self._request.addfinalizer(partial(mocked.stop))

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
                with open(out_csv, encoding="utf-8", newline="") as f:
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
                with open(out_csv, encoding="utf-8", newline="") as f:
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

    def test_resource_usage_unavailable_marks_successful_row_warn(self) -> None:
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
                with open(out_csv, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert rows[0]["status"] == "warn"
        assert "resource monitor: resource usage unavailable" in rows[0]["error"]
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
                with open(out_csv, encoding="utf-8", newline="") as f:
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
                with open(out_csv, encoding="utf-8", newline="") as f:
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

        row_metrics = client_metrics._compute_profile_row_metrics(profile, 0.5)

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
                with open(out_csv, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["status"]) == ("ok")
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            assert (field) not in (rows[0])
        assert (rows[0]["model_logical_mflop_per_request_torch_profiler_eager"]) == ("nan")
        assert ("compute_profile_plan_not_found") in (rows[0]["compute_profile_error_torch_profiler_eager"])
