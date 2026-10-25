"""Regression evidence for failed counters, collector warnings, and /ready aborts."""
import csv
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.artifact_layout import case_sidecar
from acprof.config import CSV_FIELDS
from acprof.host import client, docker_runtime, orchestrator, runtime_images
from acprof.host.client import ClientRunner
from acprof.host.client_config import ClientConfig
from acprof.host.detect import TaskInfo
from acprof.monitors import energy_cpu, energy_nvml, resource_readers, resource_usage


def test_io_counter_reset_is_not_reported_as_zero():
    counters = {"read": 100, "write": 200}
    readers = resource_readers._ContainerReaders(
        io=lambda: (counters["read"], counters["write"])
    )
    with patch.object(resource_readers, "_resolve_container_metric_readers", return_value=readers), patch.object(
        resource_readers, "_prepare_cpu_frequency_reader", return_value=lambda: (None, None)
    ):
        monitor = resource_usage.ResourceUsageMonitor(sample_hz=1, container_name="case")
        try:
            monitor.start()
            counters.update(read=50, write=250)
            result, error, _ = monitor.stop()
        finally:
            monitor.close()

    assert math.isnan(result.container_io_read_bytes)
    assert result.container_io_write_bytes == 50.0
    assert "container_io_read_bytes counter reset: 100 -> 50" in error


def test_resource_errors_keep_distinct_gpu_queries_and_other_causes():
    with patch.object(resource_readers, "_resolve_container_metric_readers",
                      return_value=resource_readers._ContainerReaders()):
        monitor = resource_usage.ResourceUsageMonitor()
    monitor._partial_platform = False
    monitor._record_runtime_error("cgroup io error")
    monitor._gpu_query_error("utilization", RuntimeError("NVML utilization failed"))
    monitor._gpu_query_error("memory", RuntimeError("NVML memory failed"))

    error = monitor._error_message()
    assert "cgroup io error" in error
    assert "gpu[utilization]: NVML utilization failed" in error
    assert "gpu[memory]: NVML memory failed" in error
    assert error.count("NVML utilization failed") == 1


@pytest.mark.parametrize("kind", ("gpu", "cpu"))
def test_full_mode_propagates_energy_monitor_errors_to_csv(tmp_path, request, kind):
    from platform_fixtures import native_policy

    native_policy(request)

    class BrokenEnergyMonitor:
        def start(self):
            return None

        def stop(self):
            if kind == "gpu":
                return energy_nvml.unavailable_gpu_result(), "Test GPU", "NVML sampler failed", []
            return energy_cpu._nan_result(), "RAPL sampler failed", []

        def apply_control_baseline(self, result, samples, trace=False):
            return None

        def close(self):
            return None

    path = tmp_path / "case.csv"
    runner = ClientRunner(ClientConfig(
        out_csv=str(path),
        warmup=0, repeat=1, repeat_in_window=1,
        idle_seconds=0, idle_cooldown_seconds=0,
        use_mips=False, gpu_mode="on" if kind == "gpu" else "off",
        gpu_device_uuid="GPU-test" if kind == "gpu" else "",
    ))
    runner.input_scale_entries = [
        {"input_scale": 1.0, "scale_label": "one", "payload": {}},
    ]
    runner.cpu_energy_mod = None
    runner.energy_mod = None
    runner.resource_usage_mod = None
    if kind == "gpu":
        runner.energy_mod = SimpleNamespace(GPUEnergyMonitor=lambda **kwargs: BrokenEnergyMonitor())
    else:
        runner.cpu_energy_mod = SimpleNamespace(CPUEnergyMonitor=lambda **kwargs: BrokenEnergyMonitor())

    with patch.object(client.requests, "get", return_value=SimpleNamespace(status_code=200, text="ok")), patch.object(
        runner, "_one_request", return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0}
    ):
        runner.main()

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["status"] == "warn"
    assert f"{kind} monitor:" in rows[0]["error"]
    assert ("NVML sampler failed" if kind == "gpu" else "RAPL sampler failed") in rows[0]["error"]
    assert float(rows[0]["latency_app_s"]) == 0.5


def test_ready_failure_materializes_full_error_matrix_without_packet_misdiagnosis(tmp_path):
    task_info = TaskInfo(
        model_id="google-bert/bert-base-uncased", pipeline_tag="fill-mask",
        task_family="nlp", runtime_backend="transformers_pipeline",
        library_name="transformers", model_revision="main",
        detection_method="hub_api",
    )

    def fake_run(cmd, check=True, capture=True, **kwargs):
        if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
            with open(kwargs["env"]["OUT_CSV"], "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerow({"status": "error", "error": "ready_failed: Connection refused"})
            case_sidecar(kwargs["env"]["OUT_CSV"], "sniff_groups").write_text(
                '{"sniff_group_id": ""}\n', encoding="utf-8"
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("acprof.host.orchestrator.start_container_session", return_value=docker_runtime.RunningContainer(
        name="case_google-bert--bert-base-uncased_1c_4g_off",
        base_url="http://127.0.0.1:8106", host_port=8106, cold_start_s=1,
    )), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
        "acprof.host.orchestrator.stop_container_session"
    ), patch("acprof.host.command.run_command", side_effect=fake_run):
        path = orchestrator.run_single_case(
            orchestrator.CaseRequest(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu='off',
                image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                output_dir=str(tmp_path),
                project_dir='.',
            ),
            workload=orchestrator.CaseWorkload(
                warmup=1,
                repeat=2,
                repeat_in_window=1,
            ),
            measurement=orchestrator.CaseMeasurement(
                require_packet_latency=False,
            ),
            plans=orchestrator.CasePlans(
                input_scales='64,128',
            ),
        )

    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {row["status"] for row in rows} == {"error"}
    assert {float(row["input_scale"]) for row in rows} == {64.0, 128.0}
    assert all("ready_failed: Connection refused" in row["error"] for row in rows)
    assert all(row["result_origin"] == "formal_attempt" for row in rows)
    sidecar = case_sidecar(path, "sniff_groups")
    groups = sidecar.read_text(encoding="utf-8").splitlines()
    assert len(groups) == len(rows)
    assert set(groups) == {'{"sniff_group_id": ""}'}
