import csv
import json
import os
import re
import tempfile
from contextlib import redirect_stderr
from functools import partial
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.config import (
    CSV_FIELDS,
    STATIC_META_FIELDS,
    STATIC_META_SCHEMA_VERSION,
)
from acprof.host import (
    docker_runtime,
    input_plan,
    model_schema,
    orchestrator,
    packet_capture,
    runtime_images,
    static_metadata,
)
from acprof.host.detect import TaskInfo


def _write_gpu_case_csv(
    path: str,
    idle_power_values: list[float],
    cpu_idle_power_values: list[float] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cpu_values = cpu_idle_power_values or [5.0 for _ in idle_power_values]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for idx, idle_power_w in enumerate(idle_power_values):
            row = {field: "nan" for field in CSV_FIELDS}
            row.update({
                "gpu_mode": "on",
                "gpu_idle_power_w": str(idle_power_w),
                "cpu_idle_power_w": str(cpu_values[idx]),
                "repeat_idx": str(idx),
                "warmup": "0",
                "status": "ok",
                "error": "",
            })
            writer.writerow(row)


def _write_cpu_case_csv(path: str, idle_power_values: list[float], gpu_mode: str = "off") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for idx, idle_power_w in enumerate(idle_power_values):
            row = {field: "nan" for field in CSV_FIELDS}
            row.update({
                "gpu_mode": gpu_mode,
                "cpu_idle_power_w": str(idle_power_w),
                "repeat_idx": str(idx),
                "warmup": "0",
                "status": "ok",
                "error": "",
            })
            writer.writerow(row)


class TestDetectEnvironment:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from platform_fixtures import native_policy
        native_policy(self._request)
        conditions = patch("acprof.host.orchestrator.record_case_conditions")
        conditions.start()
        self._request.addfinalizer(partial(conditions.stop))
        output = tmp_path
        self.output_dir = str(output)

    def test_host_mem_total_bytes_uses_physical_page_count(self) -> None:
        values = {
            "SC_PAGE_SIZE": 4096,
            "SC_PHYS_PAGES": 8_000_000,
        }

        with patch(
            "acprof.host.static_metadata.os.sysconf",
            side_effect=lambda name: values[name],
        ):
            total = static_metadata._host_mem_total_bytes()

        assert (total) == (32_768_000_000)

    def test_host_swap_metadata_reads_capacity_usage_type_and_swappiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meminfo_path = os.path.join(tmp, "meminfo")
            swaps_path = os.path.join(tmp, "swaps")
            swappiness_path = os.path.join(tmp, "swappiness")
            with open(meminfo_path, "w", encoding="utf-8") as f:
                f.write("SwapTotal:       2097148 kB\n")
                f.write("SwapFree:        2096124 kB\n")
            with open(swaps_path, "w", encoding="utf-8") as f:
                f.write("Filename Type Size Used Priority\n")
                f.write("/swapfile file 2097148 1024 -2\n")
            with open(swappiness_path, "w", encoding="utf-8") as f:
                f.write("60\n")

            metadata = static_metadata._host_swap_metadata(
                proc_meminfo_path=meminfo_path,
                proc_swaps_path=swaps_path,
                swappiness_path=swappiness_path,
            )

        assert (metadata["host_swap_total_bytes"]) == (2_147_479_552)
        assert (metadata["host_swap_used_bytes_at_start"]) == (1_048_576)
        assert (metadata["host_swap_type"]) == ("file")
        assert (metadata["host_vm_swappiness"]) == (60)

    def test_host_swap_metadata_reports_none_when_swap_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meminfo_path = os.path.join(tmp, "meminfo")
            swaps_path = os.path.join(tmp, "swaps")
            swappiness_path = os.path.join(tmp, "swappiness")
            with open(meminfo_path, "w", encoding="utf-8") as f:
                f.write("SwapTotal:             0 kB\n")
                f.write("SwapFree:              0 kB\n")
            with open(swaps_path, "w", encoding="utf-8") as f:
                f.write("Filename Type Size Used Priority\n")
            with open(swappiness_path, "w", encoding="utf-8") as f:
                f.write("0\n")

            metadata = static_metadata._host_swap_metadata(
                proc_meminfo_path=meminfo_path,
                proc_swaps_path=swaps_path,
                swappiness_path=swappiness_path,
            )

        assert (metadata["host_swap_total_bytes"]) == (0)
        assert (metadata["host_swap_used_bytes_at_start"]) == (0)
        assert (metadata["host_swap_type"]) == ("none")
        assert (metadata["host_vm_swappiness"]) == (0)

    def test_docker_storage_metadata_uses_daemon_root_backing_filesystem(self) -> None:
        with patch(
            "acprof.host.static_metadata._docker_root_dir",
            return_value="/var/lib/docker",
        ), patch(
            "acprof.host.static_metadata.shutil.disk_usage",
            return_value=SimpleNamespace(total=1_000, used=400, free=600),
        ) as disk_usage, patch(
            "acprof.host.static_metadata._docker_mount_metadata",
            return_value=("/dev/nvme0n1p2", "ext4"),
        ), patch(
            "acprof.host.static_metadata._block_device_storage_type",
            return_value="nvme_ssd",
        ):
            metadata = static_metadata._docker_storage_metadata()

        disk_usage.assert_called_once_with("/var/lib/docker")
        assert (metadata) == ({
                "docker_storage_total_bytes": 1_000,
                "docker_storage_available_bytes_at_start": 600,
                "docker_storage_filesystem": "ext4",
                "docker_storage_device": "/dev/nvme0n1p2",
                "docker_storage_type": "nvme_ssd",
            })

    @pytest.mark.parametrize('device_metadata_case', range(4))
    def test_block_device_type_uses_transport_and_rotational_flag(self, device_metadata_case) -> None:
        cases = (
            ({"tran": "nvme", "rota": False}, "nvme_ssd"),
            ({"tran": "sata", "rota": False}, "ssd"),
            ({"tran": "sata", "rota": True}, "hdd"),
            ({"tran": None, "rota": None}, "unknown"),
        )
        with patch(
            "acprof.host.static_metadata.shutil.which",
            return_value="/usr/bin/lsblk",
        ):
            (device_metadata, expected) = tuple(cases)[device_metadata_case]
            with patch(
                "acprof.host.command.run_command",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"blockdevices": [device_metadata]}),
                    stderr="",
                ),
            ):
                assert (static_metadata._block_device_storage_type("/dev/test")) == (expected)

    def test_docker_storage_metadata_is_unknown_when_daemon_root_is_unavailable(self) -> None:
        with patch(
            "acprof.host.static_metadata._docker_root_dir",
            return_value=None,
        ), patch(
            "acprof.host.static_metadata.shutil.disk_usage",
        ) as disk_usage:
            metadata = static_metadata._docker_storage_metadata()

        disk_usage.assert_not_called()
        assert (metadata["docker_storage_total_bytes"]) is None
        assert (metadata["docker_storage_available_bytes_at_start"]) is None
        assert (metadata["docker_storage_type"]) == ("unknown")

    def test_select_nlp_torch_index_url_uses_cu124_for_cuda_12_4_driver(self) -> None:
        with patch("acprof.host.runtime_images.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "acprof.host.command.run_command",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="Driver Version: 550.78    CUDA Version: 12.4\n",
                stderr="",
            ),
        ):
            assert (runtime_images.select_nlp_torch_index_url()) == (runtime_images.CUDA124_NLP_TORCH_INDEX_URL)

    def test_select_nlp_torch_index_url_respects_explicit_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"ACPROF_NLP_TORCH_INDEX_URL": "https://example.invalid/torch"},
            clear=True,
        ):
            assert (runtime_images.select_nlp_torch_index_url()) == ("https://example.invalid/torch")


    def test_runtime_container_is_offline_and_does_not_receive_hf_token(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="0123456789abcdef",
            detection_method="hub_api",
        )
        commands = []
        ready_response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "status": "ok",
                "model_id": task_info.model_id,
                "device": "cpu",
                "load_time_s": 1.0,
            },
        )

        with patch(
            "acprof.host.command.run_command",
            side_effect=lambda cmd, **_kwargs: (
                commands.append(cmd)
                or SimpleNamespace(returncode=0, stdout="b" * 64 if cmd[:2] == ["docker", "run"] else "", stderr="")
            ),
        ), patch("requests.get", return_value=ready_response):
            docker_runtime.start_container_session(
                task_info=task_info,
                cpu=1,
                mem=2,
                gpu="off",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                container_name="offline-test",
                log_prefix="[test]",
            )

        docker_run = next(cmd for cmd in commands if cmd[:3] == ["docker", "run", "-d"])
        assert (docker_run[docker_run.index("-p") + 1]) == ("127.0.0.1:8104:8002")
        assert ("HF_HUB_OFFLINE=1") in (docker_run)
        assert ("TRANSFORMERS_OFFLINE=1") in (docker_run)
        assert ("MODEL_LOCAL_PATH=/models/model-snapshot") in (docker_run)
        assert ("HF_TOKEN") not in (docker_run)
        assert ("HUGGING_FACE_HUB_TOKEN") not in (docker_run)

    def test_cold_start_breakdown_uses_container_startup_timestamps(self) -> None:
        metrics = docker_runtime._cold_start_breakdown(
            {
                "load_time_s": 9.0,
                "startup_timing": {
                    "server_process_started_at_epoch_s": 101.0,
                    "server_setup_s": 0.5,
                    "cuda_init_s": 0.25,
                    "model_load_completed_at_epoch_s": 104.0,
                    "model_load_s": 2.25,
                },
            },
            docker_started_at_epoch_s=100.0,
            ready_received_at_epoch_s=105.0,
        )

        assert (metrics["cold_start_container_launch_s"]) == (1.0)
        assert (metrics["cold_start_server_setup_s"]) == (0.5)
        assert (metrics["cold_start_cuda_init_s"]) == (0.25)
        assert (metrics["cold_start_model_load_s"]) == (2.25)
        assert (metrics["cold_start_ready_wait_s"]) == (1.0)
        assert (metrics["cold_start_started_at"]) != ("nan")
        assert (metrics["cold_start_ready_at"]) != ("nan")

    def test_start_container_session_surfaces_ready_processing_errors(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="0123456789abcdef",
            detection_method="hub_api",
        )
        ready_response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "ok"},
        )

        def fake_run(cmd, **_kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout="b" * 64 if cmd[:2] == ["docker", "run"] else "",
                stderr="",
            )

        with patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ), patch(
            "acprof.host.container_state.inspect_container_state",
            return_value={},
        ), patch(
            "acprof.host.container_state.container_startup_exit_error",
            return_value="container exited after ready handling",
        ), patch(
            "acprof.host.docker_runtime._cold_start_breakdown",
            side_effect=RuntimeError("timing parser bug"),
        ), patch(
            "requests.get",
            return_value=ready_response,
        ), pytest.raises(RuntimeError, match="timing parser bug"):
            docker_runtime.start_container_session(
                task_info=task_info,
                cpu=1,
                mem=2,
                gpu="off",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                container_name="ready-processing-test",
                log_prefix="[test]",
            )

    def test_start_container_session_reports_oom_before_ready_timeout(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        commands = []

        def fake_run(cmd, **_kwargs):
            commands.append(cmd)
            stdout = "[server] Loading model\n" if cmd[:2] == ["docker", "logs"] else "b" * 64
            if cmd[:2] == ["docker", "ps"]:
                stdout = ""
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        container_state = {
            "Status": "exited",
            "Running": False,
            "Restarting": False,
            "OOMKilled": True,
            "ExitCode": 137,
            "Error": "",
        }
        with patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ), patch(
            "acprof.host.container_state.inspect_container_state",
            return_value=container_state,
        ), patch(
            "requests.get",
            side_effect=ConnectionError("connection refused"),
        ):
            with pytest.raises(RuntimeError) as raised:
                docker_runtime.start_container_session(
                    task_info=task_info,
                    cpu=1,
                    mem=2,
                    gpu="off",
                    image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                    container_name="oom-test",
                    log_prefix="[test]",
                )

        message = str(raised.value)
        assert ("container_oom_killed") in (message)
        assert ("memory_limit=2g") in (message)
        assert ("exit_code=137") in (message)
        assert (["docker", "rm", "-f", "b" * 64]) in (commands)

    def test_windows_process_is_not_relabelled_from_docker_wsl_kernel(self) -> None:
        with patch("acprof.host.static_metadata.platform.system", return_value="Windows"), patch(
            "acprof.host.static_metadata.platform.release", return_value="11"
        ), patch.dict("acprof.host.static_metadata.os.environ", {}, clear=True), patch(
            "acprof.host.command.run_command",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="6.6.87.2-microsoft-standard-WSL2\n",
                stderr="",
            ),
        ):
            assert (static_metadata._detect_environment()) == ("windows11")

    def test_detect_environment_linux_ubuntu_without_wsl(self) -> None:
        with patch("acprof.host.static_metadata.platform.system", return_value="Linux"), patch(
            "acprof.host.static_metadata.platform.freedesktop_os_release",
            return_value={"ID": "ubuntu", "VERSION_ID": "24.04"},
        ), patch.dict("acprof.host.static_metadata.os.environ", {}, clear=True), patch(
            "acprof.host.command.run_command",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="docker unavailable"),
        ):
            assert (static_metadata._detect_environment()) == ("ubuntu24.04")

    def test_collect_static_meta_includes_environment(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
            parameter_count=110_106_428,
            parameter_bytes=440_425_712,
            precision_dtype="FP32",
            parameter_dtype_counts={"FP32": 110_106_428},
            quantized=False,
            model_license="apache-2.0",
            model_metadata_source="huggingface_hub",
            model_resolution={"schema_version": 1, "status": "candidate", "loader": "Transformers Auto/pipeline"},
        )

        with patch("acprof.host.static_metadata._detect_environment", return_value="windows11+wsl"), patch(
            "acprof.host.static_metadata._get_gpu_name", return_value="Test GPU"
        ), patch(
            "acprof.host.static_metadata._get_gpu_mem_total_bytes", return_value=987654321
        ), patch(
            "acprof.host.static_metadata._host_mem_total_bytes", return_value=64_000_000_000
        ), patch(
            "acprof.host.static_metadata._host_swap_metadata",
            return_value={
                "host_swap_total_bytes": 2_000_000_000,
                "host_swap_used_bytes_at_start": 100_000_000,
                "host_swap_type": "file",
                "host_vm_swappiness": 60,
            },
        ), patch(
            "acprof.host.static_metadata._docker_model_cache_bytes", return_value=123
        ) as cache_size, patch("acprof.host.static_metadata._docker_image_size_bytes", return_value=456), patch(
            "acprof.host.static_metadata._docker_storage_metadata",
            return_value={
                "docker_storage_total_bytes": 1_000_000,
                "docker_storage_available_bytes_at_start": 600_000,
                "docker_storage_filesystem": "ext4",
                "docker_storage_device": "/dev/nvme0n1p2",
                "docker_storage_type": "nvme_ssd",
            },
        ), patch(
            "acprof.host.static_metadata._cpu_power_metadata",
            return_value=("rapl", "rapl_cgroup_cpu_share"),
        ), patch(
            "acprof.host.static_metadata._cpu_frequency_policy_metadata",
            return_value=("performance", "on"),
        ):
            meta = static_metadata.collect_static_meta(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                batch_size=1,
                input_scale_type="seq_length",
                run_command="acprof run --model google-bert/bert-base-uncased",
                cgroup_version="v2",
                cgroup_collection_mode="strict_v2",
            )
            disabled_meta = static_metadata.collect_static_meta(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                batch_size=1,
                input_scale_type="seq_length",
                run_command=(
                    "acprof run --model google-bert/bert-base-uncased "
                    "--compute-profile-tool none"
                ),
                compute_profile_enabled=False,
            )
            previous_calls = cache_size.call_count
            mounted_meta = static_metadata.collect_static_meta(
                task_info=task_info, batch_size=1, input_scale_type="seq_length",
                image_info=runtime_images.ImageInfo(tag="mounted-fixture", runtime_environment={
                    "model_store": {"model_artifact_bytes": 2345},
                    "model_download": {"endpoint": "https://hf-mirror.com"},
                }),
            )
            assert (cache_size.call_count) == (previous_calls)

        assert (mounted_meta.model_storage_mode) == ("mounted")
        assert (mounted_meta.model_cache_bytes) == (2345)
        assert (mounted_meta.model_artifact_bytes) == (2345)
        assert (mounted_meta.runtime_image_bytes) == (456)
        assert (mounted_meta.total_deployment_bytes) == (2801)
        assert (meta.model_storage_mode) == ("baked")
        assert (meta.model_artifact_bytes) is None
        assert (meta.runtime_image_bytes) is None
        assert (meta.total_deployment_bytes) == (456)
        assert (meta.environment) == ("windows11+wsl")
        assert (meta.model_resolution) == (task_info.model_resolution)
        assert (meta.model_resolution) is not (task_info.model_resolution)
        assert (meta.run_command) == ("acprof run --model google-bert/bert-base-uncased")
        assert (meta.gpu_mem_total_bytes) == (987654321)
        assert (meta.host_mem_total_bytes) == (64_000_000_000)
        assert (meta.host_swap_total_bytes) == (2_000_000_000)
        assert (meta.host_swap_used_bytes_at_start) == (100_000_000)
        assert (meta.host_swap_type) == ("file")
        assert (meta.host_vm_swappiness) == (60)
        assert (meta.docker_storage_total_bytes) == (1_000_000)
        assert (meta.docker_storage_available_bytes_at_start) == (600_000)
        assert (meta.docker_storage_filesystem) == ("ext4")
        assert (meta.docker_storage_device) == ("/dev/nvme0n1p2")
        assert (meta.docker_storage_type) == ("nvme_ssd")
        assert (meta.cpu_power_source) == ("rapl")
        assert (meta.vcpu_power_method) == ("rapl_cgroup_cpu_share")
        assert (meta.cpu_governor) == ("performance")
        assert (meta.cpu_boost) == ("on")
        assert (meta.cgroup_version) == ("v2")
        assert (meta.cgroup_collection_mode) == ("strict_v2")
        assert (meta.parameter_count) == (110_106_428)
        assert (meta.parameter_bytes) == (440_425_712)
        assert (meta.model_cache_bytes) == (123)
        assert (meta.precision_dtype) == ("FP32")
        assert (meta.inference_precision_by_device) == ({"cpu": "FP32", "gpu": "FP16"})
        assert not (meta.quantized)
        assert (meta.model_license) == ("apache-2.0")
        assert (meta.input_format["json_schema"]["required"]) == (["text"])
        assert ("n_results") in (meta.output_format["json_schema"]["properties"])
        assert (disabled_meta.compute_profile_tools) == ([])
        assert not (disabled_meta.compute_profiles_retained)
        assert (disabled_meta.compute_profile_provenance) == ("disabled")
        assert (meta.execution_profile_schema_version) == (1)
        assert (meta.execution_profile_tools) == ([])
        assert not (meta.execution_profiles_retained)
        assert (meta.execution_profile_provenance) == ("disabled")

    def test_static_meta_compute_profile_fields_follow_host_metadata(self) -> None:
        assert (STATIC_META_SCHEMA_VERSION) == (7)
        assert ("parameter_bytes") in (STATIC_META_FIELDS)
        assert ("model_cache_bytes") in (STATIC_META_FIELDS)
        assert ("model_weight_bytes") not in (STATIC_META_FIELDS)
        assert ("gpu_mem_total_bytes") in (STATIC_META_FIELDS)
        assert ("host_mem_total_bytes") in (STATIC_META_FIELDS)
        assert ("host_swap_total_bytes") in (STATIC_META_FIELDS)
        assert ("host_swap_used_bytes_at_start") in (STATIC_META_FIELDS)
        assert ("host_swap_type") in (STATIC_META_FIELDS)
        assert ("host_vm_swappiness") in (STATIC_META_FIELDS)
        assert ("docker_storage_total_bytes") in (STATIC_META_FIELDS)
        assert ("cgroup_version") in (STATIC_META_FIELDS)
        assert ("cgroup_collection_mode") in (STATIC_META_FIELDS)
        assert (STATIC_META_FIELDS.index("gpu_mem_total_bytes")) < (STATIC_META_FIELDS.index("environment"))
        assert (STATIC_META_FIELDS.index("docker_storage_type")) < (STATIC_META_FIELDS.index("environment"))
        assert (STATIC_META_FIELDS.index("cgroup_collection_mode")) < (STATIC_META_FIELDS.index("cpu_power_source"))
        assert (STATIC_META_FIELDS.index("cpu_boost")) < (STATIC_META_FIELDS.index("compute_profile_tools"))
        assert ("compute_profile_schema_version") not in (STATIC_META_FIELDS)
        assert (STATIC_META_FIELDS.index("compute_profile_provenance")) < (STATIC_META_FIELDS.index("execution_profile_schema_version"))
        assert ("massif_sampling_strategy") in (STATIC_META_FIELDS)
        assert ("nsys_sampling_strategy") in (STATIC_META_FIELDS)
        assert (STATIC_META_FIELDS[-1]) == ("execution_profile_provenance")

    def test_enrich_static_meta_preserves_native_json_types(self) -> None:
        base = static_metadata.StaticMeta(
            model_name="model",
            model_revision="main",
            task_family="nlp",
            pipeline_tag="fill-mask",
            runtime_backend="transformers_pipeline",
            image_tag="image",
            batch_size=1,
            input_scale_type="seq_length",
            run_command="acprof run",
            model_download_url="https://example.invalid/model",
            gpu="GPU",
            gpu_mem_total_bytes=123,
            host_mem_total_bytes=1_024,
            host_swap_total_bytes=512,
            host_swap_used_bytes_at_start=64,
            host_swap_type="file",
            host_vm_swappiness=60,
            model_cache_bytes=456,
            docker_image_bytes=789,
            docker_storage_total_bytes=10_000,
            docker_storage_available_bytes_at_start=4_000,
            docker_storage_filesystem="ext4",
            docker_storage_device="/dev/nvme0n1p2",
            docker_storage_type="nvme_ssd",
            environment="ubuntu24.04",
            cgroup_version="v2",
            cgroup_collection_mode="strict_v2",
            cpu_power_source="rapl",
            vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance",
            cpu_boost="off",
        )

        enriched = static_metadata.enrich_static_meta(
            base,
            {
                "compute_profile_tools": ["torch_profiler_eager", "ncu"],
                "ncu_metrics": ["gpu__time_duration.sum", "metric.sum"],
                "compute_profiles_retained": True,
            },
        )

        assert (enriched.compute_profile_tools) == (["torch_profiler_eager", "ncu"])
        assert (enriched.ncu_metrics) == (["gpu__time_duration.sum", "metric.sum"])
        assert (enriched.compute_profiles_retained)
        assert (enriched.run_command) == (base.run_command)

        execution_enriched = static_metadata.enrich_static_meta(
            enriched,
            {
                "execution_profile_schema_version": 1,
                "execution_profile_tools": ["massif", "nsys"],
                "execution_profiles_retained": True,
                "execution_profile_provenance": "collected",
            },
        )
        assert (execution_enriched.execution_profile_schema_version) == (1)
        assert (execution_enriched.execution_profile_tools) == (["massif", "nsys"])
        assert (execution_enriched.execution_profiles_retained)
        assert (execution_enriched.execution_profile_provenance) == ("collected")

    def test_write_static_meta_json_includes_enriched_fields_atomically(self) -> None:
        meta = static_metadata.StaticMeta(
            model_name="model",
            model_revision="main",
            task_family="nlp",
            pipeline_tag="fill-mask",
            runtime_backend="transformers_pipeline",
            image_tag="image",
            batch_size=1,
            input_scale_type="seq_length",
            run_command="acprof run --model model",
            model_download_url="https://example.invalid/model",
            gpu="GPU",
            gpu_mem_total_bytes=123,
            host_mem_total_bytes=1_024,
            host_swap_total_bytes=512,
            host_swap_used_bytes_at_start=64,
            host_swap_type="file",
            host_vm_swappiness=60,
            model_cache_bytes=456,
            docker_image_bytes=789,
            docker_storage_total_bytes=10_000,
            docker_storage_available_bytes_at_start=4_000,
            docker_storage_filesystem="ext4",
            docker_storage_device="/dev/nvme0n1p2",
            docker_storage_type="nvme_ssd",
            environment="ubuntu24.04",
            cgroup_version="v2",
            cgroup_collection_mode="strict_v2",
            cpu_power_source="rapl",
            vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance",
            cpu_boost="off",
            parameter_count=42,
            parameter_bytes=168,
            precision_dtype="FP32",
            parameter_dtype_counts={"FP32": 42},
            input_format={"media_type": "application/json"},
            output_format={"media_type": "application/json"},
            quantized=False,
            model_license="mit",
            compute_profile_tools=["torch_profiler_eager", "ncu"],
            compute_profile_provenance="direct",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "static_meta.json")
            with patch("acprof.host.static_metadata.os.fsync", wraps=os.fsync) as fsync:
                static_metadata.write_static_meta_json(meta, path)
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            leftovers = [
                name
                for name in os.listdir(tmp)
                if name.startswith(".static_meta.json.")
            ]

        assert (list(payload)) == (STATIC_META_FIELDS)
        assert (payload["model_resolution"]) == ({})
        assert ("compute_profile_schema_version") not in (payload)
        assert (payload["parameter_count"]) == (42)
        assert (payload["parameter_bytes"]) == (168)
        assert (payload["model_cache_bytes"]) == (456)
        assert ("model_weight_bytes") not in (payload)
        assert (payload["host_mem_total_bytes"]) == (1_024)
        assert (payload["host_swap_total_bytes"]) == (512)
        assert (payload["host_swap_used_bytes_at_start"]) == (64)
        assert (payload["host_swap_type"]) == ("file")
        assert (payload["host_vm_swappiness"]) == (60)
        assert (payload["docker_storage_total_bytes"]) == (10_000)
        assert (payload["cgroup_version"]) == ("v2")
        assert (payload["cgroup_collection_mode"]) == ("strict_v2")
        assert (payload["docker_storage_available_bytes_at_start"]) == (4_000)
        assert (payload["docker_storage_type"]) == ("nvme_ssd")
        assert (payload["parameter_dtype_counts"]) == ({"FP32": 42})
        assert not (payload["quantized"])
        assert (payload["compute_profile_tools"]) == (["torch_profiler_eager", "ncu"])
        assert (payload["compute_profile_provenance"]) == ("direct")
        assert (leftovers) == ([])
        assert fsync.call_count >= 2

    def test_write_static_meta_json_rejects_nonfinite_values_atomically(self) -> None:
        meta = static_metadata.StaticMeta(
            model_name="model",
            model_revision="main",
            task_family="nlp",
            pipeline_tag="fill-mask",
            runtime_backend="transformers_pipeline",
            image_tag="image",
            batch_size=1,
            input_scale_type="seq_length",
            run_command="acprof run --model model",
            model_download_url="https://example.invalid/model",
            gpu="GPU",
            gpu_mem_total_bytes=None,
            model_cache_bytes=0,
            docker_image_bytes=0,
            environment="ubuntu24.04",
            cpu_power_source="unavailable",
            vcpu_power_method="unavailable",
            cpu_governor="unknown",
            cpu_boost="unknown",
            ncu_fma_flop_weight=float("nan"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "static_meta.json")
            previous = '{"previous":true}\n'
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(previous)

            with pytest.raises(
                ValueError,
                match="Out of range float values are not JSON compliant",
            ):
                static_metadata.write_static_meta_json(meta, path)

            with open(path, "r", encoding="utf-8") as stream:
                persisted = stream.read()

        assert persisted == previous


    def test_compute_plan_adds_static_flops_by_input_scale(self) -> None:
        meta = static_metadata.StaticMeta(
            model_name="model",
            model_revision="main",
            task_family="nlp",
            pipeline_tag="fill-mask",
            runtime_backend="transformers_pipeline",
            image_tag="image",
            batch_size=1,
            input_scale_type="seq_length",
            run_command="acprof run",
            model_download_url="https://example.invalid/model",
            gpu="GPU",
            gpu_mem_total_bytes=123,
            model_cache_bytes=456,
            docker_image_bytes=789,
            environment="ubuntu24.04",
            cpu_power_source="rapl",
            vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance",
            cpu_boost="off",
        )
        plan = {
            "static_metadata": {
                "compute_profile_tools": ["torch_profiler_eager"],
                "torch_profiler_eager_flop_semantics": (
                    "logical_operator_shape_flops"
                ),
            },
            "profiles": {
                "gpu": {
                    "torch_profiler_eager": {
                        "flop_semantics": "logical_operator_shape_flops",
                        "entries": [
                            {
                                "input_scale": 64,
                                "model_logical_mflop_per_request_torch_profiler_eager": 12.5,
                                "error": "",
                            },
                            {
                                "input_scale": 128,
                                "model_logical_mflop_per_request_torch_profiler_eager": 25.25,
                                "error": "",
                            },
                        ],
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "compute_profile_plan.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(plan, f)
            enriched = static_metadata.enrich_static_meta_from_compute_plan(
                meta,
                path,
            )

        assert (enriched.static_flops) == ({
                "source": "torch_profiler_eager",
                "profile": "gpu",
                "semantics": "logical_operator_shape_flops",
                "unit": "FLOP/request",
                "input_scale_type": "seq_length",
                "batch_size": 1,
                "values": [
                    {"input_scale": 64, "flops_per_request": 12_500_000},
                    {"input_scale": 128, "flops_per_request": 25_250_000},
                ],
            })
        assert (enriched.static_macs) is None

    def test_compute_plan_rejects_nonfinite_metadata(self) -> None:
        meta = static_metadata.StaticMeta(
            model_name="model", model_revision="main", task_family="nlp",
            pipeline_tag="fill-mask", runtime_backend="transformers_pipeline",
            image_tag="image", batch_size=1, input_scale_type="seq_length",
            run_command="acprof run", model_download_url="https://example.invalid/model",
            gpu="GPU", gpu_mem_total_bytes=123, model_cache_bytes=456,
            docker_image_bytes=789, environment="ubuntu24.04",
            cpu_power_source="rapl", vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance", cpu_boost="off",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "compute_profile_plan.json")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write('{"static_metadata":{"batch_size":NaN}}')
            with patch("sys.stdout", new_callable=StringIO) as output:
                enriched = static_metadata.enrich_static_meta_from_compute_plan(meta, path)

        assert enriched.batch_size == 1
        assert "invalid compute profile plan JSON" in output.getvalue()

    def test_execution_plan_rejects_oversized_metadata(self) -> None:
        meta = static_metadata.StaticMeta(
            model_name="model", model_revision="main", task_family="nlp",
            pipeline_tag="fill-mask", runtime_backend="transformers_pipeline",
            image_tag="image", batch_size=1, input_scale_type="seq_length",
            run_command="acprof run", model_download_url="https://example.invalid/model",
            gpu="GPU", gpu_mem_total_bytes=123, model_cache_bytes=456,
            docker_image_bytes=789, environment="ubuntu24.04",
            cpu_power_source="rapl", vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance", cpu_boost="off",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "execution_profile_plan.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({
                    "static_metadata": {"batch_size": 2},
                    "padding": "x" * (4 * 1024 * 1024),
                }, stream)
            with patch("sys.stdout", new_callable=StringIO) as output:
                enriched = static_metadata.enrich_static_meta_from_execution_plan(meta, path)

        assert enriched.batch_size == 1
        assert "exceeds the 4 MiB read limit" in output.getvalue()

    def test_cpu_frequency_policy_metadata_reads_governor_and_boost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpu0_cpufreq = os.path.join(tmp, "cpu0", "cpufreq")
            cpu1_cpufreq = os.path.join(tmp, "cpu1", "cpufreq")
            cpufreq = os.path.join(tmp, "cpufreq")
            os.makedirs(cpu0_cpufreq)
            os.makedirs(cpu1_cpufreq)
            os.makedirs(cpufreq)

            for path in (
                os.path.join(cpu0_cpufreq, "scaling_governor"),
                os.path.join(cpu1_cpufreq, "scaling_governor"),
            ):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("performance\n")
            with open(os.path.join(cpufreq, "boost"), "w", encoding="utf-8") as f:
                f.write("1\n")

            with patch("acprof.host.static_metadata.CPU_SYSFS_ROOT", tmp):
                assert (static_metadata._cpu_frequency_policy_metadata()) == (("performance", "on"))

    def test_manual_nlp_scales_write_effective_scale_plan(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        session = docker_runtime.RunningContainer(
            name="probe_google-bert--bert-base-uncased_1c_4g_off",
            base_url="http://127.0.0.1:8106",
            host_port=8106,
            cold_start_s=1.0,
        )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.input_plan._start_probe_session",
            return_value=session,
        ), patch(
            "acprof.host.input_plan.stop_container_session",
        ), patch(
            "acprof.host.input_plan._post_probe_payload",
            return_value={
                "effective_input_scale": 254.0,
                "truncated_by_limit": False,
                "reason": "within_model_limit",
                "payload": {"text": "hello [MASK]", "params": {}},
            },
        ):
            planned = input_plan.plan_input_scales(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
                input_scales="64",
            )

            assert (planned.scales) == ([254.0])
            assert (planned.source) == ("manual")
            assert (planned.plan_file) is not None
            assert planned.plan_file is not None
            assert (os.path.exists(planned.plan_file))
            with open(planned.plan_file, "r", encoding="utf-8") as f:
                plan = json.load(f)

        assert (plan["entries"][0]["input_scale"]) == (254.0)
        assert (plan["entries"][0]["payload"]) == ({"text": "hello [MASK]", "params": {}})

    @pytest.mark.parametrize('task_family', ('cv', 'timeseries'))
    def test_manual_non_nlp_scales_write_reusable_payload_plan(self, task_family) -> None:
        class FakeWorkloadGenerator:
            def generate(self, scale: float) -> dict:
                return {"value": float(scale)}

            def effective_input_scale(
                self,
                scale: float,
                payload: dict | None = None,
            ) -> float:
                return float(scale)

            def scale_label(self, scale: float) -> str:
                return f"scale{scale:g}"

            def max_input_scale(self) -> float:
                return 2048.0

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.workloads.get_generator",
            return_value=FakeWorkloadGenerator(),
        ), patch.object(input_plan, "_start_probe_session", return_value=SimpleNamespace(name="probe")), patch.object(
            input_plan, "stop_container_session"
        ), patch.object(input_plan, "_request_scale_meta", return_value={
            "max_effective_input_scale": 512, "input_scale_type": "context_length",
            "reason": "test model context limit",
        }
        ):
            task_info = TaskInfo(
                model_id=f"test/{task_family}",
                pipeline_tag="image-classification" if task_family == "cv" else "time-series-forecasting",
                task_family=task_family,
                runtime_backend="transformers_pipeline" if task_family == "cv" else "chronos",
                library_name="transformers",
                model_revision="main",
                detection_method="unit",
            )

            planned = input_plan.plan_input_scales(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
                input_scales="1,2",
            )

            assert (planned.scales) == ([1.0, 2.0])
            assert (planned.source) == ("manual")
            assert (planned.plan_file) == (os.path.join(tmp, "input_scale_plan.json"))
            assert planned.plan_file is not None
            with open(planned.plan_file, "r", encoding="utf-8") as f:
                plan = json.load(f)

            assert (plan["entries"]) == ([
                    {
                        "input_scale": 1.0,
                        "scale_label": "scale1",
                        "input_metadata": {},
                        "payload": {"value": 1.0},
                    },
                    {
                        "input_scale": 2.0,
                        "scale_label": "scale2",
                        "input_metadata": {},
                        "payload": {"value": 2.0},
                    },
                ])
            assert (plan["schema_version"]) == (2)
            assert re.search(r"^[0-9a-f]{64}$", planned.plan_sha256)

    def test_auto_audio_scales_use_workload_manifest_defaults(self) -> None:
        class FakeWorkloadGenerator:
            def default_input_scales(self) -> list[float]:
                return [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]

        task_info = TaskInfo(
            model_id="test/audio",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        expected = input_plan.PlannedInputScales(
            scales=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            source="workload_spec",
            plan_file="/tmp/input_scale_plan.json",
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.workloads.get_generator",
            return_value=FakeWorkloadGenerator(),
        ), patch.object(
            input_plan,
            "_plan_audio_scales",
            return_value=expected,
        ) as plan_audio:
            planned = input_plan.plan_input_scales(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
            )

        assert (planned) is (expected)
        assert (plan_audio.call_args.kwargs["scales"]) == ([1.0, 2.0, 5.0, 10.0, 20.0, 30.0])

    def test_audio_scale_meta_records_fixed_frontend_and_decoder_limit(self) -> None:
        response = {
            "input_scale_type": "duration_s",
            "required_sampling_rate": 16000,
            "max_short_form_duration_s": 30,
            "model_input_num_samples": 480000,
            "model_input_frames": 3000,
            "short_form_fixed_padding": True,
            "fixed_frontend_num_samples": 480000,
            "fixed_frontend_num_frames": 3000,
            "frontend_feature_bins": 128,
            "encoder_positions": 1500,
            "decoder_output_token_limit": 448,
            "model_type": "whisper",
            "reason": "fixed frontend; 448 is an output limit",
        }
        session = docker_runtime.RunningContainer(
            name="probe",
            base_url="http://127.0.0.1:1",
            host_port=1,
            cold_start_s=0.0,
        )
        with patch.object(
            input_plan,
            "_request_scale_meta",
            return_value=response,
        ):
            metadata = input_plan._request_audio_scale_meta(session, {})

        assert (metadata["short_form_fixed_padding"])
        assert (metadata["fixed_frontend_num_samples"]) == (480000)
        assert (metadata["fixed_frontend_num_frames"]) == (3000)
        assert (metadata["frontend_feature_bins"]) == (128)
        assert (metadata["decoder_output_token_limit"]) == (448)

    def test_audio_output_schema_describes_effective_scale_and_nullable_tokens(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        _, output_format = model_schema._model_io_formats(task_info)
        properties = output_format["json_schema"]["properties"]

        assert (properties["effective_input_scale"]) == ({"type": "number"})
        assert (properties["output_token_count"]) == ({"type": ["integer", "null"]})

    def test_workload_spec_is_rejected_for_unimplemented_families(self) -> None:
        task_info = TaskInfo(
            model_id="test/forecast",
            pipeline_tag="time-series-forecasting",
            task_family="timeseries",
            runtime_backend="chronos",
            library_name="chronos",
            model_revision="main",
            detection_method="unit",
        )
        with tempfile.TemporaryDirectory() as tmp, pytest.raises(ValueError, match="workload-spec is not declared for time-series-forecasting"):
            input_plan.plan_input_scales(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
                workload_spec_path="/tmp/not-used.json",
            )

    def test_audio_manifest_limit_is_checked_before_starting_probe(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            input_plan,
            "_start_probe_session",
        ) as start_probe, pytest.raises(ValueError, match="long-form workload"):
            input_plan._plan_audio_scales(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                scales=[31.0],
                batch_size=1,
                output_dir=tmp,
                source="manual",
                workload_spec_path=None,
            )
        start_probe.assert_not_called()

    def test_materialized_plan_v2_records_workload_and_input_metadata(self) -> None:
        class FakeWorkloadGenerator:
            def generate(self, scale: float) -> dict:
                return {"value": float(scale), "params": {"mode": "fixed"}}

            def effective_input_scale(self, scale: float, payload=None) -> float:
                return float(scale)

            def scale_label(self, scale: float) -> str:
                return f"scale{scale:g}"

            def input_metadata(self, scale: float, payload=None) -> dict:
                return {"input_num_samples": int(scale * 100)}

            def plan_metadata(self) -> dict:
                return {"workload_id": "fixture-v1"}

        task_info = TaskInfo(
            model_id="test/audio",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        constraints = {"max_short_form_duration_s": 30}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.workloads.get_generator",
            return_value=FakeWorkloadGenerator(),
        ):
            planned = input_plan._materialize_scale_plan(
                task_info=task_info,
                scales=[1.0, 2.0],
                batch_size=1,
                output_dir=tmp,
                source="unit",
                model_constraints=constraints,
            )
            assert planned.plan_file is not None
            with open(planned.plan_file, "r", encoding="utf-8") as plan_file:
                plan = json.load(plan_file)

        assert (plan["schema_version"]) == (2)
        assert (plan["workload"]) == ({"workload_id": "fixture-v1"})
        assert (plan["model_constraints"]) == (constraints)
        assert (plan["entries"][0]["input_metadata"]["input_num_samples"]) == (100)
        assert (planned.workload["model_constraints"]) == (constraints)
        assert re.search(r"^[0-9a-f]{64}$", planned.plan_sha256)

    def test_run_single_case_passes_container_name_to_client(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        captured_env = {}

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.dict(
            "acprof.host.orchestrator.os.environ",
            {"ACPROF_WECOM_WEBHOOK_URL": "https://example.invalid/secret"},
        ), patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
                cold_start_started_at="2026-08-23T10:00:00.000+08:00",
                cold_start_ready_at="2026-08-23T10:00:01.000+08:00",
                cold_start_container_launch_s=0.1,
                cold_start_server_setup_s=0.2,
                cold_start_cuda_init_s=0.0,
                cold_start_model_load_s=0.6,
                cold_start_ready_wait_s=0.1,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=self.output_dir,
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=1,
                input_scales="64",
                require_packet_latency=False,
            )

        assert (captured_env["CONTAINER_NAME"]) == ("case_google-bert--bert-base-uncased_1c_4g_off")
        assert (captured_env["USE_MIPS"]) == ("1")
        assert (captured_env["COLD_START_STARTED_AT"]) == ("2026-08-23T10:00:00.000+08:00")
        assert (captured_env["COLD_START_CONTAINER_LAUNCH_S"]) == ("0.1")
        assert (captured_env["COLD_START_MODEL_LOAD_S"]) == ("0.6")
        assert (captured_env["COLD_START_READY_WAIT_S"]) == ("0.1")
        assert ("ACPROF_WECOM_WEBHOOK_URL") not in (captured_env)

    def test_run_single_case_passes_compute_profile_plan_to_client(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        captured_env = {}

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=self.output_dir,
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=1,
                input_scales="64",
                compute_profile_plan_file="results/test-unit/compute_profile_plan.json",
                require_packet_latency=False,
            )

        assert (captured_env["COMPUTE_PROFILE_PLAN_FILE"]) == ("results/test-unit/compute_profile_plan.json")

    def test_run_single_case_passes_execution_profile_plan_to_client(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        captured_env = {}

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch(
            "acprof.host.orchestrator._resolve_packet_latency_runtime",
            return_value=None,
        ), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=self.output_dir,
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=1,
                input_scales="64",
                execution_profile_plan_file=(
                    "results/test-unit/execution_profile_plan.json"
                ),
                require_packet_latency=False,
            )

        assert (captured_env["EXECUTION_PROFILE_PLAN_FILE"]) == ("results/test-unit/execution_profile_plan.json")

    def test_run_single_case_passes_idle_debug_settings_to_client(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        captured_env = {}

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=self.output_dir,
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=1,
                input_scales="64",
                idle_cooldown_seconds=4.5,
                idle_debug=True,
                require_packet_latency=False,
            )

        assert (captured_env["IDLE_DEBUG"]) == ("1")
        assert (captured_env["IDLE_COOLDOWN_SECONDS"]) == ("4.5")
        assert (captured_env["IDLE_DIAG_PATH"]) == (os.path.join(
                os.path.dirname(captured_env["OUT_CSV"]),
                "debug_idle_diag",
                os.path.basename(captured_env["OUT_CSV"]) + ".idle_diag.jsonl",
            ))

    def test_run_single_case_passes_auto_repeat_window_settings_to_client(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        captured_env = {}

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_gpu_case_csv(captured_env["OUT_CSV"], [10.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ) as start_container, patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="on",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=self.output_dir,
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=0,
                repeat_window_seconds=10.0,
                request_timeout_seconds=123.5,
                input_scales="64",
                require_packet_latency=False,
            )

        assert (captured_env["REPEAT_IN_WINDOW"]) == ("0")
        assert (captured_env["REPEAT_WINDOW_SECONDS"]) == ("10.0")
        assert (captured_env["REQUEST_TIMEOUT_SECONDS"]) == ("123.5")
        assert (start_container.call_args.kwargs["request_timeout_seconds"]) == (123.5)

    def test_run_single_case_preserves_manual_repeat_window_to_client(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        captured_env = {}

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_gpu_case_csv(captured_env["OUT_CSV"], [10.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="on",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=self.output_dir,
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=1000,
                repeat_window_seconds=10.0,
                input_scales="64",
                require_packet_latency=False,
            )

        assert (captured_env["REPEAT_IN_WINDOW"]) == ("1000")
        assert (captured_env["REPEAT_WINDOW_SECONDS"]) == ("10.0")

    @pytest.mark.parametrize('recorded_failure_case', range(2), ids=['None', "Failure('predict', 'inference_failed', 'typed inference error')"])
    def test_run_single_case_aborts_when_client_exits_nonzero(self, recorded_failure_case) -> None:
        from acprof.artifact_layout import case_sidecar
        from acprof.failures import Failure, RuntimeFailure
        recorded_failure = None
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                if recorded_failure is not None:
                    case_sidecar(kwargs["env"]["OUT_CSV"], "runtime_failures").write_text(
                        json.dumps({"failures": [recorded_failure.to_dict()]}))
                return SimpleNamespace(returncode=7, stdout="", stderr="idle unstable")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch(
            "acprof.host.orchestrator.container_runtime_oom_error",
            return_value=None,
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            recorded_failure = tuple((None, Failure('predict', 'inference_failed', 'typed inference error')))[recorded_failure_case]
            with pytest.raises(
                RuntimeFailure if recorded_failure else orchestrator.EnergyProfilingError
            ) as raised:
                orchestrator.run_single_case(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu="on",
                    image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                    output_dir=self.output_dir,
                    project_dir=".",
                    warmup=0,
                    repeat=1,
                    repeat_in_window=0,
                    input_scales="64",
                    require_packet_latency=False,
                )
            if recorded_failure:
                assert (raised.value.failure) == (recorded_failure)
            else:
                assert ("client.py exited with code 7") in (str(raised.value))

    def test_run_single_case_records_runtime_oom_before_mips_failure(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                out_csv = kwargs["env"]["OUT_CSV"]
                successful_row = {field: "nan" for field in CSV_FIELDS}
                successful_row.update({
                    "cpu_cores": "2",
                    "mem_cap_gb": "2",
                    "gpu_mode": "on",
                    "input_scale": "64",
                    "repeat_idx": "0",
                    "warmup": "0",
                    "repeat_in_window": "1",
                    "latency_app_s": "1.25",
                    "cpu_idle_power_w": "5.0",
                    "gpu_idle_power_w": "10.0",
                    "status": "ok",
                    "error": "",
                })
                failed_row = {field: "nan" for field in CSV_FIELDS}
                failed_row.update({
                    "cpu_cores": "2",
                    "mem_cap_gb": "2",
                    "gpu_mode": "on",
                    "input_scale": "128",
                    "repeat_idx": "0",
                    "warmup": "0",
                    "repeat_in_window": "1",
                    "status": "error",
                    "error": "RemoteDisconnected during inference",
                })
                with open(out_csv, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    writer.writerows([successful_row, failed_row])
                return SimpleNamespace(
                    returncode=orchestrator.MIPS_EXIT_CODE,
                    stdout="",
                    stderr="container is not running",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        container_state = {
            "Status": "exited",
            "Running": False,
            "Restarting": False,
            "OOMKilled": True,
            "ExitCode": 137,
            "Error": "",
        }
        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_2c_2g_on",
                base_url="http://127.0.0.1:8204",
                host_port=8204,
                cold_start_s=1.0,
            ),
        ), patch(
            "acprof.host.orchestrator._resolve_packet_latency_runtime",
            return_value=None,
        ), patch(
            "acprof.host.container_state.inspect_container_state",
            return_value=container_state,
        ) as inspect_state, patch(
            "acprof.host.orchestrator.stop_container_session"
        ) as stop_container, patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=2,
                mem=2,
                gpu="on",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=1,
                input_scales="64,128,256",
                require_packet_latency=False,
            )

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        assert (len(rows)) == (3)
        assert (rows[0]["status"]) == ("ok")
        assert (rows[0]["error"]) == ("")
        assert (rows[1]["status"]) == ("error")
        assert ("RemoteDisconnected") in (rows[1]["error"])
        assert ("container_runtime_oom") in (rows[1]["error"])
        assert (rows[2]["status"]) == ("error")
        assert ("container_runtime_oom") in (rows[2]["error"])
        assert ("docker_oom_killed=true") in (rows[2]["error"])
        assert ("container_exit_code=137") in (rows[2]["error"])
        inspect_state.assert_called_once_with(
            "case_google-bert--bert-base-uncased_2c_2g_on"
        )
        stop_container.assert_called_once()

    def test_run_single_case_still_aborts_non_oom_mips_failure(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                return SimpleNamespace(
                    returncode=orchestrator.MIPS_EXIT_CODE,
                    stdout="",
                    stderr="perf permission denied",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch(
            "acprof.host.orchestrator._resolve_packet_latency_runtime",
            return_value=None,
        ), patch(
            "acprof.host.orchestrator.container_runtime_oom_error",
            return_value=None,
        ), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            with pytest.raises(orchestrator.MIPSProfilingError):
                orchestrator.run_single_case(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu="off",
                    image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                    output_dir=self.output_dir,
                    project_dir=".",
                    warmup=0,
                    repeat=1,
                    repeat_in_window=1,
                    input_scales="64",
                    require_packet_latency=False,
                )

    def test_client_error_context_rejects_nonfinite_json(self, tmp_path, capsys) -> None:
        path = tmp_path / "client-error.json"
        path.write_text('{"input_scale": NaN}', encoding="utf-8")

        assert orchestrator._load_client_error_context(str(path)) == {}
        assert "could not read structured client error context" in capsys.readouterr().err

    def test_client_error_context_rejects_oversized_json(self, tmp_path, capsys) -> None:
        path = tmp_path / "client-error.json"
        path.write_text(json.dumps({"padding": "x" * (4 * 1024 * 1024)}), encoding="utf-8")

        assert orchestrator._load_client_error_context(str(path)) == {}
        assert "4 MiB read limit" in capsys.readouterr().err

    def test_run_single_case_records_request_timeout_and_returns(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                with open(
                    kwargs["env"]["CLIENT_ERROR_PATH"],
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "error_type": "client_request_timeout",
                            "input_scale": 64.0,
                            "request_timeout_s": 120.0,
                            "request_phase": "auto_repeat_window_warmup",
                            "request_id": "case_scale64_auto_warmup0",
                        },
                        f,
                    )
                return SimpleNamespace(
                    returncode=orchestrator.CLIENT_REQUEST_TIMEOUT_EXIT_CODE,
                    stdout="",
                    stderr="slow inference",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch(
            "acprof.host.orchestrator._resolve_packet_latency_runtime",
            return_value=None,
        ), patch(
            "acprof.host.orchestrator.stop_container_session"
        ) as stop_container, patch(
            "acprof.host.orchestrator.container_runtime_oom_error",
            return_value=None,
        ), patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=1,
                repeat=2,
                repeat_in_window=0,
                request_timeout_seconds=120.0,
                input_scales="64,128",
                require_packet_latency=False,
            )

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        assert (len(rows)) == (6)
        assert ({row["status"] for row in rows}) == ({"error"})
        assert (all("client_request_timeout" in row["error"] for row in rows))
        trigger_rows = [row for row in rows if float(row["input_scale"]) == 64.0]
        skipped_rows = [row for row in rows if float(row["input_scale"]) == 128.0]
        assert (all("reason=triggering_scale_probe_timed_out" in row["error"] for row in trigger_rows))
        assert (all("triggering_request_latency_s>120" in row["error"] for row in trigger_rows))
        assert (all("reason=skipped_after_prior_scale_timeout" in row["error"] for row in skipped_rows))
        assert (all("planned_request_attempted=false" in row["error"] for row in skipped_rows))
        assert (sorted({float(row["input_scale"]) for row in rows})) == ([64.0, 128.0])
        stop_container.assert_called_once()

    def test_run_single_case_preserves_completed_rows_before_request_timeout(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                out_csv = kwargs["env"]["OUT_CSV"]
                row = {field: "nan" for field in CSV_FIELDS}
                row.update({
                    "cpu_cores": "1",
                    "mem_cap_gb": "4",
                    "gpu_mode": "off",
                    "input_scale": "64",
                    "repeat_idx": "0",
                    "warmup": "0",
                    "repeat_in_window": "1",
                    "latency_app_s": "12.5",
                    "cpu_idle_power_w": "5.0",
                    "status": "ok",
                    "error": "",
                })
                with open(out_csv, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    writer.writerow(row)
                with open(
                    kwargs["env"]["CLIENT_ERROR_PATH"],
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "error_type": "client_request_timeout",
                            "input_scale": 128.0,
                            "request_timeout_s": 300.0,
                            "request_phase": "measurement_repeat",
                            "measurement_repeat_idx": 0,
                            "request_id": "case_scale128_r0:0",
                        },
                        f,
                    )
                return SimpleNamespace(
                    returncode=orchestrator.CLIENT_REQUEST_TIMEOUT_EXIT_CODE,
                    stdout="",
                    stderr="slow inference",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch(
            "acprof.host.orchestrator._resolve_packet_latency_runtime",
            return_value=None,
        ), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch(
            "acprof.host.orchestrator.container_runtime_oom_error",
            return_value=None,
        ), patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=1,
                input_scales="64,128",
                require_packet_latency=False,
            )

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        assert (len(rows)) == (2)
        assert (rows[0]["input_scale"]) == ("64")
        assert (rows[0]["latency_app_s"]) == ("12.5")
        assert (rows[0]["status"]) == ("ok")
        assert (rows[0]["error"]) == ("")
        assert (rows[1]["input_scale"]) == ("128")
        assert (rows[1]["status"]) == ("error")
        assert ("client_request_timeout") in (rows[1]["error"])
        assert ("planned_request_attempted=true") in (rows[1]["error"])
        assert ("measurement_row_completed=false") in (rows[1]["error"])
        assert ("triggering_request_latency_s>300") in (rows[1]["error"])

    def test_run_single_case_writes_error_rows_when_container_start_fails(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator.start_container_session",
            side_effect=RuntimeError(
                "container_oom_killed during startup "
                "(container=unit-test, memory_limit=2g, status=exited, exit_code=137)"
            ),
        ):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=2,
                gpu="on",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=1,
                repeat=2,
                input_scales="85,170",
            )

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        assert (len(rows)) == (6)
        assert ({row["status"] for row in rows}) == ({"error"})
        assert (all(row["cpu_cores"] == "1" for row in rows))
        assert (all(row["mem_cap_gb"] == "2" for row in rows))
        assert (all(row["gpu_mode"] == "on" for row in rows))
        assert (sorted({float(row["input_scale"]) for row in rows})) == ([85.0, 170.0])
        assert (sorted((row["warmup"], row["repeat_idx"]) for row in rows)) == ([
                ("0", "0"),
                ("0", "0"),
                ("0", "1"),
                ("0", "1"),
                ("1", "0"),
                ("1", "0"),
            ])
        assert (all(
                "container_start_failed: container_oom_killed during startup"
                in row["error"]
                for row in rows
            ))

    def test_run_single_case_accepts_stable_idle_power_case_csv(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                _write_gpu_case_csv(kwargs["env"]["OUT_CSV"], [10.0, 10.2, 10.1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="on",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=0,
                repeat=3,
                repeat_in_window=1,
                input_scales="64",
                require_packet_latency=False,
            )

        assert (csv_path.endswith(".csv"))

    def test_run_single_case_warns_when_gpu_idle_power_csv_is_unstable(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                _write_gpu_case_csv(kwargs["env"]["OUT_CSV"], [10.0, 10.7, 10.2])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        stdout = StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run), redirect_stderr(stdout):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="on",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=0,
                repeat=3,
                repeat_in_window=1,
                input_scales="64",
                require_packet_latency=False,
            )

        message = stdout.getvalue()
        assert (csv_path.endswith(".csv"))
        assert ("[energy][WARN]") in (message)
        assert ("gpu_idle_power_w") in (message)
        assert ("6.8%") in (message)
        assert ("5.0%") in (message)
        assert ("--idle-seconds") in (message)
        assert ("GPU processes") in (message)

    def test_run_single_case_warns_when_cpu_idle_power_csv_is_unstable(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                _write_cpu_case_csv(kwargs["env"]["OUT_CSV"], [5.0, 5.4, 5.1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        stdout = StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_container_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run), redirect_stderr(stdout):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=0,
                repeat=3,
                repeat_in_window=1,
                input_scales="64",
                require_packet_latency=False,
            )

        message = stdout.getvalue()
        assert (csv_path.endswith(".csv"))
        assert ("[energy][WARN]") in (message)
        assert ("cpu_idle_power_w") in (message)
        assert ("7.7%") in (message)
        assert ("5.0%") in (message)
        assert ("--idle-seconds") in (message)
        assert ("host background processes") in (message)

    def test_resolve_packet_latency_runtime_requires_local_linux_tools(self) -> None:
        with patch("acprof.host.packet_capture.shutil.which", return_value=None):
            runtime = packet_capture._resolve_packet_latency_runtime(
                project_dir="/repo",
                pcap_file="/repo/results/sniff_case.pcap",
                sniff_iface="docker0",
            )

        assert (runtime) is None

    def test_resolve_packet_latency_runtime_uses_tcpdump_without_sudo_when_capable(self) -> None:
        def fake_which(name: str) -> str | None:
            return {
                "tcpdump": "/usr/bin/tcpdump",
                "tshark": "/usr/bin/tshark",
            }.get(name)

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd[:2] == ["getcap", "/usr/bin/tcpdump"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout="/usr/bin/tcpdump cap_net_admin,cap_net_raw=eip\n",
                    stderr="",
                )
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        with patch("acprof.host.packet_capture.shutil.which", side_effect=fake_which), patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            runtime = packet_capture._resolve_packet_latency_runtime(
                project_dir="/repo",
                pcap_file="/repo/results/sniff_case.pcap",
                sniff_iface="docker0",
            )

        assert (runtime) is not None
        assert runtime is not None
        assert (runtime.mode) == ("local")
        assert (runtime.tcpdump_cmd[0]) == ("/usr/bin/tcpdump")
        assert ("sudo") not in (runtime.tcpdump_cmd)

    def test_resolve_packet_latency_runtime_requires_administrator_setup(self):
        with patch('acprof.host.packet_capture.shutil.which', side_effect=lambda name: '/usr/bin/' + name), patch(
            'acprof.host.packet_capture.os.geteuid', return_value=1000,
        ), patch('acprof.host.command.run_command', return_value=SimpleNamespace(
            returncode=0, stdout='', stderr='',
        )) as run:
            with pytest.raises(packet_capture.PacketLatencyError, match='capture capability'):
                packet_capture._resolve_packet_latency_runtime('/repo', '/tmp/test.pcap', 'docker0')
        assert (run.call_args_list[0].args[0]) == (['getcap', '/usr/bin/tcpdump'])
        assert (run.call_count) == (1)

    def test_run_single_case_fails_when_packet_latency_runtime_unavailable(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        with patch(
            "acprof.host.orchestrator.start_container_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_container_session"
        ):
            with pytest.raises(packet_capture.PacketLatencyError) as raised:
                orchestrator.run_single_case(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu="off",
                    image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                    output_dir=self.output_dir,
                    project_dir=".",
                    warmup=0,
                    repeat=1,
                    repeat_in_window=1,
                    input_scales="64",
                )

        message = str(raised.value)
        assert ("packet latency is required") in (message)
        assert ("tcpdump") in (message)
        assert ("tshark") in (message)
        assert ("sudo setcap") in (message)

    def test_finalize_case_preserves_packet_latency_sidecar_on_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pcap_file = os.path.join(tmp, "case.pcap")
            lat_json = os.path.join(tmp, "packet_latency.json")
            out_csv = os.path.join(tmp, "result.csv")
            with open(pcap_file, "wb") as stream:
                stream.write(b"pcap")
            with open(lat_json, "wb") as stream:
                stream.write(b'{"previous": true}\n')
            with open(lat_json, "rb") as stream:
                original = stream.read()
            payload = {
                "schema_version": 2,
                "requests": {"window:0": {"latency_s": 0.25}},
            }
            process = SimpleNamespace(
                terminate=lambda: None,
                wait=lambda timeout=None: 0,
            )
            runtime = packet_capture.PacketLatencyRuntime(
                mode="local",
                tcpdump_cmd=["tcpdump"],
                parse_cmd=["parse-pcap"],
            )

            def fail_after_partial_write(value, stream, **kwargs):
                del value, kwargs
                stream.write('{"partial":')
                raise OSError("disk full")

            with patch("acprof.host.orchestrator.time.sleep"), patch(
                "acprof.host.orchestrator.host_command.run_command",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(payload),
                    stderr="",
                ),
            ), patch(
                "acprof.host.orchestrator.json.dump",
                side_effect=fail_after_partial_write,
            ):
                with pytest.raises(OSError, match="disk full"):
                    orchestrator._finalize_case(
                        process,
                        runtime,
                        False,
                        0,
                        "",
                        pcap_file,
                        lat_json,
                        out_csv,
                        True,
                        "basic",
                        "off",
                    )

            with open(lat_json, "rb") as stream:
                assert stream.read() == original

    def test_assert_packet_latency_csv_complete_rejects_nan_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_case.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write("latency_s,status\nnan,ok\n")

            with pytest.raises(packet_capture.PacketLatencyError) as raised:
                orchestrator._assert_packet_latency_csv_complete(csv_path)

        assert ("latency_s is missing") in (str(raised.value))

    def test_assert_packet_latency_csv_complete_ignores_timeout_error_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_case.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write("latency_s,status\n0.25,ok\nnan,error\n")

            orchestrator._assert_packet_latency_csv_complete(
                csv_path,
                ignore_error_rows=True,
            )
