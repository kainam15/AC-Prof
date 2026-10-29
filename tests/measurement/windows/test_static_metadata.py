import json
import os
import tempfile
from functools import partial
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.config import STATIC_META_FIELDS, STATIC_META_SCHEMA_VERSION
from acprof.host import machine_metadata, platform_metadata, static_metadata
from acprof.host.detect import TaskInfo
from acprof.host.images import runtime as runtime_images


class TestStaticMetadata:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from tests.support.platform_fixtures import native_policy
        native_policy(self._request)
        conditions = patch("acprof.host.orchestrator.record_case_conditions")
        conditions.start()
        self._request.addfinalizer(partial(conditions.stop))
        self.output_dir = str(tmp_path)

    def test_host_mem_total_bytes_uses_physical_page_count(self) -> None:
        values = {
            "SC_PAGE_SIZE": 4096,
            "SC_PHYS_PAGES": 8_000_000,
        }

        with patch(
            "acprof.host.machine_metadata.os.sysconf",
            side_effect=lambda name: values[name],
        ):
            total = machine_metadata.host_mem_total_bytes()

        assert (total) == (32_768_000_000)

    def test_host_swap_metadata_reads_capacity_usage_type_and_swappiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meminfo_path = os.path.join(tmp, "meminfo")
            swaps_path = os.path.join(tmp, "swaps")
            swappiness_path = os.path.join(tmp, "swappiness")
            with open(meminfo_path, "w", encoding="utf-8") as f:
                f.write("SwapTotal:       2097148 kB\nSwapFree:        2096124 kB\n")
            with open(swaps_path, "w", encoding="utf-8") as f:
                f.write("Filename Type Size Used Priority\n/swapfile file 2097148 1024 -2\n")
            with open(swappiness_path, "w", encoding="utf-8") as f:
                f.write("60\n")

            metadata = machine_metadata.host_swap_metadata(
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
                f.write("SwapTotal:             0 kB\nSwapFree:             0 kB\n")
            with open(swaps_path, "w", encoding="utf-8") as f:
                f.write("Filename Type Size Used Priority\n")
            with open(swappiness_path, "w", encoding="utf-8") as f:
                f.write("0\n")

            metadata = machine_metadata.host_swap_metadata(
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
            "acprof.host.machine_metadata.docker_root_dir",
            return_value="/var/lib/docker",
        ), patch(
            "acprof.host.machine_metadata.shutil.disk_usage",
            return_value=SimpleNamespace(total=1_000, used=400, free=600),
        ) as disk_usage, patch(
            "acprof.host.machine_metadata.docker_mount_metadata",
            return_value=("/dev/nvme0n1p2", "ext4"),
        ), patch(
            "acprof.host.machine_metadata.block_device_storage_type",
            return_value="nvme_ssd",
        ):
            metadata = machine_metadata.docker_storage_metadata()

        disk_usage.assert_called_once_with("/var/lib/docker")
        assert (metadata) == ({
                "docker_storage_total_bytes": 1_000,
                "docker_storage_available_bytes_at_start": 600,
                "docker_storage_filesystem": "ext4",
                "docker_storage_device": "/dev/nvme0n1p2",
                "docker_storage_type": "nvme_ssd",
            })

    @pytest.mark.parametrize(("device_metadata", "expected"), [
        ({"tran": "nvme", "rota": False}, "nvme_ssd"),
        ({"tran": "sata", "rota": False}, "ssd"),
        ({"tran": "sata", "rota": True}, "hdd"),
        ({"tran": None, "rota": None}, "unknown"),
    ], ids=["nvme", "sata-ssd", "sata-hdd", "unknown"])
    def test_block_device_type_uses_transport_and_rotational_flag(self, device_metadata, expected) -> None:
        with patch(
            "acprof.host.machine_metadata.shutil.which",
            return_value="/usr/bin/lsblk",
        ):
            with patch(
                "acprof.system.command.run_command",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"blockdevices": [device_metadata]}),
                    stderr="",
                ),
            ):
                assert (machine_metadata.block_device_storage_type("/dev/test")) == (expected)

    def test_docker_storage_metadata_is_unknown_when_daemon_root_is_unavailable(self) -> None:
        with patch(
            "acprof.host.machine_metadata.docker_root_dir",
            return_value=None,
        ), patch(
            "acprof.host.machine_metadata.shutil.disk_usage",
        ) as disk_usage:
            metadata = machine_metadata.docker_storage_metadata()

        disk_usage.assert_not_called()
        assert (metadata["docker_storage_total_bytes"]) is None
        assert (metadata["docker_storage_available_bytes_at_start"]) is None
        assert (metadata["docker_storage_type"]) == ("unknown")

    def test_select_nlp_torch_index_url_uses_cu124_for_cuda_12_4_driver(self) -> None:
        with patch("acprof.host.images.runtime.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "acprof.system.command.run_command",
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

    def test_windows_process_is_not_relabelled_from_docker_wsl_kernel(self) -> None:
        with patch("acprof.host.platform_metadata.platform.system", return_value="Windows"), patch(
            "acprof.host.platform_metadata.platform.release", return_value="11"
        ), patch.dict("acprof.platform.os.environ", {}, clear=True):
            assert (platform_metadata.operating_system_label()) == ("windows11")

    def test_detect_environment_linux_ubuntu_without_wsl(self) -> None:
        with patch("acprof.host.platform_metadata.platform.system", return_value="Linux"), patch(
            "acprof.host.platform_metadata.platform.freedesktop_os_release",
            return_value={"ID": "ubuntu", "VERSION_ID": "24.04"},
        ), patch.dict("acprof.platform.os.environ", {}, clear=True):
            assert (platform_metadata.operating_system_label()) == ("ubuntu24.04")

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

        with patch("acprof.host.static_metadata.operating_system_label", return_value="windows11+wsl"), patch(
            "acprof.host.static_metadata.probe_gpu_metadata", return_value={
                "schema_version": 1, "device": {}, "errors": [],
                "fields": {"name": {"value": "Test GPU", "status": "available", "source": "fixture"},
                           "memory_total_bytes": {"value": 987654321, "status": "available", "source": "fixture"}},
            }
        ), patch(
            "acprof.host.machine_metadata.host_mem_total_bytes", return_value=64_000_000_000
        ), patch(
            "acprof.host.machine_metadata.host_swap_metadata",
            return_value={
                "host_swap_total_bytes": 2_000_000_000,
                "host_swap_used_bytes_at_start": 100_000_000,
                "host_swap_type": "file",
                "host_vm_swappiness": 60,
            },
        ), patch(
            "acprof.host.machine_metadata.docker_model_cache_bytes", return_value=123
        ) as cache_size, patch("acprof.host.machine_metadata.docker_image_size_bytes", return_value=456), patch(
            "acprof.host.machine_metadata.docker_storage_metadata",
            return_value={
                "docker_storage_total_bytes": 1_000_000,
                "docker_storage_available_bytes_at_start": 600_000,
                "docker_storage_filesystem": "ext4",
                "docker_storage_device": "/dev/nvme0n1p2",
                "docker_storage_type": "nvme_ssd",
            },
        ), patch(
            "acprof.host.machine_metadata.cpu_power_metadata",
            return_value=("rapl", "rapl_cgroup_cpu_share"),
        ), patch(
            "acprof.host.machine_metadata.cpu_frequency_policy_metadata",
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
            with open(path, encoding="utf-8") as f:
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

            with open(path, encoding="utf-8") as stream:
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

            with patch("acprof.host.machine_metadata.CPU_SYSFS_ROOT", tmp):
                assert (machine_metadata.cpu_frequency_policy_metadata()) == (("performance", "on"))
