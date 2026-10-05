import csv
import json
import os
import subprocess
import tempfile
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import acprof.host.profilers.compute_parsers as host_profilers_compute_parsers
import acprof.host.profilers.tool_discovery as host_profilers_tool_discovery
from acprof.container.handlers import transformers_pipeline_load_kwargs
from acprof.host import compute_profile, profiler_support
from acprof.host.detect import TaskInfo
from acprof.host.profilers import advisor, compute_parsers, ncu, tool_discovery, torch


def _write_input_scale_plan(directory: str, input_scale: float = 8.0) -> str:
    path = os.path.join(directory, "input_scale_plan.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"schema_version": 2,
                "model_id": "google-bert/bert-base-uncased",
                "task_family": "nlp",
                "pipeline_tag": "fill-mask",
                "entries": [
                    {
                        "input_scale": input_scale,
                        "scale_label": f"seq{input_scale:g}",
                        "payload": {
                            "text": "hello [MASK]",
                            "params": {},
                        },
                    }
                ],
            },
            f,
        )
    return path


def _ncu_resume_csv_text() -> str:
    return "\n".join([
        (
            '"ID","Kernel Name","smsp__sass_thread_inst_executed_op_'
            'ffma_pred_on.sum","sm__ops_path_tensor_src_fp16_dst_fp32.sum",'
            '"gpu__time_duration.sum"'
        ),
        '"","","inst","FLOP","usec"',
        '"0","kernel_a","10","100","1000"',
        '"1","kernel_b","20","300","2000"',
    ])


class TestComputeProfile:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        selection = patch('acprof.host.gpu_device.resolve_gpu_device', return_value={'uuid': 'GPU-fixture'})
        selection.start()
        self._request.addfinalizer(partial(selection.stop))

    def test_transformers_handler_forces_eager_only_when_requested(self) -> None:
        assert (transformers_pipeline_load_kwargs(None)) == ({})
        assert (transformers_pipeline_load_kwargs({
                "attention_implementation": "eager",
            })) == ({"model_kwargs": {"attn_implementation": "eager"}})
        with pytest.raises(ValueError, match="must be 'eager'"):
            transformers_pipeline_load_kwargs({
                "attention_implementation": "sdpa",
            })

    def test_input_scale_plan_is_required(self) -> None:
        with pytest.raises(ValueError, match="input_scale_plan_file is required"):
            compute_profile.load_input_scale_plan_entries("")

        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "input_scale_plan.json")
            with pytest.raises(FileNotFoundError, match="input scale plan not found"):
                compute_profile.load_input_scale_plan_entries(missing)

    @pytest.mark.parametrize('schema_version', (2,))
    def test_current_input_plan_reuses_the_exact_payload(self, schema_version) -> None:
        payload = {
            "audio_base64": "UklGRg==",
            "audio_format": "wav",
            "sample_rate": 16000,
            "params": {"asr_task": "transcribe"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "input_scale_plan.json")
            plan = {"schema_version": 2,
                "entries": [
                    {
                        "input_scale": 1.0,
                        "scale_label": "dur1s",
                        "payload": payload,
                    }
                ]
            }
            if schema_version == 2:
                plan.update({
                    "schema_version": 2,
                    "workload": {"workload_id": "fixture"},
                    "model_constraints": {"max_short_form_duration_s": 30},
                })
                plan["entries"][0]["input_metadata"] = {
                    "input_num_samples": 16000
                }
            with open(path, "w", encoding="utf-8") as plan_file:
                json.dump(plan, plan_file)

            entries = compute_profile.load_input_scale_plan_entries(path)

            assert (entries[0]["payload"]) == (payload)

    def test_compute_container_is_offline_and_does_not_receive_hf_token(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="0123456789abcdef",
            detection_method="hub_api",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            payload_file = os.path.join(tmp_dir, "payloads.json")
            profile_root = os.path.join(tmp_dir, "profiles")
            os.makedirs(profile_root)
            with open(payload_file, "w", encoding="utf-8") as f:
                f.write("{}")

            with profiler_support.profiler_container_command(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu=1,
                mem=2,
                use_gpu=False,
                payload_file=payload_file,
                profile_root=profile_root,
                tool_mount_roots=(),
            ) as cmd:
                cmd = list(cmd)

        assert (f"{os.path.abspath(payload_file)}:/payloads/input_scale_plan.json:ro") in (cmd)
        package_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "acprof"
        )
        assert (f"{package_root}:/app/acprof:ro") in (cmd)
        assert ("HF_HUB_OFFLINE=1") in (cmd)
        assert ("TRANSFORMERS_OFFLINE=1") in (cmd)
        assert ("MODEL_LOCAL_PATH=/models/model-snapshot") in (cmd)
        assert ("HF_TOKEN") not in (cmd)
        assert ("HUGGING_FACE_HUB_TOKEN") not in (cmd)

    def test_profiler_container_command_scopes_model_store_mount(self) -> None:
        task_info = SimpleNamespace(
            model_id="fixture",
            model_revision="main",
            task_family="structured",
            pipeline_tag="tabular-regression",
            runtime_backend="onnxruntime",
            runtime_profile_id="onnxruntime-cpu",
            model_store={"entry_id": "fixture"},
        )
        mount = SimpleNamespace(args=["--mount", "fixture"], close=Mock())

        with patch("acprof.host.model_store.acquire_mount", return_value=mount), patch(
            "acprof.host.model_store.retain_mount_for_cleanup_debt"
        ) as retain:
            with profiler_support.profiler_container_command(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu=1,
                mem=2,
                use_gpu=False,
                payload_file="/tmp/payload.json",
                profile_root="/tmp/profiles",
                tool_mount_roots=(),
            ) as command:
                assert ("--mount") in (command)
                mount.close.assert_not_called()

        mount.close.assert_called_once_with()
        retain.assert_not_called()

    def test_profiler_container_command_retains_mount_if_execution_is_interrupted(self) -> None:
        task_info = SimpleNamespace(
            model_id="fixture",
            model_revision="main",
            task_family="structured",
            pipeline_tag="tabular-regression",
            runtime_backend="onnxruntime",
            runtime_profile_id="onnxruntime-cpu",
            model_store={"entry_id": "fixture"},
        )
        mount = SimpleNamespace(args=[], close=Mock())

        with patch("acprof.host.model_store.acquire_mount", return_value=mount), patch(
            "acprof.host.model_store.retain_mount_for_cleanup_debt"
        ) as retain, pytest.raises(subprocess.TimeoutExpired):
            with profiler_support.profiler_container_command(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu=1,
                mem=2,
                use_gpu=False,
                payload_file="/tmp/payload.json",
                profile_root="/tmp/profiles",
                tool_mount_roots=(),
            ):
                raise subprocess.TimeoutExpired(["docker", "run"], 1)

        retain.assert_called_once_with(mount)
        mount.close.assert_not_called()

    def test_parse_advisor_report_sums_self_gflop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "advisor.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Function", "Self GFLOP"])
                writer.writeheader()
                writer.writerow({"Function": "a", "Self GFLOP": "1.5"})
                writer.writerow({"Function": "b", "Self GFLOP": "2.25"})

            assert (compute_parsers.parse_advisor_self_gflop_csv(report_path)) == (3.75) or round(abs((compute_parsers.parse_advisor_self_gflop_csv(report_path)) - (3.75)), 7) == 0

    def test_parse_advisor_report_skips_native_csv_preamble(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "advisor.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                f.write('sep=,\n\n')
                f.write('"Intel(R) Advisor Command Line Tool\\n')
                f.write('Copyright (C) 2009-2025 Intel Corporation. All rights reserved."\n')
                f.write('"Survey Data version=1.1.0","delimiter=,"\n\n')
                writer = csv.DictWriter(f, fieldnames=["ID", "Self GFLOP", "Module"])
                writer.writeheader()
                writer.writerow({"ID": "1", "Self GFLOP": "1.5", "Module": "libtorch_cpu.so"})
                writer.writerow({"ID": "2", "Self GFLOP": "< 0.001", "Module": "libtorch_cpu.so"})
                writer.writerow({"ID": "3", "Self GFLOP": "2.25", "Module": "libtorch_cpu.so"})

            assert (compute_parsers.parse_advisor_self_gflop_csv(report_path)) == (3.75) or round(abs((compute_parsers.parse_advisor_self_gflop_csv(report_path)) - (3.75)), 7) == 0

    def test_parse_ncu_raw_csv_sums_flop_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ncu.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"],
                )
                writer.writeheader()
                writer.writerow({
                    "Kernel Name": "kernel_a",
                    "Metric Name": "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                    "Metric Unit": "FLOP",
                    "Metric Value": "1,000",
                })
                writer.writerow({
                    "Kernel Name": "kernel_b",
                    "Metric Name": "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
                    "Metric Unit": "FLOP",
                    "Metric Value": "2.5K",
                })
                writer.writerow({
                    "Kernel Name": "kernel_b",
                    "Metric Name": (
                        "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_on.sum"
                    ),
                    "Metric Unit": "FLOP",
                    "Metric Value": "1.5K",
                })
                writer.writerow({
                    "Kernel Name": "kernel_b",
                    "Metric Name": (
                        "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum"
                    ),
                    "Metric Unit": "FLOP",
                    "Metric Value": "1K",
                })
                writer.writerow({
                    "Kernel Name": "kernel_b",
                    "Metric Name": "sm__ops_path_tensor_src_int8_dst_int32.sum",
                    "Metric Unit": "OP",
                    "Metric Value": "9K",
                })
                writer.writerow({
                    "Kernel Name": "kernel_c",
                    "Metric Name": "gpu__time_duration.sum",
                    "Metric Unit": "nsecond",
                    "Metric Value": "10",
                })

            assert (host_profilers_compute_parsers.parse_ncu_profile_csv(report_path)["total_flops_per_request"]) == (3500.0) or round(abs((host_profilers_compute_parsers.parse_ncu_profile_csv(report_path)["total_flops_per_request"]) - (3500.0)), 7) == 0

    def test_ncu_atomic_text_write_syncs_file_and_directory(self) -> None:
        real_fsync = os.fsync
        fsync_calls = []

        def record_fsync(fd):
            fsync_calls.append(fd)
            return real_fsync(fd)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.artifacts.os.fsync",
            side_effect=record_fsync,
        ):
            path = os.path.join(tmp, "ncu.csv")
            ncu._write_text_atomic(path, "new")

        assert len(fsync_calls) == 2

    def test_parse_ncu_raw_csv_weights_sass_fma_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ncu.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"],
                )
                writer.writeheader()
                writer.writerow({
                    "Kernel Name": "kernel_a",
                    "Metric Name": "smsp__sass_thread_inst_executed_op_fadd_pred_on",
                    "Metric Unit": "inst",
                    "Metric Value": "10",
                })
                writer.writerow({
                    "Kernel Name": "kernel_b",
                    "Metric Name": "smsp__sass_thread_inst_executed_op_ffma_pred_on",
                    "Metric Unit": "inst",
                    "Metric Value": "10",
                })

            assert (host_profilers_compute_parsers.parse_ncu_profile_csv(report_path)["total_flops_per_request"]) == (30.0) or round(abs((host_profilers_compute_parsers.parse_ncu_profile_csv(report_path)["total_flops_per_request"]) - (30.0)), 7) == 0

    def test_parse_ncu_wide_csv_sums_metric_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ncu_wide.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID",
                    "Kernel Name",
                    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                    "gpu__time_duration.sum",
                ])
                writer.writerow(["", "", "inst", "inst", "nsecond"])
                writer.writerow(["0", "kernel_a", "10", "20", "100"])
                writer.writerow(["1", "kernel_b", "1K", "2K", "200"])

            assert (host_profilers_compute_parsers.parse_ncu_profile_csv(report_path)["total_flops_per_request"]) == (10 + 20 * 2 + 1000 + 2000 * 2) or round(abs((host_profilers_compute_parsers.parse_ncu_profile_csv(report_path)["total_flops_per_request"]) - (10 + 20 * 2 + 1000 + 2000 * 2)), 7) == 0

    def test_parse_ncu_long_csv_returns_normalized_structured_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ncu_long.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "ID",
                        "Kernel Name",
                        "Metric Name",
                        "Metric Unit",
                        "Metric Value",
                    ],
                )
                writer.writeheader()
                metrics = [
                    ("0", "kernel_a", "smsp__sass_thread_inst_executed_op_fadd_pred_on", "inst", "10"),
                    ("0", "kernel_a", "smsp__sass_thread_inst_executed_op_ffma_pred_on", "inst", "20"),
                    ("0", "kernel_a", "sm__ops_path_tensor_src_fp16_dst_fp32.sum", "FLOP", "100"),
                    ("0", "kernel_a", "gpu__time_duration.sum", "usec", "1000"),
                    ("1", "kernel_b", "smsp__sass_thread_inst_executed_op_fadd_pred_on", "inst", "30"),
                    ("1", "kernel_b", "smsp__sass_thread_inst_executed_op_ffma_pred_on", "inst", "40"),
                    ("1", "kernel_b", "sm__ops_path_tensor_src_fp16_dst_fp32.sum", "FLOP", "300"),
                    ("1", "kernel_b", "gpu__time_duration.sum", "msecond", "2"),
                ]
                for launch_id, kernel, metric, unit, value in metrics:
                    writer.writerow({
                        "ID": launch_id,
                        "Kernel Name": kernel,
                        "Metric Name": metric,
                        "Metric Unit": unit,
                        "Metric Value": value,
                    })

            parsed = compute_parsers.parse_ncu_profile_csv(
                report_path,
                repeat=2,
            )

        assert (parsed["scalar_flops_per_request"]) == (80.0) or round(abs((parsed["scalar_flops_per_request"]) - (80.0)), 7) == 0
        assert (parsed["tensor_flops_per_request"]) == (200.0) or round(abs((parsed["tensor_flops_per_request"]) - (200.0)), 7) == 0
        assert (parsed["total_flops_per_request"]) == (280.0) or round(abs((parsed["total_flops_per_request"]) - (280.0)), 7) == 0
        assert (parsed["tensor_share_pct"]) == ((400.0 / 560.0) * 100.0) or round(abs((parsed["tensor_share_pct"]) - ((400.0 / 560.0) * 100.0)), 7) == 0
        assert (parsed["kernel_launch_count_per_request"]) == (1.0) or round(abs((parsed["kernel_launch_count_per_request"]) - (1.0)), 7) == 0
        assert (parsed["kernel_time_sum_ms_per_request"]) == (1.5) or round(abs((parsed["kernel_time_sum_ms_per_request"]) - (1.5)), 7) == 0

    def test_parse_ncu_wide_csv_normalizes_duration_and_launches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ncu_wide.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID",
                    "Kernel Name",
                    "CC",
                    "launch__sm_count",
                    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                    "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
                    "gpu__time_duration.sum",
                ])
                writer.writerow(["", "", "", "SM", "inst", "FLOP", "nsecond"])
                writer.writerow(["0", "kernel_a", "8.9", "24", "10", "100", "1000"])
                writer.writerow(["1", "kernel_b", "8.9", "24", "20", "200", "3000"])

            parsed = compute_parsers.parse_ncu_profile_csv(
                report_path,
                repeat=2,
            )

        assert (parsed["scalar_flops_per_request"]) == (30.0) or round(abs((parsed["scalar_flops_per_request"]) - (30.0)), 7) == 0
        assert (parsed["tensor_flops_per_request"]) == (150.0) or round(abs((parsed["tensor_flops_per_request"]) - (150.0)), 7) == 0
        assert (parsed["total_flops_per_request"]) == (180.0) or round(abs((parsed["total_flops_per_request"]) - (180.0)), 7) == 0
        assert (parsed["kernel_launch_count_per_request"]) == (1.0) or round(abs((parsed["kernel_launch_count_per_request"]) - (1.0)), 7) == 0
        assert (parsed["kernel_time_sum_ms_per_request"]) == (0.002) or round(abs((parsed["kernel_time_sum_ms_per_request"]) - (0.002)), 7) == 0
        assert (parsed["gpu_compute_capability"]) == ("8.9")
        assert (parsed["gpu_sm_count"]) == (24.0)

    def test_torch_profile_rejects_unverified_eager_result(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        runner_result = {
            "model_logical_mflop_per_request_torch_profiler_eager": 123.0,
            "attention_implementation": "sdpa",
            "attention_implementation_verified": False,
        }

        with tempfile.TemporaryDirectory() as tmp, patch(
                "acprof.host.profilers.torch.profiler_container_command",
            return_value=nullcontext(["docker"]),
        ), patch(
            "acprof.host.profilers.torch.run_command",
            return_value=SimpleNamespace(
                returncode=0,
                stdout=json.dumps(runner_result),
                stderr="",
            ),
        ):
            entry = torch._run_torch_profiler_for_entry(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu=1,
                mem=4,
                use_gpu=True,
                payload_file=os.path.join(tmp, "payloads.json"),
                profile_root=tmp,
                entry={"input_scale": 8.0},
                repeat=1,
            )

        assert (entry["model_logical_mflop_per_request_torch_profiler_eager"]) is None
        assert ("attention_implementation_not_verified") in (entry["error"])

    def test_ncu_entry_exposes_exact_csv_metric_keys_and_relative_report(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        csv_text = "\n".join([
            (
                '"ID","Kernel Name","smsp__sass_thread_inst_executed_op_'
                'ffma_pred_on.sum","sm__ops_path_tensor_src_fp16_dst_fp32.sum",'
                '"gpu__time_duration.sum"'
            ),
            '"","","inst","FLOP","usec"',
            '"0","kernel_a","10","100","1000"',
            '"1","kernel_b","20","300","2000"',
        ])
        collect_result = SimpleNamespace(
            returncode=0,
            stdout=(
                '{"gpu_compute_capability":"8.9","gpu_sm_count":24}\n'
            ),
            stderr="",
        )
        import_result = SimpleNamespace(
            returncode=0,
            stdout=csv_text,
            stderr="",
        )

        with tempfile.TemporaryDirectory() as tmp:
            profile_root = os.path.join(tmp, "compute_profiles")
            os.makedirs(profile_root)
            with patch(
                    "acprof.host.profilers.ncu.profiler_container_command",
                return_value=nullcontext(["docker"]),
            ), patch(
                "acprof.host.profilers.ncu._ncu_collect_filter_args",
                return_value=[],
            ), patch(
                "acprof.host.profilers.ncu._ncu_section_args",
                return_value=[],
            ), patch(
                "acprof.host.profilers.ncu.run_command",
                side_effect=[collect_result, import_result],
            ):
                result = ncu._run_ncu_for_entry(
                    ncu_bin="/usr/bin/ncu",
                    ncu_metrics=[
                        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                        "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
                        "gpu__time_duration.sum",
                    ],
                    task_info=task_info,
                    image_tag="acprof-test:latest",
                    cpu=1,
                    mem=4,
                    payload_file=os.path.join(tmp, "payloads.json"),
                    profile_root=profile_root,
                    tool_mount_roots=(),
                    entry={"input_scale": 8.0},
                    repeat=2,
                )

            assert (os.path.isfile(os.path.join(profile_root, "ncu_scale_8.csv")))

        assert (result["gpu_executed_mflop_per_request_ncu"]) == (0.00023) or round(abs((result["gpu_executed_mflop_per_request_ncu"]) - (0.00023)), 7) == 0
        assert (result["gpu_executed_tensor_mflop_per_request_ncu"]) == (0.0002) or round(abs((result["gpu_executed_tensor_mflop_per_request_ncu"]) - (0.0002)), 7) == 0
        assert (result["gpu_executed_scalar_mflop_per_request_ncu"]) == (0.00003) or round(abs((result["gpu_executed_scalar_mflop_per_request_ncu"]) - (0.00003)), 7) == 0
        assert (result["gpu_executed_tensor_share_pct_ncu"]) == ((400.0 / 460.0) * 100.0) or round(abs((result["gpu_executed_tensor_share_pct_ncu"]) - ((400.0 / 460.0) * 100.0)), 7) == 0
        assert (result["gpu_kernel_launch_count_per_request_ncu"]) == (1.0)
        assert (result["gpu_kernel_time_sum_ms_per_request_ncu"]) == (1.5)
        assert ("gpu_profile_report_ncu") not in (result)
        assert (result["report"]) == (os.path.join("compute_profiles", "ncu_scale_8.csv"))

    def test_ncu_parse_failure_preserves_report_until_artifacts_are_discarded(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        command_result = SimpleNamespace(
            returncode=0,
            stdout="not a parseable NCU CSV",
            stderr="",
        )

        with tempfile.TemporaryDirectory() as tmp:
            profile_root = os.path.join(tmp, "compute_profiles")
            os.makedirs(profile_root)
            with patch(
                    "acprof.host.profilers.ncu.profiler_container_command",
                return_value=nullcontext(["docker"]),
            ), patch(
                "acprof.host.profilers.ncu._ncu_collect_filter_args",
                return_value=[],
            ), patch(
                "acprof.host.profilers.ncu._ncu_section_args",
                return_value=[],
            ), patch(
                "acprof.host.profilers.ncu.run_command",
                side_effect=[command_result, command_result],
            ):
                result = ncu._run_ncu_for_entry(
                    ncu_bin="/usr/bin/ncu",
                    ncu_metrics=["flop_count_sp"],
                    task_info=task_info,
                    image_tag="acprof-test:latest",
                    cpu=1,
                    mem=4,
                    payload_file=os.path.join(tmp, "payloads.json"),
                    profile_root=profile_root,
                    tool_mount_roots=(),
                    entry={"input_scale": 8.0},
                    repeat=1,
                )

            report = os.path.join("compute_profiles", "ncu_scale_8.csv")
            assert ("gpu_profile_report_ncu") not in (result)
            assert (result["report"]) == (report)
            assert (os.path.isfile(os.path.join(tmp, report)))

            profiles = {
                "gpu": {
                    "ncu": {
                        "tool": "ncu",
                        "entries": [result],
                    },
                },
            }
            compute_profile._strip_discarded_profile_paths(profiles)

        assert ("gpu_profile_report_ncu") not in (result)
        assert (result["report"]) is None

    def test_select_ncu_metrics_combines_sass_and_float_tensor_aggregates(self) -> None:
        fadd = "smsp__sass_thread_inst_executed_op_fadd_pred_on"
        ffma = "smsp__sass_thread_inst_executed_op_ffma_pred_on"
        tensor = "sm__ops_path_tensor_src_fp16_dst_fp32.sum"

        metrics = ncu._select_ncu_flop_metrics([
            fadd,
            f"{ffma}.sum",
            "flop_count_sp",
            tensor,
            "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_on.sum",
            "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum",
            "sm__ops_path_tensor_src_int8_dst_int32.sum",
        ])

        assert (metrics) == ([fadd, f"{ffma}.sum", tensor])

    def test_resolve_ncu_metrics_reports_privilege_failure(self) -> None:
        calls = []

        def fake_run(cmd, check=False):
            calls.append(cmd)
            if "--query-metrics-mode" in cmd:
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "==ERROR== Invalid option --query-metrics-mode suffix. "
                        "Please specify along with --metrics."
                    ),
                )
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Device NVIDIA GeForce RTX 4060 Laptop GPU\n"
                    "==ERROR== ERR_NVGPUCTRPERM - The user does not have "
                    "permission to access NVIDIA GPU Performance Counters\n"
                ),
                stderr="",
            )

        with patch("acprof.host.profilers.ncu.run_command", side_effect=fake_run):
            metrics, error = ncu._resolve_ncu_metrics("/opt/ncu")

        assert (metrics) == ([])
        assert ("ERR_NVGPUCTRPERM") in (error)
        assert (len(calls)) == (2)

    def test_resolve_ncu_metrics_retries_query_in_gpu_container(self) -> None:
        calls = []
        container_base_cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "--cap-add=SYS_ADMIN",
            "--cap-add=SYS_PTRACE",
            "acprof-test:latest",
        ]
        fadd = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
        tensor = "sm__ops_path_tensor_src_fp16_dst_fp32.sum"

        def fake_run(cmd, check=False):
            calls.append(cmd)
            if cmd[:len(container_base_cmd)] == container_base_cmd:
                return SimpleNamespace(
                    returncode=0,
                    stdout="\n".join([
                        fadd,
                        tensor,
                        (
                            "sm__ops_path_tensor_src_fp16_dst_fp32_"
                            "sparsity_on.sum"
                        ),
                        "sm__ops_path_tensor_src_int8_dst_int32.sum",
                    ]),
                    stderr="",
                )
            if "--query-metrics-mode" in cmd:
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="unsupported query mode",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="",
                stderr=(
                    "==ERROR== ERR_NVGPUCTRPERM - The user does not have "
                    "permission to access NVIDIA GPU Performance Counters"
                ),
            )

        with patch("acprof.host.profilers.ncu.run_command", side_effect=fake_run):
            metrics, error = ncu._resolve_ncu_metrics(
                "/opt/ncu",
                container_base_cmd=container_base_cmd,
            )

        assert (metrics) == ([fadd, tensor])
        assert (error) == ("")
        assert (len(calls)) == (3)
        assert (calls[-1]) == ([
                *container_base_cmd,
                "/opt/ncu",
                "--query-metrics",
                "--query-metrics-mode",
                "all",
            ])

    def test_gpu_profile_builds_privileged_container_for_metric_query(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        query = {}

        def fake_resolve(ncu_bin, *, container_base_cmd=None, include_host=True):
            query["ncu_bin"] = ncu_bin
            if container_base_cmd is None:
                return [], "host query unavailable"
            assert not include_host
            query["container_base_cmd"] = container_base_cmd
            return list(compute_parsers.NCU_SASS_FLOP_WEIGHTS), ""

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.profilers.ncu._resolve_ncu_metrics",
            side_effect=fake_resolve,
        ), patch(
            "acprof.host.profilers.ncu._run_ncu_for_entry",
            return_value={
                "input_scale": 8.0,
                "tool": "ncu",
                "gpu_executed_mflop_per_request_ncu": 1.0,
                "error": "",
            },
        ):
            ncu._profile_gpu_entries(
                entries=[{"input_scale": 8.0}],
                ncu_bin="/opt/nvidia/nsight-compute/2025.1.0/ncu",
                ncu_root=None,
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu=2,
                mem=4,
                payload_file=os.path.join(tmp, "payloads.json"),
                profile_root=tmp,
                repeat=1,
            )

        assert (query["ncu_bin"]) == ("/opt/nvidia/nsight-compute/2025.1.0/ncu")
        container_base_cmd = query["container_base_cmd"]
        assert ("--gpus") in (container_base_cmd)
        assert ("--cap-add=SYS_ADMIN") in (container_base_cmd)
        assert ("--cap-add=SYS_PTRACE") in (container_base_cmd)
        assert ("--security-opt=seccomp=unconfined") in (container_base_cmd)
        assert (container_base_cmd[-1]) == ("acprof-test:latest")

    def test_ncu_resume_reuses_csv_recovers_report_and_collects_only_missing(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="revision-1",
            detection_method="hub_api",
        )
        metrics = [
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
        ]
        collected = []

        with tempfile.TemporaryDirectory() as tmp:
            profile_root = os.path.join(tmp, "compute_profiles")
            os.makedirs(profile_root)
            with open(
                os.path.join(profile_root, "ncu_scale_1.csv"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(_ncu_resume_csv_text())
            with open(
                os.path.join(profile_root, "ncu_scale_20.ncu-rep"),
                "wb",
            ) as f:
                f.write(b"existing report")

            def fake_export(**kwargs):
                assert (kwargs["report_base"].endswith("ncu_scale_20"))
                ncu._write_text_atomic(
                    kwargs["host_csv"],
                    _ncu_resume_csv_text(),
                )
                return True, ""

            def fake_collect(**kwargs):
                scale = float(kwargs["entry"]["input_scale"])
                collected.append(scale)
                assert (scale) == (30.0)
                _report_base, host_csv, _host_report, _checkpoint = (
                    ncu._ncu_artifact_paths(profile_root, scale)
                )
                ncu._write_text_atomic(
                    host_csv,
                    _ncu_resume_csv_text(),
                )
                return ncu._ncu_entry_from_csv(
                    entry=kwargs["entry"],
                    host_csv=host_csv,
                    profile_root=profile_root,
                    repeat=kwargs["repeat"],
                    runner_payload={
                        "gpu_compute_capability": "8.9",
                        "gpu_sm_count": 24,
                    },
                )

            with patch(
                "acprof.host.profilers.ncu._resolve_ncu_metrics",
                return_value=(metrics, ""),
            ), patch(
                "acprof.host.profilers.ncu._export_ncu_report",
                side_effect=fake_export,
            ) as export_report, patch(
                "acprof.host.profilers.ncu._run_ncu_for_entry",
                side_effect=fake_collect,
            ):
                result = ncu._profile_gpu_entries(
                    entries=[
                        {"input_scale": 1.0},
                        {"input_scale": 20.0},
                        {"input_scale": 30.0},
                    ],
                    ncu_bin="/opt/nvidia/nsight-compute/ncu",
                    ncu_root=None,
                    task_info=task_info,
                    image_tag="acprof-test:latest",
                    cpu=2,
                    mem=4,
                    payload_file=os.path.join(tmp, "payloads.json"),
                    profile_root=profile_root,
                    repeat=1,
                    resume_existing=True,
                )

            assert (collected) == ([30.0])
            assert (export_report.call_count) == (1)
            assert ([entry["input_scale"] for entry in result["entries"]]) == ([1.0, 20.0, 30.0])
            assert (all(
                ncu._ncu_entry_complete(entry)
                for entry in result["entries"]
            ))
            for scale in (1, 20, 30):
                assert (os.path.isfile(os.path.join(
                    profile_root,
                    f"ncu_scale_{scale}.checkpoint.json",
                )))

    def test_ncu_resume_rejects_checkpoint_from_different_repeat(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="revision-1",
            detection_method="hub_api",
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile_root = os.path.join(tmp, "compute_profiles")
            os.makedirs(profile_root)
            host_csv = os.path.join(profile_root, "ncu_scale_1.csv")
            ncu._write_text_atomic(host_csv, _ncu_resume_csv_text())
            entry = ncu._ncu_entry_from_csv(
                entry={"input_scale": 1.0},
                host_csv=host_csv,
                profile_root=profile_root,
                repeat=1,
            )
            checkpoint_path = os.path.join(
                profile_root,
                "ncu_scale_1.checkpoint.json",
            )
            ncu._write_ncu_checkpoint(
                checkpoint_path=checkpoint_path,
                task_info=task_info,
                image_tag="acprof-test:latest",
                input_scale=1.0,
                repeat=1,
                metrics=["metric", compute_parsers.NCU_DURATION_METRIC],
                host_csv=host_csv,
                entry=entry,
            )

            resumed = ncu._resume_ncu_for_entry(
                ncu_bin="/opt/ncu",
                ncu_metrics=["metric", compute_parsers.NCU_DURATION_METRIC],
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu=1,
                mem=4,
                payload_file=os.path.join(tmp, "payloads.json"),
                profile_root=profile_root,
                tool_mount_roots=(),
                entry={"input_scale": 1.0},
                repeat=2,
            )

        assert (resumed) is None

    def test_vendor_mode_missing_tools_write_nan_profiles_with_errors(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        with tempfile.TemporaryDirectory() as tmp, patch(
                "acprof.host.compute_profile.find_executable",
            return_value=None,
        ), patch(
            "acprof.host.compute_profile.run_command",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
                compute_profile_tool="vendor",
            )

            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
            assert not (os.path.exists(os.path.join(tmp, "compute_profiles")))
            assert not (os.path.exists(
                    os.path.join(tmp, "compute_profile_payloads.json")
                ))

        assert ("advisor_not_found") in (plan["profiles"]["cpu"]["intel_advisor"]["error"])
        assert ("ncu_not_found") in (plan["profiles"]["gpu"]["ncu"]["error"])
        assert (plan["profiles"]["cpu"]["intel_advisor"]["entries"][0][
                "model_mflop_per_request"
            ]) is (None)
        assert (plan["profiles"]["gpu"]["ncu"]["entries"][0][
                "gpu_executed_mflop_per_request_ncu"
            ]) is (None)

    def test_default_compute_profile_mode_writes_disabled_plan_without_probes(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
                "acprof.host.compute_profile.find_executable",
            side_effect=AssertionError("none mode must not discover tools"),
        ), patch(
            "acprof.host.profilers.torch._profile_torch_entries",
            side_effect=AssertionError("none mode must not run Torch"),
        ), patch(
            "acprof.host.profilers.ncu._profile_gpu_entries",
            side_effect=AssertionError("none mode must not run NCU"),
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=1,
                ncu_repeat=1,
                keep_profiles=True,
            )
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
            assert not (os.path.exists(os.path.join(tmp, "compute_profiles")))

        assert (plan["compute_profile_tool_mode"]) == ("none")
        assert (plan["profiles"]) == ({})
        assert (plan["static_metadata"]["compute_profile_tools"]) == ([])
        assert not (plan["static_metadata"]["compute_profiles_retained"])
        assert (plan["static_metadata"]["compute_profile_provenance"]) == ("disabled")

    def test_both_compute_profile_uses_torch_on_each_device_and_gpu_ncu(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        calls = []

        def fake_find_executable(root, names):
            calls.append(("find", names))
            if "ncu" in names:
                return "/opt/nvidia/nsight-compute/2024.1.1/ncu"
            raise AssertionError("advisor should not be resolved in both mode")

        def fake_torch_profile(**kwargs):
            calls.append(("torch", kwargs["profile_key"], kwargs["use_gpu"]))
            return {
                "tool": "torch_profiler_eager",
                "repeat": kwargs["repeat"],
                "error": "",
                "entries": [
                    {
                        "input_scale": 8.0,
                        "tool": "torch_profiler_eager",
                        "model_logical_mflop_per_request_torch_profiler_eager": 123.0,
                        "error": "",
                    }
                ],
            }

        def fake_gpu_profile(**kwargs):
            calls.append(("gpu", kwargs["ncu_bin"]))
            return {
                "tool": "ncu",
                "repeat": kwargs["repeat"],
                "error": "",
                "entries": [
                    {
                        "input_scale": 8.0,
                        "tool": "ncu",
                        "gpu_executed_mflop_per_request_ncu": 300.0,
                        "error": "",
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
                "acprof.host.compute_profile.find_executable",
            side_effect=fake_find_executable,
        ), patch(
            "acprof.host.profilers.torch._profile_torch_entries",
            side_effect=fake_torch_profile,
        ), patch(
            "acprof.host.profilers.advisor._profile_cpu_entries",
            side_effect=AssertionError("vendor CPU profiler should not run in both mode"),
        ), patch(
            "acprof.host.profilers.ncu._profile_gpu_entries",
            side_effect=fake_gpu_profile,
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
                compute_profile_tool="both",
            )

            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)

        assert (calls) == ([
                ("find", ("ncu", "nv-nsight-cu-cli")),
                ("torch", "cpu", False),
                ("torch", "gpu", True),
                ("gpu", "/opt/nvidia/nsight-compute/2024.1.1/ncu"),
            ])
        assert (plan["compute_profile_tool_mode"]) == ("both")
        assert (plan["profiles"]["cpu"]["torch_profiler_eager"]["tool"]) == ("torch_profiler_eager")
        assert (plan["profiles"]["gpu"]["ncu"]["tool"]) == ("ncu")
        assert ("tool") not in (plan["profiles"]["cpu"])
        assert ("tool") not in (plan["profiles"]["gpu"])


    def test_both_mode_keeps_torch_and_ncu_failures_independent(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        calls = []

        def fake_torch_profile(**kwargs):
            calls.append(("torch", kwargs["profile_key"]))
            raise RuntimeError("torch probe failed")

        def fake_gpu_profile(**kwargs):
            calls.append(("ncu", kwargs["repeat"]))
            return {
                "tool": "ncu",
                "repeat": kwargs["repeat"],
                "metrics": ["gpu__time_duration.sum"],
                "error": "",
                "entries": [{
                    "input_scale": 8.0,
                    "tool": "ncu",
                    "gpu_executed_mflop_per_request_ncu": 42.0,
                    "error": "",
                }],
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
                "acprof.host.compute_profile.find_executable",
            return_value="/usr/bin/ncu",
        ), patch(
            "acprof.host.profilers.torch._profile_torch_entries",
            side_effect=fake_torch_profile,
        ), patch(
            "acprof.host.profilers.ncu._profile_gpu_entries",
            side_effect=fake_gpu_profile,
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=3,
                torch_profiler_repeat=2,
                keep_profiles=False,
                compute_profile_tool="both",
            )
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)

        assert (calls) == ([("torch", "cpu"), ("torch", "gpu"), ("ncu", 3)])
        assert ("schema_version") not in (plan)
        assert ("torch_profiler_eager_failed") in (plan["profiles"]["gpu"]["torch_profiler_eager"]["error"])
        assert (plan["profiles"]["gpu"]["ncu"]["entries"][0][
                "gpu_executed_mflop_per_request_ncu"
            ]) == (42.0)
        metadata = plan["static_metadata"]
        assert ("compute_profile_schema_version") not in (metadata)
        assert (metadata["compute_profile_tools"]) == (["torch_profiler_eager", "ncu"])
        assert (metadata["torch_profiler_eager_repeat_cpu"]) == (2)
        assert (metadata["torch_profiler_eager_repeat_gpu"]) == (2)
        assert (metadata["ncu_repeat"]) == (3)
        assert not (metadata["compute_profiles_retained"])

    def test_vendor_compute_profile_mode_keeps_missing_tool_errors(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        with tempfile.TemporaryDirectory() as tmp, patch(
                "acprof.host.compute_profile.find_executable",
            return_value=None,
        ), patch(
            "acprof.host.compute_profile.run_command",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
                compute_profile_tool="vendor",
            )

            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)

        assert ("advisor_not_found") in (plan["profiles"]["cpu"]["intel_advisor"]["error"])
        assert ("ncu_not_found") in (plan["profiles"]["gpu"]["ncu"]["error"])

    def test_compute_profile_resource_overrides_are_used(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        calls = []

        def fake_cpu_profile(**kwargs):
            calls.append(("cpu", kwargs["cpu"], kwargs["mem"]))
            return {"tool": "intel_advisor", "repeat": 1, "error": "", "entries": []}

        def fake_gpu_profile(**kwargs):
            calls.append(("gpu", kwargs["cpu"], kwargs["mem"]))
            return {"tool": "ncu", "repeat": 1, "error": "", "entries": []}

        with tempfile.TemporaryDirectory() as tmp, patch(
                "acprof.host.compute_profile.find_executable",
            return_value="/usr/bin/tool",
        ), patch(
            "acprof.host.profilers.advisor._profile_cpu_entries",
            side_effect=fake_cpu_profile,
        ), patch(
            "acprof.host.profilers.ncu._profile_gpu_entries",
            side_effect=fake_gpu_profile,
        ):
            compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
                compute_profile_cpus=8,
                compute_profile_mem=16,
                compute_profile_tool="vendor",
            )

        assert (calls) == ([("cpu", 8, 16), ("gpu", 8, 16)])

    def test_compute_profile_default_resources_use_host_capacity(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        calls = []

        def fake_cpu_profile(**kwargs):
            calls.append(("cpu", kwargs["cpu"], kwargs["mem"]))
            return {"tool": "intel_advisor", "repeat": 1, "error": "", "entries": []}

        with tempfile.TemporaryDirectory() as tmp, patch(
                "acprof.host.compute_profile.find_executable",
            return_value="/usr/bin/tool",
        ), patch(
            "acprof.host.compute_profile._host_logical_cpus",
            return_value=12,
        ), patch(
            "acprof.host.compute_profile._host_memory_gb_fraction",
            return_value=48,
        ), patch(
            "acprof.host.profilers.advisor._profile_cpu_entries",
            side_effect=fake_cpu_profile,
        ):
            compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
                compute_profile_tool="vendor",
            )

        assert (calls) == ([("cpu", 12, 48)])

    def test_find_executable_searches_default_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            tool_discovery,
            "DEFAULT_TOOL_SEARCH_ROOTS",
            (tmp,),
        ):
            bin_dir = os.path.join(tmp, "latest", "bin64")
            os.makedirs(bin_dir)
            advisor_path = os.path.join(bin_dir, "advisor")
            with open(advisor_path, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\n")

            assert (compute_profile.find_executable(None, ("advisor", "advixe-cl"))) == (advisor_path)

    def test_tool_mount_root_uses_profiler_install_root(self) -> None:
        advisor_bin = "/opt/intel/oneapi/advisor/2025.5/bin64/advisor"
        ncu_bin = "/opt/nvidia/nsight-compute/2025.1.0/ncu"

        assert (host_profilers_tool_discovery.tool_mount_root(advisor_bin, None)) == ("/opt/intel/oneapi/advisor/2025.5")
        assert (host_profilers_tool_discovery.tool_mount_root(ncu_bin, None)) == ("/opt/nvidia/nsight-compute/2025.1.0")

    def test_debian_ncu_mount_roots_include_target_symlink_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib_root = os.path.join(tmp, "usr", "lib", "nsight-compute")
            arch_root = os.path.join(tmp, "usr", "lib", "x86_64-linux-gnu", "nsight-compute")
            target_root = os.path.join(lib_root, "target")
            real_target = os.path.join(arch_root, "target", "linux-desktop-glibc_2_11_3-x64")
            os.makedirs(target_root)
            os.makedirs(real_target)
            os.symlink(real_target, os.path.join(target_root, "linux-desktop-glibc_2_11_3-x64"))
            ncu_path = os.path.join(lib_root, "ncu")
            with open(ncu_path, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\n")

            assert (tool_discovery.tool_mount_roots(ncu_path, None)) == ([lib_root, arch_root])


class TestComputeProfileProgress:
    def _collect(self, directory, **kwargs):
        input_plan = _write_input_scale_plan(directory)
        with open(input_plan, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["entries"].append({
            **payload["entries"][0],
            "input_scale": 16.0,
            "scale_label": "seq16",
        })
        with open(input_plan, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        with patch.object(
            compute_profile, "find_executable", return_value=None,
        ), patch.object(
            compute_profile, "_executable_version", return_value="unknown",
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=TaskInfo(
                    model_id="google-bert/bert-base-uncased",
                    pipeline_tag="fill-mask",
                    task_family="nlp",
                    runtime_backend="transformers_pipeline",
                    library_name="transformers",
                    model_revision="main",
                    detection_method="hub_api",
                ),
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=kwargs.pop("gpu_list", ["off", "on"]),
                output_dir=directory,
                input_scale_plan_file=input_plan,
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=2,
                ncu_repeat=2,
                torch_profiler_repeat=2,
                keep_profiles=False,
                **kwargs,
            )
        with open(plan_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _torch_profile(**kwargs):
        return {
            "tool": torch.TORCH_PROFILER_TOOL,
            "repeat": kwargs["repeat"],
            "error": "",
            "entries": [{
                "input_scale": entry["input_scale"],
                "tool": torch.TORCH_PROFILER_TOOL,
                "model_logical_mflop_per_request_torch_profiler_eager": 42.0,
                "error": "",
            } for entry in kwargs["entries"]],
        }

    def test_each_complete_tool_notifies_before_next_tool_and_excludes_callback_time(self):
        order = []
        completions = []
        elapsed = [0.0]

        def profile_stage(profiler, kwargs):
            order.append(("start", profiler))
            for entry in kwargs["entries"]:
                order.append(("scale", profiler, entry["input_scale"]))
                elapsed[0] += 2.5
            order.append(("released", profiler))
            return self._torch_profile(**kwargs)

        def notify(completion):
            order.append(("complete", completion.profiler))
            completions.append(completion)
            elapsed[0] += 100.0

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            torch, "_profile_torch_entries",
            side_effect=lambda **kwargs: profile_stage(
                "GPU Torch" if kwargs["use_gpu"] else "CPU Torch", kwargs,
            ),
        ), patch.object(
            ncu, "_profile_gpu_entries",
            side_effect=lambda **kwargs: profile_stage("NCU", kwargs),
        ), patch.object(
            compute_profile.time, "perf_counter", side_effect=lambda: elapsed[0],
        ):
            self._collect(tmp, compute_profile_tool="both", progress_callback=notify)

        expected = []
        for profiler in ("CPU Torch", "GPU Torch", "NCU"):
            expected.extend([
                ("start", profiler),
                ("scale", profiler, 8.0),
                ("scale", profiler, 16.0),
                ("released", profiler),
                ("complete", profiler),
            ])
        assert (order) == (expected)
        assert ([event.elapsed_seconds for event in completions]) == ([5.0] * 3)
        assert ([event.total_samples for event in completions]) == ([2] * 3)
        assert ([event.status for event in completions]) == (["success"] * 3)

    def test_missing_vendor_tools_still_notify_each_failed_stage(self):
        completions = []
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._collect(
                tmp, compute_profile_tool="vendor", progress_callback=completions.append,
            )

        assert ([event.profiler for event in completions]) == (["CPU Advisor", "NCU"])
        assert ([event.status for event in completions]) == (["failed", "failed"])
        assert ([event.total_samples for event in completions]) == ([2, 2])
        assert ([event.error_samples for event in completions]) == ([2, 2])
        assert (plan["profiles"]["cpu"]["intel_advisor"]["error"]) == ("advisor_not_found")
        assert (plan["profiles"]["gpu"]["ncu"]["error"]) == ("ncu_not_found")

    def test_raised_tool_failure_notifies_and_continues_to_next_device(self):
        completions = []

        def torch_profile(**kwargs):
            if not kwargs["use_gpu"]:
                raise RuntimeError("CPU probe failed")
            return self._torch_profile(**kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            torch, "_profile_torch_entries", side_effect=torch_profile,
        ):
            plan = self._collect(
                tmp, compute_profile_tool="torch", progress_callback=completions.append,
            )

        assert ([event.profiler for event in completions]) == (["CPU Torch", "GPU Torch"])
        assert ([event.status for event in completions]) == (["failed", "success"])
        assert ("CPU probe failed") in (completions[0].detail)
        assert (completions[0].error_samples) == (2)
        assert (plan["profiles"]["gpu"][torch.TORCH_PROFILER_TOOL]["error"]) == ("")

    def test_notification_failure_does_not_change_plan_or_stop_next_stage(self):
        attempted = []

        def broken_callback(event):
            attempted.append(event.profiler)
            raise RuntimeError("notification failed")

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            torch, "_profile_torch_entries", side_effect=self._torch_profile,
        ):
            expected = self._collect(tmp, compute_profile_tool="torch")
            actual = self._collect(
                tmp, compute_profile_tool="torch", progress_callback=broken_callback,
            )

        assert (attempted) == (["CPU Torch", "GPU Torch"])
        assert (actual) == (expected)

    @pytest.mark.parametrize('mode,gpu_list', (('none', ['off', 'on']), ('ncu', ['off'])))
    def test_disabled_and_inapplicable_tools_do_not_notify(self, mode, gpu_list):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            torch, "_profile_torch_entries",
        ) as torch_call, patch.object(
            ncu, "_profile_gpu_entries",
        ) as ncu_call, patch.object(
            advisor, "_profile_cpu_entries",
        ) as advisor_call:
            completions = []
            self._collect(
                tmp, compute_profile_tool=mode, gpu_list=gpu_list,
                progress_callback=completions.append,
            )
            assert (completions) == ([])
            torch_call.assert_not_called()
            ncu_call.assert_not_called()
            advisor_call.assert_not_called()
