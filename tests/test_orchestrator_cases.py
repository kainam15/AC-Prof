import csv
import json
import os
import tempfile
from contextlib import redirect_stderr
from functools import partial
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.config import CSV_FIELDS
from acprof.host import cases, docker_runtime, orchestrator
from acprof.host.detect import TaskInfo
from acprof.host.images import runtime as runtime_images
from acprof.host.measurement import capture as packet_capture, error_rows


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


class TestOrchestratorCases:
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
            "acprof.host.orchestrator.start_service_session",
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
            "acprof.host.orchestrator.stop_service_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='off',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=self.output_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=1,
                    repeat_in_window=1,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64',
                ),
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
            "acprof.host.orchestrator.start_service_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_service_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='off',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=self.output_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=1,
                    repeat_in_window=1,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64',
                    compute_profile_plan_file='results/test-unit/compute_profile_plan.json',
                ),
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
            "acprof.host.orchestrator.start_service_session",
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
            "acprof.host.orchestrator.stop_service_session"
        ), patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='off',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=self.output_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=1,
                    repeat_in_window=1,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64',
                    execution_profile_plan_file='results/test-unit/execution_profile_plan.json',
                ),
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
            "acprof.host.orchestrator.start_service_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_service_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='off',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=self.output_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=1,
                    repeat_in_window=1,
                ),
                measurement=cases.CaseMeasurement(
                    idle_cooldown_seconds=4.5,
                    idle_debug=True,
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64',
                ),
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
            "acprof.host.orchestrator.start_service_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ) as start_container, patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_service_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='on',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=self.output_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=1,
                    repeat_in_window=0,
                    repeat_window_seconds=10.0,
                    request_timeout_seconds=123.5,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64',
                ),
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
            "acprof.host.orchestrator.start_service_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_service_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='on',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=self.output_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=1,
                    repeat_in_window=1000,
                    repeat_window_seconds=10.0,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64',
                ),
            )

        assert (captured_env["REPEAT_IN_WINDOW"]) == ("1000")
        assert (captured_env["REPEAT_WINDOW_SECONDS"]) == ("10.0")

    @pytest.mark.parametrize("typed_failure", [False, True], ids=["no_recorded_failure", "typed_inference_failure"])
    def test_run_single_case_aborts_when_client_exits_nonzero(self, typed_failure) -> None:
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
            "acprof.host.orchestrator.start_service_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_service_session"
        ), patch(
            "acprof.host.container_state.container_runtime_oom_error",
            return_value=None,
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            if typed_failure:
                recorded_failure = Failure("predict", "inference_failed", "typed inference error")
            with pytest.raises(
                RuntimeFailure if recorded_failure else orchestrator.EnergyProfilingError
            ) as raised:
                orchestrator.run_single_case(
                    cases.CaseRequest(
                        task_info=task_info,
                        cpu=1,
                        mem=4,
                        gpu='on',
                        image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                        output_dir=self.output_dir,
                        project_dir='.',
                    ),
                    workload=cases.CaseWorkload(
                        warmup=0,
                        repeat=1,
                        repeat_in_window=0,
                    ),
                    measurement=cases.CaseMeasurement(
                        require_packet_latency=False,
                    ),
                    plans=cases.CasePlans(
                        input_scales='64',
                    ),
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
            "acprof.host.orchestrator.start_service_session",
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
            "acprof.host.orchestrator.stop_service_session"
        ) as stop_container, patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            csv_path = orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=2,
                    mem=2,
                    gpu='on',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=tmp_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=1,
                    repeat_in_window=1,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64,128,256',
                ),
            )

            with open(csv_path, encoding="utf-8", newline="") as f:
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
            "acprof.host.orchestrator.start_service_session",
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
            "acprof.host.container_state.container_runtime_oom_error",
            return_value=None,
        ), patch(
            "acprof.host.orchestrator.stop_service_session"
        ), patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            with pytest.raises(orchestrator.MIPSProfilingError):
                orchestrator.run_single_case(
                    cases.CaseRequest(
                        task_info=task_info,
                        cpu=1,
                        mem=4,
                        gpu='off',
                        image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                        output_dir=self.output_dir,
                        project_dir='.',
                    ),
                    workload=cases.CaseWorkload(
                        warmup=0,
                        repeat=1,
                        repeat_in_window=1,
                    ),
                    measurement=cases.CaseMeasurement(
                        require_packet_latency=False,
                    ),
                    plans=cases.CasePlans(
                        input_scales='64',
                    ),
                )

    def test_client_error_context_rejects_nonfinite_json(self, tmp_path, capsys) -> None:
        path = tmp_path / "client-error.json"
        path.write_text('{"input_scale": NaN}', encoding="utf-8")

        assert error_rows.load_client_error_context(str(path)) == {}
        assert "could not read structured client error context" in capsys.readouterr().err

    def test_client_error_context_rejects_oversized_json(self, tmp_path, capsys) -> None:
        path = tmp_path / "client-error.json"
        path.write_text(json.dumps({"padding": "x" * (4 * 1024 * 1024)}), encoding="utf-8")

        assert error_rows.load_client_error_context(str(path)) == {}
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
            "acprof.host.orchestrator.start_service_session",
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
            "acprof.host.orchestrator.stop_service_session"
        ) as stop_container, patch(
            "acprof.host.container_state.container_runtime_oom_error",
            return_value=None,
        ), patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            csv_path = orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='off',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=tmp_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=1,
                    repeat=2,
                    repeat_in_window=0,
                    request_timeout_seconds=120.0,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64,128',
                ),
            )

            with open(csv_path, encoding="utf-8", newline="") as f:
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
            "acprof.host.orchestrator.start_service_session",
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
            "acprof.host.orchestrator.stop_service_session"
        ), patch(
            "acprof.host.container_state.container_runtime_oom_error",
            return_value=None,
        ), patch(
            "acprof.host.command.run_command",
            side_effect=fake_run,
        ):
            csv_path = orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='off',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=tmp_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=1,
                    repeat_in_window=1,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64,128',
                ),
            )

            with open(csv_path, encoding="utf-8", newline="") as f:
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
            "acprof.host.orchestrator.start_service_session",
            side_effect=RuntimeError(
                "container_oom_killed during startup "
                "(container=unit-test, memory_limit=2g, status=exited, exit_code=137)"
            ),
        ):
            csv_path = orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=2,
                    gpu='on',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=tmp_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=1,
                    repeat=2,
                ),
                plans=cases.CasePlans(
                    input_scales='85,170',
                ),
            )

            with open(csv_path, encoding="utf-8", newline="") as f:
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
            "acprof.host.orchestrator.start_service_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_service_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run):
            csv_path = orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='on',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=tmp_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=3,
                    repeat_in_window=1,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64',
                ),
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
            "acprof.host.orchestrator.start_service_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_service_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run), redirect_stderr(stdout):
            csv_path = orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='on',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=tmp_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=3,
                    repeat_in_window=1,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64',
                ),
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
            "acprof.host.orchestrator.start_service_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_service_session"
        ), patch("acprof.host.command.run_command", side_effect=fake_run), redirect_stderr(stdout):
            csv_path = orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu='off',
                    image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                    output_dir=tmp_dir,
                    project_dir='.',
                ),
                workload=cases.CaseWorkload(
                    warmup=0,
                    repeat=3,
                    repeat_in_window=1,
                ),
                measurement=cases.CaseMeasurement(
                    require_packet_latency=False,
                ),
                plans=cases.CasePlans(
                    input_scales='64',
                ),
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
        with patch("acprof.host.measurement.capture.shutil.which", return_value=None):
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

        with patch("acprof.host.measurement.capture.shutil.which", side_effect=fake_which), patch(
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
        with patch('acprof.host.measurement.capture.shutil.which', side_effect=lambda name: '/usr/bin/' + name), patch(
            'acprof.host.measurement.capture.os.geteuid', return_value=1000,
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
            "acprof.host.orchestrator.start_service_session",
            return_value=docker_runtime.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator.stop_service_session"
        ):
            with pytest.raises(packet_capture.PacketLatencyError) as raised:
                orchestrator.run_single_case(
                    cases.CaseRequest(
                        task_info=task_info,
                        cpu=1,
                        mem=4,
                        gpu='off',
                        image_info=runtime_images.ImageInfo(tag='acprof-test:latest'),
                        output_dir=self.output_dir,
                        project_dir='.',
                    ),
                    workload=cases.CaseWorkload(
                        warmup=0,
                        repeat=1,
                        repeat_in_window=1,
                    ),
                    plans=cases.CasePlans(
                        input_scales='64',
                    ),
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
                        True,
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
