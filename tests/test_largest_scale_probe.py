import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from acprof.cli.probe import main as probe_main
from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import RunningContainer
from acprof.host.input_plan import PlannedInputScales
from acprof.host.largest_scale_probe import (
    PROBE_SUMMARY_NAME,
    load_largest_scale_entry,
    run_largest_scale_probe,
    select_minimum_resources,
    write_probe_summary,
)
from acprof.host.orchestrator import ImageInfo


def _task_info() -> TaskInfo:
    return TaskInfo(
        model_id="demo/model",
        pipeline_tag="fill-mask",
        task_family="nlp",
        runtime_backend="transformers_pipeline",
        library_name="transformers",
        model_revision="1" * 40,
        detection_method="manual",
    )


def _write_plan(root: Path) -> Path:
    path = root / "input_scale_plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "entries": [
                    {
                        "input_scale": 64,
                        "scale_label": "tokens",
                        "payload": {"text": "small"},
                    },
                    {
                        "input_scale": 512,
                        "scale_label": "tokens",
                        "payload": {"text": "largest"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_minimum_resource_selection_prefers_cpu_only() -> None:
    selected = select_minimum_resources(
        [8, 1, 4],
        [16, 2, 8],
        ["on", "off"],
    )
    assert (selected) == ((1, 2, "off"))
    assert (select_minimum_resources([2], [8], ["on"])) == ((2, 8, "on"))

def test_largest_materialized_scale_is_selected() -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        plan = _write_plan(Path(temporary_dir))

        entry = load_largest_scale_entry(plan)

    assert (entry["input_scale"]) == (512.0)
    assert (entry["payload"]) == ({"text": "largest"})


@pytest.mark.parametrize("input_scale", (True, "512"))
def test_largest_scale_plan_rejects_non_numeric_scales(input_scale) -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "input_scale_plan.json"
        path.write_text(json.dumps({
            "schema_version": 2,
            "entries": [{"input_scale": input_scale, "payload": {"text": "bad"}}],
        }), encoding="utf-8")

        with pytest.raises(RuntimeError, match="input_scale"):
            load_largest_scale_entry(path)


def test_largest_scale_plan_rejects_unknown_schema_before_probe() -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "input_scale_plan.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "entries": [{"input_scale": 512, "payload": {"text": "old"}}],
        }), encoding="utf-8")

        with pytest.raises(RuntimeError, match="schema_version"):
            load_largest_scale_entry(path)


def test_largest_scale_plan_read_is_bounded() -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "input_scale_plan.json"
        path.write_text(json.dumps({
            "schema_version": 2,
            "entries": [{"input_scale": 512, "payload": {"text": "ok"}}],
            "padding": "x" * (4 * 1024 * 1024),
        }), encoding="utf-8")

        with pytest.raises(RuntimeError, match="4 MiB"):
            load_largest_scale_entry(path)


def test_probe_summary_write_syncs_file_and_directory() -> None:
    real_fsync = os.fsync
    fsync_calls: list[int] = []

    def record_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    with tempfile.TemporaryDirectory() as temporary_dir, patch(
        "acprof.artifacts.os.fsync",
        side_effect=record_fsync,
    ):
        path = Path(temporary_dir) / PROBE_SUMMARY_NAME
        write_probe_summary(path, {"schema_version": 3, "status": "ok"})

    assert len(fsync_calls) == 2

@patch("acprof.host.largest_scale_probe.stop_container_session")
@patch("acprof.host.largest_scale_probe.start_container_session")
@patch("acprof.host.largest_scale_probe.requests.post")
def test_probe_times_exactly_one_largest_request_and_writes_summary(
    post: Mock,
    start_container: Mock,
    stop_container: Mock,
) -> None:
    response = Mock()
    response.status_code = 200
    response.text = ""
    response.json.return_value = {
        "effective_input_scale": 512,
        "output_length": 1,
    }
    post.return_value = response
    start_container.return_value = RunningContainer(
        name="probe-container",
        base_url="http://127.0.0.1:8002",
        host_port=8002,
        cold_start_s=12.5,
        cold_start_container_launch_s=0.5,
        cold_start_server_setup_s=1.0,
        cold_start_cuda_init_s=0.0,
        cold_start_model_load_s=10.0,
        cold_start_ready_wait_s=1.0,
    )

    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        plan = _write_plan(output_dir)
        planned = PlannedInputScales(
            scales=[64.0, 512.0],
            source="manual",
            plan_file=str(plan),
        )

        summary = run_largest_scale_probe(
            task_info=_task_info(),
            image_info=ImageInfo(tag="acprof-nlp-demo--model:latest"),
            planned_input_scales=planned,
            cpu_list=[4, 1],
            mem_list=[8, 2],
            gpu_list=["on", "off"],
            batch_size=1,
            output_dir=output_dir,
        )
        saved_text = (output_dir / PROBE_SUMMARY_NAME).read_text(
            encoding="utf-8"
        )
        saved = json.loads(saved_text)

    assert (post.call_count) == (1)
    assert (post.call_args.args[0]) == ("http://127.0.0.1:8002/predict")
    assert (post.call_args.kwargs["json"]) == ({"text": "largest"})
    assert (post.call_args.kwargs["timeout"]) is None
    assert (summary["status"]) == ("ok")
    assert (summary["schema_version"]) == (3)
    assert (summary["resource"]["cpu_cores"]) == (1)
    assert (summary["resource"]["mem_gb"]) == (2)
    assert (summary["resource"]["gpu_mode"]) == ("off")
    assert (summary["input"]["planned_scale"]) == (512.0)
    assert (summary["input"]["effective_scale"]) == (512.0)
    assert (summary["input"]["batch_size"]) == (1)
    assert (summary["cold_start"]["total_s"]) == (12.5)
    assert (summary["timing"]["request_s"]) > (0.0)
    assert (summary["timing"]["ready_plus_request_s"]) > (12.5)
    assert (summary["timing"]["request_timeout_s"]) is None
    assert (start_container.call_args.kwargs["request_timeout_seconds"]) is None
    assert (saved["status"]) == ("ok")
    assert (saved["timing"]["request_timeout_s"]) is None
    assert ("NaN") not in (saved_text)
    stop_container.assert_called_once_with(
        start_container.return_value,
        log_prefix="[largest-probe]",
    )

@patch("acprof.host.largest_scale_probe.stop_container_session")
@patch("acprof.host.largest_scale_probe.start_container_session")
@patch("acprof.host.largest_scale_probe.requests.post")
def test_request_timeout_is_persisted_without_formal_csv(
    post: Mock,
    start_container: Mock,
    stop_container: Mock,
) -> None:
    post.side_effect = requests.Timeout("too slow")
    start_container.return_value = RunningContainer(
        name="probe-container",
        base_url="http://127.0.0.1:8002",
        host_port=8002,
        cold_start_s=3.0,
    )

    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        plan = _write_plan(output_dir)
        summary = run_largest_scale_probe(
            task_info=_task_info(),
            image_info=ImageInfo(tag="image:latest"),
            planned_input_scales=PlannedInputScales(
                scales=[64.0, 512.0],
                source="manual",
                plan_file=str(plan),
            ),
            cpu_list=[1],
            mem_list=[2],
            gpu_list=["off"],
            batch_size=1,
            output_dir=output_dir,
            timeout_seconds=0.01,
        )

        assert (summary["status"]) == ("timeout")
        assert ("超过 0.01 秒") in (summary["error"])
        assert (summary["timing"]["request_timeout_s"]) == (0.01)
        assert (start_container.call_args.kwargs["request_timeout_seconds"]) == (0.01)
        assert ((output_dir / PROBE_SUMMARY_NAME).is_file())
        assert not ((output_dir / "result_all.csv").exists())
    stop_container.assert_called_once()

@patch("acprof.host.largest_scale_probe.stop_container_session")
@patch("acprof.host.largest_scale_probe.start_container_session")
@patch("acprof.host.largest_scale_probe.requests.post")
def test_startup_oom_advances_to_first_viable_memory(
    post: Mock,
    start_container: Mock,
    stop_container: Mock,
) -> None:
    owned_session = RunningContainer(
        name="probe-4gb", base_url="http://127.0.0.1:8002",
        host_port=8002, cold_start_s=7.0,
    )
    start_container.side_effect = [RuntimeError("container_oom_killed during startup"), owned_session]
    response = Mock(status_code=200, text="")
    response.json.return_value = {"effective_input_scale": 512}
    post.return_value = response

    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        plan = _write_plan(output_dir)
        summary = run_largest_scale_probe(
            task_info=_task_info(),
            image_info=ImageInfo(tag="image:latest"),
            planned_input_scales=PlannedInputScales(
                scales=[64.0, 512.0],
                source="manual",
                plan_file=str(plan),
            ),
            cpu_list=[1, 4],
            mem_list=[8, 2, 4],
            gpu_list=["off", "on"],
            batch_size=1,
            output_dir=output_dir,
            timeout_seconds=30,
        )

    assert (summary["status"]) == ("ok")
    assert (summary["resource"]["mem_gb"]) == (4)
    assert (summary["memory_probe"]["minimum_viable_mem_gb"]) == (4)
    assert ([item["status"] for item in summary["memory_probe"]["attempts"]]) == (["startup_oom", "ok"])
    assert ([call.kwargs["mem"] for call in start_container.call_args_list]) == ([2, 4])
    assert (post.call_count) == (1)
    stop_container.assert_called_once_with(
        owned_session,
        log_prefix="[largest-probe]",
    )

@patch("acprof.host.largest_scale_probe.stop_container_session")
@patch("acprof.host.largest_scale_probe.start_container_session")
@patch("acprof.host.largest_scale_probe.requests.post")
def test_runtime_memory_oom_advances_and_successful_attempt_supplies_timing(
    post: Mock,
    start_container: Mock,
    stop_container: Mock,
) -> None:
    start_container.side_effect = [
        RunningContainer(
            name="probe-2gb",
            base_url="http://127.0.0.1:8002",
            host_port=8002,
            cold_start_s=3.0,
        ),
        RunningContainer(
            name="probe-4gb",
            base_url="http://127.0.0.1:8004",
            host_port=8004,
            cold_start_s=5.0,
        ),
    ]
    oom_response = Mock(
        status_code=500,
        text='{"error":"MemoryError: cannot allocate memory"}',
    )
    success_response = Mock(status_code=200, text="")
    success_response.json.return_value = {"effective_input_scale": 512}
    post.side_effect = [oom_response, success_response]

    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        plan = _write_plan(output_dir)
        summary = run_largest_scale_probe(
            task_info=_task_info(),
            image_info=ImageInfo(tag="image:latest"),
            planned_input_scales=PlannedInputScales(
                scales=[64.0, 512.0],
                source="manual",
                plan_file=str(plan),
            ),
            cpu_list=[1],
            mem_list=[2, 4, 8],
            gpu_list=["on"],
            batch_size=1,
            output_dir=output_dir,
            timeout_seconds=30,
        )

    assert (summary["resource"]["mem_gb"]) == (4)
    assert ([item["status"] for item in summary["memory_probe"]["attempts"]]) == (["runtime_oom", "ok"])
    assert (summary["cold_start"]["total_s"]) == (5.0)
    assert (post.call_count) == (2)
    assert (stop_container.call_count) == (2)

@patch("acprof.host.largest_scale_probe.stop_container_session")
@patch("acprof.host.largest_scale_probe.start_container_session")
@patch("acprof.host.largest_scale_probe.requests.post")
def test_cuda_oom_stops_host_memory_scan(
    post: Mock,
    start_container: Mock,
    stop_container: Mock,
) -> None:
    start_container.return_value = RunningContainer(
        name="probe-2gb",
        base_url="http://127.0.0.1:8002",
        host_port=8002,
        cold_start_s=3.0,
    )
    post.return_value = Mock(
        status_code=500,
        text='{"error":"CUDA out of memory"}',
    )

    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        plan = _write_plan(output_dir)
        summary = run_largest_scale_probe(
            task_info=_task_info(),
            image_info=ImageInfo(tag="image:latest"),
            planned_input_scales=PlannedInputScales(
                scales=[64.0, 512.0],
                source="manual",
                plan_file=str(plan),
            ),
            cpu_list=[1],
            mem_list=[2, 4, 8],
            gpu_list=["on"],
            batch_size=1,
            output_dir=output_dir,
            timeout_seconds=30,
        )

    assert (summary["status"]) == ("cuda_oom")
    assert (summary["resource"]["mem_gb"]) is None
    assert (len(summary["memory_probe"]["attempts"])) == (1)
    assert (start_container.call_count) == (1)
    assert (post.call_count) == (1)
    stop_container.assert_called_once()

@patch("acprof.host.largest_scale_probe.stop_container_session")
@patch("acprof.host.largest_scale_probe.start_container_session")
@patch("acprof.host.largest_scale_probe.requests.post")
def test_all_memory_candidates_oom_without_claiming_a_minimum(
    post: Mock,
    start_container: Mock,
    stop_container: Mock,
) -> None:
    start_container.side_effect = [
        RuntimeError("container_oom_killed during startup"),
        RuntimeError("container_oom_killed during startup"),
    ]

    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        plan = _write_plan(output_dir)
        summary = run_largest_scale_probe(
            task_info=_task_info(),
            image_info=ImageInfo(tag="image:latest"),
            planned_input_scales=PlannedInputScales(
                scales=[64.0, 512.0],
                source="manual",
                plan_file=str(plan),
            ),
            cpu_list=[1],
            mem_list=[2, 4],
            gpu_list=["off"],
            batch_size=1,
            output_dir=output_dir,
            timeout_seconds=30,
        )

    assert (summary["status"]) == ("oom")
    assert (summary["resource"]["mem_gb"]) is None
    assert (summary["memory_probe"]["minimum_viable_mem_gb"]) is None
    assert (summary["resource"]["last_attempt_mem_gb"]) == (4)
    assert ("2GB,4GB") in (summary["error"])
    post.assert_not_called()
    stop_container.assert_not_called()

@patch("acprof.cli.probe.require_cgroup_prerequisites", return_value="v2")
@patch("acprof.cli.probe.require_native_docker")
@patch("acprof.cli.probe.require_collection_host")
@patch("acprof.cli.probe.bootstrap_project_env")
@patch("acprof.host.detect.detect_task", return_value=_task_info())
@patch("acprof.cli.probe.plan_input_scales")
@patch("acprof.cli.probe.run_largest_scale_probe")
def test_cli_delegates_planning_and_probe_with_requested_matrix(
    run_probe: Mock,
    plan_scales: Mock,
    _detect_task: Mock,
    _bootstrap: Mock,
    _native_linux: Mock,
    _native_docker: Mock,
    _cgroup: Mock,
) -> None:
    def fake_run(**kwargs):
        summary_path = Path(kwargs["output_dir"]) / PROBE_SUMMARY_NAME
        return {
            "status": "ok",
            "timing": {},
            "artifacts": {"summary": str(summary_path)},
        }

    run_probe.side_effect = fake_run

    built_image = ImageInfo(tag="acprof-nlp-demo--model:latest")
    with tempfile.TemporaryDirectory() as temporary_dir, patch("acprof.host.command.run_command", return_value=Mock(returncode=1, stdout="", stderr="No such image"),
    ), patch("acprof.host.runtime_images.build_runtime_image", return_value=built_image,
    ) as build_image:
        plan_path = Path(temporary_dir) / "planned.json"
        plan_scales.return_value = PlannedInputScales(
            scales=[64.0, 512.0],
            source="manual",
            plan_file=str(plan_path),
        )

        returncode = probe_main(
            [
                "--model",
                "demo/model",
                "--revision", "a" * 40,
                "--cpus",
                "1,4",
                "--mems",
                "2,8",
                "--gpus",
                "off,on",
                "--batch-size",
                "3",
                "--input-scales",
                "64,512",
                "--output-dir",
                temporary_dir,
                "--skip-build",
            ]
        )
        assert (_detect_task.call_args.kwargs["revision"]) == ("a" * 40)
        summary_path = Path(run_probe.call_args.kwargs["output_dir"]) / PROBE_SUMMARY_NAME
        saved = json.loads(summary_path.read_text(encoding="utf-8"))

    assert (returncode) == (0)
    build_image.assert_called_once()
    expected_task = _task_info()
    expected_task.runtime_profile_id = "nlp-cu128"
    expected_task.model_resolution = {
        "schema_version": 1, "status": "candidate", "task": "fill-mask",
        "backend": "transformers_pipeline", "library": "transformers", "artifact_format": "transformers",
        "loader": "Transformers Auto/pipeline", "operation": "predict", "model_revision": "1" * 40,
        "metadata_files": [], "model_type": None, "runtime_profile": "nlp-cu128",
        "interface_kind": "standard", "pipeline_task": "fill-mask",
        "code_revision": None, "code_files": [], "model_spec": {},
        "model_id": "demo/model", "transformers_version": "4.57.6", "trust_remote_code": False,
        "requested_devices": ["off", "on"],
        "dependency_preflight": {
            "schema_version": 1, "revision": "1" * 40, "runtime_profile": "nlp-cu128",
            "lock": "dockerfiles/locks/nlp-cu128.txt", "source_sha256": {}, "dependencies": [],
        },
        "precision": {
            "off": {"dtype": "FP32", "torch_dtype": "float32", "supported_dtypes": ["FP32", "FP16"],
                    "device": "cpu", "runtime_profile": "nlp-cu128", "evidence": []},
            "on": {"dtype": "FP16", "torch_dtype": "float16", "supported_dtypes": ["FP32", "FP16"],
                   "device": "gpu", "runtime_profile": "nlp-cu128", "evidence": []},
        },
    }
    assert (build_image.call_args.args[0]) == (expected_task)
    assert (plan_scales.call_args.kwargs["image_info"]) is (built_image)
    assert (run_probe.call_args.kwargs["image_info"]) is (built_image)
    assert (plan_scales.call_args.kwargs["cpu_list"]) == ([1, 4])
    assert (plan_scales.call_args.kwargs["mem_list"]) == ([2, 8])
    assert (plan_scales.call_args.kwargs["gpu_list"]) == (["off", "on"])
    assert (run_probe.call_args.kwargs["batch_size"]) == (3)
    assert (run_probe.call_args.kwargs["timeout_seconds"]) is None
    assert (saved["timing"]["command_s"]) >= (0.0)
