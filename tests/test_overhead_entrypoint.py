"""Exercise source state -> device restoration -> launch -> requests -> saved report."""
import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts import measure_overhead


class TestOverheadEntrypoint:
    def run_case(self, fail=False, *, extra_args=(), expected_cpu=2, expected_mem=4,
                 expected_scale=2, invalid=False, window_failure=False, publication_failure=False):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            plan = source / "input_scale_plan.json"
            plan.write_text(json.dumps({"schema_version": 2, "entries": [
                {"input_scale": 2, "payload": {"scale": 2}},
                {"input_scale": 8, "payload": {"scale": 8}},
            ]}))
            options = {"gpus": "on", "cpus": "1,2", "mems": "2,4", "gpu_device": "GPU-recorded",
                       "cpuset_cpus": "1-2", "request_timeout_seconds": 17, "idle_seconds": 0,
                       "idle_cooldown_seconds": 0, "measurement_environment": {"ACPROF_RUNTIME_THREADS": "2"}}
            task = {"model_id": "fixture/model", "pipeline_tag": "tabular-regression", "task_family": "structured",
                    "runtime_backend": "onnxruntime", "library_name": "onnxruntime", "model_revision": "pinned",
                    "detection_method": "manual"}
            (source / "run_state.json").write_text(json.dumps({
                "schema_version": 1, "status": "complete", "run_id": "source-id", "options": options,
                "runtime": {"task": task, "image": {"tag": "sha256:fixture"}},
                "artifacts": {plan.name: hashlib.sha256(plan.read_bytes()).hexdigest()}}))
            stack.enter_context(patch.dict(os.environ, {"ACPROF_GPU_DEVICE": "GPU-caller", "DEVICE_INDEX": "1"}))
            stack.enter_context(patch("os.sched_getaffinity", return_value={1, 2, 3}))
            query = stack.enter_context(patch("acprof.host.gpu_device.resolve_gpu_device",
                                               return_value={"uuid": "GPU-recorded", "index": 3}))
            stack.enter_context(patch("acprof.host.env_utils.bootstrap_project_env"))
            stack.enter_context(patch("acprof.host.run_state.host_identity", return_value={}))
            stack.enter_context(patch("acprof.host.runtime_images.require_image_identity"))
            session = SimpleNamespace(name="owned", base_url="http://fixture.invalid",
                                      gpu_device={"uuid": "GPU-recorded", "index": 3})
            def launch(*_args, **kwargs):
                assert _args[1:3] == (expected_cpu, expected_mem)
                assert (kwargs) == ({"cpuset_cpus": "1-2", "request_timeout_seconds": 17})
                assert (os.environ["ACPROF_GPU_DEVICE"]) == ("GPU-recorded")
                assert (os.environ["ACPROF_RUNTIME_THREADS"]) == ("2")
                return session
            start = stack.enter_context(patch("acprof.host.docker_runtime.start_container_session", side_effect=launch))
            stop = stack.enter_context(patch("acprof.host.docker_runtime.stop_container_session"))
            stack.enter_context(patch("acprof.host.hardware_conditions.observe_conditions",
                                      return_value={"cpu_affinity": ["1-2"], "errors": []}))
            monitor = Mock()
            monitor.stop.return_value = (None, "", [1, 2])
            monitor.sampling_boundary_snapshot.return_value = None
            stack.enter_context(patch("acprof.monitors.resource_usage.ResourceUsageMonitor", return_value=monitor))
            response = Mock(status_code=200)
            response.json.return_value = {}
            failures = [*[response] * 5, RuntimeError("fixture request failure")] if window_failure else (
                RuntimeError("fixture request failure") if fail else None)
            post = stack.enter_context(patch("requests.post", side_effect=failures, return_value=response))
            if publication_failure:
                stack.enter_context(patch("scripts.measure_overhead.atomic_write_json",
                                          side_effect=OSError("fixture disk full")))
            arguments = [str(source), "--gpu", "on", "--rounds", "3", "--requests", "2", "--modes", "none,basic",
                         "--sample-hz", "20", "--output-dir", str(output), *extra_args]
            if invalid:
                with pytest.raises(SystemExit) as failure:
                    measure_overhead.main(arguments)
                assert failure.value.code == 2
                start.assert_not_called()
                stop.assert_not_called()
                post.assert_not_called()
                assert not (output / "overhead.json").exists()
                return
            if fail or window_failure:
                with pytest.raises(RuntimeError, match="fixture request failure"):
                    measure_overhead.main(arguments)
            else:
                assert (measure_overhead.main(arguments)) == (0)
            start.assert_called_once()
            stop.assert_called_once_with(session, "[overhead]")
            query.assert_called_once_with("GPU-recorded")
            assert (post.call_args.kwargs["timeout"]) == (17)
            assert (os.environ["ACPROF_GPU_DEVICE"]) == ("GPU-caller")
            if publication_failure:
                assert not (output / "overhead.json").exists()
                assert not list(output.glob("*.boundaries.json"))
                return
            report = json.loads((output / "overhead.json").read_text())
            assert (report["successful"]) == (not (fail or window_failure))
            assert (report["cpuset_cpus"]) == ("1-2")
            assert (report["gpu_device_uuid"]) == ("GPU-recorded")
            assert (report["cpu_cores"], report["mem_cap_gb"], report["input_scale"]) == (
                expected_cpu, expected_mem, expected_scale)
            assert post.call_args.kwargs["json"] == {"scale": expected_scale}
            assert report["payload_sha256"] == hashlib.sha256(
                json.dumps({"scale": expected_scale}, sort_keys=True).encode()).hexdigest()
            if not fail and not window_failure:
                assert (len(report["rounds"])) == (6)
            sidecars = sorted(output.glob("*.boundaries.json"))
            if "--window-boundaries" not in extra_args:
                assert sidecars == []
                assert "window_boundaries" not in report
            else:
                assert report["window_boundaries"]["enabled"] is True
                assert len(sidecars) == (1 if window_failure else 6)
                for path in sidecars:
                    diagnostic = json.loads(path.read_text())
                    assert path.name == diagnostic["window_id"] + ".boundaries.json"
                    assert diagnostic["successful"] == (not window_failure)
                    assert diagnostic["request_window"]["completed_requests"] == (0 if window_failure else 2)
                if not window_failure:
                    assert {row["window_boundary_file"] for row in report["rounds"]} == {
                        path.name for path in sidecars}

    def test_entrypoint_restores_constraints_and_saves_successful_report(self):
        self.run_case()

    def test_entrypoint_request_failure_cleans_container_and_saves_failure(self):
        self.run_case(fail=True)

    def test_boundary_flag_writes_one_sidecar_per_comparison_window(self):
        self.run_case(extra_args=("--window-boundaries",))

    def test_boundary_flag_preserves_failed_window_and_container_cleanup(self):
        self.run_case(extra_args=("--window-boundaries",), window_failure=True)

    @pytest.mark.parametrize("window_failure", [False, True])
    def test_final_report_failure_preserves_workload_error(self, capsys, window_failure):
        self.run_case(fail=not window_failure, window_failure=window_failure, publication_failure=True,
                      extra_args=("--window-boundaries",))
        assert "final report failed: fixture disk full" in capsys.readouterr().err

    @pytest.mark.parametrize(("arguments", "cpu", "mem", "scale"), [
        (("--cpu", "1"), 1, 4, 2),
        (("--mem", "2"), 2, 2, 2),
        (("--input-scale", "8"), 2, 4, 8),
        (("--cpu", "1", "--mem", "2", "--input-scale", "8"), 1, 2, 8),
    ])
    def test_selected_source_coordinates_reach_launch_payload_and_report(self, arguments, cpu, mem, scale):
        self.run_case(extra_args=arguments, expected_cpu=cpu, expected_mem=mem, expected_scale=scale)

    @pytest.mark.parametrize("arguments", [
        ("--cpu", "0"), ("--cpu", "3"), ("--mem", "0"), ("--mem", "8"),
        ("--input-scale", "3"), ("--input-scale", "nan"), ("--input-scale", "inf"),
    ])
    def test_invalid_source_coordinates_fail_before_runtime(self, arguments):
        self.run_case(extra_args=arguments, invalid=True)
