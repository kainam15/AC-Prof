"""Exercise source state -> device restoration -> launch -> requests -> saved report."""
import hashlib
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import measure_overhead


class OverheadEntrypointTests(unittest.TestCase):
    def run_case(self, fail=False):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            plan = source / "input_scale_plan.json"
            plan.write_text(json.dumps({"schema_version": 2, "entries": [{"input_scale": 2, "payload": {}}]}))
            options = {"gpus": "on", "cpus": "2", "mems": "4", "gpu_device": "GPU-recorded",
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
            stack.enter_context(patch("acprof.host.docker_runtime.require_image_identity"))
            session = SimpleNamespace(name="owned", base_url="http://fixture.invalid",
                                      gpu_device={"uuid": "GPU-recorded", "index": 3})
            def launch(*_args, **kwargs):
                self.assertEqual(kwargs, {"cpuset_cpus": "1-2", "request_timeout_seconds": 17})
                self.assertEqual(os.environ["ACPROF_GPU_DEVICE"], "GPU-recorded")
                self.assertEqual(os.environ["ACPROF_RUNTIME_THREADS"], "2")
                return session
            start = stack.enter_context(patch("acprof.host.docker_runtime._start_container_session", side_effect=launch))
            stop = stack.enter_context(patch("acprof.host.docker_runtime._stop_container_session"))
            stack.enter_context(patch("acprof.host.hardware_conditions.observe_conditions",
                                      return_value={"cpu_affinity": ["1-2"], "errors": []}))
            monitor = Mock()
            monitor.stop.return_value = (None, "", [1, 2])
            stack.enter_context(patch("acprof.monitors.resource_usage.ResourceUsageMonitor", return_value=monitor))
            response = Mock(status_code=200)
            response.json.return_value = {}
            post = stack.enter_context(patch("requests.post", side_effect=RuntimeError("fixture request failure") if fail else None,
                                             return_value=response))
            arguments = [str(source), "--gpu", "on", "--rounds", "3", "--requests", "2", "--modes", "none,basic",
                         "--sample-hz", "20", "--output-dir", str(output)]
            if fail:
                with self.assertRaisesRegex(RuntimeError, "fixture request failure"):
                    measure_overhead.main(arguments)
            else:
                self.assertEqual(measure_overhead.main(arguments), 0)
            start.assert_called_once()
            stop.assert_called_once_with(session, "[overhead]")
            query.assert_called_once_with("GPU-recorded")
            self.assertEqual(post.call_args.kwargs["timeout"], 17)
            self.assertEqual(os.environ["ACPROF_GPU_DEVICE"], "GPU-caller")
            report = json.loads((output / "overhead.json").read_text())
            self.assertEqual(report["successful"], not fail)
            self.assertEqual(report["cpuset_cpus"], "1-2")
            self.assertEqual(report["gpu_device_uuid"], "GPU-recorded")
            if not fail:
                self.assertEqual(len(report["rounds"]), 6)

    def test_entrypoint_restores_constraints_and_saves_successful_report(self):
        self.run_case()

    def test_entrypoint_request_failure_cleans_container_and_saves_failure(self):
        self.run_case(fail=True)
