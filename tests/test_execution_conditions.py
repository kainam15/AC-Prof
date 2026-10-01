"""Recorded device constraints must win over caller environment and pinned scopes."""
import os
import unittest
from unittest.mock import patch

from acprof.host.execution_conditions import ExecutionConditions
from acprof.host.gpu_device import gpu_device_scope, pin_gpu_device, resolve_gpu_device


class ExecutionConditionsTests(unittest.TestCase):
    def options(self, **changes):
        return {"gpus": "off,on", "gpu_device": "GPU-recorded", "cpuset_cpus": "3,1-2",
                "request_timeout_seconds": 17, "measurement_environment": {"ACPROF_RUNTIME_THREADS": "2"},
                **changes}

    def test_uuid_affinity_threads_timeout_and_caller_state(self):
        def query(selector):
            return {"uuid": selector, "index": 4 if selector == "GPU-recorded" else 1}

        with patch.dict(os.environ, {"ACPROF_GPU_DEVICE": "GPU-caller", "DEVICE_INDEX": "1",
                                     "ACPROF_RUNTIME_THREADS": "8"}), gpu_device_scope(), \
                patch("acprof.host.gpu_device.resolve_gpu_device", side_effect=query), \
                patch("os.sched_getaffinity", return_value={1, 2, 3, 4}):
            pin_gpu_device("GPU-caller")
            conditions = ExecutionConditions.from_options(self.options(), gpu="on")
            with conditions.activate() as device:
                self.assertEqual(resolve_gpu_device()["uuid"], "GPU-recorded")
                self.assertEqual(device["index"], 4)
                self.assertEqual(os.environ["DEVICE_INDEX"], "4")
                self.assertEqual(os.environ["ACPROF_RUNTIME_THREADS"], "2")
                self.assertEqual(conditions.container_options, {"cpuset_cpus": "1-3", "request_timeout_seconds": 17})
            self.assertEqual(resolve_gpu_device()["uuid"], "GPU-caller")
            self.assertEqual(os.environ["ACPROF_GPU_DEVICE"], "GPU-caller")
            self.assertEqual(os.environ["ACPROF_RUNTIME_THREADS"], "8")

    def test_unavailable_cpu_set_rejects_before_gpu_query(self):
        conditions = ExecutionConditions.from_options(self.options(), gpu="on")
        with patch("os.sched_getaffinity", return_value={1}), patch("acprof.host.execution_conditions.pin_gpu_device") as gpu:
            with self.assertRaisesRegex(ValueError, "affinity"):
                with conditions.activate():
                    self.fail("must not execute")
            gpu.assert_not_called()

    def test_missing_uuid_and_invalid_timeout_are_not_reconstructed(self):
        for changes in ({"gpu_device": "1"}, {"gpu_device": ""}, {"request_timeout_seconds": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ExecutionConditions.from_options(self.options(**changes), gpu="on")

    def test_missing_device_restores_environment(self):
        conditions = ExecutionConditions.from_options(self.options(cpuset_cpus=""), gpu="on")
        with patch.dict(os.environ, {"ACPROF_GPU_DEVICE": "GPU-caller"}), \
                patch("acprof.host.execution_conditions.pin_gpu_device", side_effect=RuntimeError("device unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                with conditions.activate():
                    self.fail("must not execute")
            self.assertEqual(os.environ["ACPROF_GPU_DEVICE"], "GPU-caller")
