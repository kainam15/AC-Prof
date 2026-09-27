"""Collector plumbing preserves Docker errors and absolute sampling cadence."""
import subprocess
import threading
import unittest
from unittest.mock import patch

from acprof.monitors.common import docker_container_pid, sample_periodically
from acprof.monitors.perf_mips import MIPSProfilingError


class MonitorCommonTests(unittest.TestCase):
    def test_pid_lookup_preserves_error_type_and_rejects_invalid_or_stopped_pid(self):
        for code, output, error in ((1, '', 'permission denied'), (0, 'not-a-pid', ''), (0, '0', '')):
            for error_type in (RuntimeError, MIPSProfilingError):
                with self.subTest(output=output, error=error, error_type=error_type), patch(
                    'acprof.monitors.common.subprocess.run',
                    return_value=subprocess.CompletedProcess([], code, output, error),
                ), self.assertRaises(error_type):
                    docker_container_pid('owned-container', error_type=error_type)

    def test_sampling_uses_absolute_deadlines_and_does_not_drift_with_callback_cost(self):
        stop = threading.Event()
        samples = []

        def sample(timestamp):
            samples.append(timestamp)
            if len(samples) == 2:
                stop.set()

        with patch('acprof.monitors.common.time.perf_counter', side_effect=[10.0, 11.0, 11.4, 12.0]), patch.object(
            stop, 'wait', return_value=False,
        ) as wait:
            sample_periodically(stop, 10.0, 1.0, sample)
        self.assertEqual(samples, [11.0, 12.0])
        self.assertEqual(len(wait.call_args_list), 2)
        self.assertAlmostEqual(wait.call_args_list[1].args[0], 0.6)

    def test_stopping_during_wait_does_not_append_an_extra_sample(self):
        stop = threading.Event()
        samples = []
        with patch('acprof.monitors.common.time.perf_counter', return_value=10.0), patch.object(
            stop, 'wait', return_value=True,
        ):
            sample_periodically(stop, 10.0, 1.0, samples.append)
        self.assertEqual(samples, [])
