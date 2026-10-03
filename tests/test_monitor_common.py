"""Collector plumbing preserves Docker errors and absolute sampling cadence."""
import subprocess
import threading
from unittest.mock import patch

import pytest

from acprof.monitors.common import docker_container_pid, sample_periodically
from acprof.monitors.perf_mips import MIPSProfilingError


@pytest.mark.parametrize('error_type_case', range(2), ids=['RuntimeError', 'MIPSProfilingError'])
@pytest.mark.parametrize('code,output,error', ((1, '', 'permission denied'), (0, 'not-a-pid', ''), (0, '0', '')))
def test_pid_lookup_preserves_error_type_and_rejects_invalid_or_stopped_pid(error_type_case, code, output, error):
    error_type = tuple((RuntimeError, MIPSProfilingError))[error_type_case]
    with patch(
        'acprof.monitors.common.run_command',
        return_value=subprocess.CompletedProcess([], code, output, error),
    ), pytest.raises(error_type):
        docker_container_pid('owned-container', error_type=error_type)

def test_sampling_uses_absolute_deadlines_and_does_not_drift_with_callback_cost():
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
    assert (samples) == ([11.0, 12.0])
    assert (len(wait.call_args_list)) == (2)
    assert (wait.call_args_list[1].args[0]) == (0.6) or round(abs((wait.call_args_list[1].args[0]) - (0.6)), 7) == 0

def test_stopping_during_wait_does_not_append_an_extra_sample():
    stop = threading.Event()
    samples = []
    with patch('acprof.monitors.common.time.perf_counter', return_value=10.0), patch.object(
        stop, 'wait', return_value=True,
    ):
        sample_periodically(stop, 10.0, 1.0, samples.append)
    assert (samples) == ([])
