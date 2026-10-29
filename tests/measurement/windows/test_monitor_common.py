"""Collector plumbing preserves Docker errors and absolute sampling cadence."""
import subprocess
import threading
from unittest.mock import patch

import pytest

from acprof.host.command import DOCKER_INSPECT_TIMEOUT_S
from acprof.monitors.common import docker_container_pid, sample_periodically
from acprof.monitors.perf_mips import MIPSProfilingError
from acprof.monitors.sampling import SamplingDiagnostics


@pytest.mark.parametrize('error_type_case', range(2), ids=['RuntimeError', 'MIPSProfilingError'])
@pytest.mark.parametrize(('code', 'output', 'error'), [(1, '', 'permission denied'), (0, 'not-a-pid', ''), (0, '0', '')])
def test_pid_lookup_preserves_error_type_and_rejects_invalid_or_stopped_pid(error_type_case, code, output, error):
    error_type = tuple((RuntimeError, MIPSProfilingError))[error_type_case]
    with patch(
        'acprof.monitors.common.run_command',
        return_value=subprocess.CompletedProcess([], code, output, error),
    ), pytest.raises(error_type):
        docker_container_pid('owned-container', error_type=error_type)


def test_pid_lookup_uses_bounded_docker_inspect_timeout():
    with patch(
        'acprof.monitors.common.run_command',
        return_value=subprocess.CompletedProcess([], 0, '1234', ''),
    ) as run:
        assert docker_container_pid('owned-container') == 1234
    assert run.call_args.kwargs['timeout'] == DOCKER_INSPECT_TIMEOUT_S


def test_pid_lookup_propagates_timeout_without_fabricating_a_pid():
    with patch(
        'acprof.monitors.common.run_command',
        side_effect=subprocess.TimeoutExpired(['docker', 'inspect'], DOCKER_INSPECT_TIMEOUT_S),
    ), pytest.raises(subprocess.TimeoutExpired):
        docker_container_pid('owned-container')

@pytest.fixture
def sampling_clock(monkeypatch):
    """Advance time through waits and callback work, without sleeping."""
    stop = threading.Event()

    class Clock:
        def __init__(self):
            self.now = 10.0
            self.waits = []

        def perf_counter(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

        def wait(self, seconds):
            self.waits.append(seconds)
            self.advance(seconds)
            if len(self.waits) > 20:
                raise AssertionError("sampling did not stop within the test budget")
            return stop.is_set()

    clock = Clock()
    monkeypatch.setattr("acprof.monitors.common.time.perf_counter", clock.perf_counter)
    monkeypatch.setattr(stop, "wait", clock.wait)
    return clock, stop


def test_sampling_preserves_first_interval_and_absolute_phase(sampling_clock):
    clock, stop = sampling_clock
    samples = []

    def sample(timestamp):
        samples.append(timestamp - 10.0)
        clock.advance(0.02)
        if len(samples) == 4:
            stop.set()

    sample_periodically(stop, 10.0, 0.05, sample)

    assert samples == pytest.approx([0.05, 0.10, 0.15, 0.20])
    assert clock.waits == pytest.approx([0.05, 0.03, 0.03, 0.03])


def test_skip_preserves_phase_and_records_expired_deadlines(sampling_clock):
    clock, stop = sampling_clock
    samples = []
    diagnostics = SamplingDiagnostics("skip", 0.05)

    def sample(timestamp):
        samples.append(timestamp - 10.0)
        clock.advance(0.12 if len(samples) == 1 else 0.005)
        if len(samples) == 3:
            stop.set()

    sample_periodically(stop, 10.0, 0.05, sample, policy="skip", diagnostics=diagnostics)
    assert samples == pytest.approx([0.05, 0.20, 0.25])
    assert diagnostics.callback_count == 3
    assert diagnostics.skipped_deadlines == 2
    assert diagnostics.overrun_count == 1
    assert diagnostics.total_callback_s == pytest.approx(0.13)
    assert diagnostics.max_callback_s == pytest.approx(0.12)
    assert diagnostics.snapshot()["protocol_version"] == "periodic-v1"


def test_burst_diagnostics_preserve_catchup_and_measure_lateness(sampling_clock):
    clock, stop = sampling_clock
    diagnostics = SamplingDiagnostics("burst", 0.05)
    samples = []

    def sample(timestamp):
        samples.append(timestamp - 10.0)
        clock.advance(0.12 if len(samples) == 1 else 0.005)
        if len(samples) == 3:
            stop.set()

    sample_periodically(stop, 10.0, 0.05, sample, diagnostics=diagnostics)
    assert samples == pytest.approx([0.05, 0.17, 0.175])
    assert diagnostics.skipped_deadlines == 0
    assert diagnostics.max_lateness_s == pytest.approx(0.07)
    assert diagnostics.total_lateness_s == pytest.approx(0.095)


def test_diagnostics_retain_callback_failure_and_propagate_it(sampling_clock):
    clock, stop = sampling_clock
    diagnostics = SamplingDiagnostics("skip", 0.05)

    def sample(timestamp):
        clock.advance(0.02)
        raise LookupError("sensor failed")

    with pytest.raises(LookupError, match="sensor failed"):
        sample_periodically(stop, 10.0, 0.05, sample, policy="skip", diagnostics=diagnostics)
    assert diagnostics.callback_errors == 1
    assert diagnostics.total_callback_s == pytest.approx(0.02)


@pytest.mark.parametrize(
    ('callback_costs', 'expected_starts', 'expected_waits'),
    [
        pytest.param(
            [0.120, 0.005, 0.005, 0.005, 0.005, 0.005],
            [0.050, 0.170, 0.175, 0.200, 0.250, 0.300],
            [0.050, 0.020, 0.045, 0.045],
            id="transient-overrun-catches-up",
        ),
        pytest.param(
            [0.080] * 6,
            [0.050, 0.130, 0.210, 0.290, 0.370, 0.450],
            [0.050],
            id="sustained-overrun-keeps-running-without-waits",
        ),
    ],
)
def test_sampling_preserves_burst_policy_after_callback_overrun(
    sampling_clock, callback_costs, expected_starts, expected_waits
):
    clock, stop = sampling_clock
    samples = []

    def sample(timestamp):
        samples.append(timestamp - 10.0)
        clock.advance(callback_costs[len(samples) - 1])
        if len(samples) == len(callback_costs):
            stop.set()

    sample_periodically(stop, 10.0, 0.05, sample)

    assert samples == pytest.approx(expected_starts)
    assert clock.waits == pytest.approx(expected_waits)


def test_sampling_catches_up_after_scheduler_delay(sampling_clock, monkeypatch):
    clock, stop = sampling_clock
    samples = []

    def delayed_wait(seconds):
        stopped = clock.wait(seconds)
        if len(clock.waits) == 1:
            clock.advance(0.07)
        return stopped

    monkeypatch.setattr(stop, "wait", delayed_wait)

    def sample(timestamp):
        samples.append(timestamp - 10.0)
        clock.advance(0.005)
        if len(samples) == 3:
            stop.set()

    sample_periodically(stop, 10.0, 0.05, sample)

    assert samples == pytest.approx([0.120, 0.125, 0.150])
    assert clock.waits == pytest.approx([0.050, 0.020])


def test_sampling_stopped_before_start_does_not_wait_or_sample(sampling_clock):
    clock, stop = sampling_clock
    samples = []
    stop.set()

    sample_periodically(stop, 10.0, 0.05, samples.append)

    assert samples == []
    assert clock.waits == []


def test_stopping_during_wait_does_not_append_an_extra_sample(sampling_clock, monkeypatch):
    clock, stop = sampling_clock
    samples = []

    def stopping_wait(seconds):
        clock.wait(seconds)
        stop.set()
        return True

    monkeypatch.setattr(stop, "wait", stopping_wait)

    sample_periodically(stop, 10.0, 0.05, samples.append)

    assert samples == []
    assert clock.waits == pytest.approx([0.05])


def test_stopping_in_overrunning_callback_prevents_catch_up(sampling_clock):
    clock, stop = sampling_clock
    samples = []

    def sample(timestamp):
        samples.append(timestamp - 10.0)
        clock.advance(0.12)
        stop.set()

    sample_periodically(stop, 10.0, 0.05, sample)

    assert samples == pytest.approx([0.05])
    assert clock.waits == pytest.approx([0.05])


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(RuntimeError("collector failed"), id="collector-error"),
        pytest.param(KeyboardInterrupt(), id="cancellation"),
    ],
)
def test_sampling_propagates_callback_failure_without_retry(sampling_clock, error):
    clock, stop = sampling_clock
    samples = []

    def sample(timestamp):
        samples.append(timestamp - 10.0)
        clock.advance(0.12)
        raise error

    with pytest.raises(type(error)) as caught:
        sample_periodically(stop, 10.0, 0.05, sample)

    assert caught.value is error
    assert samples == pytest.approx([0.05])
    assert clock.waits == pytest.approx([0.05])


def test_sampling_without_start_timestamp_does_not_publish_samples(sampling_clock, monkeypatch):
    clock, stop = sampling_clock
    samples = []

    def bounded_wait(seconds):
        clock.wait(seconds)
        if len(clock.waits) == 2:
            stop.set()
        return stop.is_set()

    monkeypatch.setattr(stop, "wait", bounded_wait)

    sample_periodically(stop, None, 0.05, samples.append)

    assert samples == []
    assert clock.waits == pytest.approx([0.05, 0.05])
