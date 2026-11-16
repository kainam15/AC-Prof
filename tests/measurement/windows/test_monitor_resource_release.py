"""Cleanup must quiesce sampling before releasing its NVML reference."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.monitors import energy_nvml, process_usage, resource_usage
from acprof.monitors.process_usage import ProcessUsageMonitor


class SamplingThread:
    def __init__(self, *, stuck=False):
        self.alive = True
        self.stuck = stuck
        self.timeouts = []

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.timeouts.append(timeout)
        if not self.stuck:
            self.alive = False


@pytest.fixture(params=['resource', 'energy'])
def gpu_monitor(request, monkeypatch):
    nvml = Mock()
    nvml.nvmlDeviceGetName.return_value = 'fixture GPU'
    if request.param == 'resource':
        monkeypatch.setattr(resource_usage, 'pynvml', nvml)
        monkeypatch.setattr(resource_usage.resource_readers, 'resolve_container_metric_readers',
                            Mock(side_effect=RuntimeError('no fixture cgroup')))
        monkeypatch.setattr(resource_usage.resource_readers, 'prepare_cpu_frequency_reader',
                            lambda *_args: lambda: (None, None))
        monitor = resource_usage.ResourceUsageMonitor(use_gpu=True)
    else:
        monkeypatch.setattr(energy_nvml, 'pynvml', nvml)
        monitor = energy_nvml.GPUEnergyMonitor()
    return SimpleNamespace(monitor=monitor, nvml=nvml, kind=request.param)


@pytest.mark.parametrize('error_type', [OSError, KeyboardInterrupt])
def test_close_releases_after_stop_boundary_failure(gpu_monitor, monkeypatch, error_type):
    monitor, nvml = gpu_monitor.monitor, gpu_monitor.nvml
    monitor._thread = thread = SamplingThread()
    monitor._t_start = 1.0
    boundary = '_read_window_counter_snapshots' if gpu_monitor.kind == 'resource' else '_read_energy_mj'
    monitor._energy_start_mj = 1
    error = error_type('boundary read failed')
    monkeypatch.setattr(monitor, boundary, Mock(side_effect=error))
    with pytest.raises(error_type) as caught:
        monitor.stop()
    assert caught.value is error

    close_error = None
    try:
        monitor.close()
    except BaseException as exc:
        close_error = exc
    assert close_error is None
    monitor.close()
    assert monitor._stop_event.is_set()
    assert not thread.is_alive()
    assert thread.timeouts == [1.0]
    nvml.nvmlShutdown.assert_called_once_with()


def test_close_keeps_nvml_owned_until_sampling_stops(gpu_monitor):
    monitor, nvml = gpu_monitor.monitor, gpu_monitor.nvml
    monitor._thread = thread = SamplingThread(stuck=True)
    with pytest.raises(RuntimeError, match='sampling thread did not stop'):
        monitor.close()
    assert thread.timeouts == [1.0]
    nvml.nvmlShutdown.assert_not_called()
    thread.stuck = False
    monitor.close()
    monitor.close()
    nvml.nvmlShutdown.assert_called_once_with()


def test_nvml_release_failure_is_reported_and_can_be_retried(gpu_monitor):
    monitor, nvml = gpu_monitor.monitor, gpu_monitor.nvml
    nvml.nvmlShutdown.side_effect = [OSError('shutdown failed'), None]
    with pytest.raises(OSError, match='shutdown failed'):
        monitor.close()
    monitor.close()
    monitor.close()
    assert nvml.nvmlShutdown.call_count == 2


@pytest.mark.parametrize('failure_stage', ['init', 'handle'])
def test_energy_monitor_releases_only_its_initialized_reference(monkeypatch, failure_stage):
    nvml = Mock()
    operation = nvml.nvmlInit if failure_stage == 'init' else nvml.nvmlDeviceGetHandleByIndex
    operation.side_effect = RuntimeError('NVML unavailable')
    monkeypatch.setattr(energy_nvml, 'pynvml', nvml)
    monitor = energy_nvml.GPUEnergyMonitor()
    monitor.close()
    monitor.close()
    assert nvml.nvmlShutdown.call_count == (0 if failure_stage == 'init' else 1)


@pytest.mark.parametrize('operation', ['stop', 'close'])
def test_process_sampler_shutdown_is_bounded_and_retryable(operation):
    monitor = ProcessUsageMonitor(pid=1, start_ticks=1)
    monitor._thread = thread = SamplingThread(stuck=True)
    monitor.nvml = nvml = Mock()
    with pytest.raises(RuntimeError, match='sampling thread did not stop'):
        getattr(monitor, operation)()
    assert thread.timeouts == [1.0]
    nvml.nvmlShutdown.assert_not_called()
    thread.stuck = False
    monitor.close()
    monitor.close()
    nvml.nvmlShutdown.assert_called_once_with()


def test_process_sampler_cannot_overwrite_an_active_thread(monkeypatch):
    monitor = ProcessUsageMonitor(pid=1, start_ticks=1)
    monitor._thread = thread = SamplingThread(stuck=True)
    monkeypatch.setattr(process_usage.threading, 'Thread', Mock(side_effect=AssertionError('second sampler spawned')))
    with pytest.raises(RuntimeError, match='already running'):
        monitor.start()
    assert monitor._thread is thread
