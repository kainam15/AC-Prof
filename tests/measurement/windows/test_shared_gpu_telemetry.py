from types import SimpleNamespace

from acprof.monitors.process_usage import ProcessUsageMonitor, reduce_process_samples


def test_process_reuses_gpu_aggregation_for_state_and_memory_ratio():
    rows = [dict(timestamp=t, members={}, gpu_util_pct=util, gpu_mem_bytes=used,
                 gpu_mem_total_bytes=1000, gpu_sm_clock_mhz=sm, gpu_memory_clock_mhz=5000,
                 gpu_temp_c=temp, gpu_pstate=state)
            for t, util, used, sm, temp, state in [(1, 10, 100, 1000, 50, 'P2'), (2, 30, 300, 1200, 60, 'P0')]]
    result = reduce_process_samples(rows)
    assert result['gpu_mem_util_avg_pct'] == 20
    assert result['gpu_mem_util_peak_pct'] == 30
    assert result['gpu_sm_clock_mhz'] == 1100
    assert result['gpu_memory_clock_mhz'] == 5000
    assert result['gpu_temp_c'] == 55
    assert result['gpu_pstate'] == 'P0'


def test_process_gpu_partial_support_and_state_rate():
    calls = []
    def clock(handle, kind):
        calls.append(kind)
        if kind == 2:
            raise RuntimeError('not supported')
        return 1200
    nvml = SimpleNamespace(NVML_CLOCK_SM=1, NVML_CLOCK_MEM=2, NVML_TEMPERATURE_GPU=0,
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=20),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=200, total=1000),
        nvmlDeviceGetClockInfo=clock, nvmlDeviceGetPerformanceState=lambda handle: 2,
        nvmlDeviceGetTemperature=lambda handle, sensor: 60)
    monitor = ProcessUsageMonitor(pid=1, start_ticks=1, use_gpu=True, device_uuid='GPU-fixture')
    monitor.nvml, monitor.gpu_handle, monitor.gpu_status = nvml, 'handle', 'available'
    for timestamp in (1, 1.05, 1.1, 2):
        monitor.sample_at(timestamp)
    result, _, _ = monitor.finish_samples()
    assert result.gpu_mem_util_avg_pct == 20
    assert result.gpu_temp_c == 60
    assert result.gpu_sm_clock_mhz == 1200
    assert result.gpu_memory_clock_mhz != result.gpu_memory_clock_mhz
    assert calls == [1, 2, 1, 2]
    assert result.metric_evidence['gpu_memory_clock_mhz']['quality'] == 'unsupported'
    assert result.metric_evidence['gpu_util_avg_pct']['scope'] == 'selected_gpu_device'


def test_new_process_counter_fields_recompute_from_retained_raw_evidence():
    from acprof.telemetry import recompute_window, window_evidence
    monitor = ProcessUsageMonitor(pid=1, start_ticks=1)
    monitor.samples = [dict(timestamp=t, boundary=True,
        members={'1:10': dict(cpu_s=cpu, cpu_user_s=cpu, cpu_system_s=0, rss=100,
                             io_read_bytes=read_bytes)}, gpu_util_pct=None, gpu_mem_bytes=None)
        for t, cpu, read_bytes in [(1, 1, 100), (2, 2, 612)]]
    monitor.finish_samples()
    data = window_evidence(identity={}, context={}, collectors={'resource': monitor.telemetry_snapshot()})
    assert recompute_window(data)['resource']['process_io_read_bytes'] == 512


def test_nested_process_retention_is_required_for_exact_replay():
    import pytest

    from acprof.telemetry import raw_capability, recompute_window, window_evidence
    process = {'status': 'recorded', 'source': 'psutil', 'retention': {'complete': False}}
    resource = {'status': 'recorded', 'retention': {'complete': True}, 'process': process}
    data = window_evidence(identity={}, context={}, collectors={'resource': resource})
    assert not raw_capability(data)['can_recompute']
    with pytest.raises(ValueError, match='truncated resource'):
        recompute_window(data)
