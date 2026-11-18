"""Docker and Process share process-tree semantics without a second sampler."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from acprof.monitors.process_usage import ProcessUsageMonitor
from acprof.monitors.resource_readers import ResourceMetricReaders
from acprof.monitors.resource_usage import ResourceUsageMonitor


def test_docker_retains_cgroup_and_collects_the_same_process_fields(monkeypatch):
    readers = ResourceMetricReaders(cpu=lambda: 1, memory=lambda: 1000)
    readers.pid = 42
    monkeypatch.setattr('acprof.monitors.resource_readers.resolve_container_metric_readers', lambda *a, **kw: readers)
    monkeypatch.setattr('acprof.process_identity.process_identity', lambda pid: {'pid': pid, 'start_ticks': 10, 'state': 'R'})
    cpu = iter((SimpleNamespace(user=3, system=1), SimpleNamespace(user=4, system=2)))
    root = SimpleNamespace(pid=42, is_running=lambda: True, create_time=lambda: 10,
        children=lambda **kw: [], oneshot=nullcontext, cpu_times=lambda: next(cpu),
        memory_info=lambda: SimpleNamespace(rss=200))
    monkeypatch.setattr('acprof.monitors.process_usage.psutil.Process', lambda pid: root)
    monkeypatch.setattr(ProcessUsageMonitor, '_read_boundary', lambda *a: {'threads': 3, 'fds': 4})
    with patch('acprof.monitors.resource_usage.time.perf_counter', return_value=12):
        monitor = ResourceUsageMonitor(container_name='fixture', cpu_cores=2, use_gpu=False)
        monitor.prepare()
        monitor._t_start = 10
        monitor._append_sample(10, boundary=True)
        result, error, _ = monitor.stop()
        monitor.close()
    assert not error
    assert result.container_mem_usage_avg_bytes == 1000
    assert result.process_rss_avg_bytes == 200
    assert result.process_cpu_time_s == 2
    assert result.process_cpu_util_avg_pct == 100
    assert result.process_threads_end == 3
    evidence = monitor.telemetry_snapshot()
    assert evidence['process']['scope'] == 'observed_process_tree'
    assert evidence['process']['pid'] == 42
    assert monitor._process_monitor._thread is None


def test_optional_process_failure_does_not_erase_required_cgroup_values(monkeypatch):
    readers = ResourceMetricReaders(cpu=lambda: 1, memory=lambda: 1000)
    readers.pid = 42
    monkeypatch.setattr('acprof.monitors.resource_readers.resolve_container_metric_readers', lambda *a, **kw: readers)
    def denied(pid):
        raise PermissionError('hidepid')
    monkeypatch.setattr('acprof.process_identity.process_identity', denied)
    monitor = ResourceUsageMonitor(container_name='fixture')
    monitor.prepare()
    monitor._t_start = 10
    monitor._append_sample(10, boundary=True)
    with patch('acprof.monitors.resource_usage.time.perf_counter', return_value=12):
        result, error, _ = monitor.stop()
    monitor.close()
    assert result.container_mem_usage_avg_bytes == 1000
    assert not error
    assert 'hidepid' in result.optional_error
    assert result.process_resource_status == 'permission_denied'


def test_cpu_access_denied_is_visible_per_metric():
    monitor = ProcessUsageMonitor(pid=1, start_ticks=1)
    monitor.status, monitor.error = 'permission_denied', 'hidepid'
    monitor.samples = [dict(timestamp=t, members={}, boundary=True) for t in (1, 2)]
    result, error, _ = monitor.finish_samples()
    assert 'hidepid' in error
    assert result.metric_evidence['process_cpu_time_s']['quality'] == 'permission_denied'
    assert result.process_cpu_time_s != result.process_cpu_time_s
