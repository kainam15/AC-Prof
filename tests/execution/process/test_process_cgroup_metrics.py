from contextlib import nullcontext
from types import SimpleNamespace

from acprof.monitors.process_usage import ProcessUsageMonitor


def observer(tmp_path, monkeypatch, *, owned=True):
    root = SimpleNamespace(pid=42, is_running=lambda: True, create_time=lambda: 10,
        children=lambda **kw: [], oneshot=nullcontext, cpu_times=lambda: SimpleNamespace(user=1, system=1),
        memory_info=lambda: SimpleNamespace(rss=200))
    monkeypatch.setattr('acprof.monitors.process_usage.psutil.Process', lambda pid: root)
    monkeypatch.setattr('acprof.process_identity.same_process', lambda identity: True)
    monkeypatch.setattr(ProcessUsageMonitor, '_read_boundary', lambda *a: {})
    (tmp_path / 'cgroup.procs').write_text('42\n')
    (tmp_path / 'cpu.stat').write_text('usage_usec 1000000\n')
    (tmp_path / 'memory.current').write_text('500\n')
    (tmp_path / 'io.stat').write_text('8:0 rbytes=100 wbytes=200 rios=1 wios=2\n')
    plan = {'method': 'cgroup_v2' if owned else 'inherited', 'verified': True, 'cgroup': str(tmp_path)}
    monitor = ProcessUsageMonitor(pid=42, start_ticks=10, resource_plan=plan)
    monitor.prepare()
    return monitor


def test_process_uses_owned_cgroup_without_container_field_substitution(tmp_path, monkeypatch):
    monitor = observer(tmp_path, monkeypatch)
    monitor.sample_at(1, boundary=True)
    (tmp_path / 'cpu.stat').write_text('usage_usec 3000000\n')
    (tmp_path / 'memory.current').write_text('900\n')
    (tmp_path / 'io.stat').write_text('8:0 rbytes=612 wbytes=200 rios=2 wios=2\n')
    monitor.sample_at(3, boundary=True)
    result, _, _ = monitor.finish_samples()
    assert result.service_cgroup_cpu_time_s == 2
    assert result.service_cgroup_mem_usage_avg_bytes == 700
    assert result.service_cgroup_mem_usage_peak_bytes == 900
    assert result.service_cgroup_io_read_bytes == 512
    assert result.service_cgroup_io_write_bytes == 0
    assert result.container_mem_usage_avg_bytes != result.container_mem_usage_avg_bytes
    assert result.metric_evidence['service_cgroup_cpu_time_s']['scope'] == 'owned_service_cgroup'
    from acprof.telemetry import recompute_window, window_evidence
    data = window_evidence(identity={}, context={}, collectors={'resource': monitor.telemetry_snapshot()})
    assert recompute_window(data)['resource']['service_cgroup_io_read_bytes'] == 512


def test_inherited_parent_cgroup_is_never_attributed_to_process(tmp_path, monkeypatch):
    monitor = observer(tmp_path, monkeypatch, owned=False)
    monitor.sample_at(1, boundary=True)
    monitor.sample_at(3, boundary=True)
    result, _, _ = monitor.finish_samples()
    assert result.service_cgroup_cpu_time_s != result.service_cgroup_cpu_time_s
    assert result.metric_evidence['service_cgroup_cpu_time_s']['quality'] == 'unsupported'


def test_cgroup_membership_loss_invalidates_counter_window(tmp_path, monkeypatch):
    monitor = observer(tmp_path, monkeypatch)
    monitor.sample_at(1, boundary=True)
    (tmp_path / 'cgroup.procs').write_text('99\n')
    monitor.sample_at(3, boundary=True)
    result, error, _ = monitor.finish_samples()
    assert result.service_cgroup_cpu_time_s != result.service_cgroup_cpu_time_s
    assert 'membership' in error
    assert result.process_rss_avg_bytes == 200


def test_cgroup_reset_warns_without_discarding_memory(tmp_path, monkeypatch):
    monitor = observer(tmp_path, monkeypatch)
    monitor.sample_at(1, boundary=True)
    (tmp_path / 'cpu.stat').write_text('usage_usec 100\n')
    monitor.sample_at(3, boundary=True)
    result, error, _ = monitor.finish_samples()
    assert result.service_cgroup_cpu_time_s != result.service_cgroup_cpu_time_s
    assert result.service_cgroup_mem_usage_avg_bytes == 500
    assert result.metric_evidence['service_cgroup_cpu_time_s']['quality'] == 'error'
    assert 'counter_decreased' in error
