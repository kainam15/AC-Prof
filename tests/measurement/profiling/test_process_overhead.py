from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.host.measurement.monitors import MonitorGroup
from acprof.monitors.interfaces import ResourceStopResult
from scripts.measure_overhead import (
    measure_profile_window,
    measure_window,
    select_source_case,
    summarize_overhead,
)


def test_source_process_inherited_limits_remain_unknown():
    entry, cpu, mem = select_source_case({'cpus': 'auto', 'mems': 'auto'}, {'entries': [{'input_scale': 1, 'payload': {}}]})
    assert cpu is None
    assert mem is None
    assert entry['input_scale'] == 1


def test_mixed_runtime_rows_cannot_be_paired():
    with pytest.raises(ValueError, match='Runtime'):
        summarize_overhead([{'round': 0, 'scenario': 'none', 'latency_app_s': 1, 'runtime_kind': 'docker'},
                            {'round': 0, 'scenario': 'basic', 'latency_app_s': 1, 'runtime_kind': 'process'}])


def test_process_counts_raw_buffers_and_reports_optional_failure_without_false_acceptance():
    monitor = Mock()
    monitor.samples = [{}, {}]
    monitor.stop.return_value = ResourceStopResult(SimpleNamespace(required_error='', metric_evidence={
        'process_io_read_bytes': {'quality': 'permission_denied', 'reason': 'denied'}}), 'optional: denied', [])
    monitor.telemetry_snapshot.return_value = {'status': 'recorded', 'samples': [{}, {}],
        'retention': {'total_samples': 2, 'retained_samples': 2, 'complete': True}}
    group = MonitorGroup()
    group.add('resource', monitor)
    response = Mock(status_code=200)
    response.json.return_value = {}
    with patch('requests.post', return_value=response):
        result = measure_window('http://fixture.invalid', {}, count=1, monitors=group, token='fixture')
    assert result['monitors'][0]['samples'] == 2
    assert result['telemetry_json_bytes'] > 0
    assert result['collector_coverage_complete'] is False
    assert result['monitors'][0]['metric_quality']['process_io_read_bytes']['quality'] == 'permission_denied'


def test_process_profile_reuses_process_observer_without_docker_or_energy(tmp_path):
    session = SimpleNamespace(runtime_kind='process', base_url='http://fixture.invalid',
        process=SimpleNamespace(pid=123, identity={'start_ticks': 456}), gpu_device={},
        resource_plan={'method': 'inherited', 'verified': True})
    with patch('acprof.monitors.process_usage.ProcessUsageMonitor') as factory, \
            patch('acprof.monitors.resource_usage.ResourceUsageMonitor') as docker, \
            patch('scripts.measure_overhead.measure_window', return_value={'latency_app_s': 1}), \
            patch('scripts.measure_overhead.time.sleep'):
        factory.return_value.stop.return_value = ResourceStopResult(None, '', [])
        measure_profile_window(session, {'payload': {}}, scenario='basic', rate=20, count=1,
            name='process', cpu=None, mem=None, gpu='off', token='fixture', output=tmp_path,
            options={'idle_seconds': 0, 'idle_cooldown_seconds': 0, 'request_timeout_seconds': 1})
    assert factory.call_args.kwargs['pid'] == 123
    assert factory.call_args.kwargs['start_ticks'] == 456
    assert factory.call_args.kwargs['resource_plan'] == session.resource_plan
    docker.assert_not_called()
