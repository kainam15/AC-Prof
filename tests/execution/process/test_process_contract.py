"""Process integration contracts without model downloads or hardware experiments."""
import json
import math
import os
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from acprof.application.resources import resource_matrix
from acprof.capabilities import Capability
from acprof.experiments.configuration import RunConfig
from acprof.host.measurement.config import ClientConfig
from acprof.host.measurement.metrics import WindowMetrics
from acprof.host.measurement.requests import WindowRequestProgress
from acprof.host.measurement.types import WindowObservations
from acprof.host.runtimes.contracts import ProcessInfo
from acprof.host.runtimes.selection import select_runtime
from acprof.monitors.process_usage import ProcessUsageMonitor, ProcessUsageResult
from acprof.monitors.resource_metrics import empty_resource_result
from acprof.platform import Environment
from acprof.result_csv import expected_measurements, measurement_key


def test_explicit_process_never_probes_docker():
    def forbidden():
        pytest.fail('explicit Process must not inspect Docker')
    selection = select_runtime('process', profiling_mode='basic', environment=Environment('container_host', 'Linux'), docker_probe=forbidden)
    assert selection.selected == 'process'


def test_auto_uses_the_docker_preflight_missing_cli_evidence(monkeypatch):
    from acprof.host import preflight
    def missing(*args, **kwargs):
        raise FileNotFoundError('docker')
    monkeypatch.setattr(preflight, 'run_command', missing)
    selection = select_runtime('auto', profiling_mode='basic', environment=Environment('vm', 'Linux'))
    assert selection.selected == 'process'
    assert selection.docker.status == 'unavailable'
    assert 'Docker CLI was not found' in selection.reason


@pytest.mark.parametrize('status', ['unavailable', 'permission_denied', 'error'])
def test_auto_falls_back_only_before_preparation_and_only_for_basic(status):
    def probe():
        return Capability(status, 'fixture Docker unavailable')
    assert select_runtime('auto', profiling_mode='basic', environment=Environment('vm', 'Linux'), docker_probe=probe).selected == 'process'
    for requested, mode in [('docker', 'basic'), ('auto', 'full')]:
        with pytest.raises(ValueError, match='Docker Runtime unavailable'):
            select_runtime(requested, profiling_mode=mode, docker_probe=probe)


def test_resource_auto_preserves_docker_matrix_and_keeps_process_unbounded():
    config = RunConfig(model='fixture', profiling_mode='basic')
    assert resource_matrix(replace(config, runtime='docker').to_namespace())[:2] == ([1, 2, 4, 8], [2, 4, 8, 16])
    assert resource_matrix(replace(config, runtime='process').to_namespace())[:2] == ([None], [None])


def test_inherited_limits_have_unambiguous_keys_and_never_zero():
    row = dict(runtime_kind='process', cpu_cores='nan', mem_cap_gb='nan', gpu_mode='off', input_scale='1', warmup='0', repeat_idx='0')
    assert measurement_key(row) in expected_measurements([None], [None], ['off'], [1], 0, 1)
    assert measurement_key(row)[:2] == ('inherited', 'inherited')
    with pytest.raises(ValueError, match="invalid cpu_cores"):
        measurement_key({**row, 'runtime_kind': 'docker'})
    with pytest.raises(ValueError, match="invalid cpu_cores"):
        measurement_key({**row, 'cpu_cores': 0})


def test_optional_process_permission_failure_retains_application_metrics():
    config = ClientConfig(runtime_kind='process', profiling_mode='basic', cpu_cores='nan', mem_cap_gb='nan')
    observed = WindowObservations(resource_result=ProcessUsageResult(**asdict(empty_resource_result()), process_resource_status='permission_denied'), resource_error='permission denied')
    values = WindowMetrics(config, {'input_scale': 1}, use_energy=False)
    values.reduce(WindowRequestProgress(count=2, latencies=[.1, .3]), observed)
    assert values.status == 'ok'
    assert values.latency_app_s == .2
    assert values.throughput == 5
    assert all(math.isnan(value) for value in values.process.values())
    assert math.isnan(values.resource['container_mem_usage_avg_bytes'])


def test_process_raw_evidence_uses_its_own_reducer_and_retention():
    from acprof.telemetry import raw_capability, recompute_window, window_evidence
    monitor = ProcessUsageMonitor(pid=os.getpid(), start_ticks=1, raw_sample_limit=2)
    monitor.samples = [dict(timestamp=t, members={'one': dict(cpu_s=cpu, rss=100)}, gpu_util_pct=None, gpu_mem_bytes=None)
                       for t, cpu in [(1, 2), (2, 3)]]
    evidence = window_evidence(identity={}, context={}, collectors={'resource': monitor.telemetry_snapshot()})
    assert raw_capability(evidence)['can_recompute']
    assert recompute_window(evidence)['resource']['process_cpu_time_s'] == 1
    monitor.samples.append(dict(timestamp=3, members={'one': dict(cpu_s=4, rss=100)}, gpu_util_pct=None, gpu_mem_bytes=None))
    evidence['collectors']['resource'] = monitor.telemetry_snapshot()
    assert not raw_capability(evidence)['can_recompute']
    with pytest.raises(ValueError, match='truncated'):
        recompute_window(evidence)


def test_service_environment_is_offline_and_respects_visible_gpu(tmp_path, monkeypatch):
    from acprof.host.runtimes.process import service_environment
    monkeypatch.setenv('HF_TOKEN', 'fixture-do-not-forward')
    monkeypatch.setenv('PYTHONPATH', '/unrelated')
    python = tmp_path / 'venv/bin/python'
    info = ProcessInfo(str(python), '/reviewed/code', '/pinned/model', 'fixture',
                       {'gpu_device': {'cuda_visible_devices': 'GPU-expected'}})
    task = SimpleNamespace(model_id='fixture/model', model_revision='1' * 40, pipeline_tag='tabular-regression',
                           task_family='structured', runtime_backend='onnxruntime', model_adapter='family-default',
                           runtime_profile_id='onnxruntime-cpu', model_spec={})
    env = service_environment(task, info, gpu='on')
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-expected'
    assert env['PYTHONPATH'] == '/reviewed/code'
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
    assert 'HF_TOKEN' not in env
    assert env['MODEL_LOCAL_PATH'] == '/pinned/model'


def test_process_validation_persists_environment_identity_without_docker_image(tmp_path, monkeypatch):
    from acprof.host.runtimes import validation
    plan = tmp_path / 'input_scale_plan.json'
    plan.write_text(json.dumps({'schema_version': 2, 'entries': [
        {'input_scale': 1, 'payload': {'features': [1, 2, 3, 4]}}
    ]}))
    task = SimpleNamespace(model_resolution={'contract': {'fields': {}}})
    info = ProcessInfo('/python', '/code', '/model', 'process-environment', {})
    stages = ('execution', 'load', 'preprocess', 'predict', 'completion', 'postprocess', 'validate_output')
    monkeypatch.setattr(validation, 'run_validation', lambda *a, **kw: {
        'status': 'ok', 'stages': [{'stage': stage, 'status': 'verified'} for stage in stages],
        'validation': {'protocol': {'status': 'verified'}, 'task': {'status': 'verified'}},
    })
    report = validation.validate_process(task_info=task, image_info=info,
        planned=SimpleNamespace(plan_file=str(plan)), cpu_list=[None], mem_list=[None],
        gpu_list=['off'], output_dir=tmp_path)
    assert report['status'] == 'ok'
    observed = task.model_resolution['runtime_validation']
    assert observed['runtime_kind'] == 'process'
    assert observed['runtime_id'] == 'process-environment'
    assert 'image_id' not in observed
    assert task.model_resolution['contract']['fields']['runtime.full']['runtime_id'] == 'process-environment'
