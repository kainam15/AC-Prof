"""Recorded Process evidence is distinct from Docker limits and measurements."""
import csv
import json

import pytest

from acprof.analysis.audit import audit_result
from acprof.config import CSV_FIELDS, STATIC_META_SCHEMA_VERSION
from acprof.platform import Environment


@pytest.fixture
def result(tmp_path):
    row = {**dict.fromkeys(CSV_FIELDS, 'nan'), 'runtime_kind': 'process', 'runtime_id': 'fixture-env',
           'environment_class': 'container_host', 'resource_scope': 'observed_process_tree',
           'resource_limit_source': 'inherited', 'gpu_mode': 'off', 'input_scale': 1,
           'repeat_idx': 0, 'warmup': 0, 'status': 'ok', 'error': '', 'task_param': '{}',
           'latency_app_s': .1, 'throughput_samples_per_s': 10,
           'process_cpu_time_s': .01, 'process_cpu_util_avg_pct': 10,
           'process_rss_avg_bytes': 100, 'process_rss_peak_bytes': 120,
           'process_resource_status': 'verified'}
    metadata = {**Environment('container_host', 'Linux').metadata(), 'schema_version': STATIC_META_SCHEMA_VERSION,
                'runtime_kind': 'process', 'runtime_id': 'fixture-env'}
    (tmp_path / 'static_meta.json').write_text(json.dumps(metadata))
    (tmp_path / 'hardware_conditions.json').write_text(json.dumps({'schema_version': 1, 'cases': {
        'inheritedc_inheritedg_off': {'resource_plan': {'method': 'inherited', 'verified': True,
                                                     'requested': {'cpu_quota': None, 'memory_gib': None}}}}}))
    def write(**changes):
        path = tmp_path / 'result_all.csv'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow({**row, **changes})
        return path
    return write


def test_audit_accepts_inherited_limits_and_rejects_false_container_metrics(result):
    report = audit_result(result())
    assert report['valid'], report['issues']
    assert report['missing_metrics']['cpu_cores'] == {'inherited_limit_not_controlled': 1}
    assert not audit_result(result(container_cpu_util_avg_pct=50))['valid']


def test_audit_rejects_a_numeric_quota_without_enforcement(result):
    report = audit_result(result(cpu_cores=4))
    assert not report['valid']
    assert 'resource_enforcement' in {issue['code'] for issue in report['issues']}


def test_runtime_identity_mismatch_is_an_audit_error(result):
    report = audit_result(result(runtime_id='other-env'))
    assert not report['valid']
    assert 'runtime_identity' in {issue['code'] for issue in report['issues']}


def test_process_plot_retains_missing_quota_and_outputs_input_scan(result, tmp_path):
    from acprof.plotting.data import prepare_df
    from acprof.plotting.process import plot_process_results
    frame = prepare_df(str(result()))
    assert frame['cpu_cores'].isna().all()
    assert 'inherited' in frame.iloc[0]['config']
    assert plot_process_results(frame, tmp_path / 'plots').is_file()


def test_process_stats_report_displays_inherited_axes(result, tmp_path):
    from acprof.analysis.uncertainty import summarize_windows
    from acprof.tui.results.reports import read_report
    with result().open() as stream:
        rows = list(csv.DictReader(stream))
    report = summarize_windows(rows, ['process_rss_peak_bytes'], include_intervals=False)
    target = tmp_path / 'stats.json'
    target.write_text(json.dumps(report))
    view = read_report(target)
    assert 'inherited' in view.rows[0].cells[0]


def test_layered_process_results_roundtrip_with_inherited_limits(result, tmp_path):
    from acprof.results.layers import publish_result_layers, read_result_layers
    manifest = publish_result_layers(result())
    fields, rows = read_result_layers(tmp_path)
    assert rows[0]['runtime_kind'] == 'process'
    assert rows[0]['cpu_cores'] == rows[0]['mem_cap_gb'] == 'nan'
    assert 'process_rss_peak_bytes' in fields
    assert set(manifest['layers']) == {'summary', 'performance', 'resources', 'energy', 'network'}
    for layer in manifest['layers'].values():
        assert 'runtime_kind' in layer['fields']
    assert 'process_rss_peak_bytes' in manifest['layers']['resources']['fields']
    assert audit_result(tmp_path / 'result_layers.json')['valid']
