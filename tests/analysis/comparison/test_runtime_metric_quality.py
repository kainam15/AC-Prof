import json

from acprof.analysis.audit import MissingReasons
from acprof.analysis.comparison import _metric_availability
from acprof.analysis.model import _summarize
from acprof.results.profile import metric_observation, metric_usable


def row(quality='partial', **changes):
    return {'process_io_read_bytes': '512', 'runtime_kind': 'process',
            'resource_metric_quality': json.dumps({'schema_version': 1, 'metrics': {
                'process_io_read_bytes': {'quality': quality, 'reason': 'permission_denied'}}}), **changes}


def test_partial_values_remain_visible_but_cannot_enter_comparison_or_summary():
    observed = row()
    assert metric_observation(observed, 'process_io_read_bytes')['scope'] == 'observed_process_tree'
    assert not _metric_availability([observed])['process_io_read_bytes']
    summary = _summarize([{'eligible': True, 'row': observed,
                           'values': {'process_io_read_bytes': 512}}], 'process_io_read_bytes')
    assert summary['value'] is None
    assert summary['missing'] == 1


def test_legacy_quality_remains_unknown_without_rewriting_history():
    assert metric_observation({}, 'process_rss_avg_bytes')['quality'] == 'unknown'
    assert metric_usable({}, 'process_rss_avg_bytes')
    assert metric_usable(row('complete'), 'process_io_read_bytes')


def test_malformed_or_conflicting_contract_is_not_measured_success():
    assert not metric_usable(row(resource_metric_quality='broken'), 'process_io_read_bytes')
    observed = row(resource_metric_quality=json.dumps({'schema_version': 1, 'metrics': {
        'process_io_read_bytes': {'quality': 'complete', 'scope': 'container_cgroup'}}}))
    assert metric_observation(observed, 'process_io_read_bytes')['quality'] == 'error'


def test_missing_reason_prefers_per_metric_permission_evidence():
    observed = row('permission_denied', process_io_read_bytes='nan')
    reason = MissingReasons({}, environment_class='native_linux').for_row(observed)
    assert reason('process_io_read_bytes') == 'permission_denied'


def test_comparison_rejects_different_gpu_state_cadence():
    from acprof.analysis.comparison import _metric_comparability_map, _metric_contracts
    def snapshot(interval):
        observed = {'gpu_temp_c': '45', 'resource_metric_quality': json.dumps({'schema_version': 1, 'metrics': {
            'gpu_temp_c': {'quality': 'complete', 'sampling': {'interval_s': interval}}}})}
        return {'metric_availability': {'gpu_temp_c': True}, 'metric_contracts': _metric_contracts([observed])}
    verdict = _metric_comparability_map(snapshot(1), snapshot(.05), 'compatible', 'compatible')
    assert verdict['gpu_temp_c'] == {'status': 'not comparable', 'reason': 'different_source_scope_window_or_sampling'}


def test_window_statistics_excludes_known_partial_readings():
    from acprof.analysis.uncertainty import _finite_metric_values
    assert _finite_metric_values([row('partial'), row('complete')], 'process_io_read_bytes') == [512]


def test_summary_does_not_blend_sampling_contracts():
    entries = []
    for interval, value in ((1, 100), (.05, 200)):
        observed = row('complete', resource_metric_quality=json.dumps({'process_io_read_bytes': {
            'quality': 'complete', 'sampling': {'interval_s': interval}}}))
        entries.append({'eligible': True, 'row': observed, 'values': {'process_io_read_bytes': value}})
    assert _summarize(entries, 'process_io_read_bytes')['value'] is None
