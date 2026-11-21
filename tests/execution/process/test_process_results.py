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


@pytest.mark.parametrize(("cpu", "case_id", "requested", "valid"), [
    ("1.0e0", "1c_4g_off", {"cpu_quota": 1, "memory_gib": 4.0}, True),
    ("1.0000000000000001", "1c_4g_off", {"cpu_quota": 1, "memory_gib": 4}, False),
    ("9007199254740993", "9007199254740992c_4g_off", {"cpu_quota": 2**53, "memory_gib": 4}, False),
    ("9007199254740993", "9007199254740993c_4g_off", {"cpu_quota": 2**53 + 1, "memory_gib": 4}, True),
    ("9007199254740993", "9007199254740993c_4g_off", {"cpu_quota": 2**53, "memory_gib": 4}, False),
    ("1", "1c_4g_off", {"cpu_quota": True, "memory_gib": 4}, False),
    ("1", "1c_4g_off", {"cpu_quota": "1", "memory_gib": 4}, False),
    ("nan", "inheritedc_4g_off", {"cpu_quota": None, "memory_gib": 4}, True),
    ("nan", "inheritedc_4g_off", {"memory_gib": 4}, False),
    ("nan", "inheritedc_4g_off", None, False),
], ids=["integral-alias", "rounded-fraction", "wrong-case", "exact-large-case", "wrong-quota",
        "boolean", "text", "explicit-inherited", "missing-axis", "null-plan-request"])
def test_process_audit_binds_explicit_requested_limits_to_the_exact_case(result, tmp_path,
                                                                        cpu, case_id, requested, valid):
    path = result(cpu_cores=cpu, mem_cap_gb="4", resource_limit_source="cgroup_v2")
    hardware = {"schema_version": 1, "cases": {case_id: {"resource_plan": {
        "method": "cgroup_v2", "verified": True, "requested": requested}}}}
    (tmp_path / "hardware_conditions.json").write_text(json.dumps(hardware))
    report = audit_result(path)
    assert report["valid"] is valid
    issues = [issue for issue in report["issues"] if issue["code"] == "resource_enforcement"]
    assert [issue["row"] for issue in issues] == ([] if valid else [2])


@pytest.mark.parametrize(("schema", "plan_change", "cpu", "accepted"), [
    ({}, {}, "nan", True),
    ({"schema_version": 1}, {}, "nan", False),
    ({"schema_version": None}, {}, "nan", False),
    ({}, {"verified": False}, "nan", False),
    ({}, {"method": "affinity"}, "nan", False),
    ({}, {"requested": None}, "nan", False),
    ({}, {"requested": {"cpu_quota": None}}, "nan", False),
    ({}, {}, "1", False),
], ids=["legacy", "schema-one", "schema-null", "unverified", "other-method", "null-request",
        "partial-request", "numeric-axis"])
def test_empty_legacy_request_only_records_two_inherited_limits(result, tmp_path, schema, plan_change, cpu, accepted):
    plan = {"verified": True, "method": "inherited", "requested": {}, **plan_change}
    case = "inheritedc_inheritedg_off" if cpu == "nan" else "1c_inheritedg_off"
    path = result(cpu_cores=cpu, resource_limit_source=plan["method"])
    hardware = {**schema, "cases": {case: {"resource_plan": plan}}}
    (tmp_path / "hardware_conditions.json").write_text(json.dumps(hardware))

    report = audit_result(path)
    assert report["valid"] is accepted
    assert ("resource_enforcement" not in {issue["code"] for issue in report["issues"]}) is accepted


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
