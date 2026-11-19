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
    from acprof.analysis.uncertainty import finite_metric_values
    assert finite_metric_values([row('partial'), row('complete')], 'process_io_read_bytes') == [512]


def test_summary_does_not_blend_sampling_contracts():
    entries = []
    for interval, value in ((1, 100), (.05, 200)):
        observed = row('complete', resource_metric_quality=json.dumps({'process_io_read_bytes': {
            'quality': 'complete', 'sampling': {'interval_s': interval}}}))
        entries.append({'eligible': True, 'row': observed, 'values': {'process_io_read_bytes': value}})
    assert _summarize(entries, 'process_io_read_bytes')['value'] is None


def test_cross_runtime_metric_delta_requires_every_formal_window_and_matching_scope():
    from acprof.analysis.comparison import _metric_comparability_map, _metric_contracts
    def snapshot(runtime, missing=False):
        names = ("latency_app_s", "latency_s")
        observed = {"runtime_kind": runtime, "latency_app_s": "0.2", "latency_s": "0.1", "packet_status": "ok",
                    "resource_metric_quality": json.dumps({"schema_version": 1, "metrics": {},
                        "collectors": {group: {"status": "available"} for group in ("client_http", "packet_latency")}})}
        return {"metric_availability": dict.fromkeys(names, True), "metric_contracts": _metric_contracts(
            [observed, {**observed, "latency_app_s": "nan"}] if missing else [observed])}
    def compare(missing=False):
        return _metric_comparability_map(snapshot("docker"), snapshot("process", missing),
                                         "incompatible", "compatible", cross_runtime=True)
    assert compare()["latency_app_s"]["status"] == "descriptive"
    assert compare()["latency_s"]["status"] == "not comparable"
    assert compare(True)["latency_app_s"]["status"] == "not comparable"



def test_runtime_pair_requires_verified_limits_pinned_workload_and_equal_inference(tmp_path):
    from copy import deepcopy

    from acprof.analysis.comparison import load_comparison_snapshot
    from acprof.analysis.conditions import PreparedComparison, compare_profiles
    from tests.support.comparison_fixtures import ComparisonFixture

    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    recorded = load_comparison_snapshot(fixture.left)
    recorded["identity"]["model_revision"] = "a" * 40
    recorded.update(inference_configuration_validation={"status": "ok"}, model_profile="fixture-inference",
                    all_formal_measured=True,
                    collector_protocol=dict.fromkeys(("cpu_energy", "cpu_instructions", "gpu_power", "packet_latency"), False),
                    verified_limits={"1c_4g_off": {"method": "cgroup_v2", "verified": True,
                                                   "requested": {"cpu_quota": 1, "memory_gib": 4}}})
    def profile(runtime, **changes):
        evidence = deepcopy(recorded)
        if runtime == "docker":
            evidence["verified_limits"] = {"1c_4g_off": {"method": "docker_cgroup_v2", "verified": True,
                                                        "observed": {"cpu_quota": 1, "memory_gib": 4}}}
        evidence.update(changes)
        evidence["conditions"]["execution_runtime"] = {"kind": runtime, "resource_scope": runtime}
        return PreparedComparison(evidence).profile(case=("1", "4", "off", "2"))
    docker = profile("docker")
    verdict = compare_profiles(docker, profile("process"))
    assert verdict["status"] == "incompatible"
    assert verdict["matched_cross_runtime"]
    assert not compare_profiles(profile("docker", verified_limits={}), profile("process")).get("matched_cross_runtime")
    for changes in ({"verified_limits": {}}, {"all_formal_measured": False}, {"model_profile": "other"},
                    {"inference_configuration_validation": {"status": "unknown"}}):
        assert not compare_profiles(docker, profile("process", **changes)).get("matched_cross_runtime")


def test_comparison_requires_observed_event_set_and_tolerates_old_or_malformed_plans():
    from acprof.analysis.comparison import _collector_protocol

    issues = []
    assert _collector_protocol({}, issues) == {}
    assert not issues
    assert _collector_protocol({"collection_plan": None}, issues) == {}
    assert issues
    def protocol(events):
        return _collector_protocol({"collection_plan": {"probes": {"cpu_instructions": {
            "status": "available", "evidence": {"events": events}}}}}, [])

    assert protocol([]) == {}
    assert protocol(["cycles", "instructions"]) != protocol(["instructions"])
    assert protocol(["cycles", "instructions"]) == protocol(["instructions", "cycles"])
