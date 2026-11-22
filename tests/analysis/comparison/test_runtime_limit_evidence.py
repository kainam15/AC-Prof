"""Only measured positive quotas can qualify a metric across Docker and Process."""
import csv
import json
import shutil

import pytest

from acprof.analysis.comparison import compare_results
from acprof.analysis.conditions import _verified_limits
from tests.analysis.comparison.test_inference_configuration_identity import configured_fixture


@pytest.fixture
def runtime_pair(tmp_path):
    fixture, _ = configured_fixture(tmp_path)
    process = tmp_path / "process"
    shutil.copytree(fixture.left, process)
    for source, runtime in ((fixture.left, "docker"), (process, "process")):
        method = "docker_cgroup_v2" if runtime == "docker" else "cgroup_v2"
        limits = {"verified": True, "method": method,
                  "observed" if runtime == "docker" else "requested": {"cpu_quota": 1, "memory_gib": 4}}
        fixture.change_json(source, "hardware_conditions.json",
                            lambda doc: doc["cases"]["1c_4g_off"].update(resource_plan=limits))
        fixture.change_json(source, "run_state.json", lambda state: state.update(run_id=runtime))

        def configure(metadata):
            metadata["runtime_environment"]["packages"] = {"torch": "fixture"}
            metadata.update(runtime_kind=runtime, runtime_id=runtime, runtime_validation={"status": "ok"},
                            collection_plan={"probes": {name: {"status": "disabled"} for name in
                                ("cpu_energy", "cpu_instructions", "gpu_power", "packet_latency")}})

        fixture.change_json(source, "static_meta.json", configure)
        csv_path = source / "result_all.csv"
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, row = reader.fieldnames, next(reader)
        row.update(runtime_kind=runtime, runtime_id=runtime, resource_limit_source=method,
                   resource_scope="observed_process_tree" if runtime == "process" else "container_cgroup",
                   resource_metric_quality=json.dumps({"schema_version": 1, "metrics": {},
                                                       "collectors": {"client_http": {"status": "available"}}}))
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerow(row)
        fixture.refresh(source)
    return fixture, process


@pytest.mark.parametrize(("runtime", "field", "json_value", "matched"), [
    ("docker", "cpu_quota", "true", False),
    ("process", "cpu_quota", "true", False),
    ("docker", "memory_gib", '"4.00000000000000001"', False),
    ("docker", "memory_gib", "4.00000000000000001", False),
    ("docker", "cpu_quota", '"10e-1"', True),
    ("process", "memory_gib", "4.0", True),
])
def test_cross_runtime_latency_requires_exact_observed_quotas(runtime_pair, runtime, field, json_value, matched):
    fixture, process = runtime_pair
    source, section = (fixture.left, "observed") if runtime == "docker" else (process, "requested")
    fixture.change_json(source, "hardware_conditions.json",
                        lambda doc: doc["cases"]["1c_4g_off"]["resource_plan"][section].update({field: "TEST_QUOTA"}))
    # Keep JSON numbers exact in the fixture before the public metadata reader sees them.
    path = source / "hardware_conditions.json"
    text = path.read_text()
    assert text.count('"TEST_QUOTA"') == 1
    path.write_text(text.replace('"TEST_QUOTA"', json_value))

    result = compare_results(fixture.left, process)

    assert result["status"] == "incompatible"
    metric = result["metric_comparability"]["latency_app_s"]
    assert metric["status"] == ("descriptive" if matched else "not comparable")
    assert (metric["reason"] == "matched_cross_runtime_metric") == matched


@pytest.mark.parametrize(("key", "cpu", "memory"), [
    ("inheritedc_4g_off", "inherited", 4),
    ("1c_inheritedg_off", 1, None),
    ("1c_4g_off", "nan", 4),
    ("1c_4g_off", 1, {"value": 4}),
])
def test_inherited_and_nonnumeric_limits_cannot_bind_cross_runtime_metrics(key, cpu, memory):
    allocation = {"verified": True, "method": "docker_cgroup_v2", "observed": {"cpu_quota": cpu, "memory_gib": memory}}
    assert not _verified_limits(allocation, key, "docker")

def test_matching_cross_runtime_latency_stays_descriptive_in_independent_comparison(runtime_pair):
    from acprof.analysis.independent_comparison import compare_experiments

    fixture, process = runtime_pair
    report = compare_experiments([fixture.left], [process], metrics=["latency_app_s"], resamples=10)
    assert report["status"] == "incompatible"
    assert any(check["metric_comparability"]["latency_app_s"]["status"] == "descriptive"
               for check in report["condition_checks"])
    group, = report["groups"]
    assert (group["comparability"], group["difference"], group["difference_ci"]) == ("not comparable", None, None)
