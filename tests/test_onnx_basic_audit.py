"""basic 容器验收器的负向协议测试；数值为测试输入，不是采集证据。"""
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from acprof.capabilities import (
    apply_collection_result,
    apply_runtime_validation,
    measurement_report,
)
from acprof.config import CSV_FIELDS
from acprof.platform import Environment
from scripts.check_onnx_basic import audit_basic_capabilities, audit_basic_rows, run_basic_e2e


def audit_rows():
    contract = {
        'schema_version': 1, 'request_count': 2,
        'variants': [{'count': 2, 'contract': {
            'task': 'tabular-regression', 'batch_size': 2,
            'input': {'rows': 4, 'feature_dim': 8}, 'output': {'shape': [4, 1]},
        }}],
    }
    return [{
        'warmup': warmup, 'repeat_idx': repeat, 'status': 'ok', 'error': '',
        'latency_app_s': '0.5', 'throughput_samples_per_s': '4',
        'container_cpu_util_avg_pct': '0', 'container_mem_usage_avg_bytes': '33554432',
        'resource_usage_iters': '2', 'repeat_in_window': '2',
        'workload_contract': json.dumps(contract),
    } for warmup, repeat in [('1', '0'), ('0', '0'), ('0', '1')]]


@pytest.mark.parametrize('host,row_environment', (('native_linux', 'native_linux'), ('wsl2', 'wsl2'), ('native_linux', 'wsl2')))
def test_saved_basic_results_preserve_host_environment_and_reject_mismatch(host, row_environment):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)

        def run(command, **_kwargs):
            stdout = ""
            if command[:2] == ["docker", "info"]:
                stdout = json.dumps({"OSType": "linux", "CgroupVersion": "2", "ServerVersion": "fixture"})
            elif command[:2] == ["docker", "port"]:
                stdout = "127.0.0.1:8002"
            elif "--rm" in command:
                for name, payload in {
                    "output_validation.json": {
                        "reference": {"known_reference_passed": True},
                        "runtime_validation": {"status": "ok", "validation": {
                            "protocol": {"status": "verified"}, "task": {"status": "verified"},
                        }},
                    },
                    "payload.json": {"input_scale": 2, "batch_size": 2},
                    "input_scale_plan.json": {"schema_version": 2, "entries": [{"input_scale": 2}]},
                }.items():
                    (output / name).write_text(json.dumps(payload), encoding="utf-8")
            elif "acprof.host.client" in command:
                rows = [{**dict.fromkeys(CSV_FIELDS, "nan"), **row,
                         "cpu_cores": "2", "mem_cap_gb": "1", "gpu_mode": "off", "input_scale": "2",
                         "environment_class": row_environment} for row in audit_rows()]
                with (output / "result_case.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, CSV_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        session = Mock()
        session.get.return_value.json.return_value = {"status": "ok"}
        session.post.return_value.json.return_value = {"output_shape": [4, 1], "n_results": 4}
        with (
            patch("scripts.check_onnx_basic.subprocess.run", side_effect=run),
            patch("scripts.check_onnx_basic.os.getuid", return_value=1000, create=True),
            patch("scripts.check_onnx_basic.os.getgid", return_value=1000, create=True),
            patch("requests.Session", return_value=session),
            patch("acprof.capabilities.detect_environment", return_value=Environment(host)),
        ):
            report = run_basic_e2e("fixture-image", output)
        assert (report["successful"]) == (host == row_environment), report.get("error")
        for name in ("static_meta.json", "run_state.json"):
            metadata = json.loads((output / name).read_text())
            assert (metadata["environment_class"]) == (host)
            assert (metadata["platform"]["environment"]) == (host)
            assert (metadata["comparability_class"]) == (host)
            assert (metadata["collection_tier"]) == ("full" if host == "native_linux" else "partial")
        audit = json.loads((output / "audit.json").read_text())
        codes = {issue["code"] for issue in audit["issues"]}
        assert ("environment_mismatch" in codes) == (host != row_environment)

def test_image_scenario_checks_processed_geometry_separately_from_source():
    rows = audit_rows()
    for row in rows:
        row['throughput_samples_per_s'] = '2'
        contract = json.loads(row['workload_contract'])
        contract['variants'][0]['contract'] = {
            'task': 'image-classification', 'batch_size': 1,
            'input': {'images': {'count': 1, 'original_resolution': [28, 28],
                                 'processed_resolution': [14, 14], 'processed_shape': [1, 3, 14, 14]}},
            'output': {'shape': [1, 2]},
        }
        row['workload_contract'] = json.dumps(contract)
    audit_basic_rows(rows, scenario='image')
    contract['variants'][0]['contract']['input']['images']['processed_resolution'] = [28, 28]
    rows[1]['workload_contract'] = json.dumps(contract)
    with pytest.raises(ValueError, match='actual workload.*processed_resolution'):
        audit_basic_rows(rows, scenario='image')

def test_text_scenario_uses_actual_tokens_and_its_own_batch_throughput():
    rows = audit_rows()
    for row in rows:
        row['throughput_samples_per_s'] = '2'
        contract = json.loads(row['workload_contract'])
        contract['variants'][0]['contract'] = {
            'task': 'text-classification', 'batch_size': 1,
            'input': {'text': {'tokens': 4, 'actual_tokens_per_sample': [4],
                              'content_tokens_per_sample': [2], 'padding': 'none', 'truncation': 'reject'}},
            'output': {'shape': [1, 2]},
        }
        row['workload_contract'] = json.dumps(contract)
    audit_basic_rows(rows, scenario='text')
    contract['variants'][0]['contract']['input']['text']['tokens'] = 2
    rows[1]['workload_contract'] = json.dumps(contract)
    with pytest.raises(ValueError, match='actual workload'):
        audit_basic_rows(rows, scenario='text')

@pytest.mark.parametrize('absent', ('protocol', 'task'))
def test_available_execution_without_protocol_or_task_evidence_is_not_success(absent):
    capability = measurement_report('basic', gpu_modes=['off'])
    layers = {name: {'status': 'verified'} for name in ('protocol', 'task') if name != absent}
    apply_runtime_validation(capability, {'devices': {'off': {'status': 'ok', 'validation': layers}}})
    apply_collection_result(capability, audit_rows())
    assert (capability.to_dict()['requested_measurements_complete'])
    with pytest.raises(ValueError, match='CPU execution.*verified'):
        audit_basic_capabilities(capability)

def test_complete_output_validation_and_metrics_allow_basic_success():
    capability = measurement_report('basic', gpu_modes=['off'])
    apply_runtime_validation(capability, {'devices': {'off': {'status': 'ok', 'validation': {
        'protocol': {'status': 'verified'}, 'task': {'status': 'verified'},
    }}}})
    apply_collection_result(capability, audit_rows())
    audit_basic_capabilities(capability)

def test_measured_zero_cpu_is_valid_and_missing_cpu_is_not():
    rows = audit_rows()
    audit_basic_rows(rows)
    rows[1]['container_cpu_util_avg_pct'] = 'nan'
    with pytest.raises(ValueError, match='required finite basic metric.*container_cpu'):
        audit_basic_rows(rows)

def test_throughput_must_match_existing_batch_latency_definition():
    rows = audit_rows()
    rows[1]['throughput_samples_per_s'] = '8'
    with pytest.raises(ValueError, match='batch_size / application latency'):
        audit_basic_rows(rows)

def test_actual_workload_cannot_drop_measured_requests():
    rows = audit_rows()
    rows[2]['repeat_in_window'] = '3'
    with pytest.raises(ValueError, match='actual workload request count'):
        audit_basic_rows(rows)

def test_wrong_output_shape_fails_even_with_valid_performance_metrics():
    rows = audit_rows()
    contract = json.loads(rows[1]['workload_contract'])
    contract['variants'][0]['contract']['output']['shape'] = [3, 1]
    rows[1]['workload_contract'] = json.dumps(contract)
    with pytest.raises(ValueError, match='actual workload differs'):
        audit_basic_rows(rows)

def test_unrequested_metrics_cannot_be_filled_with_zero():
    rows = audit_rows()
    rows[0]['cpu_energy_total_j'] = '0'
    with pytest.raises(ValueError, match='unrequested full metric'):
        audit_basic_rows(rows)
