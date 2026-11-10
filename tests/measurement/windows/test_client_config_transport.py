"""Public wire contract, independently specified and exercised in real children."""
import json
import os
import subprocess
import sys
from dataclasses import asdict, fields, replace

import pytest

from acprof.host.measurement.config import ClientConfig

# Each field needs an explicit wire name, non-default value and expected encoding.
# Do not derive these expectations from the production codec or its registry.
CONTRACT = {
    "runtime_kind": ("ACPROF_EXECUTION_RUNTIME", "process", "process"),
    "process_pid": ("ACPROF_PROCESS_PID", 123, "123"),
    "process_start_ticks": ("ACPROF_PROCESS_START_TICKS", 456, "456"),
    "resource_plan": ("ACPROF_RESOURCE_PLAN", {"method": "fixture", "verified": True}, '{"method": "fixture", "verified": true}'),
    "process_startup_s": ("ACPROF_PROCESS_STARTUP_S", "1.25", "1.25"),
    "model_revision": ("MODEL_REVISION", "revision-7", "revision-7"),
    "task_family": ("TASK_FAMILY", "structured", "structured"),
    "pipeline_tag": ("PIPELINE_TAG", "tabular-classification", "tabular-classification"),
    "runtime_backend": ("RUNTIME_BACKEND", "onnx", "onnx"),
    "image_tag": ("IMAGE_TAG", "fixture:7", "fixture:7"),
    "cpu_cores": ("CPU_CORES", 3, "3"),
    "mem_cap_gb": ("MEM_CAP_GB", 7, "7"),
    "gpu_mode": ("GPU_MODE", "on", "on"),
    "base_url": ("BASE_URL", "http://127.0.0.1:9137", "http://127.0.0.1:9137"),
    "endpoint": ("ENDPOINT", "/fixture", "/fixture"),
    "batch_size": ("BATCH_SIZE", 7, "7"),
    "warmup": ("WARMUP", 3, "3"),
    "repeat": ("REPEAT", 9, "9"),
    "repeat_in_window": ("REPEAT_IN_WINDOW", 4, "4"),
    "repeat_window_seconds": ("REPEAT_WINDOW_SECONDS", 3.125, "3.125"),
    "auto_warmup_requests": ("AUTO_WARMUP_REQUESTS", 17, "17"),
    "request_timeout_seconds": ("REQUEST_TIMEOUT_SECONDS", 1.2345678901234567, "1.2345678901234567"),
    "cold_start_s": ("COLD_START_S", "1.75", "1.75"),
    "cold_start_started_at": ("COLD_START_STARTED_AT", "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z"),
    "cold_start_ready_at": ("COLD_START_READY_AT", "2026-01-01T01:00:02Z", "2026-01-01T01:00:02Z"),
    "cold_start_container_launch_s": ("COLD_START_CONTAINER_LAUNCH_S", "0.25", "0.25"),
    "cold_start_server_setup_s": ("COLD_START_SERVER_SETUP_S", "0.5", "0.5"),
    "cold_start_cuda_init_s": ("COLD_START_CUDA_INIT_S", "0.75", "0.75"),
    "cold_start_model_load_s": ("COLD_START_MODEL_LOAD_S", "1.0", "1.0"),
    "cold_start_ready_wait_s": ("COLD_START_READY_WAIT_S", "0.125", "0.125"),
    "out_csv": ("OUT_CSV", "fixture results/结果.csv", "fixture results/结果.csv"),
    "case_name": ("CASE_NAME", "fixture-case", "fixture-case"),
    "container_name": ("CONTAINER_NAME", "fixture-container", "fixture-container"),
    "sniff_groups_path": ("SNIFF_GROUPS_PATH", "fixture groups.json", "fixture groups.json"),
    "idle_debug": ("IDLE_DEBUG", True, "1"),
    "idle_diag_path": ("IDLE_DIAG_PATH", "fixture idle.json", "fixture idle.json"),
    "client_error_path": ("CLIENT_ERROR_PATH", "fixture error.json", "fixture error.json"),
    "idle_debug_trace_interval_s": ("IDLE_DEBUG_TRACE_INTERVAL_S", 0.625, "0.625"),
    "use_mips": ("USE_MIPS", True, "1"),
    "profiling_mode": ("PROFILING_MODE", "basic", "basic"),
    "dram_energy": ("DRAM_ENERGY", "off", "off"),
    "sample_hz": ("SAMPLE_HZ", 37.5, "37.5"),
    "sampling_policy": ("ACPROF_SAMPLING_POLICY", "skip", "skip"),
    "raw_sample_limit": ("ACPROF_RAW_SAMPLE_LIMIT", 543, "543"),
    "run_id": ("ACPROF_RUN_ID", "run-7", "run-7"),
    "case_id": ("ACPROF_CASE_ID", "case-7", "case-7"),
    "case_attempt_id": ("ACPROF_CASE_ATTEMPT_ID", "attempt-7", "attempt-7"),
    "window_records_dir": ("ACPROF_WINDOW_RECORDS_DIR", "fixture windows", "fixture windows"),
    "telemetry_dir": ("ACPROF_TELEMETRY_DIR", "fixture telemetry", "fixture telemetry"),
    "result_root": ("ACPROF_RESULT_ROOT", "fixture results", "fixture results"),
    "idle_seconds": ("IDLE_SECONDS", 1.125, "1.125"),
    "device_index": ("DEVICE_INDEX", 2, "2"),
    "gpu_device_uuid": ("GPU_DEVICE_UUID", "GPU-fixture", "GPU-fixture"),
    "idle_cooldown_seconds": ("IDLE_COOLDOWN_SECONDS", 2.125, "2.125"),
    "input_scales_str": ("INPUT_SCALES", "1,3.5", "1,3.5"),
    "input_scale_plan_file": ("INPUT_SCALE_PLAN_FILE", "fixture inputs.json", "fixture inputs.json"),
    "compute_profile_plan_file": ("COMPUTE_PROFILE_PLAN_FILE", "fixture compute.json", "fixture compute.json"),
    "execution_profile_plan_file": ("EXECUTION_PROFILE_PLAN_FILE", "fixture execution.json", "fixture execution.json"),
    "input_scale_order": ("INPUT_SCALE_ORDER", [3.5, 1.0], "[3.5, 1.0]"),
}
EVIDENCE_GROUP = ("run_id", "case_id", "case_attempt_id", "window_records_dir", "telemetry_dir")
AMBIENT_DEFAULTS = {"AUTO_WARMUP_REQUESTS", "SNIFF_GROUPS_PATH", "IDLE_DEBUG_TRACE_INTERVAL_S"}
READER = '''
import json
from dataclasses import asdict
from acprof.host.measurement.config import ClientConfig
try:
    config = ClientConfig.from_env()
except ValueError as error:
    print(json.dumps({"error": str(error)}))
    raise SystemExit(2)
print(json.dumps(asdict(config)))
'''


def read_child(environment):
    names = {entry[0] for entry in CONTRACT.values()}
    clean = {key: value for key, value in os.environ.items() if key not in names}
    return subprocess.run([sys.executable, "-c", READER], env={**clean, **environment},
                          capture_output=True, text=True, timeout=15)


def test_every_field_has_an_independent_transport_example():
    assert set(CONTRACT) == {item.name for item in fields(ClientConfig)}
    assert len({entry[0] for entry in CONTRACT.values()}) == len(CONTRACT)
    defaults = ClientConfig()
    assert all(getattr(defaults, name) != value for name, (_, value, _) in CONTRACT.items())


@pytest.mark.parametrize("name", CONTRACT)
def test_each_nondefault_field_crosses_a_real_process_boundary(name):
    wire, value, expected = CONTRACT[name]
    values = {name: value}
    if name == "runtime_kind":
        values.update(process_pid=123, process_start_ticks=456, profiling_mode="basic")
    if name in EVIDENCE_GROUP:
        values.update({field: CONTRACT[field][1] for field in EVIDENCE_GROUP})
    config = ClientConfig(**values)
    config.validate()
    encoded = config.to_env()
    assert encoded[wire] == expected
    child = read_child(encoded)
    assert child.returncode == 0, child.stderr + child.stdout
    assert json.loads(child.stdout) == asdict(config)


def test_default_empty_and_full_configurations_round_trip():
    config = ClientConfig()
    encoded = config.to_env()
    assert set(encoded) == {entry[0] for entry in CONTRACT.values()} - AMBIENT_DEFAULTS
    assert encoded["CPU_CORES"] == "nan"
    assert encoded["MEM_CAP_GB"] == "nan"
    assert encoded["INPUT_SCALE_ORDER"] == ""
    for environment in ({}, encoded):
        child = read_child(environment)
        assert child.returncode == 0, child.stderr + child.stdout
        assert json.loads(child.stdout) == asdict(config)
    full = ClientConfig(**{name: value for name, (_, value, _) in CONTRACT.items()})
    full.validate()
    child = read_child(full.to_env())
    assert child.returncode == 0, child.stderr + child.stdout
    assert json.loads(child.stdout) == asdict(full)


@pytest.mark.parametrize(("field", "key", "ambient", "expected"), [
    ("auto_warmup_requests", "AUTO_WARMUP_REQUESTS", "19", 19),
    ("sniff_groups_path", "SNIFF_GROUPS_PATH", "ambient.json", "ambient.json"),
    ("idle_debug_trace_interval_s", "IDLE_DEBUG_TRACE_INTERVAL_S", "0.375", 0.375),
])
def test_only_conditional_defaults_inherit_ambient_values(field, key, ambient, expected):
    config = ClientConfig()
    child = read_child({key: ambient, "WARMUP": "99", **config.to_env()})
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == asdict(replace(config, **{field: expected}))
    explicit = replace(config, **{field: CONTRACT[field][1]})
    child = read_child({key: ambient, **explicit.to_env()})
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == asdict(explicit)


@pytest.mark.parametrize("changes", [
    {"BATCH_SIZE": "bad"}, {"WARMUP": "-1"}, {"REPEAT_WINDOW_SECONDS": "nan"},
    {"ACPROF_RESOURCE_PLAN": "[]"}, {"ACPROF_RESOURCE_PLAN": "{"},
    {"INPUT_SCALE_ORDER": "{}"}, {"INPUT_SCALE_ORDER": '["bad"]'},
    {"ACPROF_RUN_ID": "only-one-identity-field"}, {"ACPROF_EXECUTION_RUNTIME": "process"},
    {"ACPROF_RAW_SAMPLE_LIMIT": "1"}, {"PROFILING_MODE": "invalid"},
])
def test_invalid_wire_values_are_rejected_by_the_child(changes):
    child = read_child({**ClientConfig().to_env(), **changes})
    assert child.returncode == 2, child.stderr + child.stdout
    assert json.loads(child.stdout)["error"]


def test_measurement_overrides_are_recorded_and_replayed_without_leaking_state(monkeypatch):
    from acprof.experiments.configuration import RunConfig
    from acprof.host.execution_conditions import source_runtime_environment
    from acprof.host.run_state import run_options

    monkeypatch.setenv("AUTO_WARMUP_REQUESTS", "17")
    monkeypatch.setenv("IDLE_DEBUG_TRACE_INTERVAL_S", "0.375")
    recorded = run_options(RunConfig.smoke("fixture/model").to_namespace())["measurement_environment"]
    assert recorded["AUTO_WARMUP_REQUESTS"] == "17"
    assert recorded["IDLE_DEBUG_TRACE_INTERVAL_S"] == "0.375"
    monkeypatch.setenv("AUTO_WARMUP_REQUESTS", "29")
    monkeypatch.setenv("IDLE_DEBUG_TRACE_INTERVAL_S", "0.875")
    with source_runtime_environment(recorded):
        child = read_child({**os.environ, **ClientConfig().to_env()})
        assert child.returncode == 0, child.stderr
        config = json.loads(child.stdout)
        assert config["auto_warmup_requests"] == 17
        assert config["idle_debug_trace_interval_s"] == 0.375
    assert os.environ["AUTO_WARMUP_REQUESTS"] == "29"
    assert os.environ["IDLE_DEBUG_TRACE_INTERVAL_S"] == "0.875"


def test_wire_normalization_is_preserved_and_unknown_variables_are_ignored():
    child = read_child({"BASE_URL": "http://fixture///", "CPU_CORES": " inherited ",
                        "MEM_CAP_GB": "none", "IDLE_DEBUG": " YES ", "USE_MIPS": "on",
                        "CASE_NAME": " case ", "INPUT_SCALE_ORDER": '["3.5", 1]',
                        "ACPROF_UNRECOGNIZED": "not-a-config-field"})
    assert child.returncode == 0, child.stderr + child.stdout
    assert json.loads(child.stdout) == asdict(ClientConfig(
        base_url="http://fixture", idle_debug=True, use_mips=True,
        case_name="case", input_scale_order=[3.5, 1.0],
    ))
