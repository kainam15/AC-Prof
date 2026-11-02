import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.experiments.configuration import RunConfig
from acprof.quality import summarize_quality
from scripts import check_hardware

PROFILER_SPECS = {
    "torch_profiler_eager": {
        "mode": "off", "measurement": "logical_flops", "kind": "compute",
        "primary": "model_logical_mflop_per_request_torch_profiler_eager",
        "error": "compute_profile_error_torch_profiler_eager",
        "compute": "torch", "execution": "none",
    },
    "ncu": {
        "mode": "on", "measurement": "gpu_flops", "kind": "compute",
        "primary": "gpu_executed_mflop_per_request_ncu",
        "error": "compute_profile_error_ncu", "compute": "ncu", "execution": "none",
    },
    "massif": {
        "mode": "off", "measurement": "cpu_heap", "kind": "execution",
        "primary": "cpu_heap_peak_bytes_massif", "error": "compute_profile_error_massif",
        "compute": "none", "execution": "massif",
    },
    "nsys": {
        "mode": "on", "measurement": "gpu_timeline", "kind": "execution",
        "primary": "host_inference_wall_time_ms_per_request_nsys",
        "error": "compute_profile_error_nsys", "compute": "none", "execution": "nsys",
    },
}


def profiler_fixture(tmp_path, tool):
    spec = PROFILER_SPECS[tool]
    config = RunConfig(
        model="demo", gpus=spec["mode"], output_dir=str(tmp_path / "results"),
        compute_profile_tool=spec["compute"], execution_profile_tool=spec["execution"],
    )
    result_dir = config.result_dir(check_hardware.ROOT)
    result_dir.mkdir(parents=True)
    entry = {"input_scale": 1.0, spec["primary"]: 1.25}
    if spec["kind"] == "compute":
        device = "gpu" if spec["mode"] == "on" else "cpu"
        plan = {"profiles": {device: {tool: {"entries": [entry]}}}}
    else:
        plan = {"profiles": [{
            "cpu_cores": 2, "mem_cap_gb": 8, "gpu_mode": spec["mode"],
            "tools": {tool: {"entries": [entry]}},
        }]}
    (result_dir / f"{spec['kind']}_profile_plan.json").write_text(json.dumps(plan))
    capability = {
        "schema_version": 3, "profiling_mode": "full",
        "requested_measurements": [spec["measurement"]],
        "measurement": {spec["measurement"]: {"status": "verified"}},
    }
    (result_dir / "capability_report.json").write_text(json.dumps(capability))
    row = {
        "warmup": "0", "status": "ok", "gpu_mode": spec["mode"], "input_scale": "1",
        "cpu_cores": "2", "mem_cap_gb": "8", spec["primary"]: 1.25, spec["error"]: "",
    }
    return config, result_dir, row


def selected_plan_entry(result_dir, tool):
    spec = PROFILER_SPECS[tool]
    path = result_dir / f"{spec['kind']}_profile_plan.json"
    plan = json.loads(path.read_text())
    if spec["kind"] == "compute":
        profile = next(iter(plan["profiles"].values()))[tool]
    else:
        profile = plan["profiles"][0]["tools"][tool]
    return path, plan, profile["entries"][0]


@pytest.mark.parametrize("tool", PROFILER_SPECS)
def test_complete_profiler_evidence_passes(tmp_path, tool):
    config, _, row = profiler_fixture(tmp_path, tool)
    missing = []
    assert check_hardware.validate_requested_profilers(config, [row], missing) == []
    assert missing == []


@pytest.mark.parametrize("tool", PROFILER_SPECS)
def test_warmup_profiler_failures_do_not_block_formal_acceptance(tmp_path, tool):
    config, _, row = profiler_fixture(tmp_path, tool)
    spec = PROFILER_SPECS[tool]
    warmup = {**row, "warmup": "1", spec["primary"]: "nan", spec["error"]: "warmup failed"}
    assert check_hardware.validate_requested_profilers(config, [warmup, row], []) == []


@pytest.mark.parametrize("tool", ["ncu", "massif", "nsys"])
def test_non_applicable_device_rows_do_not_require_profiler_fields(tmp_path, tool):
    config, _, row = profiler_fixture(tmp_path, tool)
    spec = PROFILER_SPECS[tool]
    other_mode = "off" if spec["mode"] == "on" else "on"
    other = {**row, "gpu_mode": other_mode, spec["primary"]: "nan", spec["error"]: "not applicable"}
    assert check_hardware.validate_requested_profilers(config, [other, row], []) == []


@pytest.mark.parametrize(("failure", "expected_code"), [
    ("capability_missing", "profiler_capability_invalid"),
    ("capability_not_requested", "profiler_capability_not_requested"),
    ("capability_status", "profiler_capability_not_verified"),
    ("plan_missing", "profiler_plan_invalid"),
    ("plan_metric", "profiler_plan_metric_missing"),
    ("plan_error", "profiler_plan_error"),
    ("csv_error", "profiler_csv_error"),
    ("no_rows", "profiler_no_applicable_rows"),
])
@pytest.mark.parametrize("tool", PROFILER_SPECS)
def test_requested_profiler_failures_are_closed(tmp_path, tool, failure, expected_code):
    config, result_dir, row = profiler_fixture(tmp_path, tool)
    spec = PROFILER_SPECS[tool]
    rows = [row]
    if failure == "capability_missing":
        (result_dir / "capability_report.json").unlink()
    elif failure in {"capability_not_requested", "capability_status"}:
        path = result_dir / "capability_report.json"
        capability = json.loads(path.read_text())
        if failure == "capability_not_requested":
            capability["requested_measurements"] = []
        else:
            capability["measurement"][spec["measurement"]]["status"] = "available"
        path.write_text(json.dumps(capability))
    elif failure == "plan_missing":
        (result_dir / f"{spec['kind']}_profile_plan.json").unlink()
    elif failure in {"plan_metric", "plan_error"}:
        path, plan, entry = selected_plan_entry(result_dir, tool)
        if failure == "plan_metric":
            entry[spec["primary"]] = "nan"
        else:
            entry[spec["error"]] = "permission denied"
        path.write_text(json.dumps(plan))
    elif failure == "csv_error":
        row[spec["error"]] = "profiler failed"
    else:
        rows = []
    issues = check_hardware.validate_requested_profilers(config, rows, [])
    assert expected_code in {issue["code"] for issue in issues}


@pytest.mark.parametrize("tool", PROFILER_SPECS)
def test_missing_formal_csv_profiler_metric_is_required(tmp_path, tool):
    config, _, row = profiler_fixture(tmp_path, tool)
    spec = PROFILER_SPECS[tool]
    row[spec["primary"]] = "nan"
    missing = []
    issues = check_hardware.validate_requested_profilers(config, [row], missing)
    assert issues == []
    assert missing == [{"row": 2, "field": spec["primary"]}]


def test_unrequested_profilers_do_not_require_artifacts(tmp_path):
    config = RunConfig(model="demo", output_dir=str(tmp_path / "missing"))
    row = {"warmup": "0", "status": "ok", "gpu_mode": "off"}
    assert check_hardware.validate_requested_profilers(config, [row], []) == []


def test_both_profiler_selections_remain_independent():
    config = RunConfig(compute_profile_tool="both", execution_profile_tool="both")
    assert check_hardware._requested_profilers(config) == [
        "torch_profiler_eager", "ncu", "massif", "nsys",
    ]


def test_run_config_writes_profiler_acceptance_failure(tmp_path):
    config = RunConfig(
        model="demo", output_dir=str(tmp_path / "results"), compute_profile_tool="torch",
    )
    primary = PROFILER_SPECS["torch_profiler_eager"]["primary"]
    row = {
        "warmup": "0", "status": "ok", "gpu_mode": "off", "input_scale": "1",
        "cpu_cores": "2", "mem_cap_gb": "8", "latency_s": "1", "latency_app_s": "1",
        "cpu_energy_total_j": "1", "cpu_instructions_per_request": "1", primary: "1",
    }
    audit = {
        "valid": True, "completion": "complete", "issues": [], **summarize_quality([]),
        "counts": {"rows": 1, "formal_ok": 1, "warmup": 0},
    }
    output = tmp_path / "acceptance"
    with (
        patch.object(check_hardware, "build_run_command", return_value=["acprof", "run"]),
        patch.object(check_hardware.subprocess, "run", return_value=SimpleNamespace(returncode=0)),
        patch.object(check_hardware, "audit_result", return_value=audit),
        patch.object(check_hardware, "read_result_csv", return_value=({}, [row])),
    ):
        result = check_hardware.run_config(config, output)
    assert result["successful"] is False
    assert result["valid"] is False
    assert "profiler_capability_invalid" in {issue["code"] for issue in result["issues"]}
    assert json.loads((output / "audit.json").read_text())["successful"] is False
