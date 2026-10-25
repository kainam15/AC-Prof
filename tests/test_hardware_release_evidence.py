"""Fail-closed tests for published native hardware evidence."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.verify_hardware_evidence import InvalidEvidence, validate_evidence

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path):
    matrix = json.loads((ROOT / "examples/hardware-validation.json").read_text())
    matrix_file = tmp_path / "matrix.json"
    matrix_file.write_text(json.dumps(matrix))
    audit = {"successful": True, "valid": True, "completion": "complete",
             "exit_code": 0, "required_metrics_missing": [], "validated_gpu_modes": ["off", "on"],
             "counts": {"formal_ok": 4}}
    report = {"kind": "hardware_matrix", "ci_commit": "a" * 40, "successful": True,
              "source": {"environment_class": "native_linux", "collection_tier": "full",
                         "platform": {"native": True}},
              "cases": [{"name": case["name"], "status": "passed",
                         "config": {**case, "gpus": "off,on"},
                         "audit": deepcopy(audit)} for case in matrix["cases"]]}
    return matrix_file, report


def validate(tmp_path, matrix_file, report):
    path = tmp_path / "hardware_matrix.json"
    path.write_text(json.dumps(report))
    return validate_evidence(path, matrix_file, "a" * 40)


def test_complete_native_evidence_is_accepted(tmp_path):
    matrix_file, report = fixture(tmp_path)
    result = validate(tmp_path, matrix_file, report)
    assert result["status"] == "verified"
    assert result["commit"] == "a" * 40
    assert len(result["cases"]) == len(report["cases"])


@pytest.mark.parametrize("tamper", [
    "wrong_commit", "failed_matrix", "not_native", "wsl", "partial",
    "missing_case", "modified_revision", "cpu_only", "failed_case",
    "missing_energy", "no_gpu_mode", "no_perf_rows", "no_audit",
])
def test_bad_hardware_artifacts_never_unlock_release(tmp_path, tamper):
    matrix_file, report = fixture(tmp_path)
    case = report["cases"][0]
    audit = case["audit"]
    if tamper == "wrong_commit":
        report["ci_commit"] = "b" * 40
    elif tamper == "failed_matrix":
        report["successful"] = False
    elif tamper == "not_native":
        report["source"]["platform"]["native"] = False
    elif tamper == "wsl":
        report["source"]["environment_class"] = "wsl2"
    elif tamper == "partial":
        report["source"]["collection_tier"] = "partial"
    elif tamper == "missing_case":
        report["cases"].pop()
    elif tamper == "modified_revision":
        case["config"]["revision"] = "b" * 40
    elif tamper == "cpu_only":
        case["config"]["gpus"] = "off"
    elif tamper == "failed_case":
        case["status"] = "not_run"
    elif tamper == "missing_energy":
        audit["required_metrics_missing"] = [{"field": "cpu_energy_total_j"}]
    elif tamper == "no_gpu_mode":
        audit["validated_gpu_modes"] = ["off"]
    elif tamper == "no_perf_rows":
        audit["counts"]["formal_ok"] = 0
    elif tamper == "no_audit":
        del case["audit"]
    with pytest.raises(InvalidEvidence):
        validate(tmp_path, matrix_file, report)


def test_hardware_workflow_is_weekly_and_never_exposes_fork_prs():
    hardware = yaml.load((ROOT / ".github/workflows/hardware.yml").read_text(), Loader=yaml.BaseLoader)
    assert hardware["on"]["schedule"]
    assert "pull_request" not in hardware["on"]
    job = hardware["jobs"]["profile"]
    assert "refs/heads/main" in job["if"]
    tracked = next(step["run"] for step in job["steps"] if "git diff --cached --quiet" in step.get("run", ""))
    assert "git rev-parse HEAD" in tracked
    assert "git diff --quiet" in tracked
    script = next(step for step in job["steps"] if "VALIDATION_PRESET" in step.get("env", {}))
    assert "pinned-matrix" in script["env"]["VALIDATION_PRESET"]
    assert "off,on" in script["env"]["VALIDATION_GPUS"]


def test_release_requires_hardware_proof_before_both_publications():
    release = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    jobs = release["jobs"]
    for name in ("publish", "pypi"):
        assert "hardware-gate" in jobs[name]["needs"]
    gate = jobs["hardware-gate"]
    assert gate["permissions"] == {"contents": "read", "actions": "read"}
    script = next(step["run"] for step in gate["steps"] if "hardware-runs.tsv" in step.get("run", ""))
    assert "head_sha" in script
    assert "head_branch" in script
    assert 'run["event"] in ("schedule", "workflow_dispatch")' in script
    assert "scripts/verify_hardware_evidence.py" in script
    assert 'exit 1' in script
    assert "hardware-proof" in [step.get("with", {}).get("name") for step in jobs["publish"]["steps"]]
