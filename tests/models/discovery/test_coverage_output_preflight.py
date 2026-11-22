"""Coverage preflight errors must leave a new output path usable for direct retry."""
import json
import subprocess
from unittest.mock import Mock

import pytest

from acprof.cli.main import main
from acprof.host.run_state import MeasurementLock, ResultDirectoryLock
from tests.execution.planning.test_resolution_decisions import candidate


@pytest.fixture
def gpu_batch(tmp_path, monkeypatch):
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({
        "schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
        "models": [{"model_id": "example/model", "revision": "a" * 40, "weight": 1}],
    }))
    output = tmp_path / "report"
    args = ["coverage", "run", str(sample), "--output-dir", str(output), "--validate-runtime", "--gpus", "on"]
    query = Mock(return_value=subprocess.CompletedProcess([], 0, "GPU-fixture,0,0000:01:00,fixture,512,Disabled\n", ""))
    detect = Mock(return_value=candidate(tag="text-generation"))
    monkeypatch.setenv("ACPROF_GPU_DEVICE", "0")
    monkeypatch.setattr("acprof.host.gpu_device.run_command", query)
    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", lambda *args: None)
    monkeypatch.setattr("acprof.host.detect.detect_task", detect)
    monkeypatch.setattr("acprof.host.automation.check_repository_access", lambda *args, **kwargs: None)
    monkeypatch.setattr("acprof.host.preparation.model_inspection.validate_model_runtime",
                        lambda *args, **kwargs: {"status": "ok"})
    return args, output, query, detect


@pytest.mark.parametrize(("fault", "message"), [
    ("selector", "GPU selector"), ("query", "fixture GPU unavailable"), ("timeout", "timed out"),
])
def test_gpu_preflight_failure_allows_direct_retry(gpu_batch, monkeypatch, capsys, fault, message):
    args, output, query, detect = gpu_batch
    if fault == "selector":
        monkeypatch.setenv("ACPROF_GPU_DEVICE", "all")
    elif fault == "query":
        query.side_effect = [subprocess.CompletedProcess([], 1, "", message), query.return_value]
    else:
        query.side_effect = [subprocess.TimeoutExpired("nvidia-smi", 10), query.return_value]

    assert main(args) == 2
    assert message in capsys.readouterr().err
    assert not output.exists()
    detect.assert_not_called()

    monkeypatch.setenv("ACPROF_GPU_DEVICE", "0")
    assert main(args) == 0
    report = json.loads((output / "coverage.json").read_text())
    assert report["configuration"]["device"] == {"kind": "gpu", "uuid": "GPU-fixture"}
    assert report["rows"][0]["runtime_status"] == "ok"
    detect.assert_called_once()


@pytest.mark.parametrize("case", ["existing", "missing-resume", "locked-resume", "busy-measurement"])
def test_prior_output_and_lock_errors_precede_gpu_preflight(gpu_batch, capsys, case):
    from contextlib import nullcontext

    args, output, query, detect = gpu_batch
    guard = nullcontext()
    if case == "existing":
        output.mkdir()
        (output / "existing").write_bytes(b"preserve")
        error = "File exists"
    elif case == "missing-resume":
        args.append("--resume")
        error = "existing report directory"
    elif case == "locked-resume":
        output.mkdir()
        args.append("--resume")
        guard = ResultDirectoryLock(output)
        error = "结果目录锁"
    else:
        guard = MeasurementLock()
        error = "测量锁"
    with guard:
        assert main(args) == 2
    assert error in capsys.readouterr().err
    query.assert_not_called()
    detect.assert_not_called()
    if case == "existing":
        assert [p.name for p in output.iterdir()] == ["existing"]
        assert (output / "existing").read_bytes() == b"preserve"


def test_competing_output_created_during_gpu_preflight_is_preserved(gpu_batch, capsys):
    args, output, query, detect = gpu_batch

    def competing_writer(*args, **kwargs):
        output.mkdir()
        (output / "existing").write_bytes(b"other writer")
        return query.return_value

    query.side_effect = competing_writer
    assert main(args) == 2
    assert "File exists" in capsys.readouterr().err
    assert [p.name for p in output.iterdir()] == ["existing"]
    assert (output / "existing").read_bytes() == b"other writer"
    detect.assert_not_called()


def test_failed_gpu_resume_preserves_frozen_files_and_can_retry(gpu_batch, capsys):
    args, output, query, detect = gpu_batch
    assert main(args) == 0
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    args.append("--resume")
    query.side_effect = [subprocess.CompletedProcess([], 1, "", "fixture driver unavailable"), query.return_value]
    assert main(args) == 2
    assert "fixture driver unavailable" in capsys.readouterr().err
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == before
    assert main(args) == 0
    detect.assert_called_once()


def test_configuration_read_failure_precedes_new_output(gpu_batch, monkeypatch):
    args, output, query, detect = gpu_batch
    monkeypatch.setattr("acprof.host.coverage_state.coverage_configuration",
                        Mock(side_effect=RuntimeError("fixture configuration reader failed")))
    with pytest.raises(RuntimeError, match="fixture configuration reader failed"):
        main(args)
    assert not output.exists()
    query.assert_called_once()
    detect.assert_not_called()
