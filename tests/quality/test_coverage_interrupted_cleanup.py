"""Restore cleanup obligations that were durable before the per-model attempt row."""
import json
from contextlib import nullcontext
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from acprof.cli.main import main
from acprof.host.coverage_state import CoverageState
from acprof.models.contract import record_runtime_validation, write_model_resolution
from tests.execution.planning.test_resolution_decisions import candidate

CONTAINER = "c" * 64
MODEL = "example/model"


@pytest.fixture
def interrupted(tmp_path, monkeypatch):
    root = tmp_path / "coverage"
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
                                 "models": [{"model_id": MODEL, "revision": "a" * 40, "weight": 1}]}))
    args = ["coverage", "run", str(sample), "--output-dir", str(root), "--validate-runtime"]
    monkeypatch.setattr("acprof.host.run_state.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.host.run_state.host_identity", lambda _: {"machine_id_sha256": "fixture"})
    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", lambda *a, **kw: None)
    monkeypatch.setattr("acprof.host.automation.check_repository_access", lambda *a, **kw: None)
    monkeypatch.setattr("acprof.host.detect.detect_task", lambda *a, **kw: candidate(tag="text-generation"))

    def run_until_interrupt(kind):
        def validate(task, directory, **kwargs):
            report = {"schema_version": 1, "mode": "full", "status": "error", "scope":
                      "isolated_minimum_scale_predict_and_postprocess_before_measurement",
                      "image_id": "sha256:" + "d" * 64, "build_fingerprint": "fixture", "payload_sha256": "e" * 64,
                      "devices": {}, "cleanup_status": "incomplete",
                      "cleanup_error": {"status": "incomplete", "container_id": CONTAINER, "final_state": "unknown"}}
            if kind == "clean":
                report.pop("cleanup_status")
                report.pop("cleanup_error")
                report["status"] = "ok"
            if kind == "unknown-id":
                report["cleanup_error"]["container_id"] = ""
            if kind != "unbound":
                record_runtime_validation(task, report)
            if kind == "wrong-model":
                task.model_resolution["provenance"]["sources"]["repository_snapshot"]["model_id"] = "different/model"
            if kind == "wrong-report":
                report["payload_sha256"] = "f" * 64
            write_model_resolution(task, directory)
            (directory / "runtime_validation.json").write_text(json.dumps(report))
            return report

        with patch("acprof.host.preparation.model_inspection.validate_model_runtime", side_effect=validate), patch.object(
                CoverageState, "record", side_effect=KeyboardInterrupt), pytest.raises(KeyboardInterrupt):
            main(args)
        origin = root / "attempts" / "attempt-000001"
        frozen = {p: p.read_bytes() for p in origin.rglob("*.json")}
        attempt = json.loads((origin / "attempt.json").read_bytes())
        assert attempt["active_model_index"] == 0
        assert not attempt["rows"]
        return args, root, frozen
    return run_until_interrupt


def assert_unchanged(frozen):
    assert all(path.read_bytes() == raw for path, raw in frozen.items())


def test_unjournaled_cleanup_requires_explicit_retry_and_original_container_proof(interrupted, capsys):
    args, root, frozen = interrupted("bound")
    events = []
    def read_inventory(command, **kwargs):
        if command == ["docker", "inspect", CONTAINER]:
            events.append("inspect")
            return CompletedProcess(command, 1, stdout="", stderr=f"No such object: {CONTAINER}")
        assert command[:4] == ["docker", "ps", "-aq", "--no-trunc"]
        events.append("inventory")
        return CompletedProcess(command, 0, stdout="", stderr="")

    def verify(*args, **kwargs):
        events.append("probe")
        return {"status": "ok"}

    with patch("acprof.host.preflight.require_native_docker"), patch(
            "acprof.system.command.run_command", side_effect=read_inventory), patch(
            "acprof.host.preparation.model_inspection.validate_model_runtime", side_effect=verify):
        assert main([*args, "--resume"]) == 2
        assert "cleanup" in capsys.readouterr().err
        assert events == []
        assert main([*args, "--resume", "--retry-stage", "cleanup"]) == 0
        assert events == ["inspect", "inventory", "probe"]
        assert main([*args, "--resume"]) == 0
        assert events == ["inspect", "inventory", "probe"]
    report = json.loads((root / "coverage.json").read_bytes())
    assert report["rows"][0]["cleanup_recovery"]["verified_container_ids"] == [CONTAINER]
    assert report["rows"][0]["cleanup_recovery"]["scoped_inventory_empty"] is True
    assert report["rows"][0]["runtime_status"] == "ok"
    assert_unchanged(frozen)


@pytest.mark.parametrize("kind", ["wrong-model", "wrong-report", "unbound", "unknown-id"])
def test_unbound_cleanup_requires_an_empty_read_only_owner_inventory(interrupted, capsys, kind):
    args, root, frozen = interrupted(kind)
    inventory = [CONTAINER]
    commands = []

    def list_containers(command, **kwargs):
        commands.append(command)
        assert command[:4] == ["docker", "ps", "-aq", "--no-trunc"]
        assert sum(arg == "--filter" for arg in command) == 3
        return CompletedProcess(command, 0, stdout="\n".join(inventory), stderr="")

    with patch("acprof.host.preflight.require_native_docker"), patch(
            "acprof.system.command.run_command", side_effect=list_containers), patch(
            "acprof.host.preparation.model_inspection.validate_model_runtime", return_value={"status": "ok"}) as probe:
        assert main([*args, "--resume"]) == 2
        assert not commands
        assert main([*args, "--resume", "--retry-stage", "cleanup"]) == 2
        assert len(commands) == 1
        assert "cleanup" in capsys.readouterr().err
        probe.assert_not_called()
        inventory.clear()  # The operator has independently resolved the original owner scope.
        assert main([*args, "--resume", "--retry-stage", "cleanup"]) == 0
        assert len(commands) == 2
        probe.assert_called_once()
    proof = json.loads((root / "coverage.json").read_bytes())["rows"][0]["cleanup_recovery"]
    assert proof["scoped_inventory_empty"] is True
    assert proof["recovered_container_ids"] == []
    assert_unchanged(frozen)

def test_interruption_after_clean_validation_still_reprobes_without_cleanup(interrupted):
    args, root, frozen = interrupted("clean")
    with patch("acprof.host.coverage_state.confirm_previous_cleanup") as confirm, patch(
            "acprof.host.preparation.model_inspection.validate_model_runtime", return_value={"status": "ok"}) as probe:
        assert main([*args, "--resume"]) == 0
        confirm.assert_not_called()
        probe.assert_called_once()
    assert json.loads((root / "coverage.json").read_text())["rows"][0]["runtime_status"] == "ok"
    assert_unchanged(frozen)


@pytest.mark.parametrize("next_probe", ["clean", "new-debt", "wrong-origin", "missing-inventory-proof"])
def test_cleanup_proven_before_second_interruption_stays_bound_to_its_debt(interrupted, next_probe):
    args, root, original = interrupted("bound")
    proof = {"status": "complete", "verified_container_ids": [CONTAINER], "scoped_inventory_empty": True}
    with patch("acprof.host.coverage_state.confirm_previous_cleanup", return_value=proof) as confirm:
        def new_validation(task, directory, **kwargs):
            if next_probe == "new-debt":
                validation = {"mode": "full", "status": "error", "image_id": "sha256:" + "d" * 64,
                              "build_fingerprint": "fixture", "payload_sha256": "e" * 64, "devices": {},
                              "cleanup_status": "incomplete", "cleanup_error": {"container_id": "b" * 64}}
                record_runtime_validation(task, validation)
                write_model_resolution(task, directory)
                (directory / "runtime_validation.json").write_text(json.dumps(validation))
                return validation
            return {"status": "ok"}
        with patch("acprof.host.preparation.model_inspection.validate_model_runtime", side_effect=new_validation), patch.object(
                CoverageState, "record", side_effect=KeyboardInterrupt), pytest.raises(KeyboardInterrupt):
            main([*args, "--resume", "--retry-stage", "cleanup"])
        confirm.assert_called_once()
        attempt_path = root / "attempts/attempt-000002/attempt.json"
        attempt = json.loads(attempt_path.read_bytes())
        assert attempt["active_cleanup_recovery"]["origin_attempt_id"] == 1
        if next_probe == "wrong-origin":
            attempt["active_cleanup_recovery"]["origin_attempt_id"] = 2
            attempt_path.write_text(json.dumps(attempt))
        elif next_probe == "missing-inventory-proof":
            attempt["active_cleanup_recovery"]["proof"].pop("scoped_inventory_empty")
            attempt_path.write_text(json.dumps(attempt))
        second_frozen = attempt_path.read_bytes()

        with patch("acprof.host.preparation.model_inspection.validate_model_runtime", return_value={"status": "ok"}) as probe:
            rc = main([*args, "--resume"])
            if next_probe == "clean":
                assert rc == 0
                probe.assert_called_once()
                assert json.loads((root / "coverage.json").read_bytes())["rows"][0]["cleanup_recovery"] == proof
            else:
                assert rc == 2
                probe.assert_not_called()
            assert confirm.call_count == 1
            if next_probe == "new-debt":
                confirm.return_value = {"status": "complete", "verified_container_ids": ["b" * 64],
                                        "scoped_inventory_empty": True}
                assert main([*args, "--resume", "--retry-stage", "cleanup"]) == 0
                row = confirm.call_args.args[0]
                assert row["cleanup_origin_attempt_id"] == 2
                assert [err["container_id"] for err in row["cleanup_errors"]] == ["b" * 64]
        assert attempt_path.read_bytes() == second_frozen
    assert_unchanged(original)
