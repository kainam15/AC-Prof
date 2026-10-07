"""Release validation leases only after positive container cleanup evidence."""
import asyncio
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_runtime_validation import SUCCESS, TestRuntimeValidation as _ValidationFixture

from acprof.host.container_lifecycle import ContainerCleanupError
from acprof.host.runtime_validation import RuntimeValidationError, validate_runtime

CID = "c" * 64


@pytest.fixture
def validation(tmp_path, monkeypatch):
    options = _ValidationFixture().fixture(tmp_path)
    mount = SimpleNamespace(args=[], close=Mock())
    retained = Mock()
    monkeypatch.setattr("acprof.host.model_store.acquire_mount", Mock(return_value=mount))
    monkeypatch.setattr("acprof.host.model_store.retain_mount_for_cleanup_debt", retained)
    monkeypatch.setattr("acprof.host.runtime_validation.recover_abandoned_containers", Mock())
    monkeypatch.setattr("acprof.host.container_state.inspect_container_state", lambda identifier: {})
    return options, mount, retained


@pytest.mark.parametrize("early", ["missing", "invalid"])
@pytest.mark.parametrize("removed", ["rm_success", "confirmed_absent"])
def test_late_cidfile_and_confirmed_cleanup_release_lease_but_keep_timeout(tmp_path, monkeypatch, validation, early, removed):
    options, mount, retained = validation
    commands = []
    first = "" if early == "missing" else ContainerCleanupError("", [{"operation": "read_cidfile"}])
    monkeypatch.setattr("acprof.host.runtime_validation.owned_container_id", Mock(side_effect=[first, CID]))

    def run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(command, 30)
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(command, 1, "", f"No such object: {CID}")
        return subprocess.CompletedProcess(command, 0 if removed == "rm_success" else 1, "", "")

    monkeypatch.setattr("acprof.host.runtime_validation.run_command", run)
    with pytest.raises(RuntimeValidationError) as caught:
        validate_runtime(**options)
    assert caught.value.failure.reason_code == "compatibility_budget_exhausted"
    report = json.loads((tmp_path / "runtime_validation.json").read_text())
    assert report["status"] == "inconclusive"
    assert "cleanup_status" not in report
    assert set(report["devices"]) == {"off"}
    assert sum(command[:2] == ["docker", "run"] for command in commands) == 1
    mount.close.assert_called_once_with()
    retained.assert_not_called()


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_uncertain_start_without_id_retains_lease(tmp_path, monkeypatch, validation, error_type):
    options, mount, retained = validation
    primary = error_type("interrupted Docker start")
    monkeypatch.setattr("acprof.host.runtime_validation.owned_container_id", lambda path: "")
    monkeypatch.setattr("acprof.host.runtime_validation.run_command", Mock(side_effect=primary))
    with pytest.raises(BaseException) as caught:
        validate_runtime(**options)
    if error_type is OSError:
        assert isinstance(caught.value, ContainerCleanupError)
        assert caught.value.run_error is primary
        report = json.loads((tmp_path / "runtime_validation.json").read_text())
        assert report["cleanup_status"] == "incomplete"
        assert report["status"] == "error"
    else:
        assert caught.value is primary
    mount.close.assert_not_called()
    retained.assert_called_once_with(mount)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit, asyncio.CancelledError])
@pytest.mark.parametrize("stage", ["read_id", "remove"])
def test_cleanup_interruption_keeps_lease_and_original_exception(monkeypatch, validation, error_type, stage):
    options, mount, retained = validation
    primary = error_type("cleanup interrupted")
    monkeypatch.setattr("acprof.host.runtime_validation.run_command",
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, SUCCESS, ""))
    monkeypatch.setattr("acprof.host.runtime_validation.owned_container_id",
                        Mock(side_effect=[CID, primary]) if stage == "read_id" else lambda path: CID)
    monkeypatch.setattr("acprof.host.runtime_validation.remove_owned_container",
                        Mock(side_effect=primary) if stage == "remove" else Mock())
    with pytest.raises(error_type) as caught:
        validate_runtime(**options)
    assert caught.value is primary
    mount.close.assert_not_called()
    retained.assert_called_once_with(mount)


def test_command_setup_failure_without_docker_attempt_releases_lease(monkeypatch, validation):
    options, mount, retained = validation
    primary = RuntimeError("invalid environment before launch")
    monkeypatch.setattr("acprof.host.runtime_validation.runtime_docker_env_args", Mock(side_effect=primary))
    monkeypatch.setattr("acprof.host.runtime_validation.owned_container_id", lambda path: "")
    run = Mock(side_effect=AssertionError("Docker must not run"))
    monkeypatch.setattr("acprof.host.runtime_validation.run_command", run)
    with pytest.raises(RuntimeError) as caught:
        validate_runtime(**options)
    assert caught.value is primary
    run.assert_not_called()
    mount.close.assert_called_once_with()
    retained.assert_not_called()
