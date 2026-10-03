"""Contract probes run outside measurements and publish only observed evidence."""
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import test_model_contract as contract_fixture
import test_runtime_validation as runtime_fixture

from acprof.host.runtime_validation import validate_runtime


class TestModelProbe:
    def probe_command(self, response):
        def run(command, **kwargs):
            if command[:2] == ["docker", "run"]:
                cidfile = Path(command[command.index("--cidfile") + 1])
                cidfile.write_text("c" * 64)
                stdout = "ACPROF_RUNTIME_VALIDATION=" + json.dumps(response)
            else:
                assert (command[1]) in ({"ps", "rm"})
                stdout = ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        return run

    def test_empty_device_set_and_mutable_image_cannot_claim_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            options = runtime_fixture.TestRuntimeValidation().fixture(Path(directory))
            with pytest.raises(ValueError):
                validate_runtime(**{**options, "gpu_list": []})
            options["image_info"].tag = "example:latest"
            with pytest.raises(ValueError):
                validate_runtime(**options)

    def test_basic_mode_does_not_claim_inference_and_only_uses_cpu(self):
        response = {"status": "ok", "mode": "basic", "stages": [
            {"stage": stage, "status": "verified"} for stage in ("import", "signature")], "inference": "not_run"}
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.runtime_validation.run_command", side_effect=self.probe_command(response),
        ) as run, patch("acprof.host.container_state.inspect_container_state", return_value={}):
            task = contract_fixture.TestModelContract().discover()
            options = runtime_fixture.TestRuntimeValidation().fixture(Path(directory))
            options.update(task_info=task, gpu_list=["off"], mode="basic")
            report = validate_runtime(**options)
            assert (report["mode"]) == ("basic")
            assert (task.model_resolution["contract"]["runtime_validation"]["status"]) == ("basic_verified")
            command = next(call.args[0] for call in run.call_args_list if call.args[0][:2] == ["docker", "run"])
            assert ("ACPROF_CONTRACT_PROBE_MODE=basic") in (command)
            assert ("--gpus") not in (command)
            with pytest.raises(ValueError):
                validate_runtime(**{**options, "gpu_list": ["on"]})

    def test_probe_is_readonly_and_updates_the_contract_report(self):
        commands = []
        stages = [{"stage": stage, "status": "verified"} for stage in
                  ("load", "preprocess", "predict", "postprocess", "validate_output")]
        response = {"status": "ok", "stages": stages, "validation": {
            "protocol": {"status": "verified"}, "task": {"status": "verified"}}}
        execute = self.probe_command(response)
        def run(command, **kwargs):
            commands.append(command)
            return execute(command, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.runtime_validation.run_command", side_effect=run,
        ), patch("acprof.host.container_state.inspect_container_state", return_value={}):
            task = contract_fixture.TestModelContract().discover()
            options = runtime_fixture.TestRuntimeValidation().fixture(Path(directory))
            options.update(task_info=task, gpu_list=["off"])
            validate_runtime(**options)
            command = next(command for command in commands if command[:2] == ["docker", "run"])
            assert ("--read-only") in (command)
            assert ("no-new-privileges") in (command)
            assert ("--gpus") not in (command)
            saved = json.loads(Path(directory, "model_resolution.json").read_text())["contract"]
            assert (saved["runtime_validation"]["status"]) == ("verified")
            assert (saved["runtime_validation"]["image_id"]) == ("sha256:" + "b" * 64)
            assert (saved["runtime_validation"]["mode"]) == ("full")
            assert (list(Path(directory).glob("*.csv"))) == ([])

    def test_incomplete_runtime_record_never_becomes_verified(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.runtime_validation.run_command", side_effect=self.probe_command({"status": "ok"}),
        ), patch("acprof.host.container_state.inspect_container_state", return_value={}):
            task = contract_fixture.TestModelContract().discover()
            options = runtime_fixture.TestRuntimeValidation().fixture(Path(directory))
            options.update(task_info=task, gpu_list=["off"])
            with pytest.raises(RuntimeError, match="contract|validation"):
                validate_runtime(**options)
            assert (task.model_resolution["contract"]["runtime_validation"]["status"]) == ("error")
