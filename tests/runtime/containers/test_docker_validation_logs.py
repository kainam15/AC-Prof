"""Real local command output exercises Docker validation log boundaries without Docker."""
import json
import subprocess
import sys
import tracemalloc
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tests.models.contracts.test_model_contract as contracts
from acprof.system import command as host_command
from tests.runtime.environment.test_runtime_validation import (
    TestRuntimeValidation as _RuntimeFixture,
)


@pytest.fixture(params=["interface", "runtime"])
def docker_validation(request, tmp_path, monkeypatch):
    from acprof.host import container_state, runtime_validation
    from acprof.host.model_store import store as model_store
    from acprof.host.preparation import interface_probe

    kind = request.param
    module = interface_probe if kind == "interface" else runtime_validation
    removed = Mock()
    lease = SimpleNamespace(args=[], close=Mock())
    monkeypatch.setattr(module, "recover_abandoned_containers", Mock())
    monkeypatch.setattr(module, "remove_owned_container", removed)
    monkeypatch.setattr(container_state, "inspect_container_state", lambda _: {"Running": True})
    monkeypatch.setattr(model_store, "acquire_mount", lambda _: lease)
    if kind == "interface":
        task = contracts.TestModelContract().discover()
        monkeypatch.setattr(module, "source_bundle", lambda _: nullcontext(
            SimpleNamespace(root=tmp_path, identity="fixture-source")))
        monkeypatch.setattr(module, "service_sources", lambda _: [])
        monkeypatch.setattr(module, "source_fingerprint", lambda *args, **kwargs: "fixture-runner")
        monkeypatch.setattr(module, "prepare_environment_image", lambda *args: SimpleNamespace(
            image_id="sha256:" + "b" * 64))
        def call():
            return module.probe_interface(task, tmp_path)
        log_path = tmp_path / "logs/interface_validation.log"
        response = {"status": "ok", "stages": [
            {"stage": stage, "status": "verified"} for stage in ("import", "signature")]}
    else:
        options = _RuntimeFixture().fixture(tmp_path)
        options["gpu_list"] = ["off"]
        def call():
            return module.validate_runtime(**options)
        log_path = tmp_path / "runtime_validation_off.log"
        response = {"status": "ok", "stages": [
            {"stage": stage, "status": "verified"} for stage in
            ("load", "preprocess", "predict", "postprocess", "validate_output")],
            "validation": {"protocol": {"status": "verified"}, "task": {"status": "verified"}}}
    outcome = SimpleNamespace(kind=kind, noise_blocks=0, response=response, cancel=False,
                              stdout=None, stderr="stderr retained\n", timeout=False,
                              call=call, path=log_path, removed=removed, root=tmp_path)
    script = tmp_path / "fixture-command.py"

    def run(command, **kwargs):
        assert command[:2] == ["docker", "run"]
        Path(command[command.index("--cidfile") + 1]).write_text("c" * 64)
        stdout = (module.RESULT_PREFIX + json.dumps(outcome.response, ensure_ascii=False) + "\n"
                  if outcome.stdout is None else outcome.stdout)
        script.write_text(
            "import sys,time\n"
            f"for _ in range({outcome.noise_blocks}): sys.stdout.buffer.write(b'x' * 65536)\n"
            f"sys.stdout.buffer.write({(chr(10) + stdout).encode()!r})\n"
            "sys.stdout.flush()\n"
            f"for _ in range({outcome.noise_blocks}): sys.stderr.buffer.write(b'y' * 65536)\n"
            f"sys.stderr.buffer.write({outcome.stderr.encode()!r})\n"
            "sys.stderr.flush()\n"
            + ("time.sleep(60)\n" if outcome.cancel or outcome.timeout else ""), encoding="utf-8")
        if outcome.cancel or outcome.timeout:
            kwargs["timeout"] = 0.5
        try:
            return host_command.run_command([sys.executable, "-I", str(script)], **kwargs)
        except subprocess.TimeoutExpired:
            if outcome.cancel:
                raise KeyboardInterrupt("fixture cancellation") from None
            raise

    monkeypatch.setattr(module, "run_command", run)
    return outcome


def test_large_stdout_and_stderr_are_persisted_with_bounded_python_memory(docker_validation):
    check = docker_validation
    peaks = []
    # Compare growth so unrelated fixed allocations (such as the bounded input
    # plan reader) do not get mistaken for buffered validation output.
    for blocks in (0, 128):
        check.noise_blocks = blocks
        tracemalloc.start()
        try:
            report = check.call()
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert report["status"] == "ok"
    assert check.path.stat().st_size > 16 * 1024 * 1024
    with check.path.open("rb") as stream:
        assert stream.read(32) == b"x" * 32
        stream.seek(-16, 2)
        assert stream.read() == b"stderr retained\n"
    assert check.removed.call_count == 2
    assert peaks[1] - peaks[0] < 2 * 1024 * 1024, f"validation memory grew from {peaks[0]:,} to {peaks[1]:,} bytes"


def test_valid_json_unicode_separators_are_not_log_boundaries(docker_validation):
    check = docker_validation
    check.response["detail"] = "中\u0085文\u2028分\u2029隔"
    report = check.call()
    result = report["validation"] if check.kind == "interface" else report["devices"]["off"]
    assert result["detail"] == check.response["detail"]
    assert check.response["detail"] in check.path.read_text()


def test_cancel_keeps_both_output_channels_after_container_cleanup(docker_validation):
    check = docker_validation
    check.cancel = True
    check.stdout = "partial stdout\n"
    check.stderr = "partial stderr\n"
    with pytest.raises(KeyboardInterrupt, match="fixture cancellation"):
        check.call()
    check.removed.assert_called_once()
    assert check.path.read_text() == "\npartial stdout\n\npartial stderr\n"


def test_stderr_cannot_supply_a_success_record(docker_validation):
    check = docker_validation
    check.stdout = "no result on stdout\n"
    prefix = "ACPROF_INTERFACE_VALIDATION=" if check.kind == "interface" else "ACPROF_RUNTIME_VALIDATION="
    check.stderr = prefix + json.dumps(check.response) + "\n"
    with pytest.raises(RuntimeError):
        check.call()
    assert "no result on stdout" in check.path.read_text()


def test_timeout_keeps_partial_output_and_runtime_phase_evidence(docker_validation):
    check = docker_validation
    check.timeout = True
    stage = {"stage": "predict", "status": "running", "detail": "中\u2028文"}
    check.stdout = 'ACPROF_RUNTIME_STAGE={"stage":"load","status":"verified"}\n'
    check.stderr = "ACPROF_RUNTIME_STAGE=" + json.dumps(stage, ensure_ascii=False) + "\n"
    with pytest.raises(subprocess.TimeoutExpired if check.kind == "interface" else RuntimeError):
        check.call()
    check.removed.assert_called_once()
    assert check.path.read_text() == "\n" + check.stdout + "\n" + check.stderr
    if check.kind == "runtime":
        report = json.loads((check.root / "runtime_validation.json").read_text())
        assert report["status"] == "inconclusive"
        evidence = report["devices"]["off"]["failure"]["evidence"]
        assert evidence["request_phase"] == "predict"
        assert evidence["model_loaded"] is True
        assert evidence["stages"][-1] == stage


def test_invalid_last_result_does_not_reuse_earlier_success(docker_validation):
    check = docker_validation
    prefix = "ACPROF_INTERFACE_VALIDATION=" if check.kind == "interface" else "ACPROF_RUNTIME_VALIDATION="
    check.stdout = prefix + json.dumps(check.response) + "\n" + prefix + "{broken"
    with pytest.raises(ValueError if check.kind == "interface" else RuntimeError):
        check.call()
    check.removed.assert_called_once()
    assert "{broken" in check.path.read_text()
