"""Exercise real project configuration through the doctor command and probe boundary."""
import json
import os
from unittest.mock import patch

import pytest

from acprof.capabilities import Capability
from acprof.cli import main as cli
from acprof.host import doctor
from acprof.host.runtimes import capabilities


@pytest.fixture
def docker_probes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(doctor.preflight, "require_collection_host", lambda **_: None)
    monkeypatch.setattr(doctor.preflight, "require_cgroup_prerequisites", lambda: "v2")
    monkeypatch.setattr(doctor.preflight, "require_native_docker", lambda: "daemon")
    monkeypatch.setattr(doctor, "_command", lambda _: "buildx")
    monkeypatch.setattr(doctor, "_resources", lambda: "bundled")
    monkeypatch.setattr(doctor, "_packet", lambda _: "ok")
    seen = []

    def perf(*, env):
        seen.append(env)
        return Capability("available", source="fixture")

    def plan(runtime, *, env, **_):
        assert runtime == "docker"
        perf(env=env)
        return {"probes": {"cpu_instructions": {"status": "available", "detail": "fixture"}}}

    monkeypatch.setattr(doctor.preflight, "probe_cpu_energy", lambda: Capability("available", source="fixture"))
    monkeypatch.setattr(doctor.preflight, "probe_perf_instructions", perf)
    monkeypatch.setattr(capabilities, "probe_collection_plan", plan)
    with patch.dict(os.environ, {"ACPROF_DOCTOR_SHELL": "inherited"}, clear=True):
        yield seen


@pytest.mark.parametrize("mode", ["full", "best_effort"])
@pytest.mark.parametrize("error", ["retired", "directory", "encoding", "nul"])
def test_invalid_project_environment_keeps_diagnostics(tmp_path, capsys, docker_probes, mode, error):
    config = tmp_path / ".env.local"
    if error == "retired":
        config.write_text("ACPROF_SUDO_PASSWORD=fixture-only\n")
    elif error == "directory":
        config.mkdir()
    elif error == "nul":
        config.write_bytes(b"ACPROF_DOCTOR_PARTIAL=first\nACPROF_DOCTOR_INVALID=fixture-only\x00\n")
    else:
        config.write_bytes(b"DOCTOR_TEST=\xff\n")

    assert cli.main(["doctor", "--runtime", "docker", "--profiling-mode", mode, "--json"]) == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert not report["ready"]
    failures = [item for item in report["checks"] if item["status"] == "unavailable"]
    assert [item["name"] for item in failures] == ["environment"]
    assert failures[0]["required"]
    assert failures[0]["remedy"]
    reason = {"retired": "ACPROF_SUDO_PASSWORD", "directory": ".env.local",
              "encoding": "utf-8", "nul": "ACPROF_DOCTOR_INVALID"}
    assert reason[error] in failures[0]["detail"]
    assert "fixture-only" not in output.out + output.err
    assert "ACPROF_DOCTOR_PARTIAL" not in os.environ
    assert len(docker_probes) == 1
    assert docker_probes[0]["ACPROF_DOCTOR_SHELL"] == "inherited"
    assert "ACPROF_DOCTOR_INVALID" not in docker_probes[0]
    assert any(item["name"] == "workspace" and item["status"] == "available" for item in report["checks"])


@pytest.mark.parametrize("mode", ["full", "best_effort"])
def test_doctor_uses_configuration_loaded_at_start(tmp_path, capsys, monkeypatch, docker_probes, mode):
    config = tmp_path / ".env.local"
    config.write_text("ACPROF_DOCTOR_FIXTURE=from-file\nACPROF_DOCTOR_SHELL=from-file\n")

    def change_file(**_):
        assert os.environ["ACPROF_DOCTOR_FIXTURE"] == "from-file"
        config.write_text("ACPROF_SUDO_PASSWORD=\n")

    monkeypatch.setattr(doctor.preflight, "require_collection_host", change_file)
    assert cli.main(["doctor", "--runtime", "docker", "--profiling-mode", mode, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ready"]
    assert [env["ACPROF_DOCTOR_FIXTURE"] for env in docker_probes] == ["from-file"]
    assert docker_probes[0]["ACPROF_DOCTOR_SHELL"] == "inherited"


def test_direct_docker_checks_load_local_settings_without_modifying_shell(tmp_path, docker_probes):
    (tmp_path / ".env").write_text("ACPROF_DOCTOR_FIXTURE=from-file\n")
    doctor.collect_checks(runtime="docker", output_dir=tmp_path)
    assert [env["ACPROF_DOCTOR_FIXTURE"] for env in docker_probes] == ["from-file"]
    assert "ACPROF_DOCTOR_FIXTURE" not in os.environ
