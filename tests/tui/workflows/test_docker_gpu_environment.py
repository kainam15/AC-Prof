"""Docker preflight must query the requested physical device and verify that Docker can expose its UUID."""
import json
import os
from dataclasses import replace
from subprocess import CompletedProcess
from unittest.mock import Mock

import pytest

from acprof.capabilities import Capability
from acprof.experiments.configuration import RunConfig
from acprof.host import gpu_device
from acprof.tui.run.diagnostics import quick_preflight
from acprof.tui.run.docker_diagnostics import DockerDiagnostic, DockerHost


@pytest.fixture
def docker_project(tmp_path, monkeypatch):
    for key in ("ACPROF_GPU_DEVICE", "DEVICE_INDEX", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("acprof.tui.run.diagnostics.inspect_docker",
                        lambda *args: DockerDiagnostic("ready", DockerHost(system="Linux")))
    monkeypatch.setattr("acprof.tui.run.diagnostics.probe_cpu_energy",
                        lambda: Capability("available", "fixture", "fixture"))
    monkeypatch.setattr("acprof.tui.run.diagnostics.probe_perf_instructions",
                        lambda **kwargs: Capability("available", "fixture", "fixture"))
    monkeypatch.setattr("acprof.tui.run.diagnostics.shutil.which", lambda cmd, **kwargs: f"/bin/{cmd}")
    plan = Mock(return_value={"probes": {}, "rapl_topology": {}})
    monkeypatch.setattr("acprof.host.runtimes.capabilities.probe_collection_plan", plan)
    monkeypatch.setattr("acprof.host.runtimes.capabilities.probe_gpu_telemetry", lambda *a, **kw: {
        "gpu_memory": Capability("available", source="nvml")})
    queries = []

    def command(cmd, *, timeout):
        program = os.path.basename(cmd[0])
        if program == "docker":
            assert cmd[1:3] == ["--host", "unix:///var/run/docker.sock"]
            if cmd[3:5] == ["image", "inspect"]:
                return CompletedProcess(cmd, 0, json.dumps({"image_id": "sha256:" + "a" * 64, "os": "linux", "architecture": "amd64"}), "")
            if cmd[3:4] == ["ps"]:
                return CompletedProcess(cmd, 0, "", "")
            assert cmd[3] == "run"
            uuid = cmd[cmd.index("--gpus") + 1].removeprefix("device=")
            return CompletedProcess(cmd, 0, uuid + "\n", "")
        if program != "nvidia-smi":
            assert program in {"ip", "getcap"}
            return CompletedProcess(cmd, 0, "fixture cap_net_raw=ep", "")
        queries.append(list(cmd))
        selector = next((opt[5:] for opt in cmd if opt.startswith("--id=")), None)
        if selector is None:
            return CompletedProcess(cmd, 0, "First GPU\nSecond GPU\n", "")
        index = selector.removeprefix("GPU-")
        if index not in {"0", "1", "2"}:
            return CompletedProcess(cmd, 1, "", "No such GPU")
        mig = "Enabled" if index == "2" else "Disabled"
        return CompletedProcess(cmd, 0, f"GPU-{index}, {index}, 01:00, Fixture, 80, {mig}\n", "")

    # Pinning and the Textual worker use the default host command runner.
    monkeypatch.setattr(gpu_device, "run_command", lambda cmd, **kwargs: command(cmd, timeout=kwargs["timeout"]))
    monkeypatch.setattr("acprof.tui.run.diagnostics.completed_command", command)
    return tmp_path, command, queries, plan


@pytest.mark.parametrize("mode", ["best_effort", "basic", "full"])
@pytest.mark.parametrize(("selector", "status", "detail"), [
    ("GPU-1", "ok", "GPU-1"), ("99", "fail", "No such GPU"), ("2", "fail", "MIG devices"),
    ("all", "fail", "one physical GPU"), ("", "fail", "one physical GPU"),
])
def test_docker_preflight_checks_the_project_device(docker_project, mode, selector, status, detail):
    project, command, queries, plan = docker_project
    (project / ".env").write_text("DEVICE_INDEX=0\n")
    (project / ".env.local").write_text(f"ACPROF_GPU_DEVICE={selector}\nCUDA_VISIBLE_DEVICES=\n")
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", profiling_mode=mode, gpus="on")
    before = os.environ.copy()
    checks = quick_preflight(config, project_dir=project, command_runner=command)
    row, = (r for r in checks if r.check_id == "gpu_device")
    assert row.status == status
    assert detail in row.detail
    assert row.source == "nvidia-smi"
    assert os.environ.copy() == before
    assert len(queries) == int(selector not in {"", "all"})
    if queries:
        assert f"--id={selector}" in queries[0]
    if status == "ok":
        assert next(r for r in checks if r.check_id == "docker_gpu").status == "ok"
        if mode != "best_effort":
            assert next(r for r in checks if r.check_id == "gpu_memory").source == "nvml"
    else:
        assert not any(r.check_id == "docker_gpu" for r in checks)
    if mode == "best_effort":
        gpu = plan.call_args.kwargs["gpu"]
        assert gpu.get("uuid", gpu.get("error")) == row.detail


@pytest.mark.parametrize("runtime", ["docker", "auto"])
def test_docker_preflight_prefers_explicit_physical_device_and_skips_cpu(docker_project, runtime):
    project, command, queries, _ = docker_project
    (project / ".env.local").write_text("ACPROF_GPU_DEVICE=99\nCUDA_VISIBLE_DEVICES=7\n")
    base = replace(RunConfig.smoke("fixture/model"), runtime=runtime)
    for mode in ("off", "on"):
        config = replace(base, gpus=mode, extra_options={**base.extra_options, "gpu_device": "1"})
        checks = quick_preflight(config, project_dir=project, command_runner=command)
        rows = [r for r in checks if r.check_id == "gpu_device"]
        if mode == "on":
            row, = rows
            assert (row.status, row.detail) == ("ok", "GPU-1")
            assert len(queries) == 1
        else:
            assert not queries
            assert not rows


def test_quick_probe_does_not_reuse_or_replace_an_enclosing_run_pin(docker_project):
    project, command, queries, _ = docker_project
    (project / ".env.local").write_text("DEVICE_INDEX=1\n")
    with gpu_device.gpu_device_scope():
        assert gpu_device.pin_gpu_device("0")["uuid"] == "GPU-0"
        config = replace(RunConfig.smoke("fixture/model"), runtime="docker", gpus="on")
        checks = quick_preflight(config, project_dir=project, command_runner=command)
        row, = (row for row in checks if row.check_id == "gpu_device")
        assert (row.status, row.detail) == ("ok", "GPU-1")
        assert gpu_device.resolve_gpu_device()["uuid"] == "GPU-0"
        assert len(queries) == 2
    assert gpu_device.selected_gpu_device() == {}


async def test_docker_preflight_rechecks_a_corrected_project_device(docker_project, monkeypatch):
    from textual.widgets import Button

    from acprof.tui.app import AcprofTui
    from tests.support.tui_waits import wait_for_condition

    project, _, _, _ = docker_project
    monkeypatch.setattr("acprof.tui.app.PROJECT_DIR", project)
    setting = project / ".env.local"
    setting.write_text("ACPROF_GPU_DEVICE=99\n")
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", gpus="on")
    app = AcprofTui(config, settings_path=project / "settings.json")

    def ready_with(status):
        return not app.preflight.check_running and any(
            row.check_id == "gpu_device" and row.status == status for row in app.preflight.checks)

    async with app.run_test(size=(120, 36)) as pilot:
        await wait_for_condition(pilot, lambda: ready_with("fail"), description="missing Docker GPU", timeout=6)
        start = app.query_one("#start-run", Button)
        assert start.disabled
        setting.write_text("ACPROF_GPU_DEVICE=GPU-1\n")
        app.preflight.check_permission_changes()
        await wait_for_condition(pilot, lambda: ready_with("ok") and not start.disabled,
                                 description="available Docker GPU restored", timeout=6)
        row, = (row for row in app.preflight.checks if row.check_id == "gpu_device")
        assert row.detail == "GPU-1"


@pytest.mark.parametrize(("failure", "reason"), [
    (FileNotFoundError("nvidia-smi not found"), "dependency_unavailable"),
    (PermissionError("nvidia-smi permission denied"), "error"),
])
def test_docker_query_failure_retains_source_and_reason(docker_project, failure, reason):
    project, _, _, _ = docker_project
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", profiling_mode="basic", gpus="on")
    checks = quick_preflight(config, project_dir=project, command_runner=Mock(side_effect=failure))
    row, = (row for row in checks if row.check_id == "gpu_device")
    assert (row.status, row.source, row.reason_code, row.detail) == ("fail", "nvidia-smi", reason, str(failure))


@pytest.mark.parametrize("mode", ["best_effort", "basic", "full"])
def test_engine_and_host_gpu_do_not_mask_container_failure(docker_project, mode):
    from acprof.tui.i18n import translate
    from acprof.tui.run.environment_view import repair_guidance
    from acprof.tui.run.global_status import build_status

    project, command, _, _ = docker_project
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", gpus="on", profiling_mode=mode)

    def fail_container(cmd, *, timeout):
        if cmd[0] == "docker" and cmd[3:4] == ["run"]:
            assert "--pull=never" in cmd
            assert "--rm" in cmd
            assert cmd[cmd.index("--gpus") + 1] == "device=GPU-0"
            assert cmd[cmd.index("--entrypoint") + 1] == "nvidia-smi"
            return CompletedProcess(cmd, 125, "", "failed to discover GPU vendor from CDI")
        return command(cmd, timeout=timeout)

    checks = quick_preflight(config, project_dir=project, command_runner=fail_container)
    rows = {row.check_id: row for row in checks}
    assert rows["docker"].status == rows["gpu_device"].status == "ok"
    row = rows["docker_gpu"]
    assert row.status == "fail"
    assert "CDI" in row.detail
    assert "NVIDIA Container Toolkit" in translate(repair_guidance(row), "en")
    status = build_status(config, issues=[], checks=checks, inventory=[], validation_pending=False,
                          running=False, inventory_running=False, current=True, completed=True)
    assert any(item.check.check_id == "docker_gpu" for item in status.blockers)


def test_first_gpu_run_without_cached_probe_can_prepare_a_model(docker_project):
    from acprof.tui.run.global_status import build_status

    project, command, _, _ = docker_project
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", gpus="on")
    def empty_cache(cmd, *, timeout):
        if cmd[0] == "docker":
            assert cmd[3:5] == ["image", "inspect"]
            return CompletedProcess(cmd, 1, "", "No such image")
        return command(cmd, timeout=timeout)

    checks = quick_preflight(config, project_dir=project, command_runner=empty_cache)
    row, = (item for item in checks if item.check_id == "docker_gpu")
    assert (row.status, row.reason_code) == ("warn", "probe_image_missing")
    status = build_status(config, issues=[], checks=checks, inventory=[], validation_pending=False,
                          running=False, inventory_running=False, current=True, completed=True)
    assert not any(item.check.check_id == "docker_gpu" for item in status.blockers)


def test_gpu_probe_does_not_run_against_unsupported_docker_endpoint(docker_project, monkeypatch):
    project, command, _, _ = docker_project
    diagnostic = DockerDiagnostic("context_unsupported", DockerHost(system="Linux"))
    monkeypatch.setattr("acprof.tui.run.diagnostics.inspect_docker", lambda *args: diagnostic)
    def host_only(cmd, *, timeout):
        assert cmd[0] != "docker"
        return command(cmd, timeout=timeout)

    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", gpus="on")
    checks = quick_preflight(config, project_dir=project, command_runner=host_only)
    assert any(row.check_id == "docker" and row.status == "fail" for row in checks)
    assert not any(row.check_id == "docker_gpu" for row in checks)
