"""Machine diagnostics stay independent of experiment selections and launch policy."""
import asyncio
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Select, Static

from acprof.capabilities import Capability
from acprof.experiments.configuration import RunConfig
from acprof.platform import Environment
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.environment_checks import scan_environment
from acprof.tui.ui.environment import display_checks


def test_machine_results_survive_unselected_gpu_and_changed_experiment():
    machine = (
        PreflightCheck("GPU power", "ok", "NVML read succeeded", check_id="gpu_power"),
        PreflightCheck("NCU", "fail", "not installed", check_id="ncu"),
    )
    controller = SimpleNamespace(
        environment_checks=machine, environment_error="", check_completed=True,
        checks=(), error="", selected_runtime="", config_matches=lambda: False,
        preflight_config=lambda: RunConfig.smoke(""), app=SimpleNamespace(environment=Environment("native_linux")),
    )
    assert display_checks(controller) == machine


@pytest.fixture
def host(monkeypatch, tmp_path):
    from acprof.tui.run import diagnostics, environment_checks

    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cpu.stat").write_text("usage_usec 123\n")
    (root / "memory.current").write_text("4096\n")
    monkeypatch.setattr(environment_checks, "current_cgroup", lambda: root)
    monkeypatch.setattr(environment_checks, "require_process_supervision", lambda: "pidfd verified")
    monkeypatch.setattr(environment_checks.socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0")])
    for module in (diagnostics, environment_checks):
        monkeypatch.setattr(module, "detect_environment", lambda: Environment("native_linux"))
    rapl = Mock(return_value=Capability("available", "package energy read", "rapl_probe"))
    perf = Mock(return_value=Capability("available", source="perf_probe"))
    monkeypatch.setattr(diagnostics, "probe_cpu_energy", rapl)
    monkeypatch.setattr(diagnostics, "probe_perf_instructions", perf)
    monkeypatch.setattr(environment_checks.shutil, "which", lambda name: "/tools/" + name)
    monkeypatch.setattr(environment_checks, "find_executable", lambda root, names: "/tools/" + names[0])
    monkeypatch.setattr(environment_checks, "find_nsys_executable", lambda root: "/tools/nsys")
    process = SimpleNamespace(cpu_times=lambda: (1, 2), memory_info=lambda: (4096, 8192),
                              children=lambda **kwargs: [])
    monkeypatch.setattr("psutil.Process", lambda: process)
    nvml = SimpleNamespace(
        nvmlInit=Mock(), nvmlShutdown=Mock(), nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda index: index, nvmlDeviceGetName=lambda handle: f"fixture GPU {handle}",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=4096),
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=0),
        nvmlDeviceGetPowerUsage=Mock(return_value=5000),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    commands = []

    def command_runner(command, *, timeout):
        commands.append(tuple(command))
        assert 0 < timeout <= 15
        if "inspect" in command:
            output = "unix:///var/run/docker.sock"
        elif "getcap" in command[0]:
            output = "/tools/tcpdump cap_net_raw=eip"
        elif "image" in command:
            output = "123456\n"
        else:
            output = "fixture version / host"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    return SimpleNamespace(runner=command_runner, commands=commands, nvml=nvml, rapl=rapl, perf=perf, root=root)


def test_scan_checks_all_host_interfaces_and_tools_without_experiment(host, tmp_path):
    rows = {row.check_id: row for row in scan_environment(project_dir=tmp_path, command_runner=host.runner)}
    for name in ("docker", "process_runtime", "process_cpu", "process_memory", "container_cpu", "container_memory",
                 "cpu_energy", "cpu_instructions", "gpu_device", "gpu_memory", "gpu_utilization", "gpu_power",
                 "ncu", "nsys", "intel_advisor", "uv", "tcpdump", "tshark"):
        assert rows[name].status == "ok", name
    assert "eth0" in rows["capture_interface"].detail
    assert "GPU 0" in rows["gpu_power"].detail
    assert "GPU 1" in rows["gpu_power"].detail
    assert "0 %" in rows["gpu_utilization"].detail, "zero is valid evidence"
    assert all(row.status not in {"declared", "not_requested"} for row in rows.values())
    assert rows["torch"].status == rows["massif"].status == "pending", "image presence is not a tool execution check"
    assert not any("run" in command or "build" in command or "pull" in command for command in host.commands)
    host.rapl.assert_called_once()
    host.perf.assert_called_once()
    host.nvml.nvmlShutdown.assert_called_once()


def test_gpu_partial_support_and_tool_timeout_do_not_hide_other_checks(host, tmp_path):
    host.nvml.nvmlDeviceGetPowerUsage.side_effect = [5000, RuntimeError("Not Supported")]
    (host.root / "memory.current").write_text("invalid counter")

    def runner(command, *, timeout):
        if command[0] == "/tools/ncu":
            raise subprocess.TimeoutExpired(command, timeout)
        return host.runner(command, timeout=timeout)

    rows = {row.check_id: row for row in scan_environment(project_dir=tmp_path, command_runner=runner)}
    assert rows["gpu_power"].status == "warn"
    assert "GPU 1: RuntimeError: Not Supported" in rows["gpu_power"].detail
    assert rows["gpu_memory"].status == rows["gpu_utilization"].status == "ok"
    assert rows["ncu"].status == rows["container_memory"].status == "fail"
    assert "TimeoutExpired" in rows["ncu"].detail
    assert rows["nsys"].status == rows["process_runtime"].status == "ok"


def test_no_gpu_reports_unavailable_for_every_gpu_metric(host, tmp_path):
    host.nvml.nvmlDeviceGetCount = lambda: 0
    rows = {row.check_id: row for row in scan_environment(project_dir=tmp_path, command_runner=host.runner)}
    for name in ("gpu_device", "gpu_memory", "gpu_utilization", "gpu_power"):
        assert rows[name].status == "fail"
        assert "no NVIDIA GPU" in rows[name].detail
    host.nvml.nvmlShutdown.assert_called_once()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_all_machine_checks_run_once_and_ignore_experiment_edits(tmp_path, monkeypatch, language):
    from acprof.tui.app import AcprofTui

    machine = [PreflightCheck("NCU", "fail", "missing host tool", check_id="ncu"),
               PreflightCheck("GPU power", "ok", "NVML read succeeded", check_id="gpu_power")]
    scan = Mock(return_value=machine)
    monkeypatch.setattr("acprof.tui.run.preflight_controller.scan_environment", scan)
    monkeypatch.setattr("acprof.tui.run.preflight_controller.quick_preflight", lambda *args, **kwargs: [])
    app = AcprofTui(RunConfig.smoke(""), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        await asyncio.wait_for(app.workers.wait_for_complete(), 10)
        await pilot.pause()
        assert not app.query_one("#start-run", Button).disabled, "an optional host tool cannot block a basic run"
        await pilot.click("#current-environment")
        await pilot.pause()
        initial = str(app.query_one("#preflight-summary", Static).content)
        assert "Runtime" not in initial
        assert "GPU" not in initial
        assert tuple(machine) == display_checks(app.preflight)
        for field, value in (("gpus", "on"), ("profiling-mode", "full"), ("runtime", "process")):
            app.query_one(f"#{field}", Select).value = value
        await pilot.pause()
        assert str(app.query_one("#preflight-summary", Static).content) == initial
        assert tuple(machine) == display_checks(app.preflight)
        assert app.preflight.display_state == "limited"
        scan.assert_called_once_with(project_dir=app.project_dir)
        await pilot.click("#preflight-retry")
        await asyncio.wait_for(app.workers.wait_for_complete(), 10)
        await pilot.pause()
        assert scan.call_count == 2
