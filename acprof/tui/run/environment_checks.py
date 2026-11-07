"""Bounded machine inventory, independent of an experiment's RunConfig.

The caller holds MeasurementLock. Never load a model, start a container, build
an image, or launch a profiler workload here.
"""
from __future__ import annotations

import math
import shutil
import socket
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from acprof.capabilities import CapabilityStatus, capability_from_error
from acprof.experiments.configuration import RunConfig
from acprof.host.profiling.tools.tool_discovery import find_executable, find_nsys_executable
from acprof.host.runtimes.process_control import require_process_supervision
from acprof.host.runtimes.resources import current_cgroup
from acprof.platform import detect_environment
from acprof.tui.i18n import message_id
from acprof.tui.run.diagnostics import PreflightCheck, _completed_command, quick_preflight

GPU_METRICS = ("gpu_device", "gpu_memory", "gpu_utilization", "gpu_power")
ENVIRONMENT_CHECK_IDS = (
    "environment", "docker", "cgroup", "uv", "process_runtime", "process_cpu", "process_memory",
    "container_cpu", "container_memory", "tcpdump", "tshark", "capture_permissions", "capture_interface",
    "cpu_energy", "cpu_instructions", *GPU_METRICS, "ncu", "nsys", "intel_advisor", "torch", "massif",
)


def _failure(identifier: str, exc: Exception, source: str) -> PreflightCheck:
    capability = capability_from_error(exc, source=source)
    return PreflightCheck(identifier, "fail", f"{type(exc).__name__}: {exc}",
                          CapabilityStatus(capability.status).value, check_id=identifier, source=source)


def _read_check(identifier: str, operation: Callable, source: str) -> PreflightCheck:
    try:
        detail = operation()
        return PreflightCheck(identifier, "ok", detail if isinstance(detail, str) else str(detail),
                              "available", check_id=identifier, source=source)
    except Exception as exc:
        return _failure(identifier, exc, source)


def _process_metric(metric: str) -> str:
    import psutil

    process = psutil.Process()
    value = process.cpu_times() if metric == "cpu" else process.memory_info()
    process.children(recursive=True)
    return message_id("run.environment.process_read", value)


def _cgroup_metric(metric: str) -> str:
    root = current_cgroup()
    if root is None:
        raise FileNotFoundError("cgroup v2 hierarchy is not visible")
    path = root / ("cpu.stat" if metric == "cpu" else "memory.current")
    value = path.read_text().strip()
    if metric == "cpu":
        counters = dict(line.split() for line in value.splitlines())
        int(counters["usage_usec"])
    else:
        int(value)
    return message_id("run.environment.cgroup_read", path, value)


def _interfaces() -> str:
    interfaces = socket.if_nameindex()
    if not interfaces:
        raise FileNotFoundError("no network interfaces found")
    return message_id("run.environment.interfaces_read", ", ".join(name for _, name in interfaces))


def _gpu_checks() -> list[PreflightCheck]:
    """Read every visible NVML device; one unsupported metric does not hide others."""
    nvml = None
    readings: dict[str, list[str]] = {name: [] for name in GPU_METRICS}
    failures: dict[str, list[str]] = {name: [] for name in GPU_METRICS}
    try:
        import pynvml

        pynvml.nvmlInit()
        nvml = pynvml
        count = nvml.nvmlDeviceGetCount()
        if count == 0:
            raise FileNotFoundError("NVML reports no NVIDIA GPU")
        for index in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
            except Exception as exc:
                for name in GPU_METRICS:
                    failures[name].append(f"GPU {index}: {type(exc).__name__}: {exc}")
                continue
            operations = {
                "gpu_device": lambda: nvml.nvmlDeviceGetName(handle),
                "gpu_memory": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).used,
                "gpu_utilization": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                "gpu_power": lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            }
            for name, operation in operations.items():
                try:
                    value = operation()
                    if name != "gpu_device" and (not math.isfinite(value) or value < 0):
                        raise ValueError(f"invalid NVML reading: {value}")
                    unit = {"gpu_memory": " B", "gpu_utilization": " %", "gpu_power": " W"}.get(name, "")
                    readings[name].append(f"GPU {index}: {value}{unit}")
                except Exception as exc:
                    failures[name].append(f"GPU {index}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        return [_failure(name, exc, "NVML") for name in GPU_METRICS]
    finally:
        if nvml is not None:
            try:
                nvml.nvmlShutdown()
            except Exception as exc:
                for name in GPU_METRICS:
                    failures[name].append(f"nvmlShutdown: {type(exc).__name__}: {exc}")
    return [PreflightCheck(
        name, "warn" if readings[name] and failures[name] else "ok" if readings[name] else "fail",
        "\n".join(readings[name] + failures[name]),
        "available" if not failures[name] else "unavailable", check_id=name, source="NVML",
    ) for name in GPU_METRICS]


def _tool_check(identifier: str, path: str | None, command_runner) -> PreflightCheck:
    if not path:
        return PreflightCheck(identifier, "fail", message_id("common.state.not_installed"),
                              "unavailable", check_id=identifier, source="host executable")
    try:
        result = command_runner((path, "--version"), timeout=5.0)
        output = (result.stdout or result.stderr).strip()
        if result.returncode != 0:
            raise RuntimeError(f"{path}: exit={result.returncode}; {output}")
        return PreflightCheck(identifier, "ok", message_id("run.environment.tool_read", path, output),
                              "available", check_id=identifier, source="--version")
    except Exception as exc:
        return _failure(identifier, exc, "--version")


def _runtime_profiler_checks(command_runner, docker_available: bool) -> list[PreflightCheck]:
    """Inspect all local runtime images; metadata is not execution evidence."""
    try:
        if not docker_available:
            raise RuntimeError("local Docker unavailable; runtime images cannot be inspected")
        result = command_runner(("docker", "image", "ls", "--filter", "label=org.acprof.environment",
                                 "--format", "{{.ID}}"), timeout=10.0)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip())
        images = tuple(dict.fromkeys(result.stdout.split()))
        detail = message_id("run.environment.runtime_tools_read", len(images))
        return [PreflightCheck(name, "pending", detail, check_id=name, source="docker image ls")
                for name in ("torch", "massif")]
    except Exception as exc:
        return [_failure(name, exc, "docker image ls") for name in ("torch", "massif")]


def scan_environment(*, project_dir: str | Path | None = None, command_runner=_completed_command) -> list[PreflightCheck]:
    """Always inspect the same host capabilities, including unselected tools."""
    # Reuse established Docker, capture-permission, RAPL and perf diagnostics.
    # This fixed diagnostic config never comes from or modifies the experiment.
    config = RunConfig(model="environment-only", runtime="docker", profiling_mode="full", gpus="off")
    rows = [row for row in quick_preflight(config, project_dir=project_dir, command_runner=command_runner)
            if row.check_id not in {"environment", "runtime", "capture_interface"}]
    rows = [_tool_check(row.check_id, shutil.which(row.check_id), command_runner)
            if row.check_id in {"tcpdump", "tshark"} else row for row in rows]
    rows.insert(0, PreflightCheck("Platform", "ok", detect_environment().label.split(" / ")[0], check_id="environment", source="platform"))
    rows.extend((
        _tool_check("uv", shutil.which("uv"), command_runner),
        _read_check("process_runtime", require_process_supervision, "process supervisor --probe"),
        _read_check("process_cpu", lambda: _process_metric("cpu"), "psutil"),
        _read_check("process_memory", lambda: _process_metric("memory"), "psutil"),
        _read_check("container_cpu", lambda: _cgroup_metric("cpu"), "cgroup v2"),
        _read_check("container_memory", lambda: _cgroup_metric("memory"), "cgroup v2"),
        _read_check("capture_interface", _interfaces, "if_nameindex"),
    ))
    if not any(row.check_id == "capture_permissions" for row in rows):
        rows.append(PreflightCheck("capture_permissions", "fail", message_id("common.state.not_installed"),
                                   "unavailable", check_id="capture_permissions", source="tcpdump"))
    rows.extend(_gpu_checks())
    for name, discover in (
        ("ncu", lambda: find_executable(None, ("ncu", "nv-nsight-cu-cli"))),
        ("nsys", lambda: find_nsys_executable(None)),
        ("intel_advisor", lambda: find_executable(None, ("advisor", "advixe-cl"))),
    ):
        try:
            rows.append(_tool_check(name, discover(), command_runner))
        except Exception as exc:
            rows.append(_failure(name, exc, "tool discovery"))
    rows.extend(_runtime_profiler_checks(command_runner, any(row.check_id == "docker" and row.status == "ok" for row in rows)))
    # Strip launch-specific runtime resolution from this presentation inventory.
    return [replace(row, runtime_kind="") for row in rows]
