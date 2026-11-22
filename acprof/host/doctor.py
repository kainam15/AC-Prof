"""Bounded prerequisite checks; no installs, image pulls or permission changes."""
from __future__ import annotations

import io
import os
import platform
import shutil
import stat
import subprocess
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path

from acprof.capabilities import CapabilityStatus
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host import preflight
from acprof.host.env_utils import load_project_env
from acprof.installation import resource_root
from acprof.platform import capability_matrix, detect_environment
from acprof.system.command import run_command
from acprof.system.uv import probe_process_uv


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str
    remedy: str = ""
    required: bool = True
    runtime_kind: str = ""


def _check(name: str, action: Callable[[], str | None], remedy: str) -> DoctorCheck:
    output = io.StringIO()
    try:
        with redirect_stdout(output), redirect_stderr(output):
            detail = action()
        return DoctorCheck(name, "available", detail or "前置检查通过")
    except (Exception, SystemExit) as exc:
        detail = output.getvalue().strip() or str(exc) or type(exc).__name__
        return DoctorCheck(name, "unavailable", detail, remedy)


def _command(arguments: list[str]) -> str:
    result = run_command(arguments, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=15, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip() or
                           f"{arguments[0]} exit={result.returncode}")
    return result.stdout.strip()


def _resources() -> str:
    from acprof.runtime.profiles import ENVIRONMENTS, environment_identity
    root = resource_root()
    required = ("dockerfiles/platform.Dockerfile", "dockerfiles/runtime.Dockerfile",
                "dockerfiles/runtime-model.Dockerfile", "dockerfiles/runtime-final.Dockerfile",
                "acprof/container/server.py", "acprof/container/runtime_manifest.py",
                "assets/audio/librispeech-clean-test-en-30s/audio.wav")
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"缺少安装资源：{relative}")
    for environment in ENVIRONMENTS.values():
        environment_identity(environment, root)
    return f"Docker 构建资源、音频素材及 {len(ENVIRONMENTS)} 个环境依赖锁可读"


def _architecture() -> str:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise RuntimeError(f"当前镜像锁只支持 linux/amd64；检测到 {machine}")
    return machine


def _capability(probe: Callable) -> str:
    result = probe()
    if result.status.value != "available":
        raise RuntimeError(f"{result.status.value}: {result.detail}")
    return f"{result.source}: available（尚未执行正式采集）"


def _packet(sniff_iface: str) -> str:
    from acprof.host.measurement.capture import (
        TCPDUMP_CAPTURE_CAPABILITY,
        tcpdump_can_capture_without_sudo,
    )
    for tool in ("tcpdump", "tshark"):
        if not shutil.which(tool):
            raise RuntimeError(f"未安装 {tool}")
    if not (Path("/sys/class/net") / sniff_iface).exists():
        raise RuntimeError(f"网卡不存在：{sniff_iface}")
    if not tcpdump_can_capture_without_sudo(str(shutil.which("tcpdump"))):
        raise RuntimeError(f"tcpdump 缺少 {TCPDUMP_CAPTURE_CAPABILITY}；见 docs/getting_started/installation.md#最小权限安装")
    _command(["tshark", "--version"])
    return f"tcpdump/tshark、capture capabilities 和 {sniff_iface} 可用；未实际抓包"


def collect_checks(*, runtime: str = "auto", profiling_mode: str = "best_effort", gpus: str = "off",
                   sniff_iface: str = "docker0", output_dir: Path | None = None,
                   probe_env: dict[str, str] | None = None) -> list[DoctorCheck]:
    gpus = normalize_gpu_mode(gpus)
    from acprof.host.runtimes.selection import select_runtime
    runtime_check = DoctorCheck("runtime", "available", "explicit Docker Runtime", runtime_kind="docker")
    if runtime != "docker":
        try:
            selected = select_runtime(runtime, profiling_mode=profiling_mode)
            runtime_check = DoctorCheck("runtime", "available", selected.reason, runtime_kind=selected.selected)
        except ValueError as exc:
            runtime_check = DoctorCheck("runtime", "unavailable", str(exc))
            if runtime == "process":
                return [runtime_check, *_workspace_checks(output_dir)]
        if runtime_check.runtime_kind == "process":
            from acprof.host.runtimes.capabilities import probe_collection_plan
            from acprof.host.runtimes.gpu import process_gpu_device
            from acprof.host.runtimes.process_control import require_process_supervision
            uv = probe_process_uv()
            checks = [runtime_check, _check("architecture", _architecture, "请使用 Linux x86_64。"),
                      _check("resources", _resources, "重新安装完整 AC-Prof。"),
                      _check("process_supervision", require_process_supervision,
                             "Process 需要可用的 pidfd 与 child-subreaper 内核接口。"),
                      DoctorCheck("uv", CapabilityStatus(uv.status).value, uv.detail, "" if uv.status == "available" else
                                  "检查 uv 的安装目录、执行权限和命令行版本。")]
            gpu = {}
            if gpus == "on":
                try:
                    gpu = process_gpu_device()
                except ValueError as exc:
                    gpu = {"status": "unavailable", "error": str(exc)}
            plan = probe_collection_plan("process", gpu=gpu, profiling_mode=profiling_mode, service_cgroup=False,
                                         env=probe_env)
            checks.extend(DoctorCheck(name, item['status'], item['detail'],
                                      required=profiling_mode != "best_effort" and name in {"process_cpu", "process_memory"}
                                      or (profiling_mode == "full" and name in {"cpu_energy", "cpu_instructions", "packet_latency"}))
                          for name, item in plan['probes'].items())
            if gpus == "on":
                gpu_ready = gpu.get("status") == "available" and bool(gpu.get("uuid"))
                checks.append(DoctorCheck(
                    "gpu", "available" if gpu_ready else "unavailable",
                    f"{gpu.get('name')} ({gpu.get('uuid')})" if gpu_ready else str(gpu.get("error") or "CUDA device unavailable"),
                    "" if gpu_ready else "检查 NVIDIA driver 与 CUDA_VISIBLE_DEVICES。",
                ))
            else:
                checks.append(DoctorCheck("gpu", "not_requested", "--gpus off"))
            checks.append(DoctorCheck("resource_limits", "available", "auto=继承；数值 CPU/内存限制需要 cgroup v2 委托权限"))
            return [*checks, *_workspace_checks(output_dir)]
    checks = [
        runtime_check,
        _check("environment", lambda: preflight.require_collection_host(profiling_mode=profiling_mode),
               "在支持的 Linux 环境使用自动采集；历史 Full 协议需要 Native Linux。"),
        _check("architecture", _architecture, "请使用 Linux x86_64；当前依赖锁不支持 ARM。"),
        _check("cgroup_v2", preflight.require_cgroup_prerequisites, "启用统一 cgroup v2 后重启。"),
        _check("docker", preflight.require_native_docker,
               "启动本机 Docker Engine；检查当前用户的 socket 权限和 Docker context。"),
        _check("buildx", lambda: _command(["docker", "buildx", "version"]),
               "安装 Docker Buildx plugin（docker-buildx-plugin）。"),
        _check("resources", _resources, "重新安装完整 AC-Prof wheel 或 standalone release。"),
    ]
    gpu = {}
    if gpus == "on":
        from acprof.host.gpu_device import DockerGpuError, require_docker_gpu, resolve_gpu_device
        try:
            gpu = resolve_gpu_device(environ=probe_env)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            gpu = {"error": str(exc)}
            checks.append(DoctorCheck("gpu", "unavailable", str(exc), "检查 NVIDIA driver 和选定设备。"))
        else:
            checks.append(DoctorCheck("gpu", "available", f"{gpu['name']} ({gpu['uuid']}, nvidia-smi)"))
        if gpu.get("uuid") and any(c.name == "docker" and c.status == "available" for c in checks):
            try:
                detail = require_docker_gpu(gpu)
            except DockerGpuError as exc:
                status = "pending" if exc.reason_code == "probe_image_missing" else "unavailable"
                checks.append(DoctorCheck("docker_gpu", status, exc.detail, exc.remedy))
            except RuntimeError as exc:
                checks.append(DoctorCheck("docker_gpu", "unavailable", str(exc), "检查探针容器的清理状态。"))
            else:
                checks.append(DoctorCheck("docker_gpu", "available", detail))
    else:
        checks.append(DoctorCheck("gpu", "not_requested", "--gpus off"))
    if profiling_mode == "best_effort":
        from acprof.host.runtimes.capabilities import probe_collection_plan
        plan = probe_collection_plan("docker", gpu=gpu, sniff_iface=sniff_iface, env=_probe_environment(probe_env))
        checks.extend(DoctorCheck(name, item["status"], f"{item.get('source', '')} {item['detail']}".strip(), required=False)
                      for name, item in plan["probes"].items())
    elif gpu.get("uuid"):
        from acprof.host.runtimes.capabilities import probe_gpu_telemetry
        hardware_energy = profiling_mode == "full" and detect_environment().native
        for name, item in probe_gpu_telemetry(gpu, hardware_energy=hardware_energy).items():
            if name != "gpu_power" or hardware_energy:
                checks.append(DoctorCheck(name, item.status.value, f"{item.source} {item.detail}".strip(),
                                          required=name == "gpu_power"))
    for name, action, remedy in (
        ("rapl", lambda: _capability(preflight.probe_cpu_energy),
         "检查 /sys/class/powercap 的 energy_uj 读取权限；或先用 --profiling-mode basic。"),
        ("perf", lambda: _capability(lambda: preflight.probe_perf_instructions(env=_probe_environment(probe_env))),
         "安装 perf 并配置 instructions 事件访问权限；或先用 --profiling-mode basic。"),
        ("packet", lambda: _packet(sniff_iface),
         "安装 tcpdump/tshark/libcap2-bin；配置 tcpdump capture capabilities 和 --sniff-iface。"),
    ):
        if profiling_mode == "best_effort":
            continue
        checks.append(_check(name, action, remedy) if profiling_mode == "full" else
                      DoctorCheck(name, "not_requested", "basic 模式不要求此能力"))
    return [*checks, *_workspace_checks(output_dir)]


def _workspace_ancestor(output_dir: Path) -> Path:
    """Find the filesystem that would hold the output, without creating it."""
    requested = output_dir.expanduser().absolute()
    destination = Path(requested.anchor)
    missing = 0
    for part in requested.parts[1:]:
        if missing:
            # Model a creatable suffix without writing it. '..' may lead back
            # to a real file, link, or mount which must then be checked.
            missing += -1 if part == ".." else 1
            continue
        child = destination / part
        try:
            mode = child.stat().st_mode
        except FileNotFoundError:
            if child.is_symlink():
                raise FileNotFoundError(f"输出路径中的符号链接目标不存在：{child}") from None
            if not os.access(destination, os.W_OK | os.X_OK):
                raise PermissionError(f"无法在此目录下创建输出目录：{destination}") from None
            missing = 1
        else:
            if not stat.S_ISDIR(mode):
                raise NotADirectoryError(f"输出路径组件不是目录：{child}")
            destination = child
    return destination.resolve(strict=True)


def _workspace_checks(output_dir):
    checks = []
    try:
        destination = _workspace_ancestor(output_dir or Path.cwd())
        free_gib = shutil.disk_usage(destination).free / 1024 ** 3
        writable = os.access(destination, os.W_OK | os.X_OK)
        checks.append(DoctorCheck("workspace", "available" if writable else "unavailable",
                                  f"{destination}；可写={writable}，剩余 {free_gib:.1f} GiB",
                                  "选择当前用户可写的 --output-dir。" if not writable else ""))
        if free_gib < 10:
            checks.append(DoctorCheck("disk_space", "warning", "可用空间不足 10 GiB",
                                      "模型与 Docker 镜像可能需要更多空间；按所选模型检查 Docker 数据盘。"))
    except (OSError, RuntimeError) as exc:
        checks.append(DoctorCheck("workspace", "unavailable", str(exc), "检查输出目录及挂载。"))
    return checks


def _probe_environment(prepared: dict[str, str] | None) -> dict[str, str]:
    if prepared is not None:
        return prepared
    environment = dict(os.environ)
    load_project_env(Path.cwd(), environ=environment)
    return environment


def report_dict(checks: list[DoctorCheck], *, profiling_mode: str, gpus: str) -> dict:
    environment = detect_environment()
    runtime = next((check.runtime_kind for check in checks if check.runtime_kind), "unknown")
    return {"schema_version": 2, "profiling_mode": profiling_mode, "gpus": gpus,
            **environment.metadata(), "native_benchmark": environment.native and runtime == "docker",
            "metric_support": capability_matrix(environment),
            "runtime_kind": runtime,
            "ready": all(not check.required or check.status in {"available", "verified", "not_requested", "warning"}
                         for check in checks),
            "scope": "prerequisites_only", "checks": [asdict(check) for check in checks]}
