"""Bounded prerequisite checks; no installs, image pulls or permission changes."""
from __future__ import annotations

import io
import json
import os
import platform
import shutil
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path

from acprof.gpu_mode import normalize_gpu_mode
from acprof.host import preflight
from acprof.host.env_utils import load_project_env
from acprof.installation import resource_root
from acprof.platform import capability_matrix, detect_environment
from acprof.system.command import run_command


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
        _tcpdump_can_capture_without_sudo,
    )
    for tool in ("tcpdump", "tshark"):
        if not shutil.which(tool):
            raise RuntimeError(f"未安装 {tool}")
    if not (Path("/sys/class/net") / sniff_iface).exists():
        raise RuntimeError(f"网卡不存在：{sniff_iface}")
    if not _tcpdump_can_capture_without_sudo(str(shutil.which("tcpdump"))):
        raise RuntimeError(f"tcpdump 缺少 {TCPDUMP_CAPTURE_CAPABILITY}；见 docs/getting_started/installation.md#最小权限安装")
    _command(["tshark", "--version"])
    return f"tcpdump/tshark、capture capabilities 和 {sniff_iface} 可用；未实际抓包"


def _gpu() -> str:
    from acprof.host.gpu_device import resolve_gpu_device
    from acprof.host.metadata.gpu import probe_gpu_metadata

    report = probe_gpu_metadata(gpu_device=resolve_gpu_device())
    name = report["fields"]["name"]["value"]
    if name is None:
        raise RuntimeError(f"GPU metadata unavailable: {report['errors']}")
    devices = f"{name} ({report['device']['uuid']}; source={report['fields']['name']['source']})"
    runtimes = json.loads(_command(["docker", "info", "--format", "{{json .Runtimes}}"] ))
    if not isinstance(runtimes, dict) or "nvidia" not in runtimes:
        raise RuntimeError("Docker 未注册 NVIDIA runtime；安装并配置 NVIDIA Container Toolkit")
    return f"{devices}；Docker 已注册 NVIDIA runtime（容器 GPU 尚未实测）"


def collect_checks(*, runtime: str = "auto", profiling_mode: str = "full", gpus: str = "off",
                   sniff_iface: str = "docker0", output_dir: Path | None = None) -> list[DoctorCheck]:
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
            checks = [runtime_check, _check("architecture", _architecture, "请使用 Linux x86_64。"),
                      _check("resources", _resources, "重新安装完整 AC-Prof。"),
                      _check("process_supervision", require_process_supervision,
                             "Process 需要可用的 pidfd 与 child-subreaper 内核接口。"),
                      DoctorCheck("uv", "available" if shutil.which("uv") else "unavailable",
                                  "Process 使用 uv 创建隔离环境", "安装 uv 并加入 PATH。")]
            gpu = process_gpu_device() if gpus == "on" else {}
            plan = probe_collection_plan("process", gpu=gpu)
            checks.extend(DoctorCheck(name, item['status'], item['detail'], required=False)
                          for name, item in plan['probes'].items())
            checks.append(DoctorCheck("resource_limits", "available", "auto=继承；数值 CPU/内存限制需要 cgroup v2 委托权限"))
            return [*checks, *_workspace_checks(output_dir)]
    checks = [
        runtime_check,
        _check("environment", lambda: preflight.require_collection_host(profiling_mode=profiling_mode),
               "WSL2 请使用 --profiling-mode basic；full 需要 Native Linux。"),
        _check("architecture", _architecture, "请使用 Linux x86_64；当前依赖锁不支持 ARM。"),
        _check("cgroup_v2", preflight.require_cgroup_prerequisites, "启用统一 cgroup v2 后重启。"),
        _check("docker", preflight.require_native_docker,
               "启动本机 Docker Engine；检查当前用户的 socket 权限和 Docker context。"),
        _check("buildx", lambda: _command(["docker", "buildx", "version"]),
               "安装 Docker Buildx plugin（docker-buildx-plugin）。"),
        _check("resources", _resources, "重新安装完整 AC-Prof wheel 或 standalone release。"),
    ]
    for name, action, remedy in (
        ("rapl", lambda: _capability(preflight.probe_cpu_energy),
         "检查 /sys/class/powercap 的 energy_uj 读取权限；或先用 --profiling-mode basic。"),
        ("perf", lambda: _capability(lambda: preflight.probe_perf_instructions(env=_probe_environment())),
         "安装 perf 并配置 instructions 事件访问权限；或先用 --profiling-mode basic。"),
        ("packet", lambda: _packet(sniff_iface),
         "安装 tcpdump/tshark/libcap2-bin；配置 tcpdump capture capabilities 和 --sniff-iface。"),
    ):
        checks.append(_check(name, action, remedy) if profiling_mode == "full" else
                      DoctorCheck(name, "not_requested", "basic 模式不要求此能力"))
    checks.append(_check("gpu", _gpu, "检查 NVIDIA driver 和 NVIDIA Container Toolkit。")
                  if gpus == "on" else DoctorCheck("gpu", "not_requested", "--gpus off"))
    return [*checks, *_workspace_checks(output_dir)]


def _workspace_checks(output_dir):
    checks = []
    destination = (output_dir or Path.cwd()).expanduser().resolve()
    while not destination.exists() and destination != destination.parent:
        destination = destination.parent
    try:
        free_gib = shutil.disk_usage(destination).free / 1024 ** 3
        writable = destination.is_dir() and os.access(destination, os.W_OK | os.X_OK)
        checks.append(DoctorCheck("workspace", "available" if writable else "unavailable",
                                  f"{destination}；可写={writable}，剩余 {free_gib:.1f} GiB",
                                  "选择当前用户可写的 --output-dir。" if not writable else ""))
        if free_gib < 10:
            checks.append(DoctorCheck("disk_space", "warning", "可用空间不足 10 GiB",
                                      "模型与 Docker 镜像可能需要更多空间；按所选模型检查 Docker 数据盘。"))
    except OSError as exc:
        checks.append(DoctorCheck("workspace", "unavailable", str(exc), "检查输出目录及挂载。"))
    return checks


def _probe_environment() -> dict[str, str]:
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
