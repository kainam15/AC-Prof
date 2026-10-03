"""TUI 只读环境检查和结果摘要；不依赖 Textual。"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from acprof.analysis.uncertainty import summarize_windows
from acprof.capabilities import Capability, measurement_requested
from acprof.experiment import RunConfig
from acprof.host.command import run_command
from acprof.host.env_utils import load_project_env
from acprof.host.packet_capture import tcpdump_capability_available
from acprof.host.preflight import probe_cpu_energy, probe_perf_instructions
from acprof.messages import join_messages, message
from acprof.platform import capability_matrix, collection_policy_error, detect_environment
from acprof.result_csv import KEY_FIELDS, require_current_fields
from acprof.tui.commands import _csv_values


@dataclass(frozen=True)
class PreflightCheck:
    """`fail` blocks the selected configuration; `warn` limits one capability.

    Unselected collectors use `not_requested`, never an availability claim.
    """

    label: str
    status: str
    detail: str
    capability_status: str = ""


def collection_preview(config: RunConfig):
    environment = detect_environment()
    matrix = capability_matrix(environment)
    selected = [name for name in ("latency", "throughput", "container_cpu", "container_memory",
                                  "cpu_energy", "cpu_instructions", "packet_latency", "gpu_power")
                if measurement_requested(config.profiling_mode, name, gpu="on" in config.gpus.split(","))]
    if "on" in config.gpus.split(","):
        selected.extend(("gpu_memory", "gpu_utilization"))
    collected = [name for name in selected if matrix[name] == "supported"]
    partial = [name for name in selected if matrix[name] == "partial"]
    missing = [name for name in matrix if matrix[name] == "unsupported"]
    missing.extend(name for name in selected if matrix[name] == "requires_native_validation")
    return message("环境：{0}\n会采（需预检）：{1}\n会降级（环境内有效）：{2}\n会缺失：{3}\n\n",
                   environment.label, ", ".join(collected) or "—", ", ".join(partial) or "—",
                   ", ".join(missing) or "—")


def _completed_command(
    command: Sequence[str],
    *,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    return run_command(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _readable_rapl_paths(
    powercap_root: str | os.PathLike[str] = "/sys/class/powercap",
) -> list[str]:
    """Use the production package-domain reader, including real read checks."""
    from acprof.monitors.energy_cpu import _discover_rapl_domains
    return sorted(domain.energy_path for domain in _discover_rapl_domains(os.fspath(powercap_root)))


def quick_preflight(
    config: RunConfig,
    *,
    project_dir: str | os.PathLike[str] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = _completed_command,
) -> list[PreflightCheck]:
    """Run read-only host checks; run.py remains the authoritative preflight."""
    checks: list[PreflightCheck] = []
    environment = detect_environment()
    policy_error = collection_policy_error(environment, profiling_mode=config.profiling_mode,
                                          compute_tool=config.compute_profile_tool,
                                          execution_tool=config.execution_profile_tool)
    checks.append(
        PreflightCheck(
            message('采集环境'),
            "fail" if policy_error else "ok",
            policy_error or environment.label,
        )
    )

    cgroup_v2 = Path("/sys/fs/cgroup/cgroup.controllers").is_file()
    checks.append(
        PreflightCheck(
            "cgroup v2",
            "ok" if cgroup_v2 else "fail",
            message('统一层级可用') if cgroup_v2 else message('未找到 cgroup.controllers'),
        )
    )

    docker_cli = shutil.which("docker")
    if not docker_cli:
        checks.append(PreflightCheck("Docker", "fail", message('未找到 docker CLI')))
    else:
        endpoint = "unknown"
        try:
            context = command_runner((docker_cli, "context", "show"), timeout=10.0)
            context_name = context.stdout.strip() if context.returncode == 0 else "default"
            inspected = command_runner(
                (
                    docker_cli,
                    "context",
                    "inspect",
                    context_name,
                    "--format",
                    '{{(index .Endpoints "docker").Host}}',
                ),
                timeout=10.0,
            )
            if inspected.returncode == 0:
                endpoint = inspected.stdout.strip()
            docker_host_override = ("" if os.environ.get("DOCKER_CONTEXT", "").strip() else
                                    os.environ.get("DOCKER_HOST", "").strip())
            if docker_host_override:
                endpoint = docker_host_override
            info = command_runner(
                (docker_cli, "info", "--format", "{{.Name}}|{{.OperatingSystem}}"),
                timeout=15.0,
            )
            native = endpoint in {
                "unix:///var/run/docker.sock",
                "unix:/var/run/docker.sock",
            }
            is_desktop = "docker desktop" in info.stdout.lower()
            status = (
                "ok"
                if info.returncode == 0 and native and not is_desktop
                else "fail"
            )
            detail = (
                f"{endpoint} · {info.stdout.strip()}"
                if info.returncode == 0
                else ((info.stderr or info.stdout).strip() or message('docker info 失败'))
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            status = "fail"
            detail = message('Docker 检查失败：{0}', exc)
        checks.append(PreflightCheck(message('本机 Docker'), status, detail))

    for tool in (("tcpdump", "tshark") if measurement_requested(config.profiling_mode, "packet_latency") else ()):
        path = shutil.which(tool)
        checks.append(
            PreflightCheck(tool, "ok" if path else "fail", path or message('未安装'))
        )
        if tool == 'tcpdump' and path:
            allowed = os.geteuid() == 0
            detail = 'CAP_NET_RAW'
            try:
                if not allowed:
                    getcap = shutil.which('getcap')
                    if getcap:
                        result = command_runner((getcap, str(Path(path).resolve())), timeout=5.0)
                        allowed = result.returncode == 0 and tcpdump_capability_available(result.stdout)
                if not allowed:
                    detail = message('缺少抓包权限；请在设置 → 连接与权限中配置 CAP_NET_RAW。')
            except (OSError, subprocess.TimeoutExpired):
                detail = message('无法检查抓包权限；请确认 getcap 可用。')
            checks.append(PreflightCheck('tcpdump permissions', 'ok' if allowed else 'fail', detail,
                                         'available' if allowed else 'permission_denied'))

    ip_cli = shutil.which("ip") if measurement_requested(config.profiling_mode, "packet_latency") else None
    if ip_cli:
        try:
            iface = command_runner(
                (ip_cli, "link", "show", config.sniff_iface),
                timeout=5.0,
            )
            checks.append(
                PreflightCheck(
                    message('抓包网卡'),
                    "ok" if iface.returncode == 0 else "fail",
                    config.sniff_iface,
                )
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(PreflightCheck(message('抓包网卡'), "fail", str(exc)))
    elif measurement_requested(config.profiling_mode, "packet_latency"):
        checks.append(PreflightCheck(message('抓包网卡'), "fail", message('未找到 ip 命令')))
    else:
        checks.append(PreflightCheck(message('抓包网卡'), "not_requested", message("本次不采集"), "not_requested"))

    rapl = probe_cpu_energy() if measurement_requested(config.profiling_mode, "cpu_energy") else Capability("not_requested", "basic", "profiling_mode")
    checks.append(
        PreflightCheck(
            "CPU RAPL",
            "ok" if rapl.status.value == "available" else "not_requested" if rapl.status.value == "not_requested" else "fail",
            f"{rapl.status.value}: {rapl.detail}", rapl.status.value,
        )
    )

    perf = Capability("not_requested", "basic", "profiling_mode")
    if measurement_requested(config.profiling_mode, "cpu_instructions"):
        probe_environ = os.environ.copy()
        try:
            load_project_env(
                project_dir if project_dir is not None else Path.cwd(), environ=probe_environ,
            )
        except ValueError as exc:
            perf = Capability("error", str(exc), "environment")
        else:
            perf = probe_perf_instructions(env=probe_environ)
    if perf.status.value == "available":
        detail = message('普通用户 perf 可用，已读到 instructions 计数并通过跨用户 PID 附加检查')
        checks.append(PreflightCheck("perf instructions", "ok", detail, perf.status.value))
    else:
        checks.append(PreflightCheck("perf instructions", "not_requested" if perf.status.value == "not_requested" else "fail", f"{perf.status.value}: {perf.detail}", perf.status.value))

    if "on" in _csv_values(config.gpus.lower()):
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                gpu = command_runner(
                    (nvidia_smi, "--query-gpu=name", "--format=csv,noheader"),
                    timeout=10.0,
                )
                detail = gpu.stdout.strip() or gpu.stderr.strip() or message('GPU 查询失败')
                status = "ok" if gpu.returncode == 0 else "fail"
            except (OSError, subprocess.TimeoutExpired) as exc:
                status, detail = "fail", str(exc)
        else:
            status, detail = "fail", message('未找到 nvidia-smi')
        checks.append(PreflightCheck("NVIDIA GPU", status, detail))

    return checks


@dataclass(frozen=True)
class ResultSummary:
    rows: int
    ok_rows: int
    error_rows: int
    warmup_rows: int
    cases: int
    groups: tuple[dict, ...] = ()
    grouping_detail: str = ""
    quality: dict = field(default_factory=dict)


SUMMARY_METRICS = (
    "latency_app_s", "throughput_samples_per_s", "container_mem_usage_peak_bytes",
    "gpu_mem_used_peak_bytes", "container_attributed_energy_eff_j", "cpu_energy_total_j", "gpu_energy_total_j",
)


def summarize_result_csv(result_csv: str | Path, *, cancelled: Callable[[], bool] = lambda: False) -> ResultSummary:
    """Use the stats window grouping/filter, without bootstrap work in a preview."""
    path = Path(result_csv).expanduser()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)) or "status" not in fields:
            raise ValueError(message("结果 CSV 缺少有效表头"))
        require_current_fields(fields)
        rows = []
        for row in reader:
            if cancelled():
                raise CancelledError()
            if None in row or any(value is None for value in row.values()):
                raise ValueError(message("结果 CSV 包含不完整的行"))
            rows.append(row)
    missing = set(KEY_FIELDS) - set(fields)
    metrics = [name for name in SUMMARY_METRICS if name in fields]
    groups = ()
    detail = ""
    if missing:
        detail = message("缺少配置身份字段，无法分组比较：{0}", ", ".join(sorted(missing)))
    elif metrics:
        groups = tuple(summarize_windows(rows, metrics, include_intervals=False, cancelled=cancelled)["groups"])
        # Unselected/inapplicable energy and device metrics stay out of the summary.
        groups = tuple(sorted((group for group in groups if group["n_windows"] or group["metric"] == "latency_app_s"),
            key=lambda group: (group["cpu_cores"], group["mem_cap_gb"], group["gpu_mode"],
                               group["input_scale"], group["environment_class"], metrics.index(group["metric"]))))
    else:
        detail = message("没有可统计的正式成功窗口。")
    from acprof.analysis.audit import audit_result
    from acprof.quality import QUALITY_FIELDS
    audit = audit_result(path)
    quality = {key: audit[key] for key in (*QUALITY_FIELDS, "run_status", "measurement_status")}
    return ResultSummary(
        rows=len(rows), ok_rows=sum(str(row["status"]).strip().lower() == "ok" for row in rows),
        error_rows=sum(str(row["status"]).strip().lower() == "error" for row in rows),
        warmup_rows=sum(str(row.get("warmup", "0")).strip() == "1" for row in rows),
        cases=len({tuple(row.get(field, "") for field in KEY_FIELDS[:3]) for row in rows}),
        groups=groups, grouping_detail=detail, quality=quality,
    )


def result_summary_text(summary: ResultSummary, path: Path) -> str:
    from acprof.tui.reports import render_quality_summary
    lines: list[str] = [message("当前选择：{0}", path), message(
        "结果已读取\n行数：{0}（成功 {1} / 错误 {2}）\n资源 case：{3}\nWarmup 行：{4}（统计排除）",
        summary.rows, summary.ok_rows, summary.error_rows, summary.cases, summary.warmup_rows),
        message("按 CPU / 内存上限 / GPU / 输入规模 / 环境分组；仅统计正式成功窗口。")]
    lines.append(render_quality_summary(summary.quality))
    if summary.grouping_detail:
        lines.append(summary.grouping_detail)
    labels = {"latency_app_s": "应用延迟", "throughput_samples_per_s": "吞吐量",
              "container_mem_usage_peak_bytes": "容器内存峰值", "gpu_mem_used_peak_bytes": "GPU 内存峰值",
              "container_attributed_energy_eff_j": "容器归因有效能耗",
              "cpu_energy_total_j": "CPU 总能耗", "gpu_energy_total_j": "GPU 总能耗"}
    previous = None
    configurations = 0
    for group in summary.groups:
        key = tuple(group[name] for name in ("cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "environment_class"))
        if key != previous:
            configurations += 1
            if configurations > 30:
                lines.append(message("摘要仅展示前 30 个配置；完整结果请使用统计报告。"))
                break
            lines.append(message("\nCPU={0:g} · MEM={1:g}GB · GPU={2} · 输入={3:g} · {4}", *key))
            previous = key
        unit = group["unit"]
        factor, unit = (1000, "ms") if unit == "s" else (1 / 1048576, "MiB") if unit == "byte" else (1, unit)
        value = "—" if group["mean"] is None else f"{group['mean'] * factor:.6g} {unit}"
        lines.append(message("{0}：{1} · 有效窗口 {2} · {3}", message(labels[group["metric"]]), value,
                             group["n_windows"], message("窗口不足" if group["reason"] == "insufficient_windows" else "窗口均值")))
    return join_messages("\n", lines)
