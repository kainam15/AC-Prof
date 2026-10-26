"""TUI 只读环境检查和结果摘要；不依赖 Textual。"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path

from acprof.analysis.uncertainty import summarize_windows
from acprof.capabilities import Capability, CapabilityStatus, measurement_requested
from acprof.experiment import RunConfig
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.command import run_command
from acprof.host.env_utils import load_project_env
from acprof.host.measurement.capture import tcpdump_capability_available
from acprof.host.preflight import probe_cpu_energy, probe_perf_instructions
from acprof.messages import join_messages, message
from acprof.platform import capability_matrix, collection_policy_error, detect_environment
from acprof.result_csv import KEY_FIELDS, require_current_fields
from acprof.tui.commands import _csv_values
from acprof.tui.i18n import message_id


@dataclass(frozen=True)
class PreflightCheck:
    """`fail` blocks the selected configuration; `warn` limits one capability.

    Unselected collectors use `not_requested`, never an availability claim.
    """

    label: str
    status: str
    detail: str
    capability_status: str = ""
    runtime_kind: str = ""


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
    return message_id('ui.0214',
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
    """Run read-only host checks; acprof run remains the authoritative preflight."""
    checks: list[PreflightCheck] = []
    environment = detect_environment()
    policy_error = collection_policy_error(environment, profiling_mode=config.profiling_mode,
                                          compute_tool=config.compute_profile_tool,
                                          execution_tool=config.execution_profile_tool)
    checks.append(
        PreflightCheck(
            message_id('ui.0213'),
            "fail" if policy_error else "ok",
            policy_error or environment.label,
        )
    )

    if config.runtime == "process":
        return checks + _process_checks(config)

    cgroup_v2 = Path("/sys/fs/cgroup/cgroup.controllers").is_file()
    checks.append(
        PreflightCheck(
            "cgroup v2",
            "ok" if cgroup_v2 else "fail",
            message_id('ui.0888') if cgroup_v2 else message_id('ui.0889'),
        )
    )

    docker_cli = shutil.which("docker")
    if not docker_cli:
        checks.append(PreflightCheck("Docker", "fail", message_id('ui.0890')))
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
                else ((info.stderr or info.stdout).strip() or message_id('ui.0891'))
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            status = "fail"
            detail = message_id('ui.0892', exc)
        checks.append(PreflightCheck(message_id('ui.0893'), status, detail))

    if config.runtime == "auto" and config.profiling_mode == "basic" and any(check.status == "fail" for check in checks[1:]):
        reasons = "; ".join(str(check.detail) for check in checks[1:] if check.status == "fail")
        checks = checks[:1] + [PreflightCheck("Runtime auto → Process", "ok", reasons)]
        return checks + _process_checks(config)
    checks.append(PreflightCheck("Runtime", "ok", "Docker", runtime_kind="docker"))

    for tool in (("tcpdump", "tshark") if measurement_requested(config.profiling_mode, "packet_latency") else ()):
        path = shutil.which(tool)
        checks.append(
            PreflightCheck(tool, "ok" if path else "fail", path or message_id('ui.0894'))
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
                    detail = message_id('ui.0261')
            except (OSError, subprocess.TimeoutExpired):
                detail = message_id('ui.0262')
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
                    message_id('ui.0667'),
                    "ok" if iface.returncode == 0 else "fail",
                    config.sniff_iface,
                )
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(PreflightCheck(message_id('ui.0667'), "fail", str(exc)))
    elif measurement_requested(config.profiling_mode, "packet_latency"):
        checks.append(PreflightCheck(message_id('ui.0667'), "fail", message_id('ui.0895')))
    else:
        checks.append(PreflightCheck(message_id('ui.0667'), "not_requested", message_id('ui.0188'), "not_requested"))

    rapl = probe_cpu_energy() if measurement_requested(config.profiling_mode, "cpu_energy") else Capability("not_requested", "basic", "profiling_mode")
    rapl_status = CapabilityStatus(rapl.status).value
    checks.append(
        PreflightCheck(
            "CPU RAPL",
            "ok" if rapl_status == "available" else "not_requested" if rapl_status == "not_requested" else "fail",
            f"{rapl_status}: {rapl.detail}", rapl_status,
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
    perf_status = CapabilityStatus(perf.status).value
    if perf_status == "available":
        detail = message_id('ui.0294')
        checks.append(PreflightCheck("perf instructions", "ok", detail, perf_status))
    else:
        checks.append(PreflightCheck("perf instructions", "not_requested" if perf_status == "not_requested" else "fail", f"{perf_status}: {perf.detail}", perf_status))

    if "on" in [normalize_gpu_mode(value) for value in _csv_values(config.gpus)]:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                gpu = command_runner(
                    (nvidia_smi, "--query-gpu=name", "--format=csv,noheader"),
                    timeout=10.0,
                )
                detail = gpu.stdout.strip() or gpu.stderr.strip() or message_id('ui.0898')
                status = "ok" if gpu.returncode == 0 else "fail"
            except (OSError, subprocess.TimeoutExpired) as exc:
                status, detail = "fail", str(exc)
        else:
            status, detail = "fail", message_id('ui.0899')
        checks.append(PreflightCheck("NVIDIA GPU", status, detail))

    return checks


def _process_checks(config):
    from dataclasses import replace

    from acprof.application.resources import resource_matrix
    from acprof.host.runtimes.process_control import require_process_supervision
    from acprof.host.runtimes.resources import preflight_resources
    checks = [PreflightCheck("Runtime", "ok", "Process: managed subprocess; basic collection", runtime_kind="process")]
    try:
        replace(config, runtime="process").validate()
        require_process_supervision()
        cpus, mems, _ = resource_matrix(replace(config, runtime="process").to_namespace())
        preflight_resources(cpus, mems, config.cpuset_cpus)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        checks.append(PreflightCheck("Process resources", "fail", str(exc)))
    else:
        checks.append(PreflightCheck("Process resources", "ok",
                                    f"CPU={config.cpus}, memory={config.mems}; auto=inherited, sharing=unknown"))
    from acprof.host.runtimes.capabilities import probe_collection_plan
    from acprof.host.runtimes.gpu import process_gpu_device
    gpu = process_gpu_device(config.to_namespace().gpu_device) if "on" in config.gpus.split(",") else {}
    plan = probe_collection_plan("process", gpu=gpu)
    for name, item in plan["probes"].items():
        requested = name in {"process_cpu", "process_memory", "gpu_memory", "gpu_utilization"}
        status = ("not_requested" if not requested or item["status"] == "not_requested" else
                  "ok" if item["status"] == "available" else "warn")
        checks.append(PreflightCheck(name, status, f"{item['status']}: {item['detail']}", item['status']))
    if shutil.which("uv") is None:
        checks.append(PreflightCheck("uv", "fail", "Process environment preparation requires uv"))
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
    "latency_app_s", "throughput_samples_per_s", "container_mem_usage_peak_bytes", "process_rss_peak_bytes",
    "gpu_mem_used_peak_bytes", "container_attributed_energy_eff_j", "cpu_energy_total_j", "gpu_energy_total_j",
)


def summarize_result_csv(result_csv: str | Path, *, cancelled: Callable[[], bool] = lambda: False) -> ResultSummary:
    """Use the stats window grouping/filter, without bootstrap work in a preview."""
    path = Path(result_csv).expanduser()
    from acprof.result_csv import open_result_text
    with open_result_text(path) as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)) or "status" not in fields:
            raise ValueError(message_id('ui.0200'))
        require_current_fields(fields)
        rows = []
        for row in reader:
            if cancelled():
                raise CancelledError()
            if None in row or any(value is None for value in row.values()):
                raise ValueError(message_id('ui.0201'))
            rows.append(row)
    missing = set(KEY_FIELDS) - set(fields)
    metrics = [name for name in SUMMARY_METRICS if name in fields]
    groups: tuple[dict, ...] = ()
    detail = ""
    if missing:
        detail = message_id('ui.0202', ", ".join(sorted(missing)))
    elif metrics:
        groups = tuple(summarize_windows(rows, metrics, include_intervals=False, cancelled=cancelled)["groups"])
        # Unselected/inapplicable energy and device metrics stay out of the summary.
        groups = tuple(sorted((group for group in groups if group["n_windows"] or group["metric"] == "latency_app_s"),
            key=lambda group: (group["cpu_cores"], group["mem_cap_gb"], group["gpu_mode"],
                               group["input_scale"], group["environment_class"], metrics.index(group["metric"]))))
    else:
        detail = message_id('ui.0579')
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
    lines: list[str] = [message_id('ui.0203', path), message_id(
        'ui.0204',
        summary.rows, summary.ok_rows, summary.error_rows, summary.cases, summary.warmup_rows),
        message_id('ui.0205')]
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
                lines.append(message_id('ui.0206'))
                break
            lines.append(message_id('ui.0207', *key))
            previous = key
        unit = group["unit"]
        factor, unit = (1000, "ms") if unit == "s" else (1 / 1048576, "MiB") if unit == "byte" else (1, unit)
        value = "—" if group["mean"] is None else f"{group['mean'] * factor:.6g} {unit}"
        lines.append(message_id('ui.0208', message(labels[group["metric"]]), value,
                             group["n_windows"], message("窗口不足" if group["reason"] == "insufficient_windows" else "窗口均值")))
    return join_messages("\n", lines)
