"""TUI 只读环境检查和结果摘要；不依赖 Textual。"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from acprof.analysis.uncertainty import summarize_windows
from acprof.capabilities import Capability, CapabilityStatus, measurement_requested
from acprof.experiments.configuration import RunConfig, RunConfigError
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.env_utils import load_project_env
from acprof.host.measurement.capture import tcpdump_capability_available
from acprof.host.preflight import probe_cpu_energy, probe_perf_instructions
from acprof.messages import join_messages, message
from acprof.platform import capability_matrix, collection_policy_error, detect_environment
from acprof.results.csv import (
    KEY_FIELDS,
    read_result_csv_snapshot,
    require_current_fields,
    verify_result_source,
)
from acprof.system.command import run_command
from acprof.tui.commands import csv_values
from acprof.tui.i18n import error_message, message_id
from acprof.tui.run.docker_diagnostics import DockerDiagnostic, inspect_docker

if TYPE_CHECKING:
    from acprof.host.runtimes.cgroups import ResourceDiagnostic
    from acprof.tui.results.profile_status import ProfilerOverview


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
    check_id: str = ""
    source: str = "quick_preflight"
    resource_diagnostic: ResourceDiagnostic | None = None
    docker_diagnostic: DockerDiagnostic | None = None
    root_cause: str = ''
    reason_code: str = ''
    affected_fields: tuple[str, ...] = ()


def permission_signature(project_dir: Path) -> tuple:
    """Cheap permission/installation identity for refocus and launch; never read measurement counters."""
    paths = {str(project_dir / ".env"), str(project_dir / ".env.local"), "/var/run/docker.sock",
             "/dev/nvidiactl", "/sys/fs/cgroup"}
    paths.update(filter(None, (shutil.which(tool) for tool in ("perf", "tcpdump", "tshark", "uv", "nvidia-smi"))))
    powercap = Path("/sys/class/powercap")
    for pattern in ("*:*/energy_uj", "*:*/*:*/energy_uj"):
        paths.update(str(path) for path in powercap.glob(pattern))
    paths.update(str(path) for path in Path("/dev").glob("nvidia[0-9]*"))
    entries: list[tuple[object, ...]] = []
    for name in sorted(paths):
        try:
            stat = os.stat(name)
            entries.append((name, stat.st_dev, stat.st_ino, stat.st_mode, stat.st_uid, stat.st_gid, stat.st_ctime_ns))
        except OSError as exc:
            entries.append((name, exc.errno))
    try:
        paranoid = Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
    except OSError as exc:
        paranoid = str(exc.errno)
    return (os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("ACPROF_GPU_DEVICE"), paranoid, tuple(entries))


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
    return message_id('run.preflight.capability_summary',
                   environment.label, ", ".join(collected) or "—", ", ".join(partial) or "—",
                   ", ".join(missing) or "—")


def completed_command(
    command: Sequence[str],
    *,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded diagnostic command with consistent decoded output."""
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
    from acprof.monitors.energy_cpu import discover_rapl_domains
    return sorted(domain.energy_path for domain in discover_rapl_domains(os.fspath(powercap_root)))


def quick_preflight(
    config: RunConfig,
    *,
    project_dir: str | os.PathLike[str] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = completed_command,
) -> list[PreflightCheck]:
    """Check prerequisites; Process may configure only an existing delegated subtree."""
    checks: list[PreflightCheck] = []
    environment = detect_environment()
    policy_error = collection_policy_error(environment, profiling_mode=config.profiling_mode,
                                          compute_tool=config.compute_profile_tool,
                                          execution_tool=config.execution_profile_tool)
    checks.append(
        PreflightCheck(
            message_id('run.preflight.environment_label'),
            "fail" if policy_error else "ok",
            policy_error or environment.label,
            check_id="environment",
            reason_code="unsupported" if policy_error else "",
        )
    )

    if config.runtime == "process":
        return checks + _process_checks(config)

    cgroup_v2 = Path("/sys/fs/cgroup/cgroup.controllers").is_file()
    checks.append(
        PreflightCheck(
            "cgroup v2",
            "ok" if cgroup_v2 else "fail",
            message_id('run.preflight.cgroup_available') if cgroup_v2 else message_id('run.preflight.cgroup_missing'),
            check_id="cgroup",
        )
    )

    docker = inspect_docker(command_runner, environment)
    checks.append(PreflightCheck(message_id('run.preflight.local_docker'),
                                 'ok' if docker.code == 'ready' else 'fail', docker.summary,
                                 check_id='docker', docker_diagnostic=docker))

    if config.runtime == "auto" and config.profiling_mode in {"best_effort", "basic"} and any(check.status == "fail" for check in checks[1:]):
        reasons = "; ".join(str(check.detail) for check in checks[1:] if check.status == "fail")
        checks = checks[:1] + [replace(checks[-1], status='warn' if docker.code != 'ready' else 'ok'),
                              PreflightCheck("Runtime auto → Process", "ok", reasons, check_id="runtime_auto")]
        return checks + _process_checks(config)
    checks.append(PreflightCheck("Runtime", "ok", "Docker", runtime_kind="docker", check_id="runtime"))

    if config.profiling_mode == "best_effort":
        from acprof.host.gpu_device import resolve_gpu_device
        return checks + _automatic_checks(config, "docker", resolve_gpu_device, project_dir=project_dir)
    for tool in (("tcpdump", "tshark") if measurement_requested(config.profiling_mode, "packet_latency") else ()):
        path = shutil.which(tool)
        checks.append(
            PreflightCheck(tool, "ok" if path else "fail", path or message_id('common.state.not_installed'), check_id=tool,
                           reason_code='' if path else 'not_installed')
        )
        if tool == 'tcpdump' and path:
            allowed = os.geteuid() == 0
            detail = 'CAP_NET_RAW'
            reason_code = ''
            try:
                if not allowed:
                    getcap = shutil.which('getcap')
                    if getcap:
                        result = command_runner((getcap, str(Path(path).resolve())), timeout=5.0)
                        allowed = result.returncode == 0 and tcpdump_capability_available(result.stdout)
                        if result.returncode:
                            reason_code = 'error'
                    else:
                        reason_code = 'dependency_unavailable'
                if not allowed:
                    detail = message_id('run.preflight.getcap_missing' if reason_code == 'dependency_unavailable' else
                                        'run.preflight.capture_permission_unknown' if reason_code == 'error' else
                                        'run.preflight.capture_permission_missing')
            except (OSError, subprocess.TimeoutExpired):
                detail = message_id('run.preflight.capture_permission_unknown')
                reason_code = 'error'
            checks.append(PreflightCheck('tcpdump permissions', 'ok' if allowed else 'fail', detail,
                                         'available' if allowed else 'permission_denied', check_id="capture_permissions",
                                         reason_code=reason_code, source='getcap' if reason_code else 'quick_preflight'))

    ip_cli = shutil.which("ip") if measurement_requested(config.profiling_mode, "packet_latency") else None
    if ip_cli:
        try:
            iface = command_runner(
                (ip_cli, "link", "show", config.sniff_iface),
                timeout=5.0,
            )
            checks.append(
                PreflightCheck(
                    message_id('run.network.capture_interface'),
                    "ok" if iface.returncode == 0 else "fail",
                    config.sniff_iface,
                    check_id="capture_interface",
                )
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(PreflightCheck(message_id('run.network.capture_interface'), "fail", str(exc), check_id="capture_interface", reason_code="error"))
    elif measurement_requested(config.profiling_mode, "packet_latency"):
        checks.append(PreflightCheck(message_id('run.network.capture_interface'), "fail", message_id('run.preflight.ip_missing'),
                                     check_id="capture_interface", source="ip", reason_code="dependency_unavailable"))
    else:
        checks.append(PreflightCheck(message_id('run.network.capture_interface'), "not_requested", message_id('run.preflight.not_collected'), "not_requested", check_id="capture_interface"))

    rapl = probe_cpu_energy() if measurement_requested(config.profiling_mode, "cpu_energy") else Capability("not_requested", "basic", "profiling_mode")
    rapl_status = CapabilityStatus(rapl.status).value
    checks.append(
        PreflightCheck(
            "CPU RAPL",
            "ok" if rapl_status == "available" else "not_requested" if rapl_status == "not_requested" else "fail",
            f"{rapl_status}: {rapl.detail}", rapl_status,
            check_id="cpu_energy", source=rapl.source,
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
        detail = message_id('run.preflight.perf_isolation_verified')
        checks.append(PreflightCheck("perf instructions", "ok", detail, perf_status, check_id="cpu_instructions", source=perf.source))
    else:
        checks.append(PreflightCheck("perf instructions", "not_requested" if perf_status == "not_requested" else "fail", f"{perf_status}: {perf.detail}", perf_status, check_id="cpu_instructions", source=perf.source))

    if "on" in [normalize_gpu_mode(value) for value in csv_values(config.gpus)]:
        nvidia_smi = shutil.which("nvidia-smi")
        reason_code = ''
        if nvidia_smi:
            try:
                gpu = command_runner(
                    (nvidia_smi, "--query-gpu=name", "--format=csv,noheader"),
                    timeout=10.0,
                )
                detail = gpu.stdout.strip() or gpu.stderr.strip() or message_id('run.preflight.gpu_query_failed')
                status = "ok" if gpu.returncode == 0 else "fail"
            except (OSError, subprocess.TimeoutExpired) as exc:
                status, detail, reason_code = "fail", str(exc), "error"
        else:
            status, detail, reason_code = "fail", message_id('run.preflight.nvidia_smi_missing'), "dependency_unavailable"
        checks.append(PreflightCheck("NVIDIA GPU", status, detail, check_id="gpu_device", reason_code=reason_code, source="nvidia-smi"))

    return checks


def _process_checks(config):
    from dataclasses import replace

    from acprof.application.resources import resource_matrix
    from acprof.host.runtimes.cgroups import ProcessResourceError, with_guidance
    from acprof.host.runtimes.process_control import require_process_supervision
    from acprof.host.runtimes.resources import preflight_resources
    checks = [PreflightCheck("Runtime", "ok", f"Process: managed subprocess; {config.profiling_mode} collection", runtime_kind="process", check_id="runtime")]
    requirement = 'process_runtime'
    try:
        replace(config, runtime="process").validate()
        require_process_supervision()
        requirement = 'process_resources'
        cpus, mems, _ = resource_matrix(replace(config, runtime="process").to_namespace())
        diagnostic = preflight_resources(cpus, mems, config.cpuset_cpus)
    except RunConfigError as exc:
        checks.extend(PreflightCheck('Configuration', 'fail', issue.reason,
                                    check_id='config_' + (issue.field or 'other'), source='configuration')
                      for issue in exc.issues)
    except ProcessResourceError as exc:
        diagnostic = with_guidance(replace(exc.diagnostic, detail="\n".join(
            part for part in (exc.diagnostic.detail, *getattr(exc, "__notes__", ())) if part)))
        checks.append(PreflightCheck("Process resources", "fail", diagnostic.summary,
                                    diagnostic.code, check_id="process_resources", resource_diagnostic=diagnostic))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        checks.append(PreflightCheck("Process resources" if requirement == 'process_resources' else "Process supervision",
                                    "fail", str(exc), check_id=requirement))
    else:
        checks.append(PreflightCheck("Process resources", "ok",
                                    f"CPU={config.cpus}, memory={config.mems}; auto=inherited, sharing=unknown", check_id="process_resources",
                                    resource_diagnostic=diagnostic))
    from acprof.host.runtimes.capabilities import probe_collection_plan
    from acprof.host.runtimes.gpu import process_gpu_device
    if config.profiling_mode == "best_effort":
        checks.extend(_automatic_checks(config, "process", process_gpu_device))
        uv = shutil.which("uv")
        checks.append(PreflightCheck("uv", "ok" if uv else "fail", uv or "Process environment preparation requires uv",
                                     check_id="uv", reason_code="" if uv else "not_installed"))
        return checks
    gpu = process_gpu_device(config.to_namespace().gpu_device) if "on" in config.gpus.split(",") else {}
    plan = probe_collection_plan("process", gpu=gpu, profiling_mode=config.profiling_mode)
    for name, item in plan["probes"].items():
        required = config.profiling_mode == "full" and name in {
            "cpu_energy", "cpu_instructions", "packet_latency", "process_cpu", "process_memory"}
        requested = required or name in {"process_cpu", "process_memory", "gpu_memory", "gpu_utilization"}
        status = ("not_requested" if not requested or item["status"] == "not_requested" else
                  "ok" if item["status"] == "available" else "fail" if required else "warn")
        checks.append(PreflightCheck(name, status, f"{item['status']}: {item['detail']}", item['status'], check_id=name, source=item.get("source", "probe_collection_plan")))
    uv = shutil.which("uv")
    checks.append(PreflightCheck("uv", "ok" if uv else "fail", uv or "Process environment preparation requires uv", check_id="uv", reason_code="" if uv else "not_installed"))
    return checks


def _automatic_checks(config, runtime, resolve_device, *, project_dir=None):
    from acprof.host.runtimes.capabilities import probe_collection_plan

    checks = []
    gpu = {}
    if "on" in config.gpus.split(","):
        try:
            gpu = resolve_device(config.extra_options.get("gpu_device"))
            if not gpu.get("uuid"):
                raise RuntimeError(gpu.get("error") or "CUDA device identity unavailable")
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            gpu = {"error": str(exc)}
            checks.append(PreflightCheck("GPU device", "fail", str(exc), check_id="gpu_device"))
        else:
            checks.append(PreflightCheck("GPU device", "ok", gpu["uuid"], check_id="gpu_device"))
    probe_env = os.environ.copy()
    load_project_env(project_dir or Path.cwd(), environ=probe_env)
    dram = config.extra_options.get("dram_energy", "auto")
    plan = probe_collection_plan(runtime, gpu=gpu, sniff_iface=config.sniff_iface, dram_energy=dram, env=probe_env,
                                 service_cgroup=config.cpus != "auto" or config.mems != "auto")
    for name, item in plan["probes"].items():
        skipped = item["status"] in {"not_applicable", "not_requested"}
        required = name == "dram_energy" and dram == "required"
        status = "not_requested" if skipped else "ok" if item["status"] == "available" else "fail" if required else "warn"
        unavailable_events = item.get("evidence", {}).get("unavailable_events", [])
        percentages = item.get("evidence", {}).get("event_running_pct", {})
        low_running_pct = name == "cpu_instructions" and bool(percentages) and min(percentages.values()) < 50
        group_fields = plan.get("fields", {}).get(name, ())
        affected = {key: probe for key, probe in plan.get("field_probes", {}).items() if key in group_fields}
        runtime_pending = name == "cgroup" and status == "ok" and bool(group_fields)
        partial = status == "ok" and bool(affected or unavailable_events or low_running_pct)
        if partial:
            status = "warn"
        fields = (tuple(affected) if name == "cpu_instructions" and item["status"] == "available"
                  else tuple(affected or group_fields))
        details = ", ".join(dict.fromkeys(probe["detail"] for probe in affected.values()))
        if name == "cpu_instructions" and unavailable_events:
            reasons = item.get("evidence", {}).get("event_statuses", {})
            labels = {
                "combination_limited": message_id("run.environment.perf.combination_limited"),
                "permission_denied": message_id("run.environment.perf.permission_denied"),
                "probe_error": message_id("run.environment.perf.probe_error"),
                "unsupported": message_id("run.environment.perf.unsupported"),
            }
            details = ", ".join(
                f"{labels.get(reasons.get(event, 'probe_error'), labels['probe_error'])}: {event}"
                for event in unavailable_events
            )
        detail = " · ".join(filter(None, (item["status"], item["detail"], details, ", ".join(fields))))
        if name == "cpu_instructions":
            if percentages and min(percentages.values()) < 99.995:
                detail += " · " + message_id(
                    "run.environment.perf.multiplexed", f"{min(percentages.values()):.1f}%"
                )
        reason_code = ("pmu_multiplexed" if partial and low_running_pct and not (affected or unavailable_events)
                       else "partial" if partial else "")
        if runtime_pending and not partial:
            # The target cgroup does not exist until the container/service is attached.
            status = "declared"
            reason_code = "runtime_pending"
            fields = tuple(field for field in group_fields if field.startswith((
                "container_io_read_", "container_io_write_", "service_cgroup_io_")))
            detail = message_id("run.environment.cgroup_runtime_pending")
        # The pcap tool_missing payload lists missing executable IDs in probe order.
        root = ""
        if name == "packet_latency" and item["status"] == "tool_missing" and item.get("source") == "pcap":
            root = {"tcpdump": "tcpdump", "tshark": "tshark", "tcpdump, tshark": "tcpdump"}.get(item["detail"], "")
        checks.append(PreflightCheck(name, status, detail,
                                    item["status"], check_id="cgroup_metrics" if name == "cgroup" else name,
                                    source=item.get("source", "probe_collection_plan"), reason_code=reason_code,
                                    affected_fields=fields, root_cause=root))
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
    profiler_overview: ProfilerOverview | None = None
    profiler_error: str = ""
    formal_ok_rows: int | None = None


SUMMARY_METRICS = (
    "latency_app_s", "throughput_samples_per_s", "container_mem_usage_peak_bytes", "process_rss_peak_bytes",
    "gpu_mem_used_peak_bytes", "container_attributed_energy_eff_j", "cpu_energy_total_j", "gpu_energy_total_j",
)


def summarize_result_csv(result_csv: str | Path, *, cancelled: Callable[[], bool] = lambda: False) -> ResultSummary:
    """Use the stats window grouping/filter, without bootstrap work in a preview."""
    if cancelled():
        raise CancelledError()
    path = Path(result_csv).expanduser()
    if path.is_dir() or (path.name == "result_layers.json" and not path.is_file()):
        from acprof.results.layout import resolve_single_result_source
        path = resolve_single_result_source(path)
    if path.is_dir():
        path /= "result_layers.json"
    snapshot = read_result_csv_snapshot(path, validate_measurements=False, cancelled=cancelled)
    fields, rows = snapshot.fields, snapshot.rows
    if "status" not in fields:
        raise ValueError(message_id('results.csv.invalid_header'))
    require_current_fields(fields)
    missing = set(KEY_FIELDS) - set(fields)
    metrics = [name for name in SUMMARY_METRICS if name in fields]
    groups: tuple[dict, ...] = ()
    detail = ""
    if missing:
        detail = message_id('results.csv.missing_identity_fields', ", ".join(sorted(missing)))
    elif metrics:
        groups = tuple(summarize_windows(rows, metrics, include_intervals=False, cancelled=cancelled)["groups"])
        # Unselected/inapplicable energy and device metrics stay out of the summary.
        groups = tuple(sorted((group for group in groups if group["n_windows"] or group["metric"] == "latency_app_s"),
            key=lambda group: ((group["cpu_cores"] is None, group["cpu_cores"] or 0.0),
                               (group["mem_cap_gb"] is None, group["mem_cap_gb"] or 0.0),
                               group["gpu_mode"], group["input_scale"], group["environment_class"],
                               metrics.index(group["metric"]))))
    else:
        detail = message_id('results.statistics.no_successful_windows')
    if cancelled():
        raise CancelledError()
    from acprof.analysis.audit import audit_result
    from acprof.quality import QUALITY_FIELDS
    audit = audit_result(snapshot.path, result_snapshot=snapshot, verify_result_snapshot=False)
    if cancelled():
        raise CancelledError()
    quality = {key: audit[key] for key in (*QUALITY_FIELDS, "run_status", "measurement_status")}
    from acprof.tui.results.profile_status import inspect_profile_result
    overview, profile_error = None, ""
    try:
        overview = inspect_profile_result(snapshot.path, result_snapshot=snapshot, cancelled=cancelled)
    except (OSError, ValueError, RuntimeError) as exc:
        profile_error = error_message(exc)
    if cancelled():
        raise CancelledError()
    summary = ResultSummary(
        rows=len(rows), ok_rows=sum(str(row["status"]).strip().lower() == "ok" for row in rows),
        error_rows=sum(str(row["status"]).strip().lower() == "error" for row in rows),
        warmup_rows=sum(str(row.get("warmup", "0")).strip() == "1" for row in rows),
        cases=len({tuple(row.get(field, "") for field in KEY_FIELDS[:3]) for row in rows}),
        groups=groups, grouping_detail=detail, quality=quality,
        profiler_overview=overview, profiler_error=profile_error,
        formal_ok_rows=sum(str(row.get('status', '')).strip().lower() == 'ok'
                           and str(row.get('warmup', '0')).strip() == '0' for row in rows),
    )
    try:
        if path.resolve() != snapshot.path:
            raise ValueError(f"result source changed: {path}")
        verify_result_source(snapshot.path, snapshot.sha256, cancelled=cancelled)
    except (ValueError, OSError) as exc:
        raise ValueError(message_id('results.summary.snapshot_unverified', str(exc))) from exc
    return summary


def result_summary_text(summary: ResultSummary, path: Path) -> str:
    from acprof.tui.results.reports import render_quality_summary
    lines: list[str] = [message_id('results.summary.selected', path), message_id(
        'results.summary.row_counts',
        summary.rows, summary.ok_rows, summary.error_rows, summary.cases, summary.warmup_rows),
        message_id('results.summary.grouping_description')]
    lines.append(render_quality_summary(summary.quality))
    if summary.grouping_detail:
        lines.append(summary.grouping_detail)
    labels = {"latency_app_s": "应用延迟", "throughput_samples_per_s": "吞吐量",
              "container_mem_usage_peak_bytes": "容器内存峰值", "process_rss_peak_bytes": "进程常驻内存峰值",
              "gpu_mem_used_peak_bytes": "GPU 内存峰值",
              "container_attributed_energy_eff_j": "容器归因有效能耗",
              "cpu_energy_total_j": "CPU 总能耗", "gpu_energy_total_j": "GPU 总能耗"}
    previous = None
    configurations = 0
    for group in summary.groups:
        key = tuple(group[name] for name in ("cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "environment_class"))
        if key != previous:
            configurations += 1
            if configurations > 30:
                lines.append(message_id('results.summary.configuration_limit'))
                break
            if key[0] is None or key[1] is None:
                lines.append(message_id(
                    'results.summary.inherited_configuration',
                    "inherited" if key[0] is None else f"{key[0]:g}",
                    "inherited" if key[1] is None else f"{key[1]:g}GB",
                    key[2], key[3], key[4],
                ))
            else:
                lines.append(message_id('results.summary.configuration', *key))
            previous = key
        unit = group["unit"]
        factor, unit = (1000, "ms") if unit == "s" else (1 / 1048576, "MiB") if unit == "byte" else (1, unit)
        value = "—" if group["mean"] is None else f"{group['mean'] * factor:.6g} {unit}"
        lines.append(message_id('results.summary.metric', message(labels[group["metric"]]), value,
                             group["n_windows"], message("窗口不足" if group["reason"] == "insufficient_windows" else "窗口均值")))
    return join_messages("\n", lines)
