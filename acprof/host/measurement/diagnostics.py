"""Idle diagnostics and NVIDIA snapshots; only called outside sampling windows."""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Any

from acprof.gpu_probe import require_gpu_identity
from acprof.host.command import run_command
from acprof.host.measurement.metrics import (
    to_float_or_nan as _to_float_or_nan,
)


def _run_json_lines(cmd: list[str], timeout: float = 2.0) -> list[dict[str, Any]]:
    result = run_command(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())
    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _collect_top_cpu_processes(limit: int = 10) -> list[dict[str, Any]]:
    result = run_command(
        [
            "ps",
            "-eo",
            "pid=,ppid=,user=,comm=,%cpu=,%mem=,args=",
            "--sort=-%cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=2.0,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())

    processes = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, user, comm, cpu_pct, mem_pct, args = parts
        processes.append({
            "pid": int(pid),
            "ppid": int(ppid),
            "user": user,
            "comm": comm,
            "cpu_pct": _to_float_or_nan(cpu_pct),
            "mem_pct": _to_float_or_nan(mem_pct),
            "args": args,
        })
        if len(processes) >= limit:
            break
    return processes


def _run_text(cmd: list[str], timeout: float = 2.0) -> str:
    result = run_command(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())
    return result.stdout


def _float_or_none(value: str) -> float | None:
    number = _to_float_or_nan(value.strip())
    return number if math.isfinite(number) else None


def _int_or_none(value: str) -> int | None:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _collect_nvidia_smi_gpu_snapshot(device_index: int = 0, *, device_uuid: str = '') -> dict[str, Any]:
    query_fields = [
        "index",
        "name",
        "pstate",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
        "clocks.gr",
        "clocks.video",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "uuid",
    ]
    output = _run_text([
        "nvidia-smi",
        f"--id={device_uuid or device_index}",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ])
    rows = list(csv.reader(output.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(query_fields):
        raise RuntimeError(f"unexpected nvidia-smi gpu rows: {output!r}")
    values = [part.strip() for part in rows[0]]
    identity = {"index": _int_or_none(values[0]), "uuid": values[14]}
    require_gpu_identity(identity, expected_uuid=device_uuid or None,
                         expected_index=None if device_uuid else device_index)
    if not isinstance(identity["index"], int) or identity["index"] < 0:
        raise RuntimeError("nvidia-smi did not resolve a current GPU index")
    return {
        **identity,
        "name": values[1],
        "pstate": values[2],
        "power_draw_w": _float_or_none(values[3]),
        "power_limit_w": _float_or_none(values[4]),
        "clocks_sm_mhz": _float_or_none(values[5]),
        "clocks_mem_mhz": _float_or_none(values[6]),
        "clocks_gr_mhz": _float_or_none(values[7]),
        "clocks_video_mhz": _float_or_none(values[8]),
        "temperature_gpu_c": _float_or_none(values[9]),
        "utilization_gpu_pct": _float_or_none(values[10]),
        "utilization_memory_pct": _float_or_none(values[11]),
        "memory_used_mib": _float_or_none(values[12]),
        "memory_total_mib": _float_or_none(values[13]),
    }


def _collect_nvidia_smi_compute_apps(device_uuid: str) -> list[dict[str, Any]]:
    output = _run_text([
        "nvidia-smi",
        f"--id={device_uuid}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    apps: list[dict[str, Any]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or "No running processes found" in line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        apps.append({
            "pid": _int_or_none(parts[0]),
            "process_name": parts[1],
            "used_memory_mib": _float_or_none(parts[2]),
        })
    return apps


def _collect_nvidia_smi_pmon(device_index: int = 0) -> list[dict[str, Any]]:
    output = _run_text(["nvidia-smi", "pmon", "-c", "1", "-i", str(device_index)])
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        gpu, pid, proc_type, sm, mem, enc, dec, jpg, ofa, *command = parts
        rows.append({
            "gpu": _int_or_none(gpu),
            "pid": _int_or_none(pid),
            "type": proc_type,
            "sm_pct": _float_or_none(sm),
            "mem_pct": _float_or_none(mem),
            "enc_pct": _float_or_none(enc),
            "dec_pct": _float_or_none(dec),
            "jpg_pct": _float_or_none(jpg),
            "ofa_pct": _float_or_none(ofa),
            "command": " ".join(command),
        })
    return rows


def collect_gpu_idle_debug_snapshot(device_index: int, *, device_uuid: str = "") -> dict[str, Any]:
    snapshot: dict[str, Any] = {"gpu_snapshot_scope": "after_gpu_idle"}
    try:
        gpu = _collect_nvidia_smi_gpu_snapshot(device_index, device_uuid=device_uuid)
        snapshot["nvidia_smi_gpu"] = gpu
    except Exception as exc:
        snapshot["nvidia_smi_gpu_error"] = repr(exc)
        snapshot["nvidia_smi_pmon_error"] = "GPU identity was not resolved"
        snapshot["nvidia_smi_compute_apps_error"] = "GPU identity was not resolved"
        return snapshot
    try:
        # pmon accepts the current index obtained from the UUID-bound snapshot.
        snapshot["nvidia_smi_pmon"] = _collect_nvidia_smi_pmon(gpu["index"])
    except Exception as exc:
        snapshot["nvidia_smi_pmon_error"] = repr(exc)
    try:
        snapshot["nvidia_smi_compute_apps"] = _collect_nvidia_smi_compute_apps(gpu["uuid"])
    except Exception as exc:
        snapshot["nvidia_smi_compute_apps_error"] = repr(exc)
    return snapshot

def collect_idle_debug_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"snapshot_scope": "after_idle"}
    try:
        snapshot["loadavg"] = list(os.getloadavg())
    except Exception as exc:
        snapshot["loadavg_error"] = repr(exc)

    try:
        snapshot["top_cpu_processes"] = _collect_top_cpu_processes()
    except Exception as exc:
        snapshot["top_cpu_processes_error"] = repr(exc)

    try:
        snapshot["docker_containers"] = _run_json_lines([
            "docker",
            "ps",
            "--format",
            "{{json .}}",
        ])
    except Exception as exc:
        snapshot["docker_containers_error"] = repr(exc)

    try:
        snapshot["docker_stats"] = _run_json_lines([
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
        ])
    except Exception as exc:
        snapshot["docker_stats_error"] = repr(exc)

    return snapshot
