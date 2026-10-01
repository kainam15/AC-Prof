"""Idle diagnostics and NVIDIA snapshots; only called outside sampling windows."""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

from acprof.host.client_metrics import (
    _to_float_or_nan,
)
from acprof.host.command import run_command


def _run_json_lines(cmd: List[str], timeout: float = 2.0) -> List[Dict[str, Any]]:
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


def _collect_top_cpu_processes(limit: int = 10) -> List[Dict[str, Any]]:
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


def _run_text(cmd: List[str], timeout: float = 2.0) -> str:
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


def _float_or_none(value: str) -> Optional[float]:
    number = _to_float_or_nan(value.strip())
    return number if math.isfinite(number) else None


def _int_or_none(value: str) -> Optional[int]:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _collect_nvidia_smi_gpu_snapshot(device_index: int = 0) -> Dict[str, Any]:
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
    ]
    output = _run_text([
        "nvidia-smi",
        f"--id={device_index}",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ])
    line = next((row.strip() for row in output.splitlines() if row.strip()), "")
    values = [part.strip() for part in line.split(",")]
    if len(values) != len(query_fields):
        raise RuntimeError(f"unexpected nvidia-smi gpu row: {line!r}")

    return {
        "index": _int_or_none(values[0]),
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


def _collect_nvidia_smi_compute_apps(device_index: int = 0) -> List[Dict[str, Any]]:
    output = _run_text([
        "nvidia-smi",
        f"--id={device_index}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    apps: List[Dict[str, Any]] = []
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


def _collect_nvidia_smi_pmon(device_index: int = 0) -> List[Dict[str, Any]]:
    output = _run_text(["nvidia-smi", "pmon", "-c", "1", "-i", str(device_index)])
    rows: List[Dict[str, Any]] = []
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


def _collect_idle_debug_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"snapshot_scope": "after_idle"}
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
