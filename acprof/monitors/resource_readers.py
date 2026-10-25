"""Discover cgroup/proc/sysfs readers before sampling and read raw counters."""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from acprof.monitors.common import docker_container_pid


@dataclass
class _ContainerReaders:
    cpu: Callable[[], float] | None = None
    memory: Callable[[], int] | None = None
    swap: Callable[[], int] | None = None
    swap_limit: Callable[[], int] | None = None
    io: Callable[[], tuple[int, int]] | None = None
    cpu_throttle: Callable[[], dict[str, float]] | None = None
    memory_events: Callable[[], dict[str, float]] | None = None
    cpu_pressure: Callable[[], dict[str, float]] | None = None
    memory_pressure: Callable[[], dict[str, float]] | None = None
    io_pressure: Callable[[], dict[str, float]] | None = None
    memory_peak: Callable[[], dict[str, float]] | None = None
    memory_stat: Callable[[], dict[str, float]] | None = None
    io_operations: Callable[[], dict[str, float]] | None = None
    pids: Callable[[], dict[str, float]] | None = None


def _read_int(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return int(f.read().strip())


def _read_cgroup_limit(path: str) -> int:
    """Read a cgroup byte limit, using -1 for the kernel's unlimited value."""
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip().lower()
    if raw == "max":
        return -1
    return int(raw)


def _read_cgroup_v2_io_stat(path: str) -> tuple[int, int]:
    read_bytes = 0
    write_bytes = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            for token in parts[1:]:
                key, separator, raw_value = token.partition("=")
                if not separator:
                    continue
                if key == "rbytes":
                    read_bytes += int(raw_value)
                elif key == "wbytes":
                    write_bytes += int(raw_value)
    return read_bytes, write_bytes


def _read_cgroup_v2_io_operations(path: str) -> dict[str, float]:
    read_ops = 0
    write_ops = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            for token in parts[1:]:
                key, separator, raw_value = token.partition("=")
                if not separator:
                    continue
                if key == "rios":
                    read_ops += int(raw_value)
                elif key == "wios":
                    write_ops += int(raw_value)
    return {"read_ops": float(read_ops), "write_ops": float(write_ops)}


def _read_cpu_stat_usage_s(path: str) -> float:
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, _, value = line.strip().partition(" ")
            if key == "usage_usec":
                return float(value) / 1_000_000.0
    raise RuntimeError(f"usage_usec missing from {path}")


def _read_flat_cgroup_stats(path: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) != 2:
                continue
            try:
                stats[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return stats


def _read_cgroup_memory_peak(path: str) -> dict[str, float]:
    return {"peak": float(_read_int(path))}


def _read_cgroup_memory_stat(path: str) -> dict[str, float]:
    stats = _read_flat_cgroup_stats(path)
    slab = stats.get("slab", float("nan"))
    if slab != slab:
        slab_reclaimable = stats.get("slab_reclaimable", float("nan"))
        slab_unreclaimable = stats.get("slab_unreclaimable", float("nan"))
        if slab_reclaimable == slab_reclaimable and slab_unreclaimable == slab_unreclaimable:
            slab = slab_reclaimable + slab_unreclaimable

    refault_anon = stats.get("workingset_refault_anon", float("nan"))
    refault_file = stats.get("workingset_refault_file", float("nan"))
    refault_total = stats.get("workingset_refault", float("nan"))
    if refault_total != refault_total and refault_anon == refault_anon and refault_file == refault_file:
        refault_total = refault_anon + refault_file

    return {
        "anon": stats.get("anon", float("nan")),
        "file": stats.get("file", float("nan")),
        "slab": slab,
        "pgfault": stats.get("pgfault", float("nan")),
        "pgmajfault": stats.get("pgmajfault", float("nan")),
        "workingset_refault": refault_total,
    }


def _read_cgroup_pids(
    current_path: str | None,
    peak_path: str | None,
    events_path: str | None,
) -> dict[str, float]:
    result = {
        "current": float("nan"),
        "peak": float("nan"),
        "max_events": float("nan"),
    }
    if current_path is not None:
        result["current"] = float(_read_int(current_path))
    if peak_path is not None:
        result["peak"] = float(_read_int(peak_path))
    if events_path is not None:
        result["max_events"] = _read_flat_cgroup_stats(events_path).get(
            "max",
            float("nan"),
        )
    return result


def _read_cgroup_v2_cpu_throttle(path: str) -> dict[str, float]:
    stats = _read_flat_cgroup_stats(path)
    return {
        "nr_periods": stats.get("nr_periods", float("nan")),
        "nr_throttled": stats.get("nr_throttled", float("nan")),
        "throttled_usec": stats.get("throttled_usec", float("nan")),
    }


def _read_cgroup_memory_events(path: str) -> dict[str, float]:
    stats = _read_flat_cgroup_stats(path)
    return {
        key: stats.get(key, float("nan"))
        for key in ("high", "max", "oom", "oom_kill")
    }


def _read_pressure_totals(path: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if not parts:
                continue
            scope = parts[0]
            if scope not in {"some", "full"}:
                continue
            for token in parts[1:]:
                key, separator, value = token.partition("=")
                if key != "total" or not separator:
                    continue
                try:
                    totals[scope] = float(value)
                except ValueError:
                    pass
                break
    return totals


def _parse_online_cpu_ids(text: str) -> list[int]:
    cpu_ids: list[int] = []
    for raw_part in text.replace("\n", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end >= start:
                cpu_ids.extend(range(start, end + 1))
        else:
            cpu_ids.append(int(part))
    return sorted(set(cpu_ids))


def _discover_cpu_ids(cpu_sysfs_root: str) -> list[int]:
    online_path = os.path.join(cpu_sysfs_root, "online")
    if os.path.exists(online_path):
        try:
            with open(online_path, encoding="utf-8") as f:
                cpu_ids = _parse_online_cpu_ids(f.read())
            if cpu_ids:
                return cpu_ids
        except Exception:
            pass

    cpu_ids = []
    try:
        for entry in os.scandir(cpu_sysfs_root):
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith("cpu") and name[3:].isdigit():
                cpu_ids.append(int(name[3:]))
    except Exception:
        return []
    return sorted(set(cpu_ids))


def _read_proc_cpuinfo_freqs_hz(proc_cpuinfo_path: str) -> list[float]:
    freqs: list[float] = []
    try:
        with open(proc_cpuinfo_path, encoding="utf-8") as f:
            for line in f:
                key, sep, value = line.partition(":")
                if sep and key.strip().lower() == "cpu mhz":
                    mhz = float(value.strip())
                    if mhz > 0:
                        freqs.append(mhz * 1_000_000.0)
    except Exception:
        return []
    return freqs


def _prepare_cpu_frequency_reader(
    cpu_sysfs_root: str = "/sys/devices/system/cpu",
    proc_cpuinfo_path: str = "/proc/cpuinfo",
) -> Callable[[], tuple[float | None, float | None]]:
    """Freeze topology/path discovery before sampling; values remain live reads."""
    paths = []
    for cpu_id in _discover_cpu_ids(cpu_sysfs_root):
        root = os.path.join(cpu_sysfs_root, f"cpu{cpu_id}", "cpufreq")
        paths.append(tuple(path for leaf in ("scaling_cur_freq", "cpuinfo_cur_freq")
                           if os.path.exists(path := os.path.join(root, leaf))))

    def read() -> tuple[float | None, float | None]:
        freqs: list[float] = []
        for candidates in paths:
            for path in candidates:
                try:
                    with open(path, encoding="utf-8") as stream:
                        khz = float(stream.read().strip())
                except (OSError, ValueError):
                    continue
                if khz > 0:
                    freqs.append(khz * 1_000.0)
                    break
        if not freqs:
            freqs = _read_proc_cpuinfo_freqs_hz(proc_cpuinfo_path)
        if not freqs:
            return None, None
        return sum(freqs) / len(freqs), max(freqs)

    return read


def _join_cgroup_path(root: str, relative: str, leaf: str) -> str:
    rel = relative.strip("/")
    return os.path.join(root, rel, leaf) if rel else os.path.join(root, leaf)


def _first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _resolve_container_metric_readers(
    container_name: str,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_root: str = "/proc",
) -> _ContainerReaders:
    if not container_name:
        return _ContainerReaders()

    pid = docker_container_pid(container_name)
    cgroup_file = os.path.join(proc_root, str(pid), "cgroup")
    with open(cgroup_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    readers = _ContainerReaders()

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            cpu_path = _join_cgroup_path(cgroup_root, parts[2], "cpu.stat")
            mem_path = _join_cgroup_path(cgroup_root, parts[2], "memory.current")
            swap_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.swap.current",
            )
            swap_limit_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.swap.max",
            )
            io_path = _join_cgroup_path(cgroup_root, parts[2], "io.stat")
            memory_peak_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.peak",
            )
            memory_stat_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.stat",
            )
            memory_events_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.events",
            )
            cpu_pressure_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "cpu.pressure",
            )
            memory_pressure_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.pressure",
            )
            io_pressure_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "io.pressure",
            )
            pids_current_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "pids.current",
            )
            pids_peak_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "pids.peak",
            )
            pids_events_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "pids.events",
            )
            if os.path.exists(cpu_path):
                readers.cpu = lambda path=cpu_path: _read_cpu_stat_usage_s(path)
                readers.cpu_throttle = (
                    lambda path=cpu_path: _read_cgroup_v2_cpu_throttle(path)
                )
            if os.path.exists(mem_path):
                readers.memory = lambda path=mem_path: _read_int(path)
            if os.path.exists(swap_path):
                readers.swap = lambda path=swap_path: _read_int(path)
            if os.path.exists(swap_limit_path):
                readers.swap_limit = (
                    lambda path=swap_limit_path: _read_cgroup_limit(path)
                )
            if os.path.exists(io_path):
                readers.io = lambda path=io_path: _read_cgroup_v2_io_stat(path)
                readers.io_operations = (
                    lambda path=io_path: _read_cgroup_v2_io_operations(path)
                )
            if os.path.exists(memory_peak_path):
                readers.memory_peak = (
                    lambda path=memory_peak_path: _read_cgroup_memory_peak(path)
                )
            if os.path.exists(memory_stat_path):
                readers.memory_stat = (
                    lambda path=memory_stat_path: _read_cgroup_memory_stat(path)
                )
            if os.path.exists(memory_events_path):
                readers.memory_events = (
                    lambda path=memory_events_path: _read_cgroup_memory_events(path)
                )
            if os.path.exists(cpu_pressure_path):
                readers.cpu_pressure = (
                    lambda path=cpu_pressure_path: _read_pressure_totals(path)
                )
            if os.path.exists(memory_pressure_path):
                readers.memory_pressure = (
                    lambda path=memory_pressure_path: _read_pressure_totals(path)
                )
            if os.path.exists(io_pressure_path):
                readers.io_pressure = (
                    lambda path=io_pressure_path: _read_pressure_totals(path)
                )
            existing_pids_paths = [
                path
                for path in (
                    pids_current_path,
                    pids_peak_path,
                    pids_events_path,
                )
                if os.path.exists(path)
            ]
            if existing_pids_paths:
                readers.pids = (
                    lambda current=(
                        pids_current_path
                        if os.path.exists(pids_current_path)
                        else None
                    ), peak=(
                        pids_peak_path
                        if os.path.exists(pids_peak_path)
                        else None
                    ), events=(
                        pids_events_path
                        if os.path.exists(pids_events_path)
                        else None
                    ): _read_cgroup_pids(current, peak, events)
                )
            return readers

    return readers
