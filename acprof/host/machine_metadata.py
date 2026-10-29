"""Preparation-time host, GPU, Docker, and CPU policy evidence probes."""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any

from acprof.host import command as host_command

CPU_SYSFS_ROOT = "/sys/devices/system/cpu"


def host_mem_total_bytes() -> int | None:
    """Return total physical host RAM in bytes when the OS exposes it."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
        total = page_size * physical_pages
        if total > 0:
            return total
    except (AttributeError, OSError, TypeError, ValueError):
        pass

    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("MemTotal:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    total = int(parts[1]) * 1024
                    return total if total > 0 else None
    except (OSError, TypeError, ValueError):
        pass
    return None


def host_swap_metadata(
    proc_meminfo_path: str = "/proc/meminfo",
    proc_swaps_path: str = "/proc/swaps",
    swappiness_path: str = "/proc/sys/vm/swappiness",
) -> dict[str, Any]:
    """Snapshot host swap capacity, usage, backing type, and policy."""
    metadata: dict[str, Any] = {
        "host_swap_total_bytes": None,
        "host_swap_used_bytes_at_start": None,
        "host_swap_type": "unknown",
        "host_vm_swappiness": None,
    }

    try:
        meminfo: dict[str, int] = {}
        with open(proc_meminfo_path, encoding="utf-8") as f:
            for line in f:
                key, separator, raw_value = line.partition(":")
                if not separator or key not in {"SwapTotal", "SwapFree"}:
                    continue
                parts = raw_value.split()
                if not parts:
                    continue
                value_kib = int(parts[0])
                if value_kib >= 0:
                    meminfo[key] = value_kib * 1024
        total = meminfo.get("SwapTotal")
        free = meminfo.get("SwapFree")
        if total is not None:
            metadata["host_swap_total_bytes"] = total
        if total is not None and free is not None:
            metadata["host_swap_used_bytes_at_start"] = max(0, total - free)
    except (OSError, TypeError, ValueError):
        pass

    swaps_read = False
    swap_types: set[str] = set()
    swap_total_kib = 0
    swap_used_kib = 0
    try:
        with open(proc_swaps_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0].lower() == "filename":
                    continue
                if len(parts) < 4:
                    continue
                filename = parts[0]
                raw_type = parts[1].strip().lower()
                size_kib = int(parts[2])
                used_kib = int(parts[3])
                if size_kib >= 0:
                    swap_total_kib += size_kib
                if used_kib >= 0:
                    swap_used_kib += used_kib
                if os.path.basename(filename).lower().startswith("zram"):
                    swap_types.add("zram")
                elif raw_type == "file":
                    swap_types.add("file")
                elif raw_type == "partition":
                    swap_types.add("partition")
                else:
                    swap_types.add("unknown")
        swaps_read = True
    except (OSError, TypeError, ValueError):
        pass

    if metadata["host_swap_total_bytes"] is None and swaps_read:
        metadata["host_swap_total_bytes"] = swap_total_kib * 1024
    if metadata["host_swap_used_bytes_at_start"] is None and swaps_read:
        metadata["host_swap_used_bytes_at_start"] = swap_used_kib * 1024
    if swaps_read:
        if not swap_types:
            metadata["host_swap_type"] = "none"
        elif len(swap_types) == 1:
            metadata["host_swap_type"] = next(iter(swap_types))
        else:
            metadata["host_swap_type"] = "mixed"

    try:
        with open(swappiness_path, encoding="utf-8") as f:
            swappiness = int(f.read().strip())
        if swappiness >= 0:
            metadata["host_vm_swappiness"] = swappiness
    except (OSError, TypeError, ValueError):
        pass
    return metadata


def docker_root_dir() -> str | None:
    """Resolve the Docker daemon data directory without assuming a default."""
    try:
        result = host_command.run_command(['docker', 'info', '--format', '{{.DockerRootDir}}'], check=False)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    root = str(result.stdout or "").strip()
    return root or None


def docker_mount_metadata(path: str) -> tuple[str, str]:
    """Return the source device and filesystem containing ``path``."""
    findmnt = shutil.which("findmnt")
    if not findmnt:
        return "unknown", "unknown"
    try:
        result = host_command.run_command([findmnt, '--json', '--target', path, '--output', 'SOURCE,FSTYPE'], check=False)
    except Exception:
        return "unknown", "unknown"
    if result.returncode != 0:
        return "unknown", "unknown"
    try:
        payload = json.loads(result.stdout)
        filesystems = payload.get("filesystems", [])
        filesystem = filesystems[0] if isinstance(filesystems, list) and filesystems else {}
        if not isinstance(filesystem, dict):
            return "unknown", "unknown"
        source = str(filesystem.get("source") or "unknown").strip() or "unknown"
        fs_type = str(filesystem.get("fstype") or "unknown").strip() or "unknown"
        return source, fs_type
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return "unknown", "unknown"


def block_device_storage_type(device: str) -> str:
    """Classify a block device from kernel-reported transport/rotation data."""
    if not device.startswith("/dev/"):
        return "unknown"
    lsblk = shutil.which("lsblk")
    if not lsblk:
        return "unknown"
    query_device = device.split("[", 1)[0]
    try:
        result = host_command.run_command([lsblk, '--json', '--output', 'KNAME,TYPE,PKNAME,ROTA,TRAN', query_device], check=False)
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    try:
        payload = json.loads(result.stdout)
        devices = payload.get("blockdevices", [])
        block_device = devices[0] if isinstance(devices, list) and devices else {}
        if not isinstance(block_device, dict):
            return "unknown"
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return "unknown"

    transport = str(block_device.get("tran") or "").strip().lower()
    rotational = block_device.get("rota")
    if isinstance(rotational, str):
        normalized = rotational.strip().lower()
        if normalized in {"0", "false", "no"}:
            rotational = False
        elif normalized in {"1", "true", "yes"}:
            rotational = True
        else:
            rotational = None
    elif isinstance(rotational, int) and not isinstance(rotational, bool):
        rotational = bool(rotational) if rotational in {0, 1} else None

    if transport == "nvme":
        return "nvme_ssd"
    if rotational is True:
        return "hdd"
    if rotational is False:
        return "ssd"
    return "unknown"


def docker_storage_metadata() -> dict[str, Any]:
    """Snapshot capacity and media metadata for Docker's backing filesystem."""
    metadata: dict[str, Any] = {
        "docker_storage_total_bytes": None,
        "docker_storage_available_bytes_at_start": None,
        "docker_storage_filesystem": "unknown",
        "docker_storage_device": "unknown",
        "docker_storage_type": "unknown",
    }
    docker_root = docker_root_dir()
    if not docker_root:
        return metadata

    try:
        usage = shutil.disk_usage(docker_root)
        total = int(usage.total)
        available = int(usage.free)
        metadata["docker_storage_total_bytes"] = total if total > 0 else None
        metadata["docker_storage_available_bytes_at_start"] = (
            available if available >= 0 else None
        )
    except (OSError, TypeError, ValueError):
        pass

    device, fs_type = docker_mount_metadata(docker_root)
    metadata["docker_storage_device"] = device
    metadata["docker_storage_filesystem"] = fs_type
    if fs_type.lower() in {"tmpfs", "ramfs"}:
        metadata["docker_storage_type"] = "memory"
    else:
        metadata["docker_storage_type"] = block_device_storage_type(device)
    return metadata


def cpu_power_metadata() -> tuple[str, str]:
    try:
        from acprof.monitors import energy_cpu

        return (
            energy_cpu.detect_cpu_power_source(),
            energy_cpu.detect_vcpu_power_method(),
        )
    except Exception:
        return "unavailable", "unavailable"


def read_sysfs_first_line(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            value = f.readline().strip()
    except OSError:
        return None
    return value or None


def summarize_cpu_policy_values(values: list[str]) -> str:
    if not values:
        return "unavailable"

    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    if len(counts) == 1:
        return values[0]

    return "mixed:" + ",".join(
        f"{value}={counts[value]}" for value in sorted(counts)
    )


def detect_cpu_governor() -> str:
    try:
        entries = os.listdir(CPU_SYSFS_ROOT)
    except OSError:
        return "unavailable"

    governors: list[str] = []
    for entry in entries:
        if not re.fullmatch(r"cpu\d+", entry):
            continue
        governor = read_sysfs_first_line(
            os.path.join(CPU_SYSFS_ROOT, entry, "cpufreq", "scaling_governor")
        )
        if governor:
            governors.append(governor)

    return summarize_cpu_policy_values(governors)


def map_boost_flag(value: str | None) -> str:
    if value == "1":
        return "on"
    if value == "0":
        return "off"
    return value or "unavailable"


def detect_cpu_boost() -> str:
    boost = read_sysfs_first_line(os.path.join(CPU_SYSFS_ROOT, "cpufreq", "boost"))
    if boost is not None:
        return map_boost_flag(boost)

    no_turbo = read_sysfs_first_line(
        os.path.join(CPU_SYSFS_ROOT, "intel_pstate", "no_turbo")
    )
    if no_turbo == "1":
        return "off"
    if no_turbo == "0":
        return "on"
    return "unavailable"


def cpu_frequency_policy_metadata() -> tuple[str, str]:
    return detect_cpu_governor(), detect_cpu_boost()


def docker_image_size_bytes(image_tag: str) -> int:
    """Get the local Docker image size in bytes."""
    result = host_command.run_command(['docker', 'image', 'inspect', image_tag, '--format', '{{.Size}}'], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"failed to inspect image size for {image_tag}: {result.stderr.strip()}")

    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"invalid docker image size for {image_tag}: {result.stdout.strip()!r}"
        ) from exc


def docker_model_cache_bytes(image_tag: str, cache_root: str = "/models/hf") -> int:
    """Measure unique regular-file logical bytes beneath the model cache root."""
    script = (
        "import os, stat\n"
        f"root = {cache_root!r}\n"
        "if not os.path.isdir(root):\n"
        "    raise SystemExit(f'model cache directory not found: {root}')\n"
        "total = 0\n"
        "seen = set()\n"
        "for dirpath, _, filenames in os.walk(root):\n"
        "    for name in filenames:\n"
        "        path = os.path.join(dirpath, name)\n"
        "        st = os.lstat(path)\n"
        "        if stat.S_ISLNK(st.st_mode):\n"
        "            continue\n"
        "        if not stat.S_ISREG(st.st_mode):\n"
        "            continue\n"
        "        key = (st.st_dev, st.st_ino)\n"
        "        if key in seen:\n"
        "            continue\n"
        "        seen.add(key)\n"
        "        total += st.st_size\n"
        "print(total)\n"
    )
    result = host_command.run_command(['docker', 'run', '--rm', '--entrypoint', 'python', image_tag, '-c', script], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to inspect model cache size for {image_tag}: {result.stderr.strip()}"
        )

    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"invalid model cache size for {image_tag}: {result.stdout.strip()!r}"
        ) from exc
