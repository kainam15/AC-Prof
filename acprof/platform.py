"""Environment identity and measurement policy, independent of runtime libraries.

Detection is local and read-only. Policy support is not evidence that a device,
permission or collector is available. Never infer historical identity here.
"""
from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping


class EnvironmentClass(str, Enum):
    NATIVE_LINUX = "native_linux"
    WSL2 = "wsl2"
    VM = "vm"
    CLOUD = "cloud"
    CONTAINER_HOST = "container_host"
    UNKNOWN = "unknown"


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    REQUIRES_NATIVE_VALIDATION = "requires_native_validation"


@dataclass(frozen=True)
class Environment:
    environment: str = "unknown"
    system: str = ""
    kernel: str = ""
    kernel_version: str = ""
    machine: str = ""
    wsl: dict = field(default_factory=dict)
    detection: tuple[str, ...] = ()

    @property
    def native(self) -> bool:
        return self.environment == "native_linux"

    @property
    def collection_tier(self) -> str:
        return {"native_linux": "full", "wsl2": "partial"}.get(self.environment, "unknown")

    @property
    def label(self) -> str:
        name = {"native_linux": "Native Linux", "wsl2": "WSL2"}.get(
            self.environment, self.environment)
        return f"{name} / {self.collection_tier.upper()}"

    def metadata(self) -> dict:
        return {"platform": {**asdict(self), "detection": list(self.detection), "native": self.native},
                "collection_tier": self.collection_tier,
                "comparability_class": self.environment,
                "environment_class": self.environment}


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def detect_environment(*, system=None, release=None, version=None, machine=None,
                       environ=None, read_text=None, exists=None) -> Environment:
    """Recognize WSL2, including custom kernels; ambiguous WSL1 stays unknown.

    VM/cloud values are reserved, not guessed from hostnames or CPU models.
    Containers never gain native identity from their Linux kernel alone.
    """
    system = platform.system() if system is None else system
    release = platform.release() if release is None else release
    version = platform.version() if version is None else version
    machine = platform.machine() if machine is None else machine
    environ = os.environ if environ is None else environ
    read_text = _read if read_text is None else read_text
    exists = os.path.exists if exists is None else exists
    evidence = []
    kind = "unknown"
    wsl = {}
    if system == "Linux":
        proc_version = read_text("/proc/version")
        kernel_text = f"{release} {version} {proc_version}".lower()
        kernel_wsl2 = "microsoft-standard" in kernel_text or "wsl2" in kernel_text
        custom_wsl2 = exists("/run/WSL")
        suspected_wsl = ("microsoft" in kernel_text or bool(environ.get("WSL_DISTRO_NAME"))
                         or bool(environ.get("WSL_INTEROP"))
                         or exists("/proc/sys/fs/binfmt_misc/WSLInterop"))
        if kernel_wsl2:
            evidence.append("kernel:wsl2")
        if custom_wsl2:
            evidence.append("/run/WSL")
        if kernel_wsl2 or custom_wsl2 or suspected_wsl:
            wsl = {"generation": 2 if kernel_wsl2 or custom_wsl2 else "unknown",
                   "distro": environ.get("WSL_DISTRO_NAME", "unknown"),
                   "interop_present": bool(environ.get("WSL_INTEROP")),
                   "proc_version": proc_version, "kernel_release": release}
            kind = "wsl2" if wsl["generation"] == 2 else "unknown"
            evidence.append("wsl_signals")
        else:
            kind = "native_linux"
            evidence.append("linux_without_wsl_or_container_signals")
        if exists("/.dockerenv") or exists("/run/.containerenv") or environ.get("container"):
            kind = "container_host"
            evidence.append("container")
    return Environment(kind, system, release, version, machine, wsl, tuple(evidence))


def recorded_identity(payload: Mapping) -> dict:
    """Fail closed for absent, inconsistent or unrecognized result provenance."""
    raw = payload.get("platform", {})
    raw = raw if isinstance(raw, Mapping) else {}
    kind = raw.get("environment", "unknown")
    if not isinstance(kind, str):
        return Environment().metadata()
    tier = {"native_linux": "full", "wsl2": "partial"}.get(kind, "unknown")
    known = (kind in {"native_linux", "wsl2"}
             and raw.get("native") is (kind == "native_linux")
             and payload.get("comparability_class") == kind
             and payload.get("collection_tier") == tier
             and payload.get("environment_class", kind) == kind)
    if not known:
        return Environment().metadata()
    return {"platform": dict(raw), "collection_tier": tier,
            "comparability_class": kind, "environment_class": kind}


_PORTABLE = ("model_loading", "functional_correctness", "preprocess", "inference", "postprocess",
             "latency", "throughput", "qps", "concurrency", "image_size", "model_size", "memory")
_GUEST = ("container_cpu", "container_memory", "cpu_utilization", "gpu_inference", "gpu_memory",
          "gpu_utilization", "nvml", "cgroup", "cpu_topology", "affinity", "cold_start")
_NATIVE_ONLY = ("rapl", "cpu_energy", "dram_energy", "pmu", "cpu_instructions", "cycles",
                "ref_cycles", "ipc", "gpu_power", "host_energy", "strict_cold_start")
_PROFILERS = ("packet_latency", "logical_flops", "gpu_flops", "cpu_flops", "cpu_heap", "gpu_timeline")


def capability_matrix(environment: Environment) -> dict[str, str]:
    names = (*_PORTABLE, *_GUEST, *_NATIVE_ONLY, *_PROFILERS)
    if environment.native:
        return dict.fromkeys(names, SupportStatus.SUPPORTED.value)
    if environment.environment == "wsl2":
        return {**dict.fromkeys(_PORTABLE, "supported"), **dict.fromkeys(_GUEST, "partial"),
                **dict.fromkeys(_NATIVE_ONLY, "unsupported"),
                **dict.fromkeys(_PROFILERS, "requires_native_validation")}
    return dict.fromkeys(names, SupportStatus.REQUIRES_NATIVE_VALIDATION.value)


def native_only_metric(metric) -> bool:
    """Central exclusion for hardware values and their derived quantities."""
    return (metric.source in {"rapl_package", "rapl_dram", "rapl_cgroup_attribution", "perf_stat", "host_cpufreq"}
            or "J" in metric.unit or metric.name.startswith("cpu_cycles_est")
            or (metric.source == "nvml_selected_device" and any(
                token in metric.name for token in ("power", "energy", "idle"))))


def collection_policy_error(environment: Environment, *, profiling_mode="basic",
                            compute_tool="none", execution_tool="none", dram_energy="auto") -> str:
    if environment.native:
        return ""
    if environment.environment != "wsl2":
        return f"Unsupported collection environment: {environment.label}; use Native Linux or WSL2."
    if (profiling_mode != "basic" or compute_tool != "none" or execution_tool != "none"
            or dram_energy == "required"):
        return ("WSL2 / PARTIAL: RAPL, DRAM energy, PMU/perf, packet and independent profilers "
                "are not enabled for WSL collection. Use --profiling-mode basic "
                "--compute-profile-tool none --execution-profile-tool none --dram-energy off, "
                "or run on Native Linux. No hardware fallback is permitted.")
    return ""
