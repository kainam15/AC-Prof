"""Observe comparison conditions before capture; never poll in measurement windows."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import atomic_write_json
from acprof.cpu_affinity import normalize_cpu_set, parse_cpu_set
from acprof.host.command import run_command

HARDWARE_FIELDS = ("host_id", "cpu_model", "cpu_affinity", "cpu_policy", "gpu", "runtime_threads")


def conditions_path(layout: ArtifactLayout) -> Path:
    # This optional extension uses the existing metadata directory contract and
    # does not rewrite the routing manifest of older v2 experiments.
    name = "metadata/hardware_conditions.json" if layout.layout_version == 2 else "hardware_conditions.json"
    return layout.contained(name)


def _read(path) -> str | None:
    try:
        return Path(path).read_text().strip() or None
    except OSError:
        return None


def _run(command) -> str | None:
    try:
        result = run_command(command, capture_output=True, text=True, timeout=10, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def observe_conditions(session) -> dict:
    record = dict.fromkeys(HARDWARE_FIELDS)
    errors = []
    machine_id = _read("/etc/machine-id")
    record["host_id"] = hashlib.sha256(machine_id.encode()).hexdigest() if machine_id else None
    cpuinfo = _read("/proc/cpuinfo") or ""
    record["cpu_model"] = sorted(set(re.findall(r"^model name\s*:\s*(.+)$", cpuinfo, re.M))) or None
    if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(session, "container_id", ""))):
        return {**record, "errors": ["container identity unavailable"]}
    raw = _run(["docker", "top", session.container_id, "-eo", "pid"])
    try:
        pids = [int(value.strip()) for value in (raw or "").splitlines()[1:]]
        masks = set()
        allowed = set()
        for pid in pids:
            for task in Path(f"/proc/{pid}/task").iterdir():
                affinity = os.sched_getaffinity(int(task.name))
                allowed.update(affinity)
                masks.add(normalize_cpu_set(",".join(str(cpu) for cpu in sorted(affinity))))
        record["cpu_affinity"] = sorted(masks) or None
        governors = {str(cpu): _read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
                     for cpu in sorted(allowed)}
        policy = {
            "governors": governors or None,
            "boost": _read("/sys/devices/system/cpu/cpufreq/boost"),
            "intel_no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        }
        # Only one of the boost interfaces may apply; absent interfaces are
        # reported distinctly from missing evidence for every applicable one.
        if policy["boost"] is not None:
            policy.pop("intel_no_turbo")
        elif policy["intel_no_turbo"] is not None:
            policy.pop("boost")
        record["cpu_policy"] = policy
    except (OSError, ValueError, AttributeError) as exc:
        errors.append(f"CPU affinity: {exc}")
    gpu = session.gpu_device
    if gpu:
        raw = _run(["nvidia-smi", f"--id={gpu.get('uuid', '')}",
                    "--query-gpu=driver_version,power.limit,persistence_mode", "--format=csv,noheader,nounits"])
        values = next(csv.reader([raw], skipinitialspace=True), []) if raw else []
        values = [None if value.strip().lower() in {"", "n/a", "[n/a]", "not supported", "[not supported]"}
                  else value.strip() for value in values]
        record["gpu"] = {"model": gpu.get("name"), "uuid": gpu.get("uuid"),
                         "driver": values[0] if len(values) == 3 else None,
                         "power_limit_w": values[1] if len(values) == 3 else None,
                         "persistence_mode": values[2] if len(values) == 3 else None}
    else:
        record["gpu"] = "not_applicable"
    import requests
    try:
        response = requests.get(session.base_url + "/meta", timeout=5, headers={"Connection": "close"})
        response.raise_for_status()
        observed = response.json().get("runtime_parameters", {}).get("effective", {}).get("threads")
        record["runtime_threads"] = observed if type(observed) is int and observed > 0 else None
    except (requests.RequestException, ValueError, AttributeError, TypeError) as exc:
        errors.append(f"runtime threads: {exc}")
    return {**record, "errors": errors}


def record_case_conditions(output_dir, case_id, session, *, cpuset_cpus="") -> None:
    cpuset_cpus = normalize_cpu_set(cpuset_cpus)
    path = conditions_path(ArtifactLayout.discover(output_dir))
    payload: dict = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "cases": {}}
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("cases"), dict):
        raise ValueError("unsupported hardware conditions schema")
    observed = observe_conditions(session)
    observed["requested_cpuset"] = cpuset_cpus or "unrestricted"
    # Runtimes may pin individual workers more narrowly than Docker's CPU set.
    # Every observed thread must stay inside the requested boundary.
    affinities = observed["cpu_affinity"]
    requested = parse_cpu_set(cpuset_cpus)
    verified = bool(affinities) and all(
        parse_cpu_set(mask) and parse_cpu_set(mask) <= requested for mask in affinities
    ) if cpuset_cpus else True
    if not verified:
        observed["errors"].append("thread affinities are unknown or outside the requested CPU set")
    payload["cases"][case_id] = observed
    atomic_write_json(path, payload)
    if not verified:
        raise RuntimeError("cannot verify requested CPU set on the running container")
