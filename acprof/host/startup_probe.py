"""Readiness-only OOM evidence; never invokes a client, profiler or CSV writer."""
from __future__ import annotations

import time
from datetime import datetime, timezone

from acprof.artifacts import atomic_write_json, read_json_object
from acprof.host import container_state, docker_runtime, runtime_identity
from acprof.host.container_lifecycle import ContainerCleanupError

PROBE_NAME = "startup_oom_pruning.json"


def probe_startup(task, image, cpu, mem, gpu, *, request_timeout_seconds, cpuset_cpus="") -> dict:
    name = f"startup-probe-{runtime_identity.model_token(task.model_id)}-{cpu}c-{mem}g-{gpu}"
    started = time.perf_counter()
    record = {"cpu_cores": cpu, "mem_cap_gb": mem, "gpu_mode": gpu,
              "container_name": name, "started_at": datetime.now(timezone.utc).isoformat(),
              "ready": False, "docker_state": None, "outcome": "error", "diagnostic": ""}
    session = None
    try:
        session = docker_runtime.start_container_session(
            task_info=task, cpu=cpu, mem=mem, gpu=gpu, image_info=image,
            container_name=name, log_prefix="[startup-probe]",
            request_timeout_seconds=request_timeout_seconds,
            **({"cpuset_cpus": cpuset_cpus} if cpuset_cpus else {}))
        record.update(ready=True, outcome="startup_feasible", container_name=session.name,
                      container_id=session.container_id,
                      docker_state=container_state.inspect_container_state(session.container_id))
    except ContainerCleanupError:
        raise
    except container_state.ContainerStartupError as exc:
        record.update(outcome=exc.outcome, docker_state=exc.state, diagnostic=str(exc))
        if exc.container_name:
            record.update(container_name=exc.container_name, container_id=exc.container_id)
    except Exception as exc:
        # Error text (including container stderr) is never OOM evidence.
        record["diagnostic"] = f"{type(exc).__name__}: {exc}"
    finally:
        if session is not None:
            docker_runtime.stop_container_session(session)
    record["duration_s"] = time.perf_counter() - started
    return record


def startup_oom_prefixes(report: dict) -> dict:
    prefixes = {}
    identity = report["identity"]
    for gpu in identity["gpus"]:
        prefix = []
        records = {r["mem_cap_gb"]: r for r in report["attempts"] if r["gpu_mode"] == gpu}
        for mem in sorted(identity["mems"]):
            record = records.get(mem, {})
            state = record.get("docker_state") or {}
            if (record.get("cpu_cores") != min(identity["cpus"])
                    or record.get("outcome") != "startup_oom" or record.get("ready") is not False
                    or state.get("OOMKilled") is not True or state.get("Running") is not False
                    or state.get("Restarting") is True):
                break
            prefix.append(mem)
        prefixes[gpu] = prefix
    return prefixes


def run_startup_probes(directory, identity, task, image, *, request_timeout_seconds) -> dict:
    from acprof.artifact_layout import ArtifactLayout
    path = ArtifactLayout.discover(directory).path(PROBE_NAME)
    if path.exists():
        report = read_json_object(path, label="startup probe report")
        if report.get("schema_version") != 2 or report.get("identity") != identity:
            raise ValueError("startup probe identity/schema changed; use a new output directory")
    else:
        report = {"schema_version": 2, "identity": identity, "status": "running", "attempts": [],
                  "scope": "startup_only", "reference_cpu_cores": min(identity["cpus"]),
                  "assumption": "Only the minimum-CPU contiguous confirmed startup-OOM prefix is inferred across CPUs."}
        atomic_write_json(path, report)
    if report["status"] == "complete":
        return report
    for gpu in identity["gpus"]:
        for mem in sorted(identity["mems"]):
            record = next((r for r in report["attempts"] if r["gpu_mode"] == gpu
                           and r["mem_cap_gb"] == mem), None)
            if record is None:
                record = probe_startup(task, image, min(identity["cpus"]), mem, gpu,
                                       request_timeout_seconds=request_timeout_seconds,
                                       **({"cpuset_cpus": identity["cpuset_cpus"]} if identity.get("cpuset_cpus") else {}))
                report["attempts"].append(record)
                atomic_write_json(path, report)
            if mem not in startup_oom_prefixes(report)[gpu]:
                break
    report["status"] = "complete"
    report["confirmed_startup_oom_prefix_gb"] = startup_oom_prefixes(report)
    atomic_write_json(path, report)
    return report
