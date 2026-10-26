"""Offline Flask inference in an owned subprocess, without Docker or root."""
from __future__ import annotations

import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acprof.config import DEFAULT_REQUEST_TIMEOUT_SECONDS, READY_TIMEOUT_S
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.runtimes.contracts import ProcessInfo
from acprof.host.runtimes.process_control import ManagedProcess


def service_environment(task, info: ProcessInfo, *, gpu="off", timeout=300) -> dict[str, str]:
    gpu = normalize_gpu_mode(gpu)
    from acprof.model_spec import encode_model_spec, task_model_spec
    from acprof.runtime_settings import runtime_environment
    # Transport credentials and unrelated host Python paths are not inherited.
    allowed = {"PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
               "NVIDIA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "TMPDIR"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PATH"] = str(Path(info.python).parent) + os.pathsep + env.get("PATH", os.defpath)
    cache = Path(info.python).parent.parent.parent / "cache"
    cache.mkdir(exist_ok=True)
    env.update({"HOME": str(cache), "XDG_CACHE_HOME": str(cache), "HF_HOME": str(cache / "hf"),
        "HF_MODULES_CACHE": str(cache / "hf-modules"), "PYTHONPATH": info.code_root,
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "MODEL_ID": task.model_id, "MODEL_REVISION": task.model_revision,
        "MODEL_LOCAL_PATH": info.model_path, "TASK_TYPE": task.pipeline_tag,
        "TASK_FAMILY": task.task_family, "RUNTIME_BACKEND": task.runtime_backend,
        "USE_GPU": "1" if gpu == "on" else "0", "ACPROF_MODEL_ADAPTER": task.model_adapter,
        "ACPROF_RUNTIME_PROFILE": task.runtime_profile_id,
        "ACPROF_MODEL_SPEC_B64": encode_model_spec(task_model_spec(task)),
        "ACPROF_REQUEST_TIMEOUT_S": str(timeout), **runtime_environment()})
    if gpu == "off":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif info.runtime_environment.get('gpu_device', {}).get('cuda_visible_devices') is not None:
        env['CUDA_VISIBLE_DEVICES'] = info.runtime_environment['gpu_device']['cuda_visible_devices']
    store = info.runtime_environment.get('model_store', {})
    if store.get('host_path'):
        env['HF_HUB_CACHE'] = str(Path(store['host_path']) / 'entries' / store['entry_id'] / 'hf')
        env['TRANSFORMERS_CACHE'] = env['HF_HUB_CACHE']
    return env


@dataclass
class RunningProcess:
    name: str
    base_url: str
    host_port: int
    process: ManagedProcess
    resource_plan: dict
    runtime_kind: str = "process"
    gpu_device: dict[str, Any] = field(default_factory=dict)
    startup: dict = field(default_factory=dict)
    lease: Any = None
    limits: Any = None


class ProcessRuntime:
    kind = "process"

    def prepare(self, task_info, project_dir, output_dir, *, rebuild=False, timeout_seconds=300):
        from acprof.host.runtimes.environment import prepare_process
        return prepare_process(task_info, project_dir, output_dir, rebuild=rebuild, timeout_seconds=timeout_seconds)

    def restore(self, snapshot):
        from acprof.host.runtimes.environment import restore_process
        return restore_process(snapshot)

    def validate(self, **kwargs):
        from acprof.host.runtimes.validation import validate_process
        return validate_process(**kwargs)

    def start(self, task_info, cpu, mem, gpu, image_info: ProcessInfo, container_name, log_prefix,
              request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS, cpuset_cpus=""):
        gpu = normalize_gpu_mode(gpu)
        import requests

        from acprof.cpu_affinity import parse_cpu_set
        from acprof.host.model_store import acquire_mount
        from acprof.host.runtimes.resources import ProcessLimits
        limits = ProcessLimits(cpu, mem, cpuset_cpus)
        lease = None
        process = None
        try:
            lease = acquire_mount(image_info.runtime_environment)
            token = uuid.uuid4().hex
            env = service_environment(task_info, image_info, gpu=gpu, timeout=request_timeout_seconds)
            started = time.monotonic()
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                reservation.listen(128)
                port = reservation.getsockname()[1]
                env.update(ACPROF_SERVER_PORT=str(port), ACPROF_SERVICE_TOKEN=token,
                           ACPROF_SERVER_FD=str(reservation.fileno()))
                process = ManagedProcess([image_info.python, "-P", "-m", "acprof.container.server"],
                    python=image_info.python, env=env, directory=Path(image_info.runtime_environment["logs_dir"]),
                    name=f"{container_name}-{token}", cpuset=",".join(map(str, sorted(parse_cpu_set(cpuset_cpus)))),
                    cgroup=limits.path, pass_fds=(reservation.fileno(), *lease.pass_fds))
            session = RunningProcess(container_name, f"http://127.0.0.1:{port}", port, process,
                                     limits.evidence(), lease=lease, limits=limits,
                                     gpu_device=image_info.runtime_environment.get('gpu_device', {}) if gpu == 'on' else {})
            deadline = time.monotonic() + READY_TIMEOUT_S
            with requests.Session() as client:
                client.trust_env = False
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"process exited before ready ({process.poll()}): {process.log_tail()}")
                    try:
                        response = client.get(session.base_url + "/ready", timeout=1)
                        body = response.json() if response.status_code == 200 else {}
                    except (requests.RequestException, ValueError):
                        body = {}
                    if body.get("service_token") == token and body.get("status") == "ok":
                        session.startup = {"process_startup_s": time.monotonic() - started,
                                           "server": body.get("startup_timing", {})}
                        limits.verify(process.pid)
                        session.resource_plan = limits.evidence()
                        print(f"{log_prefix} Process ready: pid={process.pid}, port={port}")
                        return session
                    time.sleep(0.05)
            raise TimeoutError(f"Process readiness timed out: {process.log_tail()}")
        except BaseException as exc:
            oom = limits.oom_killed()
            if process is not None:
                try:
                    process.close()
                except BaseException:
                    from acprof.host.model_store import retain_mount_for_cleanup_debt
                    if lease is not None:
                        retain_mount_for_cleanup_debt(lease)
                    raise
            if lease is not None:
                lease.close()
            limits.close()
            if oom and isinstance(exc, Exception):
                raise RuntimeError(f"runtime_oom: owned cgroup memory.events oom_kill increased; {exc}") from exc
            raise

    def stop(self, session, *, log_prefix=None):
        try:
            session.process.close()
        except BaseException:
            from acprof.host.model_store import retain_mount_for_cleanup_debt
            retain_mount_for_cleanup_debt(session.lease)
            raise
        session.lease.close()
        session.limits.close()

    def failure(self, session, memory, returncode):
        if session.limits.oom_killed():
            return "runtime_oom: owned cgroup memory.events oom_kill increased"
        code = session.process.poll()
        return f"process_exit: returncode={code}; {session.process.log_tail()}" if code is not None else ""
