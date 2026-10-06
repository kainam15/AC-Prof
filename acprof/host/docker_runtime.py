"""Owned inference service sessions and their cold-start timing."""
from __future__ import annotations

import datetime
import math
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from acprof.config import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    READY_POLL_INTERVAL_S,
    READY_TIMEOUT_S,
    SERVER_PORT,
)
from acprof.host import command as host_command, container_state
from acprof.host.container_lifecycle import (
    container_owner_labels,
    recover_abandoned_containers,
    remove_owned_container,
)
from acprof.host.container_state import (
    ContainerStartupError,
)
from acprof.host.detect import TaskInfo
from acprof.host.env_utils import hf_offline_docker_env_args
from acprof.host.gpu_device import gpu_docker_args, normalize_gpu_mode, resolve_gpu_device
from acprof.host.runtime_images import ImageInfo
from acprof.runtime_settings import runtime_docker_env_args


@dataclass
class RunningContainer:
    name: str
    base_url: str
    host_port: int
    cold_start_s: float
    cold_start_started_at: str = "nan"
    cold_start_ready_at: str = "nan"
    cold_start_container_launch_s: float = float("nan")
    cold_start_server_setup_s: float = float("nan")
    cold_start_cuda_init_s: float = float("nan")
    cold_start_model_load_s: float = float("nan")
    cold_start_ready_wait_s: float = float("nan")
    gpu_device: Dict[str, Any] = field(default_factory=dict)
    container_id: str = ""
    _model_store_mount: Any = field(default=None, repr=False, compare=False)


def _parse_csv_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nonnegative_float_or_nan(value: Any) -> float:
    parsed = _parse_csv_float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else float("nan")


def _iso_from_epoch(epoch_s: float) -> str:
    if not math.isfinite(epoch_s):
        return "nan"
    return datetime.datetime.fromtimestamp(
        epoch_s,
        tz=datetime.timezone.utc,
    ).astimezone().isoformat(timespec="milliseconds")


def _cold_start_breakdown(
    body: Optional[Dict[str, Any]],
    docker_started_at_epoch_s: float,
    ready_received_at_epoch_s: float,
) -> Dict[str, Any]:
    timing = body.get("startup_timing", {}) if isinstance(body, dict) else {}
    if not isinstance(timing, dict):
        timing = {}

    process_started_at = _parse_csv_float(
        timing.get("server_process_started_at_epoch_s")
    )
    model_load_completed_at = _parse_csv_float(
        timing.get("model_load_completed_at_epoch_s")
    )
    container_launch_s = (
        process_started_at - docker_started_at_epoch_s
        if math.isfinite(process_started_at)
        and process_started_at >= docker_started_at_epoch_s
        else float("nan")
    )
    ready_wait_s = (
        ready_received_at_epoch_s - model_load_completed_at
        if math.isfinite(model_load_completed_at)
        and ready_received_at_epoch_s >= model_load_completed_at
        else float("nan")
    )
    model_load_s = _nonnegative_float_or_nan(timing.get("model_load_s"))
    if not math.isfinite(model_load_s) and isinstance(body, dict):
        model_load_s = _nonnegative_float_or_nan(body.get("load_time_s"))

    return {
        "cold_start_started_at": _iso_from_epoch(docker_started_at_epoch_s),
        "cold_start_ready_at": _iso_from_epoch(ready_received_at_epoch_s),
        "cold_start_container_launch_s": container_launch_s,
        "cold_start_server_setup_s": _nonnegative_float_or_nan(
            timing.get("server_setup_s")
        ),
        "cold_start_cuda_init_s": _nonnegative_float_or_nan(
            timing.get("cuda_init_s")
        ),
        "cold_start_model_load_s": model_load_s,
        "cold_start_ready_wait_s": ready_wait_s,
    }


def cold_start_client_env(session: RunningContainer) -> Dict[str, str]:
    return {
        "COLD_START_STARTED_AT": session.cold_start_started_at,
        "COLD_START_READY_AT": session.cold_start_ready_at,
        "COLD_START_CONTAINER_LAUNCH_S": str(
            session.cold_start_container_launch_s
        ),
        "COLD_START_SERVER_SETUP_S": str(session.cold_start_server_setup_s),
        "COLD_START_CUDA_INIT_S": str(session.cold_start_cuda_init_s),
        "COLD_START_MODEL_LOAD_S": str(session.cold_start_model_load_s),
        "COLD_START_READY_WAIT_S": str(session.cold_start_ready_wait_s),
        "COLD_START_S": f"{session.cold_start_s:.6f}",
    }


def service_port(cpu: int, mem: int) -> int:
    return SERVER_PORT + cpu * 100 + mem


def _launch_container(command: List[str]) -> str:
    """Track ownership even if Docker creates a container but fails to start it."""
    with tempfile.TemporaryDirectory(prefix="acprof-container-") as directory:
        cidfile = Path(directory) / "container.cid"
        try:
            result = host_command.run_command([*command[:3], '--cidfile', str(cidfile), *command[3:]], check=False)
            if result.returncode != 0:
                raise RuntimeError(f"docker run failed: {result.stderr.strip()}")
            identifier = result.stdout.strip()
            if not re.fullmatch(r"[0-9a-f]{64}", identifier):
                raise RuntimeError("docker run did not return an immutable container ID")
            return identifier
        except BaseException:
            identifier = cidfile.read_text().strip() if cidfile.is_file() else ""
            if re.fullmatch(r"[0-9a-f]{64}", identifier):
                remove_owned_container(identifier, host_command.run_command)
            raise


def start_container_session(
    task_info: TaskInfo,
    cpu: int,
    mem: int,
    gpu: str,
    image_info: ImageInfo,
    container_name: str,
    log_prefix: str,
    request_timeout_seconds: float | None = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    cpuset_cpus: str = "",
) -> RunningContainer:
    import requests

    gpu = normalize_gpu_mode(gpu)
    if request_timeout_seconds is not None:
        request_timeout_seconds = float(request_timeout_seconds)
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be finite and positive, or None")
    from acprof.cpu_affinity import normalize_cpu_set
    cpuset_cpus = normalize_cpu_set(cpuset_cpus)
    completion_timeout = "none" if request_timeout_seconds is None else f"{request_timeout_seconds:g}"
    host_port = service_port(cpu, mem)

    container_name = f"{container_name}-{uuid.uuid4().hex[:12]}"

    gpu_flag = []
    gpu_device = {}
    use_gpu = 0
    if gpu == "on":
        gpu_device = resolve_gpu_device()
        gpu_flag = gpu_docker_args(gpu_device)
        use_gpu = 1

    owner = container_owner_labels()
    recover_abandoned_containers(owner, host_command.run_command)
    labels = [part for key, value in owner.items() for part in ("--label", f"{key}={value}")]
    from acprof.host.model_store import acquire_mount, retain_mount_for_cleanup_debt

    model_store_mount = acquire_mount(image_info.runtime_environment)
    docker_cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        *labels,
        f"--cpus={cpu}",
        *([f"--cpuset-cpus={cpuset_cpus}"] if cpuset_cpus else []),
        f"--memory={mem}g",
        *gpu_flag,
        "-e", f"MODEL_ID={task_info.model_id}",
        "-e", f"MODEL_REVISION={task_info.model_revision or 'main'}",
        "-e", f"TASK_FAMILY={task_info.task_family}",
        "-e", f"TASK_TYPE={task_info.pipeline_tag}",
        "-e", f"RUNTIME_BACKEND={task_info.runtime_backend}",
        "-e", f"USE_GPU={use_gpu}",
        *hf_offline_docker_env_args(),
        *model_store_mount.args,
        "-e", f"ACPROF_REQUEST_TIMEOUT_S={completion_timeout}",
        *runtime_docker_env_args(),
        "-p", f"127.0.0.1:{host_port}:{SERVER_PORT}",
        image_info.tag,
    ]

    t0_wall = time.time()
    t0 = time.perf_counter()
    container_id = ""
    try:
        container_id = _launch_container(docker_cmd)
    except BaseException:
        # _launch_container owns best-effort cleanup when Docker created an ID.
        # If launch itself fails, absence cannot be re-proven here without that
        # ID, so retain the lease conservatively until process exit.
        retain_mount_for_cleanup_debt(model_store_mount)
        raise

    base_url = f"http://127.0.0.1:{host_port}"
    deadline = time.perf_counter() + READY_TIMEOUT_S

    def fail_startup(reason: str, *, timed_out: bool = False) -> None:
        state = container_state.inspect_container_state(container_id)
        try:
            logs = host_command.run_command(['docker', 'logs', container_id, '--tail', '200'], check=False, timeout=15)
            diagnostic = ((logs.stdout or "") + "\n" + (logs.stderr or "")).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            diagnostic = f"container log unavailable: {type(exc).__name__}"
        if diagnostic:
            print(diagnostic[-8000:], file=sys.stderr)
        raise ContainerStartupError(
            reason + ("; container_log_tail=" + diagnostic[-4000:] if diagnostic else ""),
            state=state, timed_out=timed_out, container_name=container_name, container_id=container_id)

    try:
        while time.perf_counter() < deadline:
            try:
                response = requests.get(
                    f"{base_url}/ready",
                    timeout=2,
                    headers={"Connection": "close"},
                )
            except (requests.exceptions.RequestException, ConnectionError):
                response = None

            if response is not None and response.status_code == 200:
                ready_received_at = time.time()
                try:
                    body = response.json()
                except ValueError:
                    body = None

                if isinstance(body, dict) and body.get("status") == "ok":
                    cold_start_s = time.perf_counter() - t0
                    breakdown = _cold_start_breakdown(
                        body,
                        t0_wall,
                        ready_received_at,
                    )
                    print(
                        f"{log_prefix} Model: {body.get('model_id')}, "
                        f"device: {body.get('device')}, load: {body.get('load_time_s')}s"
                    )
                    print(f"{log_prefix} Server ready. cold_start={cold_start_s:.3f}s")
                    return RunningContainer(
                        name=container_name,
                        container_id=container_id,
                        base_url=base_url,
                        host_port=host_port,
                        cold_start_s=cold_start_s,
                        gpu_device=gpu_device,
                        _model_store_mount=model_store_mount,
                        **breakdown,
                    )

                if response.text.strip() == "ok":
                    cold_start_s = time.perf_counter() - t0
                    breakdown = _cold_start_breakdown(
                        None,
                        t0_wall,
                        ready_received_at,
                    )
                    print(f"{log_prefix} Server ready. cold_start={cold_start_s:.3f}s")
                    return RunningContainer(
                        name=container_name,
                        container_id=container_id,
                        base_url=base_url,
                        host_port=host_port,
                        cold_start_s=cold_start_s,
                        gpu_device=gpu_device,
                        _model_store_mount=model_store_mount,
                        **breakdown,
                    )

            startup_exit_error = container_state.container_startup_exit_error(container_name, mem)
            if startup_exit_error:
                print(f"{log_prefix} Container exited before server became ready: {startup_exit_error}")
                fail_startup(startup_exit_error)
            time.sleep(READY_POLL_INTERVAL_S)

        cold_start_s = time.perf_counter() - t0
        print(f"{log_prefix} Server not ready after {READY_TIMEOUT_S}s. cold_start={cold_start_s:.3f}s")
        startup_exit_error = container_state.container_startup_exit_error(container_name, mem)
        fail_startup(
            startup_exit_error
            or f"server not ready after {READY_TIMEOUT_S}s for container {container_name}",
            timed_out=True,
        )
    except BaseException:
        try:
            remove_owned_container(container_id, host_command.run_command)
        except BaseException:
            retain_mount_for_cleanup_debt(model_store_mount)
            raise
        model_store_mount.close()
        raise


def stop_container_session(session: RunningContainer, log_prefix: Optional[str] = None) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", session.container_id):
        raise ValueError("refusing to remove a container without its owned immutable ID")
    if log_prefix:
        print(f"{log_prefix} Stopping container...")
    remove_owned_container(session.container_id, host_command.run_command, stop=True)
    mount, session._model_store_mount = session._model_store_mount, None
    if mount is not None:
        mount.close()
