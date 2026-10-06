"""计算与执行 profiler 共用的容器参数、负载计划及产物写入工具。"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Sequence

from acprof.artifacts import atomic_write, read_input_scale_plan

if TYPE_CHECKING:
    from acprof.host.detect import TaskInfo
from acprof.host.env_utils import hf_offline_docker_env_args
from acprof.host.gpu_device import gpu_docker_args
from acprof.runtime_settings import runtime_docker_env_args

CONTAINER_INPUT_SCALE_PLAN_FILE = "/payloads/input_scale_plan.json"


def format_scale_value(scale: float) -> str:
    value = float(scale)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def parse_last_json_line(text: str) -> Dict[str, Any]:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def load_input_scale_plan_entries(
    input_scale_plan_file: str,
) -> List[Dict[str, Any]]:
    if not input_scale_plan_file:
        raise ValueError("input_scale_plan_file is required for compute profiling")
    if not os.path.isfile(input_scale_plan_file):
        raise FileNotFoundError(
            f"input scale plan not found: {input_scale_plan_file}"
        )

    plan = read_input_scale_plan(input_scale_plan_file)
    raw_entries = plan.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(
            f"invalid input scale plan (missing entries): {input_scale_plan_file}"
        )

    entries: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"invalid input scale plan entry at index {idx}: {entry!r}"
            )
        raw_scale = entry.get("input_scale")
        payload = entry.get("payload")
        if raw_scale is None or not isinstance(payload, dict):
            raise ValueError(
                f"input scale plan entry missing input_scale/payload "
                f"at index {idx}"
            )
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
            raise ValueError(
                f"input scale plan entry has invalid input_scale at index {idx}: {raw_scale!r}"
            )
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"input scale plan entry has non-positive/non-finite input_scale "
                f"at index {idx}: {raw_scale!r}"
            )
        entries.append({
            "input_scale": scale,
            "scale_label": str(
                entry.get("scale_label") or format_scale_value(scale)
            ),
            "payload": payload,
        })
    return entries


class _ProfilerMountScope:
    def __init__(self, mount: Any):
        self._mount = mount

    def __enter__(self):
        return self._mount

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is not None and issubclass(
            exc_type, (subprocess.TimeoutExpired, KeyboardInterrupt)
        ):
            from acprof.host.model_store import retain_mount_for_cleanup_debt

            retain_mount_for_cleanup_debt(self._mount)
        else:
            self._mount.close()
        return False


@contextmanager
def profiler_container_command(
    *,
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    use_gpu: bool,
    payload_file: str,
    profile_root: str,
    tool_mount_roots: Sequence[str],
) -> Iterator[List[str]]:
    from acprof.host.model_store import acquire_mount
    from acprof.installation import resource_root
    package_root = str(resource_root() / "acprof")
    cmd = [
        "docker", "run", "--rm",
        f"--cpus={cpu}",
        f"--memory={mem}g",
        "-v", f"{os.path.abspath(payload_file)}:{CONTAINER_INPUT_SCALE_PLAN_FILE}:ro",
        "-v", f"{os.path.abspath(profile_root)}:/profiles",
        "-e", f"MODEL_ID={task_info.model_id}",
        "-e", f"MODEL_REVISION={task_info.model_revision or 'main'}",
        "-e", f"TASK_FAMILY={task_info.task_family}",
        "-e", f"TASK_TYPE={task_info.pipeline_tag}",
        "-e", f"RUNTIME_BACKEND={task_info.runtime_backend}",
        "-e", f"USE_GPU={1 if use_gpu else 0}",
        *hf_offline_docker_env_args(),
        "-e", "HOME=/tmp",
        "-e", f"OMP_NUM_THREADS={max(1, int(cpu))}",
        "-e", f"MKL_NUM_THREADS={max(1, int(cpu))}",
        "-e", f"OPENBLAS_NUM_THREADS={max(1, int(cpu))}",
        "-e", f"NUMEXPR_NUM_THREADS={max(1, int(cpu))}",
        "-e", f"TORCH_NUM_THREADS={max(1, int(cpu))}",
        # Profiler slowdown had no request deadline; an explicit setting may override it.
        "-e", "ACPROF_REQUEST_TIMEOUT_S=none",
        *runtime_docker_env_args(),
    ]
    if not task_info.runtime_profile_id and os.path.isdir(package_root):
        cmd.extend(["-v", f"{package_root}:/app/acprof:ro"])
    for tool_mount_root in tool_mount_roots:
        abs_root = os.path.abspath(tool_mount_root)
        cmd.extend(["-v", f"{abs_root}:{abs_root}:ro"])
    if use_gpu:
        cmd.extend([
            *gpu_docker_args(),
            "--cap-add=SYS_ADMIN",
            "--cap-add=SYS_PTRACE",
            "--security-opt=seccomp=unconfined",
        ])
    mount_index = cmd.index("HOME=/tmp") - 1
    model_store_mount = acquire_mount({
        "model_store": getattr(task_info, "model_store", {}),
    })
    cmd[mount_index:mount_index] = model_store_mount.args
    cmd.append(image_tag)
    with _ProfilerMountScope(model_store_mount):
        yield cmd


def profile_runner_args(entry: Dict[str, Any], repeat: int, mode: str) -> List[str]:
    return [
        "python", "-m", "acprof.container.compute_profile_runner",
        "--payload-file", CONTAINER_INPUT_SCALE_PLAN_FILE,
        "--input-scale", format_scale_value(float(entry["input_scale"])),
        "--repeat", str(max(1, int(repeat))),
        "--profile-mode", mode,
    ]


def write_profile_json(path: str, payload: Dict[str, Any]) -> None:
    """Publish legacy profiler JSON (including unavailable NaN) after collection."""
    atomic_write(path, lambda stream: json.dump(payload, stream, ensure_ascii=True, indent=2))


COMPUTE_THREAD_ENV_NAMES = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
}


def safe_filename_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return token or "unknown"


def relative_artifact(path: str, output_dir: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(output_dir))


def docker_env(cmd: Sequence[str], name: str, value: str) -> List[str]:
    if not cmd:
        return []
    return [*cmd[:-1], "-e", f"{name}={value}", cmd[-1]]


def execution_thread_environment(cmd: Sequence[str]) -> List[str]:
    """Keep execution probes aligned with the normal matrix runtime config."""
    filtered: List[str] = []
    index = 0
    while index < len(cmd):
        value = str(cmd[index])
        if value == "-e" and index + 1 < len(cmd):
            assignment = str(cmd[index + 1])
            if assignment.split("=", 1)[0] in COMPUTE_THREAD_ENV_NAMES:
                index += 2
                continue
        filtered.append(value)
        index += 1
    # The old quota-derived profiler default is removed above. An explicit
    # legacy request must still match the normal server, just like generic settings.
    if filtered and 'TORCH_NUM_THREADS' in os.environ:
        filtered = docker_env(filtered, 'TORCH_NUM_THREADS', os.environ['TORCH_NUM_THREADS'])
    return filtered
