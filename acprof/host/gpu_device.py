"""在测量窗口外解析物理 GPU；容器与主机采集器共享同一 UUID。"""
from __future__ import annotations

import csv
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from subprocess import CompletedProcess

from acprof.system.command import run_command

_SELECTED: ContextVar[dict | None] = ContextVar("acprof_gpu_device", default=None)


@contextmanager
def gpu_device_scope():
    """Limit a pinned selection to one CLI invocation, including embedded callers."""
    token = _SELECTED.set(None)
    try:
        yield
    finally:
        _SELECTED.reset(token)


def selected_gpu_device() -> dict:
    return dict(_SELECTED.get() or {})


def resolve_gpu_device(selector: str | None = None, *, environ: Mapping[str, str] | None = None,
                       command_runner: Callable[..., CompletedProcess[str]] | None = None) -> dict:
    """An explicit probe environment resolves a fresh request, outside the run's pin."""
    if selector is None and environ is None and _SELECTED.get() is not None:
        return selected_gpu_device()
    environment = os.environ if environ is None else environ
    selector = str(selector if selector is not None else
                   environment.get("ACPROF_GPU_DEVICE", environment.get("DEVICE_INDEX", "0"))).strip()
    if not (selector.isdecimal() or selector.startswith("GPU-")) or any(c in selector for c in ",\n\r "):
        raise ValueError("GPU selector must be one physical GPU index or UUID; all and MIG are unsupported")
    query = command_runner or partial(run_command, env=environ)
    result = query(
        ["nvidia-smi", f"--id={selector}",
         "--query-gpu=uuid,index,pci.bus_id,name,memory.total,mig.mode.current",
         "--format=csv,noheader,nounits"],
        timeout=10,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot resolve GPU {selector}: {result.stderr.strip()}")
    rows = list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise RuntimeError(f"Expected exactly one physical GPU for {selector}")
    uuid, index, bus, name, memory_mib, mig = (value.strip() for value in rows[0])
    if not uuid.startswith("GPU-") or mig.lower() == "enabled":
        raise RuntimeError("MIG devices cannot be attributed using physical-device NVML energy")
    return {"uuid": uuid, "index": int(index), "pci_bus_id": bus, "name": name,
            "memory_total_bytes": int(float(memory_mib) * 1024 ** 2)}


def pin_gpu_device(selector: str | None = None) -> dict:
    device = resolve_gpu_device(selector)
    _SELECTED.set(device)
    return dict(device)


def gpu_docker_args(device: dict | None = None) -> list[str]:
    device = resolve_gpu_device() if device is None else device
    return ["--gpus", f"device={device['uuid']}",
            "-e", f"NVIDIA_VISIBLE_DEVICES={device['uuid']}", "-e", "CUDA_VISIBLE_DEVICES=0"]


class DockerGpuError(RuntimeError):
    """A failed container probe retains both Docker's evidence and a recovery step."""

    def __init__(self, detail: str, *, reason_code: str = "passthrough_failed"):
        self.detail = detail
        self.reason_code = reason_code
        self.remedy = {
            "probe_image_missing": "Continue image preparation, or pre-cache the locked base image to check now.",
            "probe_image_invalid": "Check local Docker image access, platform labels and the installed AC-Prof locks.",
            "passthrough_failed": ("Check NVIDIA Container Toolkit and the Docker GPU/CDI configuration on the local "
                                   "engine; retry docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L."),
        }[reason_code]
        super().__init__(f"Docker GPU preflight: {detail}. {self.remedy}")


def _local_probe_image(run) -> str:
    from acprof.host.images.environment import platform_fingerprint, verify_labels
    from acprof.host.images.runtime import inspect_identity
    from acprof.network_policy import docker_hub_image
    from acprof.runtime.profiles import PLATFORMS, PYTHON_BASE_IMAGE

    inspect = partial(inspect_identity, command_runner=run)
    reference = docker_hub_image(PYTHON_BASE_IMAGE)
    if (image := inspect(reference)) and (image.get("os"), image.get("architecture")) == ("linux", "amd64"):
        return image["image_id"]
    for platform in PLATFORMS.values():
        fingerprint = platform_fingerprint(platform)
        name = f"acprof-platform-{platform.platform_id}:{fingerprint[:20]}"
        if image := inspect(name):
            verify_labels(image, {"platform_id": platform.platform_id, "platform_build_fingerprint": fingerprint},
                          "platform")
            if (image.get("os"), image.get("architecture")) != ("linux", "amd64"):
                raise DockerGpuError(f"Expected linux/amd64 for {name}", reason_code="probe_image_invalid")
            return image["image_id"]
    raise DockerGpuError(f"No local locked AC-Prof platform or base image. To verify now: docker pull {reference}",
                         reason_code="probe_image_missing")


def require_docker_gpu(device: dict, *, command_runner=None, image_id: str | None = None) -> str:
    """Prove visibility with the selected UUID and a locked or prepared image."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from uuid import uuid4

    from acprof.host.container_lifecycle import (
        container_owner_labels,
        owned_container_id,
        owner_container_ids,
        remove_owned_container,
    )
    from acprof.host.preflight import NATIVE_DOCKER_SOCKET

    runner = command_runner or run_command

    def run(cmd, *, timeout, check=False):
        # All callers have resolved the native engine. Keep probe and cleanup
        # on that endpoint even if another shell switches the default context.
        return runner(["docker", "--host", f"unix://{NATIVE_DOCKER_SOCKET}", *cmd[1:]], timeout=timeout)

    try:
        try:
            image = _local_probe_image(run) if image_id is None else image_id
        except DockerGpuError:
            raise
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            raise DockerGpuError(str(exc), reason_code="probe_image_invalid") from exc
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
            raise DockerGpuError("Docker returned an invalid local image ID", reason_code="probe_image_invalid")
        with TemporaryDirectory(prefix="acprof-gpu-check-") as directory:
            cidfile = Path(directory) / "container.cid"
            command = ["docker", "run", "--rm", "--pull=never", "--cidfile", str(cidfile),
                       "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                       "--user=65532:65532", "--pids-limit=128", *gpu_docker_args(device)]
            owner = {**container_owner_labels(), "org.acprof.gpu-probe": uuid4().hex}
            for key, value in owner.items():
                command += ["--label", f"{key}={value}"]
            command += ["--entrypoint", "nvidia-smi", image, "--query-gpu=uuid", "--format=csv,noheader"]
            finished = False
            try:
                result = run(command, timeout=15)
                if result.returncode:
                    raise DockerGpuError((result.stderr or result.stdout).strip() or f"docker run exited {result.returncode}")
                if result.stdout.split() != [device["uuid"]]:
                    raise DockerGpuError(f"Expected only {device['uuid']}; container reported {result.stdout.strip()!r}")
                finished = True
            finally:
                if identifier := owned_container_id(cidfile):
                    remove_owned_container(identifier, run)
                elif not finished:
                    # The daemon may have created the container before the CLI
                    # wrote its cidfile. Remove only IDs bearing this probe's nonce.
                    for identifier in owner_container_ids(owner, run):
                        remove_owned_container(identifier, run)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DockerGpuError(str(exc)) from exc
    return f"{device['uuid']} (docker run --gpus, {image})"
