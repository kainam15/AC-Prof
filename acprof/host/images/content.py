"""Read-only, capability-gated containerd content checks; never elevate or mutate stores."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass

from acprof.host.images.errors import DockerPreparationError
from acprof.host.preparation.preparation_log import redact_text
from acprof.system import command as host_command

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ROW = re.compile(r"^(\S+)\s+\S+\s+(sha256:[0-9a-f]{64})\s+(complete|incomplete|unavailable)\s+\((\d+)/(\d+|\?)\)")


@dataclass(frozen=True)
class ImageContent:
    status: str
    image: str
    image_id: str = ""
    present: int | None = None
    required: int | None = None
    detail: str = ""
    engine_version: str = ""
    repair_reference: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def docker_endpoint() -> str:
    """Honor Docker CLI context precedence, including remote/Desktop/rootless endpoints."""
    if not os.environ.get("DOCKER_CONTEXT") and os.environ.get("DOCKER_HOST"):
        return os.environ["DOCKER_HOST"]
    result = host_command.run_command(["docker", "context", "inspect", "--format", "{{json .Endpoints.docker.Host}}"], timeout=15)
    if result.returncode:
        return ""
    value = json.loads(result.stdout)
    return value if isinstance(value, str) else ""


def check_content(image: str, image_id: str = "") -> ImageContent:
    """Check compressed content availability for AC-Prof's linux/amd64 platform.

    Docker inspect / runnable snapshots alone cannot prove blob availability.
    Match containerd's target digest to the Docker descriptor before trusting it.
    Unsupported stores, permissions, architectures and contexts remain unverified.
    """
    version = ""
    try:
        ctr = shutil.which("ctr")
        if not ctr:
            return ImageContent("unverified", image, image_id, detail="ctr is unavailable")
        endpoint = docker_endpoint()
        if endpoint not in {"unix:///var/run/docker.sock", "unix:///run/docker.sock"}:
            return ImageContent("unverified", image, image_id, detail="Docker context has no verified local containerd endpoint")
        info = host_command.run_command(["docker", "info", "--format", "{{json .}}"], timeout=15)
        if info.returncode:
            raise ValueError(info.stderr or info.stdout)
        daemon = json.loads(info.stdout)
        version = daemon.get("ServerVersion", "")
        if (not any("io.containerd.snapshotter.v1" in row for row in daemon.get("DriverStatus", []))
                or "rootless" in str(daemon.get("SecurityOptions", []))):
            return ImageContent("unverified", image, image_id, detail="Docker store has no supported content checker", engine_version=version)
        if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
            return ImageContent("unverified", image, image_id, detail="ctr default platform is not linux/amd64", engine_version=version)
        inspected = host_command.run_command(["docker", "image", "inspect", image], timeout=15)
        if inspected.returncode:
            raise ValueError(inspected.stderr or inspected.stdout)
        metadata = json.loads(inspected.stdout)[0]
        if image_id and metadata.get("Id") != image_id:
            raise DockerPreparationError("docker_identity_changed", "Docker image identity changed during content check", image=image)
        image_id = metadata["Id"]
        target = metadata.get("Descriptor", {}).get("digest", "")
        if not _DIGEST.fullmatch(target):
            return ImageContent("unverified", image, image_id, detail="Docker did not expose an OCI target descriptor", engine_version=version)
        checked = host_command.run_command([
            ctr, "--address", "/var/run/docker/containerd/containerd.sock", "--namespace", "moby",
            "images", "check", f"target.digest=={target}",
        ], timeout=20)
        if checked.returncode:
            raise ValueError(checked.stderr or checked.stdout)
        rows = [match for line in checked.stdout.splitlines() if (match := _ROW.match(line)) and match[2] == target]
        states = {(row[3], int(row[4]), None if row[5] == "?" else int(row[5])) for row in rows}
        if len(states) != 1:
            raise ValueError("containerd returned no matching content result or inconsistent aliases")
        status, present, required = states.pop()
        if status == "complete" and (not required or present != required):
            raise ValueError("containerd returned inconsistent content counts")
        reference = next((ref for ref in metadata.get("RepoDigests", []) if ref.endswith("@" + target)), "")
        detail = f"OCI content {status} ({present}/{required if required is not None else '?'})"
        if version_needs_upgrade(version):
            detail += "; Docker Engine 29.9.0 fixes pulls skipping missing blobs (moby/moby#53615)"
        return ImageContent(status, image, image_id, present, required, detail, version, reference)
    except DockerPreparationError:
        raise
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError) as exc:
        return ImageContent("unverified", image, image_id, detail=redact_text(str(exc)), engine_version=version)


def version_needs_upgrade(version: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$", version)
    return bool(match and tuple(map(int, match.groups())) < (29, 9, 0))


def require_content(image: str, image_id: str = "", *, component: str = "environment") -> ImageContent:
    from acprof.host.images.progress import emit_image_progress

    emit_image_progress(component, image, "checking")
    result = check_content(image, image_id)
    emit_image_progress(component, image, "checking", content=result.to_dict())
    print(f"[docker-content] {image}: {result.status}; {result.detail}", flush=True)
    if result.status in {"incomplete", "unavailable"}:
        raise DockerPreparationError("docker_content_incomplete", result.detail, image=image, content=result.to_dict())
    return result


def repair_content(content: dict) -> None:
    """Repull the exact registry digest, retain cached layers, then independently recheck."""
    from acprof.host.images.operations import pull_image

    image, image_id = content["image"], content["image_id"]
    before = check_content(image, image_id)
    if before.status == "complete":
        return
    if before.status not in {"incomplete", "unavailable"} or not before.repair_reference:
        raise DockerPreparationError("docker_repair_unverified", before.detail, image=image, content=before.to_dict())
    if os.environ.get("ACPROF_REQUIRE_PREPARED_IMAGE") == "1" or os.environ.get("ACPROF_MAX_DOWNLOAD", "").strip():
        raise DockerPreparationError("docker_repair_forbidden", "Prepared-image/budget mode requires separate download preparation", image=image)
    pull_image(before.repair_reference, component="environment")
    after = require_content(image, image_id)
    if after.status != "complete":
        raise DockerPreparationError("docker_repair_unverified", after.detail, image=image, content=after.to_dict())
