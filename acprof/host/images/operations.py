"""Docker operations retain diagnostics and use structured progress where supported."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import httpx

from acprof.host.images.errors import (
    DockerPreparationError,
    frontend_error,
    frontend_failed,
    reason_code,
)
from acprof.host.images.progress import DockerOutput
from acprof.system import command as host_command


def run_docker(command: list[str], *, component: str, image: str, operation: str, step: str = "",
               frontend: str = ""):
    output = DockerOutput(component, image, "building" if operation == "build" else "pulling")
    output.start(detail=step, frontend=frontend, cli=operation == "pull")
    succeeded = cancelled = False
    try:
        result = host_command.run_command(command, capture_output=False, output_line=output.feed,
                                         env_overrides={"BUILDKIT_PROGRESS": "rawjson"} if operation == "build" else {})
        # Alternate command adapters may provide retained output instead of callbacks.
        if not output.seen:
            for line in ((result.stdout or "") + "\n" + (result.stderr or "")).splitlines():
                if line:
                    output.feed(line)
        succeeded = result.returncode == 0 and not output.error
    except KeyboardInterrupt:
        cancelled = True
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        if frontend_failed(str(exc), output.step, frontend):
            raise frontend_error(str(exc), frontend) from exc
        raise DockerPreparationError(reason_code(str(exc), operation), str(exc), image=image, step=step) from exc
    finally:
        output.finish(succeeded=succeeded)
        if cancelled:
            output.transition("cancelled")
    if result.returncode or output.error:
        # The caller owns pull policy: a failed pull can still lead to a build.
        if operation != "pull":
            output.transition("failed", detail=output.detail[:2048])
        if frontend_failed(output.detail, output.step, frontend):
            raise frontend_error(output.detail, frontend, exit_code=result.returncode)
        summary = ("Docker 构建失败: " if operation == "build" else "Docker 拉取失败: ") + output.detail
        raise DockerPreparationError(reason_code(output.detail, operation), summary,
                                    image=image, step=output.step or step, exit_code=result.returncode)
    output.transition("verifying")
    return result


def _anonymous_socket() -> str | None:
    """Keep Docker CLI credential helpers, TLS and SSH contexts authoritative."""
    from acprof.host.images.content import docker_endpoint
    try:
        config = Path(os.environ.get("DOCKER_CONFIG", str(Path.home() / ".docker"))) / "config.json"
        settings = json.loads(config.read_text()) if config.exists() else {}
        if any(settings.get(key) for key in ("auths", "credsStore", "credHelpers")):
            return None
        endpoint = docker_endpoint()
        return endpoint.removeprefix("unix://") if endpoint.startswith("unix:///") else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _engine_uses_offsets(client: httpx.Client) -> bool:
    """Inspect this daemon once; unknown stores cannot prove a fresh byte counter."""
    try:
        response = client.get("http://docker/info", timeout=15.0)
        response.raise_for_status()
        info = response.json()
    except (httpx.HTTPError, ValueError):
        return True
    if not isinstance(info, dict):
        return True
    status = info.get("DriverStatus")
    if status is not None and (not isinstance(status, list) or any(not isinstance(row, list) for row in status)):
        return True
    if any("io.containerd.snapshotter.v1" in row for row in status or []):
        return True
    driver = info.get("Driver")
    return not isinstance(driver, str) or driver not in {
        "overlay2", "btrfs", "zfs", "aufs", "devicemapper", "vfs", "fuse-overlayfs", "windowsfilter",
    }


def pull_image(reference: str, *, component: str) -> None:
    socket = _anonymous_socket()
    if socket is None:
        run_docker(["docker", "pull", "--platform", "linux/amd64", reference],
                   component=component, image=reference, operation="pull", step="docker pull")
        return
    output = DockerOutput(component, reference, "pulling")
    output.start(detail="docker pull")
    succeeded = cancelled = False
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=socket),
                          timeout=httpx.Timeout(120.0, connect=15.0, read=None)) as client:
            output.accounting.engine_offsets = _engine_uses_offsets(client)
            with client.stream("POST", "http://docker/images/create", params={"fromImage": reference, "platform": "linux/amd64"}) as response:
                if response.status_code >= 400:
                    detail = next(response.iter_bytes(chunk_size=12000), b"").decode("utf-8", errors="replace")
                    output.log(detail)
                    code = "docker_auth_failed" if response.status_code in {401, 403} else reason_code(detail, "pull")
                    raise DockerPreparationError(code, detail, image=reference, step="docker pull", exit_code=response.status_code)
                for line in _response_lines(response):
                    output.feed(line)
                if output.error:
                    raise DockerPreparationError(reason_code(output.error, "pull"), output.detail, image=reference, step="docker pull")
                succeeded = True
    except KeyboardInterrupt:
        cancelled = True
        raise
    except httpx.HTTPError as exc:
        raise DockerPreparationError("docker_network_failed", str(exc), image=reference, step="docker pull") from exc
    finally:
        output.finish(succeeded=succeeded)
        if cancelled:
            output.transition("cancelled")
    output.transition("verifying")


def _response_lines(response):
    pending = b""
    # Do not batch to 16 KiB: small daemon events must reach the UI immediately.
    for chunk in response.iter_bytes():
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if len(line) > 65536:
                raise DockerPreparationError("docker_progress_invalid", "Docker progress line exceeds 64 KiB")
            yield line.decode("utf-8", errors="replace")
        if len(pending) > 65536:
            raise DockerPreparationError("docker_progress_invalid", "Docker progress line exceeds 64 KiB")
    if pending:
        yield pending.decode("utf-8", errors="replace")
