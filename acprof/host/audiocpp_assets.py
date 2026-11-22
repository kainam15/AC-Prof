"""Prepare the upstream release executables outside inference and measurement."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from acprof.artifacts import file_sha256
from acprof.network_policy import DEPENDENCY_USER_AGENT, DownloadSource

_RELEASE = "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.6"
_ASSETS = {
    "cpu": ("audio-v0.9.6-bin-ubuntu-x64-cpu-portable.tar.gz", 75883690,
            "7e447eb6b6c5dffbde9c972c11ff6f4b70442d2301ada2e10e1d9c5ec9c8ac8c"),
    "cuda": ("audio-v0.9.6-bin-ubuntu-x64-cuda12.8-colab.tar.gz", 232585375,
             "0cd6629b4e328a21bcf4a2359fe7ed615e1dc562073d01c8723c45cb7b724ffe"),
}


def selected_assets(task) -> dict:
    if getattr(task, "runtime_backend", "") != "audio_cpp":
        return {}
    from acprof.models.inference_configuration import configuration_device

    requested = (getattr(task, "model_resolution", {}) or {}).get("requested_devices") or ["off", "on"]
    devices = {"cuda" if configuration_device(item) == "on" else "cpu" for item in requested}
    records = {device: {"filename": name, "url": f"{_RELEASE}/{name}", "size": size, "sha256": sha}
            for device, (name, size, sha) in _ASSETS.items() if device in devices}
    local = os.environ.get("ACPROF_AUDIOCPP_CPU_SERVER")
    if "cpu" in records and local:
        path = Path(local).expanduser().resolve(strict=True)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"ACPROF_AUDIOCPP_CPU_SERVER requires an executable: {path}")
        records["cpu"] = {"path": str(path), "size": path.stat().st_size, "sha256": file_sha256(path)}
    return records


def _archive(record: dict) -> Path:
    cache = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "acprof/audiocpp" / record["filename"]


def _verified(path: Path, record: dict) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size != record["size"] or file_sha256(path) != record["sha256"]:
        raise ValueError(f"audio.cpp archive differs from its pinned SHA256 or size: {path}")
    return True


def download_sources(assets: dict, *, process: bool) -> list[DownloadSource]:
    sources = []
    for device, record in assets.items():
        if "path" in record:
            continue
        if device == "cpu":
            # The pinned CPU release references GLIBC_2.43, including in libggml-base.
            # Docker runtimes use the project's pinned Debian trixie platform.
            libc, version = platform.libc_ver() if process else ("glibc", "2.41")
            if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 43):
                raise RuntimeError(
                    f"audio.cpp CPU portable v0.9.6 requires glibc >= 2.43; runtime has {libc} {version}. "
                    "Set ACPROF_AUDIOCPP_CPU_SERVER to a compatible v0.9.6 CPU build (docs/models/adaptation.md)."
                )
        cached = _verified(_archive(record), record)
        sources.append(DownloadSource("native", record["url"], 0 if cached else record["size"],
            cached_bytes=record["size"] if cached else 0, cache_status="hit" if cached else "miss",
            stage="dependencies", network_context="host", source_kind="official",
            detail="audio.cpp release SHA256 " + record["sha256"]))
    return sources


def _fetch(record: dict) -> Path:
    path = _archive(record)
    if _verified(path, record):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(record["url"], headers={"User-Agent": DEPENDENCY_USER_AGENT})
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        pending = Path(output.name)
        try:
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=60) as stream:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > record["size"]:
                        raise ValueError("audio.cpp release archive exceeds the pinned size")
                    digest.update(chunk)
                    output.write(chunk)
            if size != record["size"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("audio.cpp release archive failed size or SHA256 verification")
            output.flush()
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
    return path


def install_assets(assets: dict, destination: Path) -> None:
    if not assets:
        return
    evidence = {}
    for device, record in assets.items():
        target = destination / device
        target.mkdir(parents=True)
        binary = target / "audiocpp_server"
        if "path" in record:
            shutil.copyfile(record["path"], binary)
            binary.chmod(0o755)
            if file_sha256(binary) != record["sha256"]:
                raise ValueError("audio.cpp native executable changed during preparation")
            evidence[device] = {"source": "local-build", **record}
            continue
        with tarfile.open(_fetch(record), mode="r|gz") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                # Deployment builds embed their package specifications. Preserve the
                # server, its adjacent shared libraries and the distribution license.
                if len(relative.parts) != 1:
                    continue
                name = relative.name
                if name not in {"audiocpp_server", "LICENSE"} and ".so" not in name:
                    continue
                if not member.isfile():
                    raise ValueError(f"unexpected audio.cpp archive member type: {member.name}")
                source = archive.extractfile(member)
                assert source is not None
                with source, (target / name).open("xb") as output:
                    shutil.copyfileobj(source, output)
                (target / name).chmod(0o755 if name == "audiocpp_server" else 0o644)
        if not binary.is_file():
            raise ValueError("audio.cpp release is missing its server executable")
        evidence[device] = {"source": record["url"], "archive_sha256": record["sha256"],
                            "sha256": file_sha256(binary)}
    (destination / "manifest.json").write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
