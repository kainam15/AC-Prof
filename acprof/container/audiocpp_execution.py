"""Run the pinned audio.cpp server on the device visible to this service."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import sysconfig
from pathlib import Path

from acprof.container.execution import inference_context, synchronize

__all__ = ["inference_context", "metadata", "prepare_device", "synchronize"]


def server_binary(device: str) -> Path:
    for root in (Path(__file__).parent / "_native", Path(sys.prefix) / "share/acprof/native"):
        binary = root / device / "audiocpp_server"
        if binary.is_file():
            return binary
    raise FileNotFoundError(f"Missing pinned audiocpp_server for {device}. Prepare the runtime for this device.")


def server_environment(binary: Path) -> dict[str, str]:
    libraries = [str(binary.parent)]
    if binary.parent.name == "cuda":
        nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
        libraries.extend(str(path) for path in sorted(nvidia.glob("*/lib")))
    if existing := os.environ.get("LD_LIBRARY_PATH"):
        libraries.append(existing)
    return {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(libraries)}


def check_binary(binary: Path, device: str) -> dict:
    build = subprocess.run([str(binary), "--version"], check=False, env=server_environment(binary),
                           capture_output=True, text=True, timeout=30)
    if build.returncode:
        raise RuntimeError(f"audio.cpp binary runtime dependencies unavailable: {build.stderr.strip()}")
    from acprof.models.audiocpp import SPEC_REVISION
    commit = re.search(r"^git: ([0-9a-f]{7,40})(?: |$)", build.stdout, re.MULTILINE)
    assets = json.loads((binary.parent.parent / "manifest.json").read_text())
    local = assets[device]["source"] == "local-build"
    # The official release is pinned by its archive SHA256; its packaged build
    # can legitimately report "git: unknown unknown". Source overrides must
    # identify the revision used by the registered package catalog.
    if ("audio.cpp 0.9.6" not in build.stdout.splitlines()
            or local and (not commit or not SPEC_REVISION.startswith(commit[1]))):
        raise ValueError(f"audio.cpp requires v0.9.6 ({SPEC_REVISION}): {build.stdout.strip()}")
    result = subprocess.run([str(binary), "--list-devices", "--json"], check=False,
                            env=server_environment(binary), capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"audio.cpp device discovery failed ({result.returncode}): {result.stderr.strip()}")
    devices = json.loads(result.stdout)["devices"]
    if not any(item.get("backend") == device for item in devices):
        raise RuntimeError(f"audio.cpp {device} device unavailable: {devices}")
    return {"build": build.stdout.strip(), "devices": devices}


def prepare_device(use_gpu: bool, threads: int = 0) -> str:
    device = "cuda" if use_gpu else "cpu"
    check_binary(server_binary(device), device)
    return device


def metadata() -> dict:
    return {"runtime": "audio.cpp"}


def _exec_owned_server() -> None:
    # The wrapper executes in its own interpreter, outside the threaded parent.
    # Kill the native server even if validation/the serving process dies abruptly.
    import ctypes

    parent = int(sys.argv[1])
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent:
        raise SystemExit("audio.cpp owner exited before exec")
    os.execv(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    _exec_owned_server()
