"""Bounded, model-free tool checks in existing local dependency environments.

The caller verifies local Docker and holds MeasurementLock. Each tool stops at
its first successful runtime; this is machine availability, not model coverage.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from acprof.host.profiling.commands import run_profiler_command
from acprof.host.runtimes.contracts import RuntimeCleanupError

TOOL_BUDGET_S = 30.0
ATTEMPT_TIMEOUT_S = 15.0

# Run with the image's Python, without importing AC-Prof or loading a model.
# Check collected events/output, following PyTorch and Valgrind's own tests.
_SELF_CHECK = '''
import json, pathlib, re, subprocess, sys, tempfile
tool = sys.argv[1]
if tool == "torch":
    import torch
    torch.set_num_threads(1)
    value = torch.ones((4, 4), device="cpu")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True, with_flops=True) as profile:
        torch.mm(value, value)
    events = [event for event in profile.key_averages() if event.key == "aten::mm"]
    flops = sum(event.flops for event in events)
    if not events or flops != 128:
        raise RuntimeError("CPU matmul events/FLOPs missing or invalid")
    result = {"version": str(torch.__version__), "events": len(events), "flops": flops}
elif tool == "massif":
    version = subprocess.check_output(["valgrind", "--version"], text=True, timeout=2).strip()
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "massif.out"
        subprocess.run(["valgrind", "--tool=massif", "--time-unit=B", "--max-snapshots=10",
                        "--massif-out-file=" + str(output), sys.executable, "-S", "-c",
                        "data = bytearray(65536); data[0] = 1"],
                       check=True, capture_output=True, text=True, timeout=10)
        report = output.read_text()
    snapshots = len(re.findall(r"^snapshot=\\d+$", report, re.M))
    peak = max((int(value) for value in re.findall(r"^mem_heap_B=(\\d+)$", report, re.M)), default=0)
    if snapshots < 1 or peak < 65536:
        raise RuntimeError("Massif snapshots/heap allocation missing or invalid")
    result = {"version": version, "snapshots": snapshots, "peak_heap_bytes": peak}
else:
    raise ValueError(tool)
print(json.dumps({"tool": tool, **result}))
'''


@dataclass(frozen=True)
class RuntimeToolCheck:
    tool: str
    status: str
    detail: str


def _images(command_runner: Callable) -> tuple[str, ...]:
    # Environment layers carry the dependencies. Avoid repeatedly testing all
    # model descendants; if base tags were pruned, use the surviving images.
    for filters in (("--filter", "label=org.acprof.image-kind=environment"), ()):
        command = ("docker", "image", "ls", "--no-trunc", "--filter", "label=org.acprof.environment",
                   *filters, "--format", "{{.ID}}")
        result = command_runner(command, timeout=10.0)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip())
        images = tuple(dict.fromkeys(result.stdout.split()))
        if any(not re.fullmatch(r"sha256:[0-9a-f]{64}", image) for image in images):
            raise ValueError("Docker returned an invalid immutable image ID")
        if images:
            return images
    return ()


def _probe(tool: str, image: str, timeout: float) -> str:
    result = run_profiler_command([
        "docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65532:65532",
        "--cpus=1", "--memory=2g", "--memory-swap=2g", "--pids-limit=128",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=128m,mode=1777", "--workdir=/tmp",
        "--env=HOME=/tmp", "--env=USER=acprof", "--env=LOGNAME=acprof",
        "--env=TORCHINDUCTOR_CACHE_DIR=/tmp/acprof-torch",
        "--env=CUDA_VISIBLE_DEVICES=", "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=OMP_NUM_THREADS=1", "--env=OPENBLAS_NUM_THREADS=1",
        "--entrypoint=python", image, "-c", _SELF_CHECK, tool,
    ], timeout=timeout, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode:
        raise RuntimeError(f"exit={result.returncode}; {(result.stderr or result.stdout).strip()[-2000:]}")
    payload = json.loads(result.stdout)
    keys = ("events", "flops") if tool == "torch" else ("snapshots", "peak_heap_bytes")
    if (not isinstance(payload, dict) or payload.get("tool") != tool
            or not isinstance(payload.get("version"), str) or not payload["version"]
            or any(type(payload.get(key)) is not int or payload[key] <= 0 for key in keys)
            or (tool == "torch" and payload["flops"] != 128)
            or (tool == "massif" and payload["peak_heap_bytes"] < 65536)):
        raise ValueError("invalid runtime self-check evidence")
    return json.dumps(payload, ensure_ascii=False)


def check_runtime_tools(command_runner: Callable[..., subprocess.CompletedProcess[str]]) -> Sequence[RuntimeToolCheck]:
    """Check both tools regardless of experiment flags; propagate cleanup debt."""
    images = _images(command_runner)
    checks = []
    for tool in ("torch", "massif"):
        deadline = time.monotonic() + TOOL_BUDGET_S
        attempts: list[str] = []
        status = "fail"
        for index, image in enumerate(images):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "pending"
                attempts.append(f"Self-check time budget exhausted; {len(images) - index} runtimes untested")
                break
            try:
                evidence = _probe(tool, image, min(ATTEMPT_TIMEOUT_S, remaining))
            except RuntimeCleanupError:
                raise  # Never start another tool after an unconfirmed cleanup.
            except subprocess.TimeoutExpired:
                attempts.append(f"{image}: self-check timed out")
            except Exception as exc:
                attempts.append(f"{image}: {type(exc).__name__}: {exc}")
            else:
                attempts.append(f"{image}: {evidence}")
                status = "ok"
                break
        checks.append(RuntimeToolCheck(tool, status, "\n".join(attempts) or "No local AC-Prof runtime image"))
    return checks
