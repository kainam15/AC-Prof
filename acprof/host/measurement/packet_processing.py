"""Bounded packet derivation, outside the completed measurement window."""

import math
import os
import subprocess

from acprof.host import command as host_command
from acprof.host.measurement.capture import PacketLatencyError


def postprocessing_timeout() -> float:
    """A per-stage budget; adjustable on resume without changing measurement identity."""
    value = os.environ.get("ACPROF_POSTPROCESS_TIMEOUT_S", "300")
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError("ACPROF_POSTPROCESS_TIMEOUT_S must be between 10 and 3600 seconds") from exc
    if not math.isfinite(timeout) or not 10 <= timeout <= 3600:
        raise ValueError("ACPROF_POSTPROCESS_TIMEOUT_S must be between 10 and 3600 seconds")
    return timeout


def run_packet_command(command: list[str], *, stage: str):
    timeout = postprocessing_timeout()
    try:
        return host_command.run_command(command, check=False, timeout=timeout, kill_process_group=True)
    except subprocess.TimeoutExpired as exc:
        raise PacketLatencyError(
            f"{stage} timed out after {timeout:g}s; completed measurements are retained. "
            "Adjust ACPROF_POSTPROCESS_TIMEOUT_S and resume to retry postprocessing only."
        ) from exc
