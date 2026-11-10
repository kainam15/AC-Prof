"""Bounded packet derivation, outside the completed measurement window."""

import math
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from acprof.artifacts import loads_finite_json, replace_file_durably
from acprof.host.measurement.capture import PacketLatencyError, packet_latency_error
from acprof.installation import module_command
from acprof.packet.protocol import require_packet_metrics
from acprof.system import command as host_command


def parse_packet_output(command: list[str], *, stage: str = "pcap parsing") -> dict:
    """Accept one finite schema-v2 JSON document only after a successful worker exit."""
    result = run_packet_command(command, stage=stage)
    if result.returncode != 0:
        raise packet_latency_error(f"{stage} failed", (result.stderr or result.stdout or "").strip())
    output = result.stdout.strip()
    if not output:
        raise packet_latency_error(f"{stage} produced no output")
    try:
        return require_packet_metrics(loads_finite_json(output))
    except (ValueError, RecursionError) as exc:
        raise packet_latency_error(f"{stage} produced invalid JSON or packet schema", str(exc)) from exc


def merge_packet_csv(csv_path: str, latency_path: str) -> None:
    """Publish only an output created by this worker invocation, never stale scratch."""
    if not Path(csv_path).is_file():
        raise packet_latency_error(f"client did not produce result CSV: {csv_path}")
    # Stay on the input filesystem for atomic replacement. Failed derivations
    # never replace the CSV; checkpointed measurement evidence remains untouched.
    with TemporaryDirectory(prefix=".packet-merge-", dir=Path(csv_path).parent) as directory:
        output = Path(directory) / "result.csv"
        result = run_packet_command(
            [*module_command("acprof.packet.merge_packet_latency"), csv_path, latency_path, str(output)],
            stage="packet latency merge",
        )
        if result.returncode != 0:
            raise packet_latency_error("packet latency merge failed", (result.stderr or result.stdout or "").strip())
        if not output.is_file():
            raise packet_latency_error(f"packet latency merge did not produce {output}")
        replace_file_durably(output, csv_path)


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
        failure = PacketLatencyError(
            f"{stage} timed out after {timeout:g}s; completed measurements are retained. "
            "Adjust ACPROF_POSTPROCESS_TIMEOUT_S and resume to retry postprocessing only."
        )
        for note in getattr(exc, "__notes__", ()):
            failure.add_note(note)
        raise failure from exc
