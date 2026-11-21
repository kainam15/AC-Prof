"""Bounded packet derivation, outside the completed measurement window."""

import math
import os
import stat
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from acprof.artifacts import loads_finite_json, replace_file_durably
from acprof.host.measurement.capture import PacketLatencyError, packet_latency_error
from acprof.installation import module_command
from acprof.packet.protocol import require_packet_metrics
from acprof.results.layout import case_sidecar
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
    """Install this worker's CSV and optional request journal after both are staged."""
    source = Path(csv_path)
    if not source.is_file():
        raise packet_latency_error(f"client did not produce result CSV: {csv_path}")
    requests = case_sidecar(source, "requests")
    with TemporaryDirectory(prefix=".packet-merge-", dir=source.parent) as directory:
        output = Path(directory) / "result.csv"
        request_output = Path(directory) / "requests.jsonl"
        publications = [(output, source)]
        if requests.exists():
            publications.append((request_output, requests))
        result = run_packet_command(
            [*module_command("acprof.packet.merge_packet_latency"), "--requests-output", str(request_output),
             csv_path, latency_path, str(output)],
            stage="packet latency merge",
        )
        if result.returncode != 0:
            raise packet_latency_error("packet latency merge failed", (result.stderr or result.stdout or "").strip())
        for staged, destination in publications:
            if staged.is_symlink() or not staged.is_file():
                raise packet_latency_error(f"packet latency merge did not produce {staged}")
            staged.chmod(stat.S_IMODE(destination.stat().st_mode))
        for staged, destination in publications:
            replace_file_durably(staged, destination)


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


def record_capture_gap(csv_path: str, reason: str, *, interrupted_groups: set[str] | None = None) -> None:
    """Only amend the postprocessing projection. Keep the original checkpoint immutable."""
    import csv
    import json

    from acprof.artifacts import atomic_write
    from acprof.results.csv import read_result_csv

    fields, rows = read_result_csv(csv_path)
    changed = False
    for row in rows:
        status = row.get("packet_status")
        interrupted = row.get("window_id") in (interrupted_groups or ())
        if status not in {"pending", "unavailable"} and not (interrupted and status in {"ok", "partial"}):
            continue
        evidence = json.loads(row["resource_metric_quality"])
        packet = evidence.get("collectors", {}).get("packet_latency")
        if not packet or packet.get("status") != "available":
            continue
        row["packet_status"] = "partial" if interrupted and status in {"ok", "partial"} else "unavailable"
        packet.update(status=row["packet_status"], reason=reason)
        row["resource_metric_quality"] = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        changed = True
    if not changed:
        return

    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write(csv_path, write)
    print(f"[sniff][WARN] {reason}")
