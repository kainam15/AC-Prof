"""Replay a source experiment or measure an existing OpenAI-compatible streaming service."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from acprof.artifacts import atomic_write_json, read_input_scale_plan
from acprof.cli.load_evidence import load_phase, publish_load_report
from acprof.experiments.load import LoadConfig
from acprof.host.detect import TaskInfo
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.execution_conditions import ExecutionConditions
from acprof.host.hardware_conditions import record_case_conditions
from acprof.host.measurement.load import run_load
from acprof.host.run_state import MeasurementLock, file_sha256, host_identity, load_run_state
from acprof.host.runtimes.service import execution_runtime
from acprof.installation import module_command, resource_root
from acprof.results.layout import ArtifactLayout, resolve_single_result_source
from acprof.system.cleanup import cleanup_on_exit
from acprof.system.command import run_command


def _stop_packet_capture(capture: subprocess.Popen[bytes]) -> None:
    def reap_after_kill() -> None:
        try:
            capture.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("tcpdump could not be reaped after kill") from error

    def force_stop() -> None:
        if capture.returncode is None:
            with cleanup_on_exit(reap_after_kill, description="packet capture reap after kill"):
                capture.kill()

    # A signal or wait failure must not bypass the remaining owned-child cleanup.
    with cleanup_on_exit(force_stop, description="packet capture forced stop"):
        capture.terminate()
        try:
            capture.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # Escalate to a bounded kill/reap when graceful termination times out.


@contextmanager
def capture_packets(path, *, interface, port=8002):
    from acprof.host.measurement.capture import require_packet_latency_prerequisites
    require_packet_latency_prerequisites(interface)
    with path.with_suffix(".log").open("w") as log:
        capture = subprocess.Popen(["tcpdump", "-i", interface, "-n", "-U", "-w", str(path),
                                    "tcp", "port", str(port)], stdout=log, stderr=log)
        with (
            cleanup_on_exit(lambda: _stop_packet_capture(capture), description="packet capture"),
            cleanup_on_exit(lambda: time.sleep(1), description="packet capture drain"),
        ):
            time.sleep(0.2)
            if capture.poll() is not None:
                raise RuntimeError("tcpdump exited before load started")
            yield
        if capture.returncode != 0 or path.stat().st_size <= 24:
            raise RuntimeError("load did not produce a valid PCAP")


def validate_packets(path, result, *, port=8002):
    command = module_command("acprof.packet.sniff_parse_pcap", python_executable=sys.executable)
    parsed = run_command([*command, str(path), str(port)], capture_output=True, text=True, check=True)
    packets = json.loads(parsed.stdout)
    records = list(packets["requests"].values())
    identifiers = [record.get("request_id") for record in records]
    successes = {row["request_id"] for row in result["requests"] if row["status"] == "ok"}
    if len(identifiers) != len(set(identifiers)) or not successes <= set(identifiers):
        raise RuntimeError("PCAP does not uniquely cover successful load requests")
    if result["protocol"]["connections"] == "reuse" and result["successful"]:
        streams = [record["tcp_stream"] for record in records if record["request_id"] in successes]
        if len(set(streams)) == len(streams):
            raise RuntimeError("PCAP contains no observed connection reuse")
    atomic_write_json(path.with_suffix(".packets.json"), packets)
    return {"successful_request_coverage": len(successes), "response_count": len(records),
            "per_request_wire_bytes": "unavailable_on_shared_streams"}


def _read_source_input_entry(
    plan_path: Path, expected_sha256: str | None, requested_scale: float | None,
) -> dict[str, Any]:
    if not expected_sha256 or file_sha256(plan_path) != expected_sha256:
        raise ValueError("source input plan identity mismatch")
    entries = read_input_scale_plan(plan_path).get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"invalid input scale plan file: {plan_path}")
    entry = entries[0] if requested_scale is None else next(
        (
            item for item in entries
            if isinstance(item, dict)
            and item.get("input_scale") is not None
            and float(item["input_scale"]) == requested_scale
        ),
        None,
    )
    if entry is None:
        raise ValueError("input scale must be present in the source experiment")
    if (not isinstance(entry, dict) or entry.get("input_scale") is None
            or not isinstance(entry.get("payload"), dict)):
        raise ValueError(f"invalid input scale plan entry: {entry!r}")
    float(entry["input_scale"])
    return entry


def _replay_resource(values: str, requested: int | None, kind: str) -> int | None:
    available: list[int | None] = [None] if kind == "process" and values == "auto" else [
        int(value) for value in values.split(",")
    ]
    if any(value is not None and value < 1 for value in available):
        raise ValueError("source resources must be positive or Process auto")
    selected = requested if requested is not None else max(available, key=lambda value: value or 0)
    if selected not in available:
        raise ValueError("load resources must be present in the source experiment")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="已完成的正式实验目录")
    parser.add_argument("--gpu", choices=("off", "on"))
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--mem", type=int)
    parser.add_argument("--input-scale", type=float)
    parser.add_argument("--scenario", choices=("serial", "concurrent", "arrival-rate"), default="serial")
    parser.add_argument("--connections", choices=("close", "reuse"), default="close")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--arrival", choices=("fixed", "poisson"), default="fixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pending", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--capture", action="store_true", help="Source replay: save and validate PCAP; reuse enables capture")
    parser.add_argument("--output-dir", type=Path, required=True)
    from acprof.cli.streaming_load import (
        add_streaming_arguments,
        run_streaming_load,
        validate_replay_arguments,
    )
    add_streaming_arguments(parser)
    args = parser.parse_args(argv)
    if args.base_url is not None:
        return run_streaming_load(args, parser)
    try:
        validate_replay_arguments(args)
        bootstrap_project_env(Path.cwd())
        args.source = resolve_single_result_source(args.source).resolve()
        state = load_run_state(args.source)
        if state.get("status") != "complete":
            raise ValueError("load requires a completed source experiment")
        options = state["options"]
        conditions = ExecutionConditions.from_options(options, gpu=args.gpu)
        config = LoadConfig(scenario=args.scenario, connections=args.connections, concurrency=args.concurrency,
                            requests=args.requests, rate=args.rate, arrival=args.arrival, seed=args.seed,
                            max_pending=args.max_pending, timeout_seconds=conditions.request_timeout_seconds).validate()
        config.schedule()  # Reject overflow before input/runtime preparation or output creation.
        if args.warmup < 0:
            raise ValueError("warmup must be nonnegative")
        snapshot = state["runtime"]
        runtime = execution_runtime(snapshot.get("kind", "docker"))
        cpu = _replay_resource(options["cpus"], args.cpu, runtime.kind)
        mem = _replay_resource(options["mems"], args.mem, runtime.kind)
        plan_path = ArtifactLayout.discover(args.source).path("input_scale_plan.json")
        expected = state["artifacts"].get(str(plan_path.relative_to(args.source)))
        entry = _read_source_input_entry(plan_path, expected, args.input_scale)
        task = TaskInfo(**snapshot["task"])
        runtime_id = snapshot["image"]["environment_id" if runtime.kind == "process" else "tag"]
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("source runtime identity is missing")
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("load output directory must be empty")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    capture = args.capture or args.connections == "reuse"
    identity = {"source_run_id": state["run_id"], "runtime_kind": runtime.kind, "runtime_id": runtime_id,
                "model_revision": task.model_revision,
                "input_scale_plan_sha256": expected, "input_scale": entry["input_scale"],
                "cpu_cores": cpu, "mem_cap_gb": mem, "conditions": asdict(conditions),
                "protocol": config.protocol(), "capture": capture, "warmup": args.warmup}
    if runtime.kind == "docker":
        identity["image_id"] = runtime_id
    report: dict[str, Any] = {"schema_version": 1, "kind": "nonstream_load_experiment", "run_id": uuid4().hex,
              "successful": False, "status": "running", "identity": identity,
              "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
              "host": host_identity(resource_root()), "scope": "HTTP load; no energy or profiler metrics; not formal experiment result layers"}
    with publish_load_report(output / "load.json", report):
        with MeasurementLock():
            image = runtime.restore(snapshot["image"], output_dir=output)
            process_gpu = image.runtime_environment.get("gpu_device", {}) if runtime.kind == "process" else None
            with conditions.activate(process_gpu=process_gpu):
                session = runtime.start(task_info=task, cpu=cpu, mem=mem, gpu=args.gpu, image_info=image,
                                        container_name="acprof-load", log_prefix="[load]", **conditions.container_options)
                try:
                    case_id = f"{cpu if cpu is not None else 'inherited'}c_{mem if mem is not None else 'inherited'}g_{args.gpu}"
                    record_case_conditions(output, case_id, session, cpuset_cpus=conditions.cpuset_cpus)
                    report["gpu_device"] = session.gpu_device
                    if args.warmup:
                        with load_phase(report, "warmup"):
                            report["warmup"] = run_load(
                                session.base_url, entry["payload"],
                                LoadConfig(requests=args.warmup, timeout_seconds=conditions.request_timeout_seconds), token="warmup",
                            )
                            if not report["warmup"]["successful"]:
                                raise RuntimeError("load warmup failed; inspect load.json warmup records")
                    from contextlib import nullcontext
                    pcap = output / "load.pcap"
                    port = session.host_port if runtime.kind == "process" else 8002
                    with capture_packets(pcap, interface=options["sniff_iface"], port=port) if capture else nullcontext():
                        with load_phase(report, "result"):
                            report["result"] = run_load(session.base_url, entry["payload"], config, token=report["run_id"])
                    if capture:
                        report["packet_validation"] = validate_packets(pcap, report["result"], port=port)
                finally:
                    runtime.stop(session, log_prefix="[load]")
        report.update(successful=report["result"]["successful"], status="complete")
    print(f"负载实验：{output / 'load.json'}；成功率 {report['result']['success_rate']:.1%}")
    return 0 if report["successful"] else 1
