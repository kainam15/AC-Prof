"""在新容器中重放源实验，执行独立的非流式 HTTP 负载协议。"""
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

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import atomic_write_json, read_input_scale_plan
from acprof.host.command import run_command
from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import start_container_session, stop_container_session
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.execution_conditions import ExecutionConditions
from acprof.host.hardware_conditions import record_case_conditions
from acprof.host.load_protocol import LoadConfig, run_load
from acprof.host.run_state import MeasurementLock, file_sha256, host_identity, load_run_state
from acprof.host.runtime_images import ImageInfo, require_image_identity
from acprof.installation import module_command, resource_root


@contextmanager
def capture_packets(path, *, interface):
    from acprof.host.packet_capture import require_packet_latency_prerequisites
    require_packet_latency_prerequisites(interface)
    with path.with_suffix(".log").open("w") as log:
        capture = subprocess.Popen(["tcpdump", "-i", interface, "-n", "-U", "-w", str(path),
                                    "tcp", "port", "8002"], stdout=log, stderr=log)
        try:
            time.sleep(0.2)
            if capture.poll() is not None:
                raise RuntimeError("tcpdump exited before load started")
            yield
        finally:
            try:
                time.sleep(1)
            finally:
                capture.terminate()
                try:
                    capture.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    capture.kill()
                    try:
                        capture.wait(timeout=5)
                    except subprocess.TimeoutExpired as kill_exc:
                        raise RuntimeError("tcpdump could not be reaped after kill") from kill_exc
        if capture.returncode != 0 or path.stat().st_size <= 24:
            raise RuntimeError("load did not produce a valid PCAP")


def validate_packets(path, result):
    command = module_command("acprof.packet.sniff_parse_pcap", python_executable=sys.executable)
    parsed = run_command([*command, str(path), "8002"], capture_output=True, text=True, check=True)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="已完成的正式实验目录")
    parser.add_argument("--gpu", choices=("off", "on"), required=True)
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
    parser.add_argument("--capture", action="store_true", help="保存并校验 PCAP；reuse 自动启用")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.source = args.source.resolve()
    try:
        bootstrap_project_env(Path.cwd())
        state = load_run_state(args.source)
        if state.get("status") != "complete":
            raise ValueError("load requires a completed source experiment")
        options = state["options"]
        conditions = ExecutionConditions.from_options(options, gpu=args.gpu)
        config = LoadConfig(scenario=args.scenario, connections=args.connections, concurrency=args.concurrency,
                            requests=args.requests, rate=args.rate, arrival=args.arrival, seed=args.seed,
                            max_pending=args.max_pending, timeout_seconds=conditions.request_timeout_seconds).validate()
        if args.warmup < 0:
            raise ValueError("warmup must be nonnegative")
        cpus, mems = [int(v) for v in options["cpus"].split(",")], [int(v) for v in options["mems"].split(",")]
        cpu, mem = args.cpu if args.cpu is not None else max(cpus), args.mem if args.mem is not None else max(mems)
        if cpu not in cpus or mem not in mems:
            raise ValueError("load resources must be present in the source experiment")
        plan_path = ArtifactLayout.discover(args.source).path("input_scale_plan.json")
        expected = state["artifacts"].get(str(plan_path.relative_to(args.source)))
        entry = _read_source_input_entry(plan_path, expected, args.input_scale)
        task, image = TaskInfo(**state["runtime"]["task"]), ImageInfo(**state["runtime"]["image"])
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("load output directory must be empty")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    capture = args.capture or args.connections == "reuse"
    identity = {"source_run_id": state["run_id"], "image_id": image.tag, "model_revision": task.model_revision,
                "input_scale_plan_sha256": expected, "input_scale": entry["input_scale"],
                "cpu_cores": cpu, "mem_cap_gb": mem, "conditions": asdict(conditions),
                "protocol": config.protocol(), "capture": capture, "warmup": args.warmup}
    report: dict[str, Any] = {"schema_version": 1, "kind": "nonstream_load_experiment", "run_id": uuid4().hex,
              "successful": False, "status": "running", "identity": identity,
              "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
              "host": host_identity(resource_root()), "scope": "HTTP load; no energy or profiler metrics; not formal experiment result layers"}
    atomic_write_json(output / "load.json", report)
    try:
        with MeasurementLock(), conditions.activate():
            require_image_identity(image.tag, image.runtime_environment)
            session = start_container_session(task, cpu, mem, args.gpu, image, "acprof-load", "[load]",
                                               **conditions.container_options)
            try:
                record_case_conditions(output, f"{cpu}c_{mem}g_{args.gpu}", session, cpuset_cpus=conditions.cpuset_cpus)
                report["gpu_device"] = session.gpu_device
                if args.warmup:
                    warmup = run_load(session.base_url, entry["payload"],
                                      LoadConfig(requests=args.warmup, timeout_seconds=conditions.request_timeout_seconds), token="warmup")
                    if not warmup["successful"]:
                        raise RuntimeError("load warmup failed")
                from contextlib import nullcontext
                pcap = output / "load.pcap"
                with capture_packets(pcap, interface=options["sniff_iface"]) if capture else nullcontext():
                    report["result"] = run_load(session.base_url, entry["payload"], config, token=report["run_id"])
                if capture:
                    report["packet_validation"] = validate_packets(pcap, report["result"])
            finally:
                stop_container_session(session, "[load]")
        report.update(successful=report["result"]["successful"], status="complete")
    except BaseException as error:
        if hasattr(error, "load_report"):
            report["result"] = error.load_report
        report.update(successful=False, status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_write_json(output / "load.json", report)
    print(f"负载实验：{output / 'load.json'}；成功率 {report['result']['success_rate']:.1%}")
    return 0 if report["successful"] else 1
