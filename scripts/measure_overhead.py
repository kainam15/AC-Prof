"""以原实验的固定镜像和输入，诊断 RAPL/NVML/cgroup 监测线程对请求的影响。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.analysis.uncertainty import (  # noqa: E402 -- 脚本先设置仓库导入路径。
    bootstrap_mean_interval,
)
from acprof.artifacts import atomic_write_json  # noqa: E402 -- 脚本先设置仓库导入路径。
from acprof.host.execution_conditions import (  # noqa: E402 -- 脚本先设置仓库导入路径。
    ExecutionConditions,
)
from acprof.host.measurement_window import (  # noqa: E402 -- 脚本先设置仓库导入路径。
    MonitorGroup,
    run_matched_control_window,
)


def summarize_overhead(rows, *, seed=0):
    groups = defaultdict(dict)
    for row in rows:
        round_index, scenario = row["round"], row["scenario"]
        value = row["latency_app_s"]
        if not math.isfinite(value) or value <= 0 or round_index in groups[scenario]:
            raise ValueError("开销实验存在无效数值或重复轮次")
        groups[scenario][round_index] = value
    baseline = groups.get("none")
    if not baseline:
        raise ValueError("缺少未启用 monitors 的基线")
    result = []
    for scenario, values in sorted(groups.items()):
        if scenario == "none":
            continue
        if set(values) != set(baseline):
            raise ValueError("对照与基线轮次不匹配")
        changes = [(values[index] / baseline[index] - 1) * 100 for index in sorted(values)]
        differences = [values[index] - baseline[index] for index in sorted(values)]
        low, high = bootstrap_mean_interval(changes, seed=seed)
        result.append({"scenario": scenario, "paired_rounds": len(changes),
                       "paired_mean_change_pct": statistics.fmean(changes),
                       "paired_mean_change_s": statistics.fmean(differences),
                       "paired_change_stdev_s": statistics.stdev(differences) if len(differences) > 1 else None,
                       "baseline_mean_s": statistics.fmean(baseline.values()),
                       "baseline_stdev_s": statistics.stdev(baseline.values()) if len(baseline) > 1 else None,
                       "latency_mean_s": statistics.fmean(values.values()),
                       "latency_stdev_s": statistics.stdev(values.values()) if len(values) > 1 else None,
                       "ci_low_pct": low, "ci_high_pct": high})
    return result


def measure_window(url, payload, *, count, monitors: MonitorGroup, token,
                   control_window=None, timeout: float = 60) -> dict[str, Any]:
    import requests

    lifecycle = time.perf_counter()
    timings = []
    contracts = defaultdict(int)
    primary_error = None
    try:
        if count < 1:
            raise ValueError("request count must be positive")
        if control_window is not None:
            control_window()
        monitors.start()
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        for index in range(count):
            before = time.perf_counter()
            response = requests.post(url + "/predict", json=payload,
                                     headers={"Connection": "close", "X-Req-Id": f"{token}:{index}"}, timeout=timeout)
            elapsed = time.perf_counter() - before
            response.raise_for_status()
            body = response.json()
            if response.status_code != 200 or not isinstance(body, dict) or body.get("error"):
                raise RuntimeError("/predict did not return a completed successful response")
            timings.append(elapsed)
            contract = body.get("workload_contract")
            if contract is not None:
                contracts[json.dumps(contract, sort_keys=True, separators=(",", ":"))] += 1
        cpu_time = time.process_time() - cpu_started
        wall_time = time.perf_counter() - wall_started
    except BaseException as error:
        primary_error = error
        raise
    finally:
        monitors.finish(len(timings), statistics.fmean(timings) if timings else math.nan)
        if primary_error is None or isinstance(primary_error, Exception):
            monitors.raise_if_failed()
    stopped = [{"monitor": type(monitors.monitors[name]).__name__, "samples": len(value[-1]), "error": value[-2]}
               for name, value in monitors.results.items() if name != "mips" and value is not None]
    if any(item["error"] or item["samples"] < 2 for item in stopped):
        raise RuntimeError(f"监测器未取得有效窗口：{stopped}")
    perf_result = asdict(monitors.results["mips"]) if monitors.results.get("mips") is not None else None
    if perf_result is not None and not (math.isfinite(perf_result["instructions_total"]) and perf_result["instructions_total"] > 0):
        raise RuntimeError("perf 未取得有效 instructions")
    return {"latency_app_s": statistics.fmean(timings), "request_count": count,
            "host_process_cpu_s": cpu_time, "request_loop_wall_s": wall_time,
            "lifecycle_wall_s": time.perf_counter() - lifecycle, "monitors": stopped,
            "perf": perf_result, "workload_contracts": [
                {"count": n, "contract": json.loads(contract)} for contract, n in contracts.items()]}


def stop_capture(capture):
    """Drain capture outside request timing, then terminate and reap even on failure."""
    try:
        time.sleep(1.0)
    finally:
        try:
            capture.terminate()
            capture.wait(timeout=5)
        except subprocess.TimeoutExpired:
            capture.kill()
            capture.wait()
        except BaseException:
            capture.kill()
            capture.wait()
            raise


def validate_capture(command, pcap, *, token, count):
    parsed = subprocess.run(command, capture_output=True, text=True)
    if parsed.returncode:
        raise RuntimeError(f"packet parser failed: {parsed.stderr}")
    data = json.loads(parsed.stdout)
    requests = data.get("requests", {})
    if len(requests) != count or any(not key.startswith(token + ":") for key in requests):
        raise RuntimeError(f"packet coverage mismatch: expected={count}, actual={len(requests)}")
    atomic_write_json(pcap.with_suffix(".packets.json"), data)
    return len(requests)


def measure_profile_window(session, entry, *, scenario, rate, count, name, cpu, mem, gpu,
                           token, output, options) -> dict[str, Any]:
    """Internal comparison; reuse collectors and the production idle lifecycle."""
    from acprof.host.packet_capture import _resolve_packet_latency_runtime
    from acprof.monitors.energy_cpu import CPUEnergyMonitor
    from acprof.monitors.energy_nvml import GPUEnergyMonitor
    from acprof.monitors.perf_mips import PerfMIPSMonitor
    from acprof.monitors.resource_usage import ResourceUsageMonitor

    idle = float(options["idle_seconds"])
    device = session.gpu_device
    monitors = MonitorGroup()
    control = None
    primary_error = None
    try:
        with ExitStack() as resources:
            if scenario == "full":
                import shutil

                from acprof.host.packet_capture import _tcpdump_can_capture_without_sudo
                tcpdump = shutil.which("tcpdump")
                if not tcpdump or not _tcpdump_can_capture_without_sudo(tcpdump):
                    raise RuntimeError("full 对照缺少现有 tcpdump capability；不会修改系统权限")
                pcap = output / f"{token}.pcap"
                runtime = _resolve_packet_latency_runtime(str(ROOT), str(pcap), options["sniff_iface"])
                if runtime is None:
                    raise RuntimeError("full 对照缺少 tcpdump/tshark")
                capture_log = resources.enter_context((output / f"{token}-tcpdump.log").open("w"))
                capture = subprocess.Popen(runtime.tcpdump_cmd, stdout=capture_log, stderr=capture_log)
                resources.callback(stop_capture, capture)
                time.sleep(0.2)
                if capture.poll() is not None:
                    raise RuntimeError("tcpdump 在请求前退出")
                if gpu == "on":
                    monitors.add("gpu", GPUEnergyMonitor(sample_hz=rate, idle_seconds=idle,
                                 device_index=device["index"], device_uuid=device["uuid"]))
                monitors.add("cpu", CPUEnergyMonitor(sample_hz=rate, idle_seconds=idle,
                             container_name=name, dram_energy=options.get("dram_energy", "auto")))
                monitors.add("mips", PerfMIPSMonitor(name))
            if scenario in {"basic", "full"}:
                monitors.add("resource", ResourceUsageMonitor(sample_hz=rate, container_name=name,
                             cpu_cores=cpu, mem_cap_gb=mem, use_gpu=gpu == "on",
                             device_index=device.get("index", 0), device_uuid=device.get("uuid", "")))
            time.sleep(float(options["idle_cooldown_seconds"]))
            if scenario == "full":
                def control():
                    run_matched_control_window(monitors, idle_seconds=idle)
            else:
                time.sleep(idle)
            result = measure_window(session.base_url, entry["payload"], count=count, monitors=monitors,
                                    token=token, control_window=control,
                                    timeout=float(options["request_timeout_seconds"]))
    except BaseException as error:
        primary_error = error
        raise
    finally:
        monitors.finish(0, math.nan)
        if primary_error is None or isinstance(primary_error, Exception):
            monitors.raise_if_failed()
    if scenario == "full":
        if capture.returncode != 0 or pcap.stat().st_size <= 24:
            raise RuntimeError("full 对照未取得有效 PCAP")
        result["pcap"] = str(pcap)
        result["packet_request_count"] = validate_capture(runtime.parse_cmd, pcap, token=token, count=count)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="已完成的新实验目录")
    parser.add_argument("--gpu", choices=("off", "on"), required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--sample-hz", default="5,20,100")
    parser.add_argument("--modes", help="内部对照场景，如 none,basic,full；此时 --sample-hz 必须为单值")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rates = [float(value) for value in args.sample_hz.split(",")]
    modes: list[str] | None = args.modes.split(",") if args.modes else None
    if modes is not None and ("none" not in modes or len(modes) < 2 or len(set(modes)) != len(modes)
                              or set(modes) - {"none", "basic", "full"} or len(rates) != 1):
        parser.error("--modes 要求 none 与 basic/full，不允许重复；采样频率须为单值")
    if args.rounds < 3 or args.requests < 1 or len(set(rates)) != len(rates) or not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        parser.error("要求至少 3 轮、正请求数及不重复的正采样频率")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("诊断输出目录必须为空")
    from acprof.host.detect import TaskInfo
    from acprof.host.docker_runtime import start_container_session, stop_container_session
    from acprof.host.env_utils import bootstrap_project_env
    from acprof.host.run_state import MeasurementLock, file_sha256, host_identity, load_run_state
    from acprof.host.runtime_images import ImageInfo, require_image_identity
    from acprof.monitors.energy_cpu import CPUEnergyMonitor
    from acprof.monitors.energy_nvml import GPUEnergyMonitor
    from acprof.monitors.resource_usage import ResourceUsageMonitor

    bootstrap_project_env(ROOT)

    source = args.source.resolve()
    state = load_run_state(source)
    if state.get("status") != "complete" or args.gpu not in state["options"]["gpus"].split(","):
        parser.error("需要已完成且包含所选设备的新实验")
    snapshot = state["runtime"]
    task, image = TaskInfo(**snapshot["task"]), ImageInfo(**snapshot["image"])
    from acprof.artifact_layout import ArtifactLayout
    plan_path = ArtifactLayout.discover(source).path("input_scale_plan.json")
    expected_hash = state["artifacts"].get(str(plan_path.relative_to(source)))
    if not expected_hash or file_sha256(plan_path) != expected_hash:
        parser.error("输入计划与原实验身份不一致")
    plan = json.loads(plan_path.read_text())
    entry = plan["entries"][0]
    cpu = max(map(int, state["options"]["cpus"].split(",")))
    mem = max(map(int, state["options"]["mems"].split(",")))
    name = "acprof-overhead-" + uuid4().hex[:12]
    conditions = ExecutionConditions.from_options(state["options"], gpu=args.gpu)
    recorded_environment = conditions.environment
    report: dict[str, Any] = {"schema_version": 1, "kind": "collection_overhead_diagnostic" if modes else "monitor_overhead_diagnostic", "successful": False,
              "source_run_id": state["run_id"], "image_id": image.tag, "model_revision": task.model_revision,
              "gpu_mode": args.gpu, "cpu_cores": cpu, "mem_cap_gb": mem, "input_scale": entry["input_scale"],
              "input_scale_plan_sha256": expected_hash, "seed": args.seed, "rounds": [],
              "payload_sha256": hashlib.sha256(json.dumps(entry["payload"], sort_keys=True).encode()).hexdigest(),
              "measurement_environment": recorded_environment, "cpuset_cpus": conditions.cpuset_cpus,
              "gpu_device_uuid": conditions.gpu_uuid, "request_timeout_seconds": conditions.request_timeout_seconds,
              "host_identity": host_identity(ROOT),
              "command": [sys.executable, *sys.argv],
              "scope": ("同一常驻模型、固定输入与线程的串行 /predict；none/basic/full 仅为内部采集器对照，"
                        "full 复用无请求对照与 PCAP/perf/RAPL/NVML/cgroup；不含 profiler、TUI、启动和离线合并成本，非正式画像"
                        if modes else "同一常驻模型的 HTTP 请求；比较监测线程，未开启 PCAP/perf/TUI，不是正式能耗实验")}
    try:
        with MeasurementLock(), conditions.activate() as device:
            report["gpu_device"] = device
            require_image_identity(image.tag, image.runtime_environment)
            session = start_container_session(task, cpu, mem, args.gpu, image, name, "[overhead]",
                                               **conditions.container_options)
            name = session.name
            try:
                from acprof.host.hardware_conditions import record_case_conditions
                record_case_conditions(output, f"{cpu}c_{mem}g_{args.gpu}", session,
                                       cpuset_cpus=conditions.cpuset_cpus)
                measure_window(session.base_url, entry["payload"], count=5, monitors=MonitorGroup(), token="warmup",
                               timeout=conditions.request_timeout_seconds)
                rng = random.Random(args.seed)
                for round_index in range(args.rounds):
                    scenarios = list(modes) if modes else [0.0, *rates]
                    rng.shuffle(scenarios)
                    for rate in scenarios:
                        if modes:
                            scenario = rate
                            result = measure_profile_window(
                                session, entry, scenario=scenario, rate=rates[0], count=args.requests,
                                name=name, cpu=cpu, mem=mem, gpu=args.gpu,
                                token=f"overhead-{round_index}-{scenario}", output=output,
                                options=state["options"])
                            report["rounds"].append({"round": round_index, "scenario": scenario, **result})
                            atomic_write_json(output / "overhead.json", report)
                            print(f"[overhead] round={round_index + 1} {scenario}: {result['latency_app_s']:.6f}s/request", flush=True)
                            continue
                        # 监测器初始化与文件输出均不计入请求计时。
                        monitors = MonitorGroup()
                        try:
                            if rate:
                                monitors.add("cpu", CPUEnergyMonitor(sample_hz=rate, container_name=name))
                                monitors.add("resource", ResourceUsageMonitor(
                                    sample_hz=rate, container_name=name, cpu_cores=cpu,
                                    mem_cap_gb=mem, use_gpu=args.gpu == "on",
                                    device_index=session.gpu_device.get("index", 0),
                                    device_uuid=session.gpu_device.get("uuid", "")))
                                if args.gpu == "on":
                                    monitors.add("gpu", GPUEnergyMonitor(sample_hz=rate,
                                        device_index=session.gpu_device["index"], device_uuid=session.gpu_device["uuid"]))
                        except BaseException:
                            monitors.finish(0, math.nan)
                            raise
                        scenario = f"monitors-{rate:g}" if rate else "none"
                        result = measure_window(session.base_url, entry["payload"], count=args.requests,
                                                monitors=monitors, token=f"overhead-{round_index}-{scenario}",
                                                timeout=conditions.request_timeout_seconds)
                        report["rounds"].append({"round": round_index, "scenario": scenario, **result})
                        atomic_write_json(output / "overhead.json", report)
                        print(f"[overhead] round={round_index + 1} {scenario}: {result['latency_app_s']:.6f}s/request", flush=True)
                report["comparisons"] = summarize_overhead(report["rounds"], seed=args.seed)
                report["successful"] = True
            finally:
                stop_container_session(session, "[overhead]")
    except BaseException as error:
        report["successful"] = False
        report["error"] = str(error)
        raise
    finally:
        atomic_write_json(output / "overhead.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
