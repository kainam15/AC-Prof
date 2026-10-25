"""Observed process-tree CPU and RSS; never substitute these for cgroup metrics."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import psutil

from acprof.monitors.common import sample_periodically
from acprof.monitors.interfaces import ResourceStopResult
from acprof.monitors.resource_metrics import ResourceUsageResult, empty_resource_result
from acprof.monitors.sampling import SamplingDiagnostics


@dataclass
class ProcessUsageResult(ResourceUsageResult):
    process_cpu_time_s: float = float("nan")
    process_cpu_util_avg_pct: float = float("nan")
    process_rss_avg_bytes: float = float("nan")
    process_rss_peak_bytes: float = float("nan")
    process_resource_status: str = "unavailable"


def reduce_process_samples(samples: list[dict]) -> dict:
    """Integrate same-identity counter deltas; do not charge pre-window child CPU."""
    if len(samples) < 2:
        return {}
    elapsed = samples[-1]["timestamp"] - samples[0]["timestamp"]
    if elapsed <= 0:
        return {}
    result: dict[str, float] = {}
    def complete(field):
        return all(sample['members'] and all(member.get(field) is not None
                    and math.isfinite(member[field]) for member in sample['members'].values()) for sample in samples)
    if complete('cpu_s'):
        total = 0.0
        for before, after in zip(samples, samples[1:], strict=False):
            for key, counters in after["members"].items():
                if key in before["members"]:
                    delta = counters["cpu_s"] - before["members"][key]["cpu_s"]
                    if delta < 0:
                        raise ValueError("process CPU counter decreased for a stable identity")
                    total += delta
        result.update(process_cpu_time_s=total, process_cpu_util_avg_pct=total / elapsed * 100)
    if complete('rss'):
        rss = [sum(member["rss"] for member in sample["members"].values()) for sample in samples]
        area = sum((b["timestamp"] - a["timestamp"]) * (x + y) / 2
                   for a, b, x, y in zip(samples, samples[1:], rss, rss[1:], strict=False))
        result.update(process_rss_avg_bytes=area / elapsed, process_rss_peak_bytes=max(rss))
    for source, average, peak in (("gpu_util_pct", "gpu_util_avg_pct", "gpu_util_peak_pct"),
                                  ("gpu_mem_bytes", "gpu_mem_used_avg_bytes", "gpu_mem_used_peak_bytes")):
        values = [sample.get(source) for sample in samples]
        if all(value is not None and math.isfinite(value) for value in values):
            finite_values = [float(value) for value in values if value is not None]
            result[average], result[peak] = sum(finite_values) / len(finite_values), max(finite_values)
    return result



class ProcessUsageMonitor:
    def __init__(self, *, pid: int, start_ticks: int, sample_hz=20.0, sampling_policy="burst",
                 raw_sample_limit=100000, device_uuid="", use_gpu=False, **kwargs):
        self.pid, self.start_ticks = pid, start_ticks
        self.dt, self.policy = 1 / sample_hz, sampling_policy
        self.limit = raw_sample_limit
        self.samples: list[dict] = []
        self.error = ""
        self.status = "available"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.diagnostics = SamplingDiagnostics(sampling_policy, self.dt)
        self.device_uuid = device_uuid
        self.use_gpu = use_gpu
        self.nvml = None
        self.gpu_handle = None
        self.gpu_status = "not_requested" if not use_gpu else "unavailable"
        self.gpu_error = ""
        self._lost_members = False
        self.gpu_queries = {name: self.gpu_status for name in ("utilization", "memory")}
        self.root = None

    def prepare(self):
        from acprof.process_identity import same_process
        try:
            if not same_process({"pid": self.pid, "start_ticks": self.start_ticks}):
                raise RuntimeError("Process observation target changed before sampling")
            self.root = psutil.Process(self.pid)
        except (PermissionError, psutil.AccessDenied) as exc:
            self.status, self.error = "permission_denied", str(exc)
        if self.use_gpu and self.device_uuid:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByUUID(self.device_uuid)
                self.gpu_status = "available"
                self.gpu_queries = dict.fromkeys(self.gpu_queries, "available")
            except Exception as exc:
                self.gpu_error = str(exc)
                from acprof.capabilities import capability_from_error
                self.gpu_status = capability_from_error(exc).to_dict()["status"]
                self.gpu_queries = dict.fromkeys(self.gpu_queries, self.gpu_status)

    def _sample(self, timestamp):
        members: dict[str, Any] = {}
        try:
            if self.root is None:
                raise psutil.AccessDenied(self.pid)
            if not self.root.is_running():
                raise psutil.NoSuchProcess(self.pid)
            for process in [self.root, *self.root.children(recursive=True)]:
                try:
                    with process.oneshot():
                        if not process.is_running():
                            continue
                        key = f"{process.pid}:{process.create_time()}"
                        values = {"cpu_s": None, "rss": None}
                        for field, reader in (("cpu_s", process.cpu_times), ("rss", process.memory_info)):
                            try:
                                value = reader()
                                values[field] = value.user + value.system if field == 'cpu_s' else value.rss
                            except psutil.AccessDenied as exc:
                                self.status, self.error = "permission_denied", f"{field}: {exc}"
                        members[key] = values
                except psutil.NoSuchProcess:
                    self._lost_members = True
            if self.samples and set(self.samples[-1]["members"]) - set(members):
                self._lost_members = True
        except psutil.AccessDenied as exc:
            members = {}
            self.status, self.error = "permission_denied", str(exc)
        except psutil.NoSuchProcess as exc:
            members = {}
            self.status, self.error = "unavailable", str(exc)
        except Exception as exc:
            members = {}
            self.status, self.error = "error", str(exc)
        sample = {"timestamp": timestamp, "members": members, "gpu_util_pct": None, "gpu_mem_bytes": None}
        if self.gpu_handle is not None:
            for key, operation in (("gpu_util_pct", lambda: self.nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu),
                                   ("gpu_mem_bytes", lambda: self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used)):
                try:
                    sample[key] = operation()
                except Exception as exc:
                    from acprof.capabilities import capability_from_error
                    self.gpu_queries["utilization" if key == "gpu_util_pct" else "memory"] = capability_from_error(exc).to_dict()["status"]
                    self.gpu_error = str(exc)
        self.samples.append(sample)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._sample(time.perf_counter())
        self._thread = threading.Thread(target=sample_periodically,
            args=(self._stop, self.samples[0]["timestamp"], self.dt, self._sample),
            kwargs={"policy": self.policy, "diagnostics": self.diagnostics}, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample(time.perf_counter())
        values = reduce_process_samples(self.samples)
        result_status = self.status
        if self._lost_members and self.status == "available":
            # Keep observed RSS, but missing terminal CPU counters prevent a complete total.
            values.update(process_cpu_time_s=float("nan"), process_cpu_util_avg_pct=float("nan"))
            result_status, self.error = "unavailable", "process-tree CPU coverage incomplete: exited descendants"
        if result_status == "available":
            result_status = "verified" if all(field in values for field in (
                "process_cpu_time_s", "process_cpu_util_avg_pct", "process_rss_avg_bytes", "process_rss_peak_bytes")) else "unavailable"
        result = ProcessUsageResult(**{**asdict(empty_resource_result(len(self.samples))), **values}, process_resource_status=result_status)
        return ResourceStopResult(result, self.error, [])

    def telemetry_snapshot(self):
        from acprof.telemetry import capture_samples
        return {"schema_version": 1, "status": "recorded" if self.samples else "unavailable",
                "process_status": self.status, "lost_members": self._lost_members, "error": self.error,
                **capture_samples(self.samples, self.limit),
                "units": {"timestamp": "s_monotonic", "cpu_s": "s", "rss": "bytes", "gpu_util_pct": "%", "gpu_mem_bytes": "bytes"},
                "source": "psutil", "scope": "observed_process_tree", "pid": self.pid,
                "start_ticks": self.start_ticks, "rss_semantics": "sum; shared pages may be counted repeatedly",
                "short_lived_processes": "sampling cannot prove complete coverage",
                "gpu": {"uuid": self.device_uuid or None, "scope": "device", "sharing": "unknown",
                        "status": self.gpu_status, "error": self.gpu_error,
                        "utilization_status": self.gpu_queries["utilization"], "memory_status": self.gpu_queries["memory"]},
                "sampling": self.diagnostics.snapshot()}

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
            self.nvml = None
