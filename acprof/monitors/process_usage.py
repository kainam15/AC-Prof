"""Observed process-tree CPU and RSS; never substitute these for cgroup metrics."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from acprof.monitors.common import sample_periodically, stop_sampling_thread
from acprof.monitors.gpu_telemetry import STATE_INTERVAL_S, gpu_metric_evidence, read_gpu_sample
from acprof.monitors.interfaces import ResourceStopResult
from acprof.monitors.process_cgroup import ProcessCgroupObserver
from acprof.monitors.process_metrics import (
    BOUNDARY_COUNTERS,
    END_GAUGES,
    boundary_metrics,
    counter_decreased,
)
from acprof.monitors.resource_metrics import (
    ResourceUsageResult,
    ResourceUsageSample,
    empty_resource_result,
    reduce_resource_samples,
)
from acprof.monitors.sampling import SamplingDiagnostics


@dataclass
class ProcessUsageResult(ResourceUsageResult):
    process_cpu_time_s: float = float("nan")
    process_cpu_util_avg_pct: float = float("nan")
    process_rss_avg_bytes: float = float("nan")
    process_rss_peak_bytes: float = float("nan")
    process_resource_status: str = "unavailable"
    process_cpu_user_time_s: float = float("nan")
    process_cpu_system_time_s: float = float("nan")
    process_io_read_bytes: float = float("nan")
    process_io_write_bytes: float = float("nan")
    process_io_read_ops: float = float("nan")
    process_io_write_ops: float = float("nan")
    process_leader_voluntary_ctx_switches: float = float("nan")
    process_leader_involuntary_ctx_switches: float = float("nan")
    process_minor_faults: float = float("nan")
    process_major_faults: float = float("nan")
    process_swap_bytes_end: float = float("nan")
    process_count_end: float = float("nan")
    process_threads_end: float = float("nan")
    process_fds_end: float = float("nan")
    process_rss_growth_bytes: float = float("nan")
    service_cgroup_cpu_time_s: float = float("nan")
    service_cgroup_mem_usage_avg_bytes: float = float("nan")
    service_cgroup_mem_usage_peak_bytes: float = float("nan")
    service_cgroup_io_read_bytes: float = float("nan")
    service_cgroup_io_write_bytes: float = float("nan")


def reduce_process_samples(samples: list[dict], *, legacy: bool = False) -> dict:
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
        pairs = 0
        decreased = counter_decreased(samples, 'cpu_s')
        if decreased and legacy:
            raise ValueError("process CPU counter decreased for a stable identity")
        for before, after in zip(samples, samples[1:], strict=False):
            for key, counters in after["members"].items():
                if key in before["members"]:
                    pairs += 1
                    delta = counters["cpu_s"] - before["members"][key]["cpu_s"]
                    total += delta
        if (pairs or legacy) and not decreased:
            result.update(process_cpu_time_s=total, process_cpu_util_avg_pct=total / elapsed * 100)
    for raw, name in (("cpu_user_s", "process_cpu_user_time_s"), ("cpu_system_s", "process_cpu_system_time_s")):
        if complete(raw):
            deltas = [counters[raw] - before['members'][key][raw]
                      for before, after in zip(samples, samples[1:], strict=False)
                      for key, counters in after['members'].items() if key in before['members']]
            if deltas and all(delta >= 0 for delta in deltas):
                result[name] = sum(deltas)
    if complete('rss'):
        rss = [sum(member["rss"] for member in sample["members"].values()) for sample in samples]
        area = sum((b["timestamp"] - a["timestamp"]) * (x + y) / 2
                   for a, b, x, y in zip(samples, samples[1:], rss, rss[1:], strict=False))
        result.update(process_rss_avg_bytes=area / elapsed, process_rss_peak_bytes=max(rss))
    gpu_samples = [ResourceUsageSample(sample['timestamp'], None, None, sample.get('gpu_util_pct'),
        sample.get('gpu_mem_bytes'), sample.get('gpu_mem_total_bytes'),
        gpu_sm_clock_mhz=sample.get('gpu_sm_clock_mhz'), gpu_memory_clock_mhz=sample.get('gpu_memory_clock_mhz'),
        gpu_pstate=sample.get('gpu_pstate'), gpu_temp_c=sample.get('gpu_temp_c')) for sample in samples]
    gpu = asdict(reduce_resource_samples(gpu_samples, float('nan'), float('nan')))
    result.update({name: value for name, value in gpu.items() if name.startswith('gpu_')})
    if legacy:
        # v1 evidence used all-or-nothing GPU utilization/memory reduction.
        for source, names in (('gpu_util_pct', ('gpu_util_avg_pct', 'gpu_util_peak_pct')),
                              ('gpu_mem_bytes', ('gpu_mem_used_avg_bytes', 'gpu_mem_used_peak_bytes'))):
            if not all(isinstance(sample.get(source), (int, float)) and math.isfinite(sample[source]) for sample in samples):
                for name in names:
                    result.pop(name, None)
    result.update(boundary_metrics(samples)[0])
    return result



class ProcessUsageMonitor:
    def __init__(self, *, pid: int, start_ticks: int, sample_hz=20.0, sampling_policy="burst",
                 raw_sample_limit=100000, device_uuid="", use_gpu=False, resource_plan=None, **kwargs):
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
        self.gpu_query_errors: dict[str, str] = {}
        self._state_sample_at: float | None = None
        self._lost_members = False
        self._new_members = False
        self.gpu_queries = {name: self.gpu_status for name in ("utilization", "memory")}
        self.root = None
        self.metric_errors: dict[str, dict[str, str]] = {}
        self.metric_evidence: dict = {}
        self.cgroup = ProcessCgroupObserver(pid, resource_plan or {})

    def prepare(self):
        from acprof.process_identity import same_process
        try:
            if not same_process({"pid": self.pid, "start_ticks": self.start_ticks}):
                raise RuntimeError("Process observation target changed before sampling")
            self.root = psutil.Process(self.pid)
            if not same_process({"pid": self.pid, "start_ticks": self.start_ticks}):
                raise RuntimeError("Process observation target changed during preparation")
        except (PermissionError, psutil.AccessDenied) as exc:
            self.status, self.error = "permission_denied", str(exc)
        self.cgroup.prepare()
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

    def _sample(self, timestamp, *, boundary=False):
        self.cgroup.sample_at(timestamp, boundary=boundary)
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
                                if field == 'cpu_s':
                                    values.update(cpu_user_s=value.user, cpu_system_s=value.system)
                            except psutil.AccessDenied as exc:
                                self.status, self.error = "permission_denied", f"{field}: {exc}"
                        if boundary:
                            values.update(self._read_boundary(process))
                        members[key] = values
                except psutil.NoSuchProcess:
                    self._lost_members = True
            if self.samples and set(self.samples[-1]["members"]) - set(members):
                self._lost_members = True
            if self.samples and set(members) - set(self.samples[-1]['members']):
                self._new_members = True
        except psutil.AccessDenied as exc:
            members = {}
            if not self.error:
                self.status, self.error = "permission_denied", str(exc)
        except psutil.NoSuchProcess as exc:
            members = {}
            self.status, self.error = "unavailable", str(exc)
        except Exception as exc:
            members = {}
            self.status, self.error = "error", str(exc)
        sample = {"timestamp": timestamp, "members": members, "boundary": boundary,
                  "gpu_util_pct": None, "gpu_mem_bytes": None}
        if self.gpu_handle is not None:
            state = self._state_sample_at is None or timestamp - self._state_sample_at >= STATE_INTERVAL_S
            sample.update(read_gpu_sample(self.nvml, self.gpu_handle, self._gpu_query_error, state=state))
            sample['gpu_mem_bytes'] = sample.pop('gpu_mem_used_bytes', None)
            if state:
                self._state_sample_at = timestamp
        self.samples.append(sample)

    def _gpu_query_error(self, name: str, error: Exception) -> None:
        from acprof.capabilities import capability_from_error
        self.gpu_queries[name] = capability_from_error(error).to_dict()['status']
        self.gpu_query_errors[name] = str(error)

    def sample_at(self, timestamp: float, *, boundary: bool = False) -> None:
        """Observe in an owning sampler; do not start an additional thread."""
        self._sample(timestamp, boundary=boundary)

    def _read_boundary(self, process) -> dict:
        values: dict[str, float] = {}
        def read(fields, operation):
            try:
                values.update(operation())
            except (OSError, psutil.Error, AttributeError, NotImplementedError, ValueError) as exc:
                if isinstance(exc, (ProcessLookupError, psutil.NoSuchProcess)):
                    self._lost_members = True
                quality = ("permission_denied" if isinstance(exc, (PermissionError, psutil.AccessDenied)) else
                           "unsupported" if isinstance(exc, (AttributeError, NotImplementedError)) else "error")
                for name in fields:
                    self.metric_errors[name] = {"quality": quality, "reason": f"{type(exc).__name__}: {exc}"}
        def io():
            counters = process.io_counters()
            return dict(zip(("io_read_bytes", "io_write_bytes", "io_read_ops", "io_write_ops"),
                            (counters.read_bytes, counters.write_bytes, counters.read_count, counters.write_count), strict=True))
        def switches():
            counters = process.num_ctx_switches()
            return {"voluntary_ctx": counters.voluntary, "involuntary_ctx": counters.involuntary}
        def faults():
            raw = Path(f"/proc/{process.pid}/stat").read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            if len(fields) < 20:
                raise ValueError('incomplete process stat')
            return {"minor_faults": int(fields[7]), "major_faults": int(fields[9])}
        def swap():
            for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                if line.startswith('VmSwap:'):
                    return {"swap_bytes": int(line.split()[1]) * 1024}
            raise NotImplementedError('VmSwap is unavailable')
        read(("io_read_bytes", "io_write_bytes", "io_read_ops", "io_write_ops"), io)
        read(("voluntary_ctx", "involuntary_ctx"), switches)
        read(("minor_faults", "major_faults"), faults)
        read(("swap_bytes",), swap)
        read(("threads",), lambda: {"threads": process.num_threads()})
        read(("fds",), lambda: {"fds": process.num_fds()})
        return values

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Process usage monitor is already running")
        self.samples = []
        self._state_sample_at = None
        self.diagnostics = SamplingDiagnostics(self.policy, self.dt)
        self._stop.clear()
        self._sample(time.perf_counter(), boundary=True)
        self._thread = threading.Thread(target=sample_periodically,
            args=(self._stop, self.samples[0]["timestamp"], self.dt, self._sample),
            kwargs={"policy": self.policy, "diagnostics": self.diagnostics}, daemon=True)
        self._thread.start()

    def stop(self):
        stop_sampling_thread(self._thread, self._stop, description="process")
        self._thread = None
        self._sample(time.perf_counter(), boundary=True)
        return self.finish_samples()

    def finish_samples(self):
        """Reduce finalized buffers without performing another hardware read."""
        values = reduce_process_samples(self.samples)
        result_status = self.status
        if self._lost_members:
            # Keep observed RSS, but missing terminal CPU counters prevent a complete total.
            values.update({name: float('nan') for name in ('process_cpu_time_s', 'process_cpu_util_avg_pct',
                'process_cpu_user_time_s', 'process_cpu_system_time_s', *BOUNDARY_COUNTERS)})
            result_status = 'unavailable'
            self.error = '; '.join(filter(None, (self.error, 'process-tree CPU coverage incomplete: exited descendants')))
        if result_status == "available":
            result_status = "verified" if all(field in values for field in (
                "process_cpu_time_s", "process_cpu_util_avg_pct", "process_rss_avg_bytes", "process_rss_peak_bytes")) else "unavailable"
        if self._new_members and result_status == 'verified':
            result_status = 'unavailable'
        self.metric_evidence = self._evidence(values)
        cgroup_values, cgroup_evidence = self.cgroup.finalize()
        values.update(cgroup_values)
        self.metric_evidence.update(cgroup_evidence)
        self.metric_evidence.update(gpu_metric_evidence(values, self.gpu_query_errors,
            requested=self.use_gpu, initialization_error=self.gpu_error))
        result = ProcessUsageResult(**{**asdict(empty_resource_result(len(self.samples))), **values,
                                    'process_resource_status': result_status, 'metric_evidence': self.metric_evidence})
        diagnostics = [self.error] if self.error else []
        diagnostics.extend(f'{name}: {evidence["reason"]}' for name, evidence in self.metric_evidence.items()
                           if evidence['quality'] in {'error', 'permission_denied', 'partial'} and evidence['reason'])
        return ResourceStopResult(result, '; '.join(dict.fromkeys(diagnostics)), [])

    def _evidence(self, values: dict) -> dict:
        from acprof.results.metrics import METRICS, row_group_fields
        boundary_errors = boundary_metrics(self.samples)[1]
        raw_fields = {**BOUNDARY_COUNTERS, **END_GAUGES, 'process_cpu_time_s': 'cpu_s',
                      'process_cpu_util_avg_pct': 'cpu_s', 'process_cpu_user_time_s': 'cpu_user_s',
                      'process_cpu_system_time_s': 'cpu_system_s'}
        for name, raw in raw_fields.items():
            if name.startswith('process_cpu_') and counter_decreased(self.samples, raw):
                boundary_errors[name] = 'counter_decreased'
        result = {}
        for name in row_group_fields('process_usage'):
            metric = METRICS[name]
            quality = 'complete' if name in values and math.isfinite(values[name]) else 'unavailable'
            reason = '' if quality == 'complete' else 'missing_counter_or_sample'
            if name in boundary_errors:
                reason = boundary_errors[name]
                quality = 'error' if reason == 'counter_decreased' else 'partial'
            if self._lost_members:
                quality, reason = 'partial', 'exited_processes; terminal counters unavailable'
            elif self._new_members and name.startswith('process_cpu_'):
                quality, reason = 'partial', 'new_processes; pre-observation CPU excluded'
            if raw_fields.get(name) in self.metric_errors:
                error = self.metric_errors[raw_fields[name]]
                quality, reason = error['quality'], error['reason']
            elif quality != 'complete' and self.status not in {'available', 'verified'}:
                quality, reason = self.status, self.error
            result[name] = {'source': metric.source, 'scope': 'observed_process_tree_leaders' if 'leader_' in name else 'observed_process_tree',
                            'window': metric.window, 'quality': quality, 'derivation': 'observed', 'reason': reason}
        return result

    def telemetry_snapshot(self):
        from acprof.telemetry import capture_samples
        return {"schema_version": 1, "status": "recorded" if self.samples else "unavailable",
                "process_status": self.status, "lost_members": self._lost_members,
                "new_members": self._new_members, "process_reduction_version": 2, "error": self.error,
                **capture_samples(self.samples, self.limit),
                "units": {"timestamp": "s_monotonic", "cpu_s": "s", "rss": "bytes", "gpu_util_pct": "%", "gpu_mem_bytes": "bytes"},
                "source": "psutil", "scope": "observed_process_tree", "pid": self.pid,
                "metric_evidence": self.metric_evidence,
                "cgroup": self.cgroup.snapshot(self.limit),
                "boundary_counter_units": {"cpu_user_s/cpu_system_s": "s", "io_*_bytes/swap_bytes": "bytes",
                                           "io_*_ops/*_ctx/*_faults/threads/fds": "count"},
                "start_ticks": self.start_ticks, "rss_semantics": "sum; shared pages may be counted repeatedly",
                "short_lived_processes": "sampling cannot prove complete coverage",
                "gpu": {"uuid": self.device_uuid or None, "scope": "device", "sharing": "unknown",
                        "status": self.gpu_status, "error": self.gpu_error,
                        "query_errors": self.gpu_query_errors, "state_interval_s": STATE_INTERVAL_S,
                        "utilization_status": self.gpu_queries["utilization"], "memory_status": self.gpu_queries["memory"]},
                "sampling": self.diagnostics.snapshot()}

    @property
    def sampling_diagnostics(self):
        return self.diagnostics

    def sampling_boundary_snapshot(self) -> dict:
        return {'start_monotonic_s': self.samples[0]['timestamp'] if self.samples else None,
                'end_monotonic_s': self.samples[-1]['timestamp'] if self.samples else None,
                'semantics': 'process_observation_timestamps',
                'reason': 'timestamp precedes procfs/NVML reads; counters bracket the request window'}

    def close(self):
        stop_sampling_thread(self._thread, self._stop, description="process")
        self._thread = None
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
            self.nvml = None
