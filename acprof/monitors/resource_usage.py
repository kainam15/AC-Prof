"""Container CPU/memory and NVML GPU utilization monitoring."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

from acprof.monitors import resource_metrics, resource_readers
from acprof.monitors.common import sample_periodically
from acprof.monitors.interfaces import ResourceStopResult
from acprof.monitors.sampling import DEFAULT_RAW_SAMPLE_LIMIT, SamplingDiagnostics, SamplingPolicy

SAMPLER_STOP_TIMEOUT_S = 1.0

try:
    import pynvml
except Exception:  # pragma: no cover - exercised by runtime fallback
    pynvml = None


class ResourceUsageMonitor:
    def __init__(
        self,
        sample_hz: float = 20.0,
        container_name: str = "",
        cpu_cores: float = 1.0,
        mem_cap_gb: float = 1.0,
        use_gpu: bool = False,
        device_index: int = 0,
        cgroup_root: str = "/sys/fs/cgroup",
        proc_root: str = "/proc",
        cpu_sysfs_root: str = "/sys/devices/system/cpu",
        proc_cpuinfo_path: str = "/proc/cpuinfo",
        device_uuid: str = "",
        sampling_policy: SamplingPolicy = "burst",
        raw_sample_limit: int = DEFAULT_RAW_SAMPLE_LIMIT,
    ) -> None:
        self.sample_hz = float(sample_hz)
        self.container_name = container_name
        self.cpu_cores = float(cpu_cores)
        self.mem_limit_bytes = float(mem_cap_gb) * float(resource_metrics.BYTES_PER_GIB)
        self.use_gpu = bool(use_gpu)
        self.device_index = int(device_index)
        self.device_uuid = device_uuid
        self.dt = 1.0 / self.sample_hz
        self.sampling_policy = sampling_policy
        self.raw_sample_limit = raw_sample_limit
        self.sampling_diagnostics = SamplingDiagnostics(sampling_policy, self.dt)
        self.cpu_sysfs_root = cpu_sysfs_root
        self.proc_cpuinfo_path = proc_cpuinfo_path
        from acprof.platform import detect_environment
        self._partial_platform = detect_environment().environment == "wsl2"
        self.gpu_query_errors: dict[str, str] = {}
        self._cpu_frequency_reader = ((lambda: (None, None)) if self._partial_platform else
                                     resource_readers._prepare_cpu_frequency_reader(
                                         cpu_sysfs_root, proc_cpuinfo_path))

        self.samples: list[resource_metrics.ResourceUsageSample] = []
        self._cpu_reader: Callable[[], float] | None = None
        self._mem_reader: Callable[[], int] | None = None
        self._swap_reader: Callable[[], int] | None = None
        self._swap_limit_reader: Callable[[], int] | None = None
        self._io_reader: Callable[[], tuple[int, int]] | None = None
        self._cpu_throttle_reader: Callable[[], dict[str, float]] | None = None
        self._memory_events_reader: Callable[[], dict[str, float]] | None = None
        self._cpu_pressure_reader: Callable[[], dict[str, float]] | None = None
        self._memory_pressure_reader: Callable[[], dict[str, float]] | None = None
        self._io_pressure_reader: Callable[[], dict[str, float]] | None = None
        self._memory_peak_reader: Callable[[], dict[str, float]] | None = None
        self._memory_stat_reader: Callable[[], dict[str, float]] | None = None
        self._io_operations_reader: Callable[[], dict[str, float]] | None = None
        self._pids_reader: Callable[[], dict[str, float]] | None = None
        self._swap_limit_bytes: int | None = None
        self._io_start: tuple[int, int] | None = None
        self._io_end: tuple[int, int] | None = None
        self._window_counters_start = resource_metrics.WindowCounterSnapshots()
        self._window_counters_end = resource_metrics.WindowCounterSnapshots()
        self._gpu_handle = None
        self._gpu_initialized = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._t_start: float | None = None
        self._t_end: float | None = None
        self._init_error = ""
        self._runtime_error = ""
        self._runtime_errors: list[str] = []
        self._closed = False

        try:
            readers = resource_readers._resolve_container_metric_readers(
                container_name,
                cgroup_root=cgroup_root,
                proc_root=proc_root,
            )
            self._cpu_reader = readers.cpu
            self._mem_reader = readers.memory
            self._swap_reader = readers.swap
            self._swap_limit_reader = readers.swap_limit
            self._io_reader = readers.io
            self._cpu_throttle_reader = readers.cpu_throttle
            self._memory_events_reader = readers.memory_events
            self._cpu_pressure_reader = readers.cpu_pressure
            self._memory_pressure_reader = readers.memory_pressure
            self._io_pressure_reader = readers.io_pressure
            self._memory_peak_reader = readers.memory_peak
            self._memory_stat_reader = readers.memory_stat
            self._io_operations_reader = readers.io_operations
            self._pids_reader = readers.pids
        except Exception as exc:
            self._init_error = str(exc)

        if self.use_gpu:
            if pynvml is None:
                self._record_runtime_error("pynvml unavailable")
            else:
                try:
                    pynvml.nvmlInit()
                    self._gpu_initialized = True
                    self._gpu_handle = (
                        pynvml.nvmlDeviceGetHandleByUUID(self.device_uuid)
                        if self.device_uuid
                        else pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                    )
                except Exception as exc:
                    self._record_runtime_error(str(exc))
                    self._gpu_handle = None

    @property
    def available(self) -> bool:
        return (
            self._cpu_reader is not None
            or self._mem_reader is not None
            or self._swap_reader is not None
            or self._swap_limit_reader is not None
            or self._io_reader is not None
            or self._cpu_throttle_reader is not None
            or self._memory_events_reader is not None
            or self._cpu_pressure_reader is not None
            or self._memory_pressure_reader is not None
            or self._io_pressure_reader is not None
            or self._memory_peak_reader is not None
            or self._memory_stat_reader is not None
            or self._io_operations_reader is not None
            or self._pids_reader is not None
            or self._gpu_handle is not None
        )

    def start(self) -> None:
        if not self.available:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Resource usage monitor is already running")

        self.samples = []
        self.sampling_diagnostics = SamplingDiagnostics(self.sampling_policy, self.dt)
        self._stop_event = threading.Event()
        self._swap_limit_bytes = None
        self._io_start = None
        self._io_end = None
        self._window_counters_start = resource_metrics.WindowCounterSnapshots()
        if self._swap_limit_reader is not None:
            try:
                self._swap_limit_bytes = self._swap_limit_reader()
            except Exception as exc:
                self._record_runtime_error(str(exc))
        if self._io_reader is not None:
            try:
                self._io_start = self._io_reader()
            except Exception as exc:
                self._record_runtime_error(str(exc))
        self._window_counters_start = self._read_window_counter_snapshots()
        self._t_start = time.perf_counter()
        self._t_end = None
        self._append_sample(self._t_start)
        self._thread = threading.Thread(
            target=sample_periodically,
            args=(self._stop_event, self._t_start, self.dt, self._append_sample),
            kwargs={"policy": self.sampling_policy, "diagnostics": self.sampling_diagnostics},
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> ResourceStopResult:
        if not self.available:
            return ResourceStopResult(resource_metrics._nan_result(), self._error_message(), [])

        if self._t_start is None:
            return ResourceStopResult(resource_metrics._nan_result(0), self._error_message(), [])

        self._t_end = time.perf_counter()
        window_counters_end = self._read_window_counter_snapshots()
        self._window_counters_end = window_counters_end
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=SAMPLER_STOP_TIMEOUT_S)
            if self._thread.is_alive():
                raise RuntimeError(
                    "resource sampling thread did not stop within "
                    f"{SAMPLER_STOP_TIMEOUT_S:.1f}s"
                )
            self._thread = None

        self._append_sample(self._t_end)
        if self._io_reader is not None:
            try:
                self._io_end = self._io_reader()
            except Exception as exc:
                self._record_runtime_error(str(exc))
        samples = [
            sample
            for sample in self.samples
            if self._t_start <= sample.timestamp <= self._t_end
        ]
        samples.sort(key=lambda item: item.timestamp)
        self.samples = samples
        result = resource_metrics.reduce_resource_samples(
            samples,
            self.cpu_cores,
            self.mem_limit_bytes,
            min_cpu_interval_s=0.5 * self.dt,
        )
        if self._swap_limit_bytes is not None:
            result.container_swap_limit_bytes = float(self._swap_limit_bytes)
        if self._io_start is not None and self._io_end is not None:
            for direction, index in (("read", 0), ("write", 1)):
                before, after = self._io_start[index], self._io_end[index]
                delta = resource_metrics.counter_delta(
                    {direction: float(before)}, {direction: float(after)}, direction
                )
                if after < before:
                    self._record_runtime_error(
                        f"container_io_{direction}_bytes counter reset: {before} -> {after}"
                    )
                setattr(result, f"container_io_{direction}_bytes", delta)
        resource_metrics.apply_window_counter_metrics(
            result,
            self._window_counters_start,
            window_counters_end,
            max(0.0, self._t_end - self._t_start),
        )
        return ResourceStopResult(
            result,
            self._error_message(),
            samples,
        )

    def telemetry_snapshot(self) -> dict:
        from dataclasses import asdict

        from acprof.telemetry import capture_samples, finite_json
        return finite_json({
            "status": "recorded" if self._t_start is not None and self.samples else "unavailable",
            "error": self._error_message(),
            **capture_samples(self.samples, self.raw_sample_limit),
            "units": {"timestamp": "s_monotonic", "container_cpu_s": "cpu_seconds",
                      "*_bytes": "bytes", "*_pct": "percent", "*_hz": "hertz",
                      "*_mhz": "megahertz", "gpu_temp_c": "celsius",
                      "gpu_pstate": "NVML_P0_to_P15"},
            "sampling": self.sampling_diagnostics.snapshot(),
            "boundaries": self.sampling_boundary_snapshot(),
            "context": {"cpu_cores": self.cpu_cores, "mem_limit_bytes": self.mem_limit_bytes,
                        "min_cpu_interval_s": 0.5 * self.dt, "swap_limit_bytes": self._swap_limit_bytes,
                        "io_start": self._io_start, "io_end": self._io_end,
                        "counters_start": asdict(self._window_counters_start),
                        "counters_end": asdict(self._window_counters_end),
                        "counter_units": {"cpu_throttle.nr_periods": "count",
                                          "cpu_throttle.nr_throttled": "count",
                                          "cpu_throttle.throttled_usec": "microseconds",
                                          "cpu_pressure.*": "microseconds",
                                          "memory_pressure.*": "microseconds",
                                          "io_pressure.*": "microseconds",
                                          "memory_events.*": "count",
                                          "memory_peak.peak": "bytes",
                                          "memory_stat.anon/file/slab": "bytes",
                                          "memory_stat.pgfault/pgmajfault/workingset_refault": "count",
                                          "io_operations.*": "count", "pids.*": "count",
                                          "io_start/io_end": "bytes"},
                        "window_duration_s": max(0.0, self._t_end - self._t_start)
                        if self._t_start is not None and self._t_end is not None else 0.0,
                        "device_uuid": self.device_uuid, "device_index": self.device_index,
                        "gpu_query_errors": self.gpu_query_errors},
        })

    def sampling_boundary_snapshot(self) -> dict:
        """Read existing timestamps only; no cgroup, proc, sysfs or NVML query."""
        return {"start_monotonic_s": self._t_start, "end_monotonic_s": self._t_end,
                "semantics": "monitor_recorded_timestamps",
                "reason": "end timestamp precedes counter snapshots, thread join and the final sample"}

    def close(self) -> None:
        if self._closed:
            return
        if self._thread is not None and self._thread.is_alive():
            self.stop()
        if self._gpu_initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._closed = True

    def _record_runtime_error(self, error: str) -> None:
        if not error or error in self._runtime_errors:
            return
        # Bound sampling-thread diagnostics without losing the first causes.
        if len(self._runtime_errors) < 32:
            self._runtime_errors.append(error)
        elif self._runtime_errors[-1] != "additional monitor errors suppressed":
            self._runtime_errors.append("additional monitor errors suppressed")
        self._runtime_error = "; ".join(self._runtime_errors)

    def _error_message(self) -> str:
        # Retain each failing NVML query instead of losing the first cause to
        # whichever collector happened to fail last.
        gpu_errors = set(self.gpu_query_errors.values())
        errors = [self._init_error] if self._init_error else []
        errors.extend(error for error in self._runtime_errors if error not in gpu_errors)
        errors.extend(f"gpu[{query}]: {error}" for query, error in self.gpu_query_errors.items())
        return "; ".join(errors)

    def _read_window_counter_snapshots(self) -> resource_metrics.WindowCounterSnapshots:
        snapshots = resource_metrics.WindowCounterSnapshots()
        readers = (
            ("cpu_throttle", self._cpu_throttle_reader),
            ("memory_events", self._memory_events_reader),
            ("cpu_pressure", self._cpu_pressure_reader),
            ("memory_pressure", self._memory_pressure_reader),
            ("io_pressure", self._io_pressure_reader),
            ("memory_peak", self._memory_peak_reader),
            ("memory_stat", self._memory_stat_reader),
            ("io_operations", self._io_operations_reader),
            ("pids", self._pids_reader),
        )
        for field, reader in readers:
            if reader is None:
                continue
            try:
                setattr(snapshots, field, reader())
            except Exception as exc:
                self._record_runtime_error(str(exc))
        return snapshots

    def _read_sample(self, timestamp: float) -> resource_metrics.ResourceUsageSample:
        container_cpu_s = None
        container_mem_usage_bytes = None
        container_swap_usage_bytes = None
        gpu_util_pct = None
        gpu_mem_used_bytes = None
        gpu_mem_total_bytes = None
        cpu_freq_avg_hz = None
        cpu_freq_peak_hz = None
        gpu_sm_clock_mhz = None
        gpu_memory_clock_mhz = None
        gpu_pstate = None
        gpu_temp_c = None

        if self._cpu_reader is not None:
            try:
                container_cpu_s = self._cpu_reader()
            except Exception as exc:
                self._record_runtime_error(str(exc))

        if self._mem_reader is not None:
            try:
                container_mem_usage_bytes = self._mem_reader()
            except Exception as exc:
                self._record_runtime_error(str(exc))

        if self._swap_reader is not None:
            try:
                container_swap_usage_bytes = self._swap_reader()
            except Exception as exc:
                self._record_runtime_error(str(exc))

        try:
            cpu_freq_avg_hz, cpu_freq_peak_hz = self._cpu_frequency_reader()
        except Exception as exc:
            self._record_runtime_error(str(exc))

        if self._gpu_handle is not None and pynvml is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util_pct = float(util.gpu)
            except Exception as exc:
                self._gpu_query_error("utilization", exc)
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_mem_used_bytes = int(mem.used)
                gpu_mem_total_bytes = int(mem.total)
            except Exception as exc:
                self._gpu_query_error("memory", exc)
            try:
                gpu_sm_clock_mhz = float(
                    pynvml.nvmlDeviceGetClockInfo(
                        self._gpu_handle,
                        pynvml.NVML_CLOCK_SM,
                    )
                )
            except Exception as exc:
                self._gpu_query_error("sm_clock", exc)
            try:
                gpu_memory_clock_mhz = float(
                    pynvml.nvmlDeviceGetClockInfo(
                        self._gpu_handle,
                        pynvml.NVML_CLOCK_MEM,
                    )
                )
            except Exception as exc:
                self._gpu_query_error("memory_clock", exc)
            try:
                gpu_pstate = resource_metrics._normalize_pstate(
                    pynvml.nvmlDeviceGetPerformanceState(self._gpu_handle)
                )
            except Exception as exc:
                self._gpu_query_error("pstate", exc)
            try:
                gpu_temp_c = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._gpu_handle,
                        pynvml.NVML_TEMPERATURE_GPU,
                    )
                )
            except Exception as exc:
                self._gpu_query_error("temperature", exc)

        return resource_metrics.ResourceUsageSample(
            timestamp,
            container_cpu_s,
            container_mem_usage_bytes,
            gpu_util_pct,
            gpu_mem_used_bytes,
            gpu_mem_total_bytes,
            cpu_freq_avg_hz,
            cpu_freq_peak_hz,
            gpu_sm_clock_mhz,
            gpu_memory_clock_mhz,
            gpu_pstate,
            gpu_temp_c,
            container_swap_usage_bytes,
        )

    def _append_sample(self, timestamp: float) -> None:
        self.samples.append(self._read_sample(timestamp))

    def _gpu_query_error(self, query: str, error: Exception) -> None:
        if (self._partial_platform and pynvml is not None
                and isinstance(error, pynvml.NVMLError_NotSupported)):
            return
        self.gpu_query_errors[query] = str(error)
        self._record_runtime_error(str(error))
