"""Container CPU/memory and NVML GPU utilization monitoring."""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from acprof.monitors import resource_metrics, resource_readers
from acprof.monitors.common import sample_periodically

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
    ) -> None:
        self.sample_hz = float(sample_hz)
        self.container_name = container_name
        self.cpu_cores = float(cpu_cores)
        self.mem_limit_bytes = float(mem_cap_gb) * float(resource_metrics.BYTES_PER_GIB)
        self.use_gpu = bool(use_gpu)
        self.device_index = int(device_index)
        self.device_uuid = device_uuid
        self.dt = 1.0 / self.sample_hz
        self.cpu_sysfs_root = cpu_sysfs_root
        self.proc_cpuinfo_path = proc_cpuinfo_path
        from acprof.platform import detect_environment
        self._partial_platform = detect_environment().environment == "wsl2"
        self.gpu_query_errors: dict[str, str] = {}
        self._cpu_frequency_reader = ((lambda: (None, None)) if self._partial_platform else
                                     resource_readers._prepare_cpu_frequency_reader(
                                         cpu_sysfs_root, proc_cpuinfo_path))

        self.samples: List[resource_metrics.ResourceUsageSample] = []
        self._cpu_reader: Optional[Callable[[], float]] = None
        self._mem_reader: Optional[Callable[[], int]] = None
        self._swap_reader: Optional[Callable[[], int]] = None
        self._swap_limit_reader: Optional[Callable[[], int]] = None
        self._io_reader: Optional[Callable[[], Tuple[int, int]]] = None
        self._cpu_throttle_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._memory_events_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._cpu_pressure_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._memory_pressure_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._io_pressure_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._memory_peak_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._memory_stat_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._io_operations_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._pids_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._swap_limit_bytes: Optional[int] = None
        self._io_start: Optional[Tuple[int, int]] = None
        self._io_end: Optional[Tuple[int, int]] = None
        self._window_counters_start = resource_metrics._WindowCounterSnapshots()
        self._gpu_handle = None
        self._gpu_initialized = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._t_start: Optional[float] = None
        self._t_end: Optional[float] = None
        self._init_error = ""
        self._runtime_error = ""
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
                self._runtime_error = "pynvml unavailable"
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
                    self._runtime_error = str(exc)
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
        self._stop_event = threading.Event()
        self._swap_limit_bytes = None
        self._io_start = None
        self._io_end = None
        self._window_counters_start = resource_metrics._WindowCounterSnapshots()
        if self._swap_limit_reader is not None:
            try:
                self._swap_limit_bytes = self._swap_limit_reader()
            except Exception as exc:
                self._runtime_error = str(exc)
        if self._io_reader is not None:
            try:
                self._io_start = self._io_reader()
            except Exception as exc:
                self._runtime_error = str(exc)
        self._window_counters_start = self._read_window_counter_snapshots()
        self._t_start = time.perf_counter()
        self._t_end = None
        self._append_sample(self._t_start)
        self._thread = threading.Thread(
            target=sample_periodically,
            args=(self._stop_event, self._t_start, self.dt, self._append_sample),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> Tuple[resource_metrics.ResourceUsageResult, str, List[resource_metrics.ResourceUsageSample]]:
        if not self.available:
            return resource_metrics._nan_result(), self._error_message(), []

        if self._t_start is None:
            return resource_metrics._nan_result(0), self._error_message(), []

        self._t_end = time.perf_counter()
        window_counters_end = self._read_window_counter_snapshots()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        self._append_sample(self._t_end)
        if self._io_reader is not None:
            try:
                self._io_end = self._io_reader()
            except Exception as exc:
                self._runtime_error = str(exc)
        samples = [
            sample
            for sample in self.samples
            if self._t_start <= sample.timestamp <= self._t_end
        ]
        samples.sort(key=lambda item: item.timestamp)
        self.samples = samples
        result = resource_metrics._result_from_samples(
            samples,
            self.cpu_cores,
            self.mem_limit_bytes,
            min_cpu_interval_s=0.5 * self.dt,
        )
        if self._swap_limit_bytes is not None:
            result.container_swap_limit_bytes = float(self._swap_limit_bytes)
        if self._io_start is not None and self._io_end is not None:
            result.container_io_read_bytes = float(
                max(0, self._io_end[0] - self._io_start[0])
            )
            result.container_io_write_bytes = float(
                max(0, self._io_end[1] - self._io_start[1])
            )
        resource_metrics._apply_window_counter_metrics(
            result,
            self._window_counters_start,
            window_counters_end,
            max(0.0, self._t_end - self._t_start),
        )
        return (
            result,
            self._error_message(),
            samples,
        )

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

    def _error_message(self) -> str:
        errors = [error for error in (self._init_error, self._runtime_error) if error]
        return "; ".join(errors)

    def _read_window_counter_snapshots(self) -> resource_metrics._WindowCounterSnapshots:
        snapshots = resource_metrics._WindowCounterSnapshots()
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
                self._runtime_error = str(exc)
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
                self._runtime_error = str(exc)

        if self._mem_reader is not None:
            try:
                container_mem_usage_bytes = self._mem_reader()
            except Exception as exc:
                self._runtime_error = str(exc)

        if self._swap_reader is not None:
            try:
                container_swap_usage_bytes = self._swap_reader()
            except Exception as exc:
                self._runtime_error = str(exc)

        try:
            cpu_freq_avg_hz, cpu_freq_peak_hz = self._cpu_frequency_reader()
        except Exception as exc:
            self._runtime_error = str(exc)

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
        self.gpu_query_errors[query] = str(error)
        if (self._partial_platform and pynvml is not None
                and isinstance(error, pynvml.NVMLError_NotSupported)):
            return
        self._runtime_error = str(error)
