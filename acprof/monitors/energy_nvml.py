import math
import threading
import time
from typing import Any

import pynvml

from acprof.config import DEFAULT_IDLE_SECONDS
from acprof.monitors.gpu_metrics import (
    EnergyResult,
    apply_gpu_energy_counter,
    median_power,
    reduce_gpu_samples,
    time_weighted_average_power,
    unavailable_gpu_result,
)
from acprof.monitors.interfaces import GPUStopResult
from acprof.monitors.sampling import (
    DEFAULT_RAW_SAMPLE_LIMIT,
    SamplingDiagnostics,
    SamplingPolicy,
    next_deadline,
)

SAMPLER_STOP_TIMEOUT_S = 1.0
COUNTER_SAMPLE_POWER_TOLERANCE = 0.25


def _get_total_power_w(handle):
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


class GPUEnergyMonitor:
    def __init__(
        self,
        sample_hz: float = 20.0,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        device_index: int = 0,
        device_uuid: str = "",
        sampling_policy: SamplingPolicy = "burst",
        raw_sample_limit: int = DEFAULT_RAW_SAMPLE_LIMIT,
    ) -> None:
        self.sample_hz = float(sample_hz)
        self.idle_seconds = float(idle_seconds)
        self.device_index = int(device_index)
        self.device_uuid = device_uuid
        self.dt = 1.0 / self.sample_hz
        self.sampling_policy = sampling_policy
        self.raw_sample_limit = raw_sample_limit
        self.sampling_diagnostics = SamplingDiagnostics(sampling_policy, self.dt)
        self.control_evidence: dict[str, Any] | None = None
        self._last_result: EnergyResult | None = None
        self._control_source_reselected = False

        self.handle = None
        self.gpu_name = "unknown"
        self.idle_power_w = float("nan")
        self.idle_trace: dict[str, Any] = {}
        self.samples: list[tuple[float, float]] = []

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._t_start: float | None = None
        self._t_end: float | None = None
        self._init_error = ""
        self._runtime_error = ""
        self._closed = False
        self._energy_start_mj: int | None = None
        self._energy_error = ""
        self.idle_energy_source = "unavailable"
        self._idle_sample_avg_w = float("nan")

        try:
            pynvml.nvmlInit()
            self.handle = (
                pynvml.nvmlDeviceGetHandleByUUID(self.device_uuid)
                if self.device_uuid
                else pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            )
            gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8", errors="ignore")
            self.gpu_name = str(gpu_name)
        except Exception as exc:
            self._init_error = str(exc)
            self.handle = None
            self.gpu_name = "unknown"


    def _build_idle_trace(
        self,
        *,
        idle_samples: list[tuple[float, float]],
        t_start: float,
        t_end: float,
    ) -> dict[str, Any]:
        powers = [power for _, power in idle_samples]
        trace: dict[str, Any] = {
            "gpu_idle_trace_schema": "nvml_gpu_idle_v1",
            "gpu_idle_baseline_method": "time_weighted_mean",
            "gpu_idle_requested_duration_s": self.idle_seconds,
            "gpu_idle_actual_duration_s": max(0.0, t_end - t_start),
            "gpu_idle_sample_hz": self.sample_hz,
            "gpu_idle_sample_count": len(idle_samples),
            "gpu_idle_power_w": self.idle_power_w,
            "gpu_idle_power_samples": [
                {"t_s": timestamp - t_start, "power_w": power}
                for timestamp, power in idle_samples
            ],
        }
        if powers:
            trace.update({
                "gpu_idle_power_min_w": min(powers),
                "gpu_idle_power_max_w": max(powers),
                "gpu_idle_power_mean_w": sum(powers) / len(powers),
                "gpu_idle_power_time_weighted_mean_w": (
                    time_weighted_average_power(idle_samples)
                ),
                "gpu_idle_power_p50_w": median_power(powers),
            })
        else:
            trace.update({
                "gpu_idle_power_min_w": float("nan"),
                "gpu_idle_power_max_w": float("nan"),
                "gpu_idle_power_mean_w": float("nan"),
                "gpu_idle_power_time_weighted_mean_w": float("nan"),
                "gpu_idle_power_p50_w": float("nan"),
            })
        if self._runtime_error:
            trace["gpu_idle_error"] = self._runtime_error
        return trace

    def apply_control_baseline(
        self,
        result: EnergyResult,
        samples: list[tuple[float, float]],
        *,
        trace: bool = False,
    ) -> float:
        """Use a monitor-matched blank window as the workload baseline."""
        self.idle_power_w = float(result.avg_power_total_w)
        self.idle_energy_source = result.energy_source
        self._idle_sample_avg_w = time_weighted_average_power(samples)
        self.control_evidence = self.telemetry_snapshot(include_control=False)
        self.idle_trace = {}
        if trace and samples:
            self.idle_trace = self._build_idle_trace(
                idle_samples=samples,
                t_start=samples[0][0],
                t_end=samples[-1][0],
            )
            self.idle_trace.update({
                "gpu_idle_trace_schema": "nvml_gpu_control_v1",
                "gpu_idle_baseline_method": (
                    "matched_control_cumulative_energy" if result.energy_source == "nvml_total_energy"
                    else "matched_control_time_weighted_mean"
                ),
                "gpu_idle_energy_source": result.energy_source,
                "gpu_idle_energy_counter_start_mj": result.counter_start_mj,
                "gpu_idle_energy_counter_end_mj": result.counter_end_mj,
                "gpu_idle_energy_duration_s": result.measurement_duration_s,
            })
        return self.idle_power_w

    def start(self) -> None:
        if self._init_error or self.handle is None:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("GPU energy monitor is already running")

        self.samples = []
        self.sampling_diagnostics = SamplingDiagnostics(self.sampling_policy, self.dt)
        self._last_result = None
        self._control_source_reselected = False
        self._runtime_error = ""
        self._stop_event = threading.Event()
        self._t_start = time.perf_counter()
        self._t_end = None
        self._energy_error = ""
        self._energy_start_mj = self._read_energy_mj()

        self._append_sample(self._t_start)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUStopResult:
        if self._init_error or self.handle is None:
            return GPUStopResult(unavailable_gpu_result(), self.gpu_name, self._init_error, [])

        if self._t_start is None:
            return GPUStopResult(unavailable_gpu_result(0, self.idle_power_w), self.gpu_name, self._runtime_error, [])

        self._t_end = time.perf_counter()
        energy_end_mj = self._read_energy_mj() if self._energy_start_mj is not None else None
        self._stop_event.set()
        self._append_sample(self._t_end)
        if self._thread is not None:
            self._thread.join(timeout=SAMPLER_STOP_TIMEOUT_S)
            if self._thread.is_alive():
                raise RuntimeError(
                    "GPU energy sampling thread did not stop within "
                    f"{SAMPLER_STOP_TIMEOUT_S:.1f}s"
                )
            self._thread = None

        samples = [
            (t, p)
            for t, p in self.samples
            if self._t_start <= t <= self._t_end
        ]
        samples.sort(key=lambda item: item[0])
        self.samples = samples

        result = reduce_gpu_samples(samples, self.idle_power_w)
        duration_s = self._t_end - self._t_start
        result.counter_start_mj = self._energy_start_mj
        result.counter_end_mj = energy_end_mj
        result.measurement_duration_s = duration_s
        if self._energy_start_mj is not None and energy_end_mj is not None:
            delta_mj = energy_end_mj - self._energy_start_mj
            if delta_mj >= 0 and duration_s > 0:
                counter_avg_w = delta_mj / 1000.0 / duration_s
                sampled_powers = [power for _, power in samples if math.isfinite(power)]
                # Some drivers return increasing but physically inconsistent
                # cumulative readings. Compare against an independently
                # sampled power envelope before accepting the counter. Leave
                # headroom for sensor update lag and missed brief peaks.
                if len(sampled_powers) >= 3 and (
                    counter_avg_w > max(sampled_powers) * (1.0 + COUNTER_SAMPLE_POWER_TOLERANCE)
                    or counter_avg_w < min(sampled_powers) * (1.0 - COUNTER_SAMPLE_POWER_TOLERANCE)
                ):
                    self._energy_error = (
                        "NVML cumulative energy rejected: counter average "
                        f"{counter_avg_w:.3f} W outside sampled-power envelope "
                        f"[{min(sampled_powers):.3f}, {max(sampled_powers):.3f}] W "
                        f"(tolerance={COUNTER_SAMPLE_POWER_TOLERANCE:.0%}); "
                        "using time-integrated power samples"
                    )
                else:
                    apply_gpu_energy_counter(result)
            else:
                self._energy_error = "cumulative energy counter decreased or window duration is invalid"
        # A cumulative workload measurement cannot be corrected with an
        # independently sampled idle baseline (or vice versa): the subtraction
        # would mix two measurement sources with different update semantics.
        # If either side distrusts the counter, prefer matched power integrals
        # on both sides and retain raw counters in diagnostic provenance.
        if self.idle_energy_source == "power_integration" and result.energy_source == "nvml_total_energy":
            sample_result = reduce_gpu_samples(samples, self.idle_power_w)
            if not math.isfinite(sample_result.energy_total_j):
                raise RuntimeError(
                    "GPU counter/control source mismatch: workload power samples unavailable"
                )
            result.energy_total_j = sample_result.energy_total_j
            result.avg_power_total_w = sample_result.avg_power_total_w
            result.energy_eff_j = sample_result.energy_eff_j
            result.avg_power_eff_w = sample_result.avg_power_eff_w
            result.energy_source = "power_integration"
            self._energy_error = (
                (self._energy_error + "; ") if self._energy_error else ""
            ) + "matched control source is power_integration; workload uses the same source"
        elif self.idle_energy_source == "nvml_total_energy" and result.energy_source == "power_integration":
            self._control_source_reselected = True
            if not math.isfinite(self._idle_sample_avg_w):
                raise RuntimeError(
                    "GPU counter/control source mismatch: idle power samples unavailable"
                )
            self.idle_power_w = self._idle_sample_avg_w
            self.idle_energy_source = "power_integration"
            result.idle_power_w = self.idle_power_w
            result.energy_eff_j = result.energy_total_j - self.idle_power_w * duration_s
            result.avg_power_eff_w = result.avg_power_total_w - self.idle_power_w
            result.peak_power_eff_w = result.peak_power_total_w - self.idle_power_w
            if self.idle_trace:
                self.idle_trace.update({
                    "gpu_idle_power_w": self.idle_power_w,
                    "gpu_idle_energy_source": "power_integration",
                    "gpu_idle_baseline_method": "matched_control_time_weighted_mean",
                    "gpu_idle_baseline_reselected_due_to_workload": True,
                })
            self._energy_error = (
                (self._energy_error + "; ") if self._energy_error else ""
            ) + "paired control baseline recomputed from time-integrated power samples"
        result.energy_fallback_reason = self._energy_error
        self._last_result = result

        return GPUStopResult(
            result,
            self.gpu_name,
            self._runtime_error,
            samples,
        )

    def telemetry_snapshot(self, *, include_control: bool = True) -> dict[str, Any]:
        from acprof.telemetry import capture_samples, finite_json
        result = self._last_result or unavailable_gpu_result()
        energy_recorded = (result.energy_source != "unavailable"
                           and math.isfinite(result.energy_total_j)
                           and math.isfinite(result.avg_power_total_w))
        return finite_json({
            "status": "recorded" if self._t_start is not None and (self.samples or energy_recorded) else "unavailable",
            "error": self._init_error or self._runtime_error,
            "capabilities": {
                "energy_totals": {"status": "recorded" if energy_recorded else "unavailable",
                                  "source": result.energy_source},
                "power_samples": {"status": "recorded" if self.samples else "unavailable",
                                  "sample_count": len(self.samples)},
                "peak_power": {"status": "recorded" if math.isfinite(result.peak_power_total_w) else "unavailable",
                               "source": "power_samples"},
            },
            **capture_samples(self.samples, self.raw_sample_limit,
                              lambda sample: {"timestamp": sample[0], "power_w": sample[1]}),
            "units": {"timestamp": "s_monotonic", "power_w": "watts", "counter_*_mj": "millijoules"},
            "sampling": self.sampling_diagnostics.snapshot(),
            "boundaries": self.sampling_boundary_snapshot(),
            "context": {"device_uuid": self.device_uuid, "device_index": self.device_index,
                        "device_name": self.gpu_name, "idle_power_w": self.idle_power_w,
                        "idle_energy_source": self.idle_energy_source,
                        "idle_sample_avg_w": self._idle_sample_avg_w,
                        "counter_start_mj": result.counter_start_mj,
                        "counter_end_mj": result.counter_end_mj,
                        "measurement_duration_s": result.measurement_duration_s,
                        "energy_source": result.energy_source,
                        "energy_fallback_reason": result.energy_fallback_reason,
                        "counter_sample_power_tolerance": COUNTER_SAMPLE_POWER_TOLERANCE,
                        "control_source_reselected": self._control_source_reselected},
            "control": self.control_evidence if include_control else None,
        })

    def sampling_boundary_snapshot(self) -> dict:
        """Read existing timestamps only; no additional NVML query."""
        return {"start_monotonic_s": self._t_start, "end_monotonic_s": self._t_end,
                "semantics": "monitor_recorded_timestamps",
                "reason": "timestamps precede counter queries; exact hardware read instants are unknown"}

    def close(self) -> None:
        if self._closed:
            return
        if self._thread is not None and self._thread.is_alive():
            self.stop()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        self._closed = True

    def _sample_loop(self) -> None:
        # Keep NVML's existing immediate first callback and thread-local phase;
        # CPU/resource collectors deliberately wait for their first interval.
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            t = time.perf_counter()
            if self._t_start is not None and t >= self._t_start:
                self._append_sample(t)
                finished = time.perf_counter()
                self.sampling_diagnostics.observe(next_t, t, finished)
            else:
                finished = time.perf_counter()
            next_t, skipped = next_deadline(next_t, finished, self.dt, self.sampling_policy)
            self.sampling_diagnostics.skipped_deadlines += skipped
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                self._stop_event.wait(sleep_s)

    def _read_energy_mj(self) -> int | None:
        try:
            value = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid cumulative energy counter")
            return int(value)
        except Exception as exc:
            self._energy_error = f"{type(exc).__name__}: {exc}"
            return None

    def _append_sample(self, timestamp: float) -> None:
        if self.handle is None:
            return
        try:
            self.samples.append((timestamp, _get_total_power_w(self.handle)))
        except Exception as exc:
            self._runtime_error = str(exc)
