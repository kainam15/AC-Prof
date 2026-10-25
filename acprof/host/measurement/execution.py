"""Own one complete measurement window behind replaceable request and collector ports.

Requests retain their existing short-lived connection protocol. Collectors are
prepared, sampled and finalized in MonitorGroup's defined order; publication and
result reduction run only after all collectors have finished.
"""
from __future__ import annotations

import datetime
import time
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, TextIO, cast

from acprof.capabilities import measurement_requested
from acprof.failures import RuntimeFailure
from acprof.host.measurement import (
    diagnostics as client_diagnostics,
    publication as client_publication,
)
from acprof.host.measurement.config import ClientConfig
from acprof.host.measurement.errors import EnergyAbort, MIPSAbort, RequestTimeoutAbort
from acprof.host.measurement.metrics import WindowMetrics, to_float_or_nan
from acprof.host.measurement.monitors import (
    MonitorCleanupError,
    MonitorGroup,
    run_matched_control_window,
)
from acprof.host.measurement.requests import (
    WindowRequestProgress,
    execute_requests,
    merge_effective_input_scale,
)
from acprof.host.measurement.types import (
    MonitorDependencies,
    WindowObservations,
    WindowResult,
    WindowSpec,
)
from acprof.host.profiling.execution_plan import ExecutionProfileResolver
from acprof.monitors.interfaces import CPUStopResult, GPUStopResult, ResourceStopResult
from acprof.telemetry import window_evidence

if TYPE_CHECKING:
    from acprof.monitors.perf_mips import MIPSResult


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class WindowExecutor:
    """Execute and finalize windows without exposing private lifecycle steps."""

    def __init__(
        self,
        config: ClientConfig,
        dependencies: MonitorDependencies,
        *,
        one_request: Callable[..., dict[str, Any]],
        cold_start_metrics: Callable[[], dict[str, Any]],
        environment_class: str,
        compute_profile_plan: dict[str, Any],
        execution_profiles: ExecutionProfileResolver,
        use_energy: bool,
        runtime_failures: list[dict[str, Any]],
        slow_latency_threshold_s: float | None = None,
    ) -> None:
        self.config = config
        self.dependencies = dependencies
        self.one_request = one_request
        self.cold_start_metrics = cold_start_metrics
        self.environment_class = environment_class
        self.compute_profile_plan = compute_profile_plan
        self.execution_profiles = execution_profiles
        self.use_energy = use_energy
        self.runtime_failures = runtime_failures
        self.slow_latency_threshold_s = slow_latency_threshold_s
        self.cpu_idle_history: list[float] = []
        self.gpu_idle_history: list[float] = []

    def execute(self, spec: WindowSpec, requests_stream: TextIO) -> WindowResult:
        group_id = spec.group_id(self.config.case_name)
        observations = WindowObservations()
        progress = WindowRequestProgress()
        metrics = WindowMetrics(
            self.config, spec.scale_entry, use_energy=self.use_energy,
            slow_latency_threshold_s=self.slow_latency_threshold_s,
        )
        try:
            self._sample(spec, group_id, progress, observations, requests_stream)
            metrics.reduce(progress, observations)
        except (RequestTimeoutAbort, EnergyAbort, MIPSAbort) as exc:
            self._preserve_uncommitted(spec, group_id, progress, observations, exc)
            raise
        except MonitorCleanupError as exc:
            self._preserve_uncommitted(spec, group_id, progress, observations, exc)
            if self._is_mips_error(exc):
                raise MIPSAbort(str(exc)) from exc
            raise
        except Exception as exc:
            if self._is_mips_error(exc):
                self._preserve_uncommitted(spec, group_id, progress, observations, exc)
                raise MIPSAbort(str(exc)) from None
            if isinstance(exc, RuntimeFailure):
                self.runtime_failures.append(exc.failure.to_dict())
            metrics.fail(exc)
        except BaseException as exc:
            self._preserve_uncommitted(spec, group_id, progress, observations, exc)
            raise

        row = metrics.build_row(
            progress, observations,
            warmup=spec.warmup,
            repeat_idx=spec.repeat_idx,
            environment_class=self.environment_class,
            cold_start=self.cold_start_metrics(),
            compute_profile_plan=self.compute_profile_plan,
            execution_profiles=self.execution_profiles,
            cpu_idle_history=self.cpu_idle_history,
            gpu_idle_history=self.gpu_idle_history,
        )
        diagnostic = self._idle_diagnostic(row, group_id, observations, metrics) if self.config.idle_debug else None
        evidence = self._evidence(spec, group_id, progress, observations)
        return WindowResult(row, diagnostic, group_id, evidence, observations.request_record)

    def _preserve_uncommitted(self, spec: WindowSpec, group_id: str, progress: WindowRequestProgress,
                              observations: WindowObservations, error: BaseException) -> None:
        if not self.config.telemetry_dir:
            return
        from pathlib import Path

        from acprof.telemetry import write_window_evidence
        from acprof.window_records import window_filename
        evidence = self._evidence(spec, group_id, progress, observations)
        evidence["context"]["uncommitted_error"] = f"{type(error).__name__}: {error}"
        try:
            write_window_evidence(Path(self.config.telemetry_dir) / window_filename(group_id), evidence)
        except Exception as storage_error:
            # Preserve cancellation/timeout priority, but retain a visible second
            # failure instead of claiming that partial raw evidence was saved.
            error.add_note(f"could not preserve uncommitted telemetry: {storage_error}")

    def _evidence(self, spec: WindowSpec, group_id: str, progress: WindowRequestProgress,
                  observations: WindowObservations) -> dict[str, Any]:
        return window_evidence(
            identity={"run_id": self.config.run_id or None, "case_id": self.config.case_id or self.config.case_name,
                      "case_attempt_id": self.config.case_attempt_id or None, "window_id": group_id},
            context={"input_scale": spec.scale, "scale_label": spec.scale_entry["scale_label"],
                     "warmup": spec.warmup, "repeat_idx": spec.repeat_idx,
                     "completed_requests": progress.count, "cpu_cores": self.config.cpu_cores,
                     "mem_cap_gb": self.config.mem_cap_gb, "gpu_mode": self.config.gpu_mode,
                     "profiling_mode": self.config.profiling_mode,
                     "model_revision": self.config.model_revision, "image_tag": self.config.image_tag,
                     "sampling_policy": self.config.sampling_policy, "sample_hz": self.config.sample_hz,
                     "environment_class": self.environment_class},
            collectors=observations.collector_evidence,
        )

    def _sample(
        self, spec: WindowSpec, group_id: str, progress: WindowRequestProgress,
        observations: WindowObservations, requests_stream: TextIO,
    ) -> None:
        monitors = MonitorGroup()
        scale = spec.scale
        payload_override = spec.scale_entry.get("payload")
        primary_error: BaseException | None = None
        try:
            observations.gpu_monitor, observations.cpu_monitor = self._attach_monitors(monitors)
            (
                observations.gpu_idle_measured_at, observations.cpu_idle_measured_at,
                observations.gpu_idle_trace, observations.idle_trace,
                observations.gpu_idle_debug_snapshot, observations.idle_debug_snapshot,
            ) = self._start_monitors(monitors, observations.gpu_monitor, observations.cpu_monitor)
            execute_requests(
                progress,
                should_send=self._should_send_request,
                one_request=self.one_request,
                merge_scale=merge_effective_input_scale,
                scale=scale,
                group_id=group_id,
                payload_override=payload_override,
                request_limit=spec.request_limit,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            self._finish_monitors(
                monitors, progress, observations, spec, group_id, requests_stream,
                primary_error,
            )

    def _finish_monitors(
        self, monitors: MonitorGroup, progress: WindowRequestProgress,
        observations: WindowObservations, spec: WindowSpec, group_id: str,
        requests_stream: TextIO, primary_error: BaseException | None,
    ) -> None:
        finalization_error: BaseException | None = None
        try:
            monitors.finish(
                progress.count,
                progress.elapsed_s / progress.count if progress.count else float("nan"),
            )
            if monitors.results.get("resource") is not None:
                observations.resource_result, observations.resource_error, observations.resource_samples = cast(
                    ResourceStopResult, monitors.results["resource"])
            if monitors.results.get("gpu") is not None:
                observations.gpu_result, _, observations.gpu_error, observations.gpu_samples = cast(
                    GPUStopResult, monitors.results["gpu"])
            if monitors.results.get("cpu") is not None:
                observations.cpu_result, observations.cpu_error, observations.cpu_samples = cast(
                    CPUStopResult, monitors.results["cpu"])
            observations.mips_result = cast("MIPSResult | None", monitors.results.get("mips"))
            # Serialization is outside every measurement window and reuses the
            # collectors' finalized buffers; unsupported test/adaptor collectors
            # explicitly report absent raw evidence rather than inventing it.
            for name, monitor in monitors.monitors.items():
                snapshot = getattr(monitor, "telemetry_snapshot", None)
                if any(operation.startswith(f"{name}.") for operation, _ in monitors.failures):
                    observations.collector_evidence[name] = {
                        "status": "unavailable", "reason": "collector_cleanup_failed", "error": monitors.error,
                    }
                    continue
                observations.collector_evidence[name] = snapshot() if snapshot is not None else {
                    "status": "unavailable", "reason": "collector_has_no_raw_evidence_protocol",
                }
            if observations.mips_result is not None:
                observations.collector_evidence["mips"] = {
                    "status": "aggregate_only", "reason": "perf_window_counters",
                    "result": asdict(observations.mips_result) if is_dataclass(observations.mips_result) else None,
                }
            window_error = f"{type(primary_error).__name__}: {primary_error}" if primary_error is not None else ""
            observations.request_record = client_publication.append_request_window(
                requests_stream, sniff_group_id=group_id,
                input_scale=progress.effective_input_scale if progress.effective_input_scale is not None else spec.scale,
                warmup=spec.warmup, repeat_idx=spec.repeat_idx,
                latencies=progress.latencies,
                error="; ".join(part for part in (window_error, monitors.error) if part),
                failed_request_id=progress.pending_request_id,
            )
        except BaseException as exc:
            finalization_error = exc
            raise
        finally:
            if primary_error is not None and (
                not isinstance(primary_error, Exception)
                or isinstance(primary_error, (RequestTimeoutAbort, EnergyAbort, MIPSAbort))
            ):
                raise primary_error
            if finalization_error is not None and not isinstance(finalization_error, Exception):
                raise finalization_error
            monitors.raise_if_failed()

    def _attach_monitors(self, monitors: MonitorGroup):
        """Register collectors in their existing order before starting sampling.

        The caller keeps ownership and executes MonitorGroup.finish even when
        collector construction raises after earlier collectors registered.
        """
        gpu_monitor = None
        cpu_monitor = None
        if measurement_requested(self.config.profiling_mode, "gpu_power", gpu=self.use_energy) and self.dependencies.gpu is not None:
            gpu_monitor = self.dependencies.gpu(
                sample_hz=self.config.sample_hz,
                idle_seconds=self.config.idle_seconds,
                device_index=self.config.device_index,
                device_uuid=self.config.gpu_device_uuid,
                sampling_policy=self.config.sampling_policy, raw_sample_limit=self.config.raw_sample_limit,
            )
            monitors.add("gpu", gpu_monitor)

        if measurement_requested(self.config.profiling_mode, "cpu_energy") and self.dependencies.cpu is not None:
            try:
                cpu_monitor = self.dependencies.cpu(
                    sample_hz=self.config.sample_hz, idle_seconds=self.config.idle_seconds,
                    container_name=self.config.container_name, dram_energy=self.config.dram_energy,
                    sampling_policy=self.config.sampling_policy, raw_sample_limit=self.config.raw_sample_limit)
                monitors.add("cpu", cpu_monitor)
            except RuntimeError as exc:
                if self.config.dram_energy == "required":
                    raise EnergyAbort(str(exc)) from exc
                raise

        if self.dependencies.resource is not None:
            resource_usage_monitor = self.dependencies.resource(
                sample_hz=self.config.sample_hz,
                container_name=self.config.container_name,
                cpu_cores=to_float_or_nan(self.config.cpu_cores),
                mem_cap_gb=to_float_or_nan(self.config.mem_cap_gb),
                use_gpu=self.use_energy,
                device_index=self.config.device_index,
                device_uuid=self.config.gpu_device_uuid,
                sampling_policy=self.config.sampling_policy, raw_sample_limit=self.config.raw_sample_limit,
            )
            monitors.add("resource", resource_usage_monitor)

        if measurement_requested(self.config.profiling_mode, "cpu_instructions") and self.config.use_mips:
            if self.dependencies.instructions is None:
                raise MIPSAbort("required instruction collector factory is unavailable")
            mips_monitor = self.dependencies.instructions(self.config.container_name)
            monitors.add("mips", mips_monitor)
        return gpu_monitor, cpu_monitor

    def _start_monitors(self, monitors, gpu_monitor, cpu_monitor):
        """Measure control/idle first and start the request monitors afterwards.

        No request is executed here. The caller's `finally` retains ownership
        of every collector, including on startup/cancellation failure.
        """
        gpu_idle_measured_at = "nan"
        cpu_idle_measured_at = "nan"
        gpu_idle_trace = {}
        idle_trace = {}
        gpu_idle_debug_snapshot = None
        idle_debug_snapshot = None
        if gpu_monitor is not None or cpu_monitor is not None:
            if self.config.idle_cooldown_seconds > 0.0:
                time.sleep(self.config.idle_cooldown_seconds)
            run_matched_control_window(monitors, idle_seconds=self.config.idle_seconds,
                                       trace=self.config.idle_debug)
            measured_at = _now_iso()
            if gpu_monitor is not None:
                gpu_idle_measured_at = measured_at
                gpu_idle_trace = dict(
                    getattr(gpu_monitor, "idle_trace", {}) or {}
                )
            if cpu_monitor is not None:
                cpu_idle_measured_at = measured_at
                idle_trace = dict(
                    getattr(cpu_monitor, "idle_trace", {}) or {}
                )

            if self.config.idle_debug:
                if gpu_monitor is not None:
                    gpu_idle_debug_snapshot = client_diagnostics.collect_gpu_idle_debug_snapshot(self.config.device_index)
                if cpu_monitor is not None:
                    idle_debug_snapshot = client_diagnostics.collect_idle_debug_snapshot()

        monitors.start()

        return (gpu_idle_measured_at, cpu_idle_measured_at, gpu_idle_trace,
                idle_trace, gpu_idle_debug_snapshot, idle_debug_snapshot)

    def _is_mips_error(self, exc: Exception) -> bool:
        from acprof.monitors.perf_mips import MIPSProfilingError
        return isinstance(exc, (MIPSAbort, MIPSProfilingError))

    def _should_send_request(self,
        completed_requests: int,
        latency_sum_s: float,
        repeat_request_limit: int,
    ) -> bool:
        if self.config.repeat_in_window > 0:
            return completed_requests < repeat_request_limit
        if completed_requests <= 0:
            return True
        return latency_sum_s < self.config.repeat_window_seconds

    def _idle_diagnostic(self, row, group_id, observations, metrics):
        """Assemble diagnostic provenance only after all monitors have finalized."""
        # Workload stop may reject a cumulative GPU counter and recompute
        # the paired idle baseline from its original power samples. Read
        # the final provenance here, not the snapshot captured before the
        # workload monitor finalized its source selection.
        if observations.gpu_monitor is not None:
            observations.gpu_idle_trace = dict(getattr(observations.gpu_monitor, "idle_trace", {}) or {})
        if observations.idle_debug_snapshot is None:
            observations.idle_debug_snapshot = client_diagnostics.collect_idle_debug_snapshot()
        return {
            "case_name": self.config.case_name,
            "gpu_mode": self.config.gpu_mode,
            "cpu_cores": self.config.cpu_cores,
            "mem_cap_gb": self.config.mem_cap_gb,
            "input_scale": row["input_scale"],
            "warmup": row["warmup"],
            "repeat_idx": row["repeat_idx"],
            "repeat_in_window": row["repeat_in_window"],
            "sniff_group_id": group_id,
            "gpu_idle_measured_at": row["gpu_idle_measured_at"],
            "gpu_device_uuid": row["gpu_device_uuid"],
            "gpu_energy_source": row["gpu_energy_source"],
            "gpu_energy_counter_start_mj": getattr(observations.gpu_result, "counter_start_mj", None),
            "gpu_energy_counter_end_mj": getattr(observations.gpu_result, "counter_end_mj", None),
            "gpu_energy_duration_s": getattr(observations.gpu_result, "measurement_duration_s", None),
            "gpu_power_samples": [[t - observations.gpu_samples[0][0], p] for t, p in observations.gpu_samples],
            "gpu_idle_power_w": to_float_or_nan(row["gpu_idle_power_w"]),
            **metrics.gpu_idle_stats,
            **observations.gpu_idle_trace,
            **(observations.gpu_idle_debug_snapshot or {}),
            "cpu_idle_measured_at": row["cpu_idle_measured_at"],
            "cpu_idle_power_w": to_float_or_nan(row["cpu_idle_power_w"]),
            **metrics.cpu_idle_stats,
            **observations.idle_trace,
            **observations.idle_debug_snapshot,
        }
