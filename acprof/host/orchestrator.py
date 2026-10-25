"""AC-Prof 资源矩阵、单项测量及结果整合。"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import atomic_write, read_json_object, replace_file_durably
from acprof.capabilities import measurement_requested, require_profiling_mode
from acprof.config import (
    CLIENT_REQUEST_TIMEOUT_EXIT_CODE,
    CSV_FIELDS,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    SERVER_PORT,
)
from acprof.host import command as host_command
from acprof.host.container_lifecycle import ContainerCleanupError
from acprof.host.container_state import container_runtime_oom_error
from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import (
    cold_start_client_env,
    service_port,
    start_container_session,
    stop_container_session,
)
from acprof.host.gpu_device import normalize_gpu_mode
from acprof.host.hardware_conditions import record_case_conditions
from acprof.host.images.identity import model_token
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import (
    _format_scale_value,
    resolve_input_scales,
    serialize_input_scales,
)
from acprof.host.measurement.capture import (
    PacketLatencyRuntime,
    _packet_latency_error,
    _resolve_packet_latency_runtime,
)
from acprof.host.measurement.energy_validation import (
    EnergyProfilingError,
    check_case_cpu_idle_power_stable,
    check_case_gpu_idle_power_stable,
    row_has_error_status,
)
from acprof.host.profiling.compute_plan import NCU_ERROR_FIELD, TORCH_ERROR_FIELD
from acprof.installation import module_command
from acprof.monitors.perf_mips import MIPS_EXIT_CODE
from acprof.pixel_metrics import PIXEL_COUNT_FIELDS, PIXEL_RATE_SOURCES
from acprof.progress_events import emit_event, measurement_boundary

if TYPE_CHECKING:
    from acprof.host.docker_runtime import RunningContainer
    from acprof.host.run_state import RunState

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatrixProgress:
    """One safely completed resource case in a matrix sweep."""

    completed_cases: int
    total_cases: int
    cpu: int
    mem: int
    gpu: str
    result_csv: Optional[str]


class MIPSProfilingError(RuntimeError):
    """Raised when required MIPS profiling cannot continue reliably."""



def _assert_packet_latency_csv_complete(
    csv_path: str,
    *,
    ignore_error_rows: bool = False,
) -> None:
    import csv

    if not os.path.exists(csv_path):
        raise _packet_latency_error(f"result CSV does not exist: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise _packet_latency_error(f"result CSV is empty: {csv_path}")
        if "latency_s" not in reader.fieldnames:
            raise _packet_latency_error(f"latency_s column is missing in {csv_path}")

        total = 0
        missing = 0
        for row in reader:
            if ignore_error_rows and row_has_error_status(row):
                continue
            total += 1
            raw_value = (row.get("latency_s") or "").strip()
            try:
                value = float(raw_value)
            except Exception:
                missing += 1
                continue

            if not math.isfinite(value) or value <= 0:
                missing += 1

    if total == 0 and not ignore_error_rows:
        raise _packet_latency_error(f"result CSV has no rows: {csv_path}")

    if missing:
        raise _packet_latency_error(
            f"latency_s is missing for {missing}/{total} row(s) in {csv_path}"
        )


@dataclass(frozen=True)
class CaseRequest:
    """Identity, resources, and locations for one container execution."""

    task_info: TaskInfo
    cpu: int
    mem: int
    gpu: str
    image_info: ImageInfo
    output_dir: str
    project_dir: str
    cpuset_cpus: str = ""


@dataclass(frozen=True)
class CaseWorkload:
    """Request counts and deadlines, independent of host collector settings."""

    batch_size: int = 1
    warmup: int = 2
    repeat: int = 5
    repeat_in_window: int = DEFAULT_REPEAT_IN_WINDOW
    repeat_window_seconds: float = DEFAULT_REPEAT_WINDOW_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS


@dataclass(frozen=True)
class CaseMeasurement:
    """Collector policy for a case; no mutable execution state."""

    sample_hz: float = 20.0
    idle_seconds: float = DEFAULT_IDLE_SECONDS
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS
    idle_debug: bool = False
    sniff_iface: str = "docker0"
    require_packet_latency: bool = True
    profiling_mode: str = "full"
    dram_energy: str = "auto"
    sampling_policy: str = "burst"
    raw_sample_limit: int = 100000


@dataclass(frozen=True)
class CasePlans:
    """Previously prepared plans consumed by the measured client."""

    input_scales: Optional[str] = None
    input_scale_plan_file: Optional[str] = None
    compute_profile_plan_file: Optional[str] = None
    execution_profile_plan_file: Optional[str] = None
    input_scale_order: Optional[List[float]] = None


@dataclass(frozen=True)
class _ClientOutcome:
    incomplete: bool = False
    completed_rows: int = 0
    reason: str = ""


def run_single_case(
    request: CaseRequest,
    workload: CaseWorkload = CaseWorkload(),
    measurement: CaseMeasurement = CaseMeasurement(),
    plans: CasePlans = CasePlans(),
    *,
    run_state: RunState | None = None,
) -> str:
    """Validate a case request and execute its owned container lifecycle."""
    from acprof.cpu_affinity import normalize_cpu_set
    from acprof.monitors.sampling import sampling_policy as require_sampling_policy

    request = replace(request, cpuset_cpus=normalize_cpu_set(request.cpuset_cpus))
    measurement = replace(
        measurement, sampling_policy=require_sampling_policy(measurement.sampling_policy)
    )
    if type(measurement.raw_sample_limit) is not int or measurement.raw_sample_limit < 2:
        raise ValueError("raw_sample_limit must be an integer of at least 2")
    require_profiling_mode(measurement.profiling_mode)
    timeout = float(workload.request_timeout_seconds)
    if timeout <= 0.0 or not math.isfinite(timeout):
        raise ValueError("request_timeout_seconds must be a finite value > 0")
    workload = replace(workload, request_timeout_seconds=timeout)
    return _CaseExecution(request, workload, measurement, plans, run_state).run()


class _CaseExecution:
    """Own the container and capture process across a single case attempt."""

    def __init__(
        self,
        request: CaseRequest,
        workload: CaseWorkload,
        measurement: CaseMeasurement,
        plans: CasePlans,
        run_state: RunState | None,
    ) -> None:
        self.request = request
        self.workload = workload
        self.measurement = measurement
        self.plans = plans
        self.run_state = run_state
        self.case = ArtifactLayout.discover(request.output_dir).case(
            request.task_info.model_id, request.cpu, request.mem, request.gpu
        )
        self.out_csv = str(self.case.csv)
        self.case_name = (
            f"case_{model_token(request.task_info.model_id)}_"
            f"{request.cpu}c_{request.mem}g_{request.gpu}"
        )
        self.require_packet_latency = measurement.require_packet_latency and measurement_requested(
            measurement.profiling_mode, "packet_latency"
        )
        self.capture: subprocess.Popen[bytes] | None = None
        self.status = "error"
        self.run_id = ""
        self.attempt_id = ""

    def run(self) -> str:
        self.case.csv.parent.mkdir(parents=True, exist_ok=True)
        if self.run_state is not None:
            checkpoint = self.run_state.processing_checkpoint(
                self.out_csv, self.request.cpu, self.request.mem, self.request.gpu
            )
            if checkpoint is not None:
                return _resume_case_processing(self.case, checkpoint, self.run_state, self.case_name)
        self.run_id = self.run_state.data["run_id"] if self.run_state is not None else uuid4().hex
        self.attempt_id = (
            self.run_state.case_attempt_id(self.out_csv)
            if self.run_state is not None else uuid4().hex
        )
        print(f"\n{'='*60}")
        print(f"[case] {self.case_name}")
        print(f"  CPU={self.request.cpu}, MEM={self.request.mem}GB, GPU={self.request.gpu}")
        print(f"  Port: {service_port(self.request.cpu, self.request.mem)}")
        print(f"{'='*60}")
        try:
            os.remove(self.case.sidecar("client_error"))
        except FileNotFoundError:
            pass
        emit_event("case_started", self.case_name)
        session = self._start_container()
        if session is None:
            return self.out_csv
        try:
            if self.run_state is not None:
                self.run_state.bind_case_container(self.out_csv, session.container_id)
            record_case_conditions(
                self.request.output_dir, self.case.case_id, session,
                cpuset_cpus=self.request.cpuset_cpus,
            )
            sniff_runtime = self._start_capture()
            outcome = self._run_client(session)
            capture_available = self.capture is not None
            if self.capture is not None:
                time.sleep(1.0)
                _stop_capture(self.capture)
                time.sleep(0.2)
                self.capture = None
            self._checkpoint_measurement(session, outcome, capture_available)
            self._finalize(outcome, capture_available, sniff_runtime)
            self.status = "error" if outcome.incomplete else _case_result_status(self.out_csv)
        except KeyboardInterrupt:
            _LOG.debug("case cancelled: %s", self.case_name)
            self.status = "cancelled"
            raise
        finally:
            self._cleanup(session, sys.exc_info()[1])
        print(f"[case] Done. Output: {self.out_csv}")
        return self.out_csv

    def _start_container(self) -> RunningContainer | None:
        try:
            return start_container_session(
                task_info=self.request.task_info,
                cpu=self.request.cpu, mem=self.request.mem, gpu=self.request.gpu,
                image_info=self.request.image_info, container_name=self.case_name,
                log_prefix="[case]", request_timeout_seconds=self.workload.request_timeout_seconds,
                cpuset_cpus=self.request.cpuset_cpus,
            )
        except ContainerCleanupError as exc:
            from acprof.artifacts import atomic_write_json

            atomic_write_json(self.case.sidecar("cleanup_error"), exc.to_dict())
            emit_event("case_finished", self.case_name, status="error")
            raise
        except RuntimeError as exc:
            error = f"container_start_failed: {exc}"
            print(f"[case] {error}", file=sys.stderr)
            self._write_error(error)
            emit_event("case_finished", self.case_name, status="error")
            return None
        except BaseException as exc:
            emit_event(
                "case_finished", self.case_name,
                status="cancelled" if isinstance(exc, KeyboardInterrupt) else "error",
            )
            raise

    def _start_capture(self) -> PacketLatencyRuntime | None:
        runtime = _resolve_packet_latency_runtime(
            project_dir=self.request.project_dir, pcap_file=str(self.case.pcap),
            sniff_iface=self.measurement.sniff_iface,
        ) if measurement_requested(self.measurement.profiling_mode, "packet_latency") else None
        if runtime is not None:
            print(f"[sniff] Starting tcpdump on {self.measurement.sniff_iface} via {runtime.mode}")
            try:
                self.capture = subprocess.Popen(
                    runtime.tcpdump_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                raise _packet_latency_error("failed to start tcpdump", repr(exc)) from exc
            time.sleep(0.3)
            if self.require_packet_latency and self.capture.poll() is not None:
                raise _packet_latency_error(
                    "tcpdump exited before workload started",
                    f"command={' '.join(runtime.tcpdump_cmd)}",
                )
        elif self.require_packet_latency:
            raise _packet_latency_error("tcpdump/tshark runtime could not be resolved")
        elif measurement_requested(self.measurement.profiling_mode, "packet_latency"):
            print("[sniff] tcpdump/tshark unavailable, skipping packet-level latency")
        else:
            print(f"[sniff] packet_latency=not_requested (profiling_mode={self.measurement.profiling_mode})")
        return runtime

    def _client_environment(self, session: RunningContainer) -> Dict[str, str]:
        request, workload, measurement, plans = self.request, self.workload, self.measurement, self.plans
        task_info = request.task_info
        scales = plans.input_scales or serialize_input_scales(
            resolve_input_scales(task_info.task_family, input_scales=None)
        )
        environment = {
            **os.environ,
            "MODEL_ID": task_info.model_id,
            "MODEL_REVISION": task_info.model_revision,
            "ACPROF_RUNTIME_PROFILE": task_info.runtime_profile_id,
            "TASK_FAMILY": task_info.task_family,
            "PIPELINE_TAG": task_info.pipeline_tag,
            "RUNTIME_BACKEND": task_info.runtime_backend,
            "IMAGE_TAG": request.image_info.tag,
            "CPU_CORES": str(request.cpu), "MEM_CAP_GB": str(request.mem), "GPU_MODE": request.gpu,
            "BASE_URL": session.base_url, "ENDPOINT": "/predict",
            "BATCH_SIZE": str(workload.batch_size), "WARMUP": str(workload.warmup),
            "REPEAT": str(workload.repeat), "REPEAT_IN_WINDOW": str(workload.repeat_in_window),
            "REPEAT_WINDOW_SECONDS": str(workload.repeat_window_seconds),
            "REQUEST_TIMEOUT_SECONDS": f"{workload.request_timeout_seconds:g}",
            **cold_start_client_env(session),
            "OUT_CSV": self.out_csv, "CASE_NAME": self.case_name, "CONTAINER_NAME": session.name,
            "USE_MIPS": "1" if measurement_requested(measurement.profiling_mode, "cpu_instructions") else "0",
            "PROFILING_MODE": measurement.profiling_mode, "DRAM_ENERGY": measurement.dram_energy,
            "INPUT_SCALE_ORDER": json.dumps(plans.input_scale_order) if plans.input_scale_order is not None else "",
            "SAMPLE_HZ": str(measurement.sample_hz),
            "ACPROF_SAMPLING_POLICY": measurement.sampling_policy,
            "ACPROF_RAW_SAMPLE_LIMIT": str(measurement.raw_sample_limit),
            "ACPROF_RUN_ID": self.run_id, "ACPROF_CASE_ID": self.case.case_id,
            "ACPROF_CASE_ATTEMPT_ID": self.attempt_id,
            "ACPROF_WINDOW_RECORDS_DIR": str(self.case.window_records(self.attempt_id)),
            "ACPROF_TELEMETRY_DIR": str(self.case.telemetry(self.attempt_id)),
            "IDLE_SECONDS": str(measurement.idle_seconds),
            "IDLE_COOLDOWN_SECONDS": str(measurement.idle_cooldown_seconds),
            "IDLE_DEBUG": "1" if measurement.idle_debug else "0",
            "IDLE_DIAG_PATH": str(self.case.idle) if measurement.idle_debug else "",
            "CLIENT_ERROR_PATH": str(self.case.sidecar("client_error")),
            "DEVICE_INDEX": str(session.gpu_device.get("index", 0)),
            "GPU_DEVICE_UUID": str(session.gpu_device.get("uuid", "")),
            "INPUT_SCALES": scales,
            "INPUT_SCALE_PLAN_FILE": plans.input_scale_plan_file or "",
            "COMPUTE_PROFILE_PLAN_FILE": plans.compute_profile_plan_file or "",
            "EXECUTION_PROFILE_PLAN_FILE": plans.execution_profile_plan_file or "",
        }
        # Keep notification credentials out of the measured subprocess.
        environment.pop("ACPROF_WECOM_WEBHOOK_URL", None)
        return environment

    def _run_client(self, session: RunningContainer) -> _ClientOutcome:
        print("[case] Running workload...")
        environment = self._client_environment(session)
        with measurement_boundary(self.case_name):
            result = host_command.run_command(
                module_command('acprof.host.client'), check=False,
                capture_output=False, env=environment,
            )
        outcome = self._client_failure(session, result.returncode) if result.returncode else _ClientOutcome()
        ready_failure = _client_ready_failure(self.out_csv)
        if ready_failure is not None:
            print(f"[case][WARN] {ready_failure}", file=sys.stderr)
            completed, _ = self._write_error(ready_failure, preserve_existing=False)
            outcome = _ClientOutcome(True, completed, "client readiness failed")
        return outcome

    def _client_failure(self, session: RunningContainer, returncode: int) -> _ClientOutcome:
        oom = container_runtime_oom_error(session.name, self.request.mem, returncode)
        if oom is not None:
            print(f"[case][WARN] {oom}", file=sys.stderr)
            completed, _ = self._write_error(
                oom, preserve_existing=True, annotate_existing_error_rows=True,
            )
            return _ClientOutcome(True, completed, "runtime OOM")
        if returncode == CLIENT_REQUEST_TIMEOUT_EXIT_CODE:
            context = _load_client_error_context(str(self.case.sidecar("client_error")))
            context.setdefault("request_timeout_s", self.workload.request_timeout_seconds)
            timeout = _timeout_context_float(context.get("request_timeout_s"), self.workload.request_timeout_seconds)
            error = (
                "client_request_timeout: connect/read inactivity exceeded "
                f"{timeout:g}s; incomplete and later rows will be marked individually"
            )
            print(f"[case][WARN] {error}", file=sys.stderr)
            completed, _ = self._write_error(error, preserve_existing=True, timeout_context=context)
            return _ClientOutcome(True, completed, "request timeout")
        from acprof.artifact_layout import case_sidecar
        from acprof.failures import Failure, RuntimeFailure, read_runtime_failures

        failure_path = case_sidecar(self.out_csv, "runtime_failures")
        if failure_path.is_file():
            failures = read_runtime_failures(failure_path)
            if failures:
                raise RuntimeFailure(Failure(**failures[-1]))
        if returncode == MIPS_EXIT_CODE:
            raise MIPSProfilingError(
                "client.py exited because MIPS profiling failed; review the "
                "[mips][ERROR] output above for the perf remediation steps."
            )
        raise EnergyProfilingError(
            f"client.py exited with code {returncode}; aborting profiling matrix. "
            "Review the client output above for the energy stability diagnostic."
        )

    def _write_error(
        self, error: str, *, preserve_existing: bool = False,
        annotate_existing_error_rows: bool = False,
        timeout_context: Dict[str, Any] | None = None,
    ) -> Tuple[int, int]:
        return _write_case_error_csv(
            task_info=self.request.task_info, out_csv=self.out_csv,
            cpu=self.request.cpu, mem=self.request.mem, gpu=self.request.gpu,
            warmup=self.workload.warmup, repeat=self.workload.repeat,
            repeat_in_window=self.workload.repeat_in_window,
            input_scales=self.plans.input_scales, error=error,
            preserve_existing=preserve_existing,
            annotate_existing_error_rows=annotate_existing_error_rows,
            timeout_context=timeout_context,
        )

    def _checkpoint_measurement(
        self, session: RunningContainer, outcome: _ClientOutcome, capture_available: bool,
    ) -> None:
        if self.run_state is None:
            return
        self.run_state.checkpoint_measurement(
            self.out_csv, self.request.cpu, self.request.mem, self.request.gpu,
            container_id=session.container_id,
            processing={
                "capture_available": capture_available, "case_incomplete": outcome.incomplete,
                "completed_rows_before_failure": outcome.completed_rows,
                "incomplete_case_reason": outcome.reason,
                "require_packet_latency": self.require_packet_latency,
                "profiling_mode": self.measurement.profiling_mode,
                "gpu": self.request.gpu, "server_port": SERVER_PORT,
            },
        )
        self.run_state.record_case_processing(self.out_csv, "running")

    def _finalize(
        self, outcome: _ClientOutcome, capture_available: bool,
        sniff_runtime: PacketLatencyRuntime | None,
    ) -> None:
        try:
            _finalize_case(
                capture_available, sniff_runtime, outcome.incomplete, outcome.completed_rows,
                outcome.reason, str(self.case.pcap), str(self.case.latency), self.out_csv,
                self.require_packet_latency, self.measurement.profiling_mode, self.request.gpu,
            )
        except BaseException:
            if self.run_state is not None:
                self.run_state.record_case_processing(self.out_csv, "failed")
            raise

    def _cleanup(self, session: RunningContainer, run_error: BaseException | None) -> None:
        try:
            try:
                if self.capture is not None and self.capture.poll() is None:
                    _stop_capture(self.capture)
            finally:
                stop_container_session(session, log_prefix="[case]")
                if self.run_state is not None:
                    self.run_state.record_case_cleanup(self.out_csv, "complete")
        except BaseException as exc:
            _LOG.debug("case cleanup failed: case=%s error_type=%s", self.case_name, type(exc).__name__)
            self.status = "error"
            if self.run_state is not None:
                self.run_state.record_case_cleanup(self.out_csv, "failed")
            if isinstance(exc, ContainerCleanupError):
                from acprof.artifacts import atomic_write_json

                exc.run_error = run_error or exc.run_error
                atomic_write_json(self.case.sidecar("cleanup_error"), exc.to_dict())
            raise
        finally:
            emit_event("case_finished", self.case_name, status=self.status)

def _resume_case_processing(case, checkpoint: dict, run_state, case_name: str) -> str:
    """Retry only derivation from a verified whole-case measurement snapshot."""
    from acprof.host.measurement.checkpoint import restore_measurement_projection

    context = checkpoint["processing"]
    out_csv = str(case.csv)
    status = "error"
    emit_event("case_started", case_name, phase="postprocessing")
    try:
        run_state.record_case_processing(out_csv, "running")
        restore_measurement_projection(case, checkpoint)
        pcap = (str(case.layout.contained(checkpoint["files"]["pcap"]["path"]))
                if "pcap" in checkpoint["files"] else str(case.pcap))
        # Commands come from trusted code. The checkpoint supplies validated data only.
        runtime = PacketLatencyRuntime(
            mode="local", tcpdump_cmd=[],
            parse_cmd=[*module_command("acprof.packet.sniff_parse_pcap"), pcap, str(context["server_port"])],
        ) if context["capture_available"] else None
        print(f"[resume] Retrying postprocessing from completed measurement: {case.case_id}")
        _finalize_case(context["capture_available"], runtime, context["case_incomplete"], context["completed_rows_before_failure"],
                       context["incomplete_case_reason"], pcap, str(case.latency), out_csv,
                       context["require_packet_latency"], context["profiling_mode"], context["gpu"])
        status = "error" if context["case_incomplete"] else _case_result_status(out_csv)
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "error"
        run_state.record_case_processing(out_csv, "failed")
        raise
    finally:
        emit_event("case_finished", case_name, status=status)
    return out_csv


def _client_ready_failure(csv_path: str) -> str | None:
    """Recognize the pre-window /ready sentinel before packet post-processing.

    The sentinel has no planned measurement key. The orchestrator replaces it
    with the full error matrix so resume and result auditing stay well formed.
    """
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            first = next(reader, None)
            second = next(reader, None)
    except (OSError, UnicodeError, csv.Error):
        return None
    if first is None or second is not None or first.get("status") != "error":
        return None
    error = str(first.get("error") or "")
    return error if error.startswith("ready_failed:") else None


def _case_result_status(csv_path: str) -> str:
    """A zero client exit code does not establish successful measurement rows."""
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as stream:
            statuses = [row.get("status") for row in csv.DictReader(stream)]
    except (OSError, UnicodeError, csv.Error):
        return "error"
    return "ok" if statuses and all(status in {"ok", "warn"} for status in statuses) else "error"


def _stop_capture(process) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _finalize_case(capture_available: bool, sniff_runtime, case_incomplete, completed_rows_before_failure, incomplete_case_reason, pcap_file, lat_json, out_csv, require_packet_latency, profiling_mode, gpu) -> None:
    """Derive and validate results after acquisition is durably checkpointed."""
    if capture_available:
        if case_incomplete and completed_rows_before_failure == 0:
            print(
                "[sniff] No completed measurements before "
                f"{incomplete_case_reason}; skipping packet-latency merge"
            )
        elif not os.path.exists(pcap_file) or os.path.getsize(pcap_file) <= 0:
            if require_packet_latency:
                raise _packet_latency_error(
                    f"pcap file is missing or empty: {pcap_file}"
                )
        else:
            print("[sniff] Parsing pcap -> packet latencies...")
            if sniff_runtime is None:
                raise _packet_latency_error("packet capture parser was not initialized")
            parse_result = host_command.run_command(sniff_runtime.parse_cmd, check=False)
            parse_output = parse_result.stdout.strip()
            if parse_result.returncode != 0:
                raise _packet_latency_error(
                    "pcap parsing failed",
                    (parse_result.stderr or parse_result.stdout or "").strip(),
                )
            if not parse_output:
                raise _packet_latency_error("pcap parser produced no output")
            try:
                latency_payload = json.loads(parse_output)
            except json.JSONDecodeError as exc:
                raise _packet_latency_error(
                    "pcap parser produced invalid JSON",
                    repr(exc),
                ) from exc
            latency_records = (
                latency_payload.get("requests")
                if isinstance(latency_payload, dict)
                and "requests" in latency_payload
                else latency_payload
            )
            if not isinstance(latency_records, dict) or not latency_records:
                raise _packet_latency_error(
                    "pcap parser did not find matching request latency records"
                )
            atomic_write(
                lat_json,
                lambda stream: json.dump(
                    latency_payload,
                    stream,
                    ensure_ascii=True,
                    indent=2,
                ),
            )

            if not os.path.exists(out_csv):
                raise _packet_latency_error(
                    f"client did not produce result CSV: {out_csv}"
                )

            print("[sniff] Merging packet latency into CSV...")
            merged_csv = out_csv + ".merged"
            merge_result = host_command.run_command([*module_command('acprof.packet.merge_packet_latency'), out_csv, lat_json, merged_csv], check=False)
            if merge_result.returncode != 0:
                raise _packet_latency_error(
                    "packet latency merge failed",
                    (merge_result.stderr or merge_result.stdout or "").strip(),
                )
            if not os.path.exists(merged_csv):
                raise _packet_latency_error(
                    f"packet latency merge did not produce {merged_csv}"
                )
            replace_file_durably(merged_csv, out_csv)
            if require_packet_latency:
                _assert_packet_latency_csv_complete(
                    out_csv,
                    ignore_error_rows=case_incomplete,
                )

    if measurement_requested(profiling_mode, "cpu_energy") and (not case_incomplete or completed_rows_before_failure > 0):
        check_case_cpu_idle_power_stable(
            out_csv,
            ignore_error_rows=case_incomplete,
        )
        if normalize_gpu_mode(gpu) == "on":
            check_case_gpu_idle_power_stable(
                out_csv,
                ignore_error_rows=case_incomplete,
            )

def _write_case_error_csv(
    *,
    task_info: TaskInfo,
    out_csv: str,
    cpu: int,
    mem: int,
    gpu: str,
    warmup: int,
    repeat: int,
    repeat_in_window: int,
    input_scales: Optional[str],
    error: str,
    preserve_existing: bool = False,
    annotate_existing_error_rows: bool = False,
    timeout_context: Optional[Dict[str, Any]] = None,
    result_origin: str = "formal_attempt",
) -> Tuple[int, int]:
    """Write missing error rows and return successful-preserved/added counts."""
    if not str(error or "").strip():
        raise ValueError("error rows require a non-empty diagnostic")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    scales = resolve_input_scales(task_info.task_family, input_scales)

    from acprof.platform import detect_environment
    environment_class = detect_environment().environment

    def make_row(scale: float, repeat_idx: int, is_warmup: bool) -> Dict[str, Any]:
        row = {field: "nan" for field in CSV_FIELDS}
        row.update({
            "environment_class": environment_class,
            "cpu_cores": str(cpu),
            "mem_cap_gb": str(mem),
            "gpu_mode": gpu,
            "input_scale": _format_scale_value(scale),
            "task_param": "",
            "repeat_idx": str(repeat_idx),
            "warmup": "1" if is_warmup else "0",
            "repeat_in_window": str(repeat_in_window),
            TORCH_ERROR_FIELD: "not_run",
            "status": "error",
            "error": error,
            "result_origin": result_origin,
        })
        if gpu == "on":
            row[NCU_ERROR_FIELD] = "not_run"
            row["compute_profile_error_nsys"] = "not_run"
        else:
            row["compute_profile_error_massif"] = "not_run"
        return row

    planned_rows: List[Dict[str, Any]] = []
    for scale in scales:
        for idx in range(warmup):
            planned_rows.append(make_row(scale, idx, True))
        for idx in range(repeat):
            planned_rows.append(make_row(scale, idx, False))

    def measurement_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
        try:
            scale = _format_scale_value(float(row.get("input_scale", "nan")))
            repeat_idx = int(row.get("repeat_idx", ""))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid measurement key in partial case CSV: {row!r}"
            ) from exc
        warmup_flag = "1" if str(row.get("warmup") or "0").strip() == "1" else "0"
        return scale, warmup_flag, repeat_idx

    planned_by_key = {
        measurement_key(row): row
        for row in planned_rows
    }
    existing_rows: List[Dict[str, Any]] = []
    existing_keys = set()
    if preserve_existing and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            optional_pixel_fields = {*PIXEL_COUNT_FIELDS, *PIXEL_RATE_SOURCES}
            missing_fields = set(CSV_FIELDS) - set(reader.fieldnames or []) - optional_pixel_fields
            if missing_fields:
                raise RuntimeError(
                    "partial case CSV is missing required fields: "
                    + ", ".join(sorted(missing_fields))
                )
            for row in reader:
                key = measurement_key(row)
                if key not in planned_by_key:
                    raise RuntimeError(
                        f"partial case CSV contains an unplanned measurement: {key!r}"
                    )
                if key in existing_keys:
                    raise RuntimeError(
                        f"partial case CSV contains a duplicate measurement: {key!r}"
                    )
                existing_keys.add(key)
                normalized_row = {
                    field: row.get(field, "nan")
                    for field in CSV_FIELDS
                }
                if (
                    annotate_existing_error_rows
                    and row_has_error_status(normalized_row)
                ):
                    existing_error = str(normalized_row.get("error") or "").strip()
                    if error not in existing_error:
                        normalized_row["error"] = (
                            f"{existing_error}; {error}"
                            if existing_error
                            else error
                        )
                existing_rows.append(normalized_row)

    missing_rows = [
        row
        for row in planned_rows
        if measurement_key(row) not in existing_keys
    ]
    if timeout_context is not None and missing_rows:
        _annotate_timeout_placeholder_rows(
            missing_rows,
            measurement_key=measurement_key,
            timeout_context=timeout_context,
        )
    rows = [*existing_rows, *missing_rows]

    def write_error_rows(stream) -> None:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if preserve_existing:
        from acprof.artifact_layout import case_sidecar
        from acprof.host.measurement.publication import reconcile_sniff_group_sidecar

        sidecar = case_sidecar(out_csv, "sniff_groups")
        if sidecar.exists():
            reconcile_sniff_group_sidecar(out_csv, sidecar)
            if missing_rows:
                def write_error_groups(stream) -> None:
                    with sidecar.open("r", encoding="utf-8", newline="") as source:
                        stream.writelines(source)
                    for _ in missing_rows:
                        stream.write('{"sniff_group_id": ""}\n')

                # As in the client, publish groups before their CSV rows. A failed
                # CSV replacement leaves a recoverable, uncommitted sidecar tail.
                atomic_write(sidecar, write_error_groups)

    if not preserve_existing:
        # A client /ready sentinel may have published one sniff-group record.
        # Replacing its CSV with planned errors must replace the parallel
        # sidecar too, or a later resume sees fewer groups than committed rows.
        from acprof.artifact_layout import case_sidecar
        sidecar = case_sidecar(out_csv, "sniff_groups")
        if sidecar.exists():
            atomic_write(
                sidecar,
                lambda stream: stream.writelines(
                    '{"sniff_group_id": ""}\n' for _ in rows
                ),
            )

    atomic_write(out_csv, write_error_rows)

    preserved_success_rows = sum(
        not row_has_error_status(row)
        for row in existing_rows
    )
    if preserve_existing:
        print(
            f"[case] Preserved existing rows: {len(existing_rows)} "
            f"(successful: {preserved_success_rows}); "
            f"wrote error rows: {len(missing_rows)}; total: {len(rows)}"
        )
    else:
        print(f"[case] Wrote error rows: {out_csv} ({len(rows)} rows)")
    return preserved_success_rows, len(missing_rows)


def _timeout_context_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else float(fallback)


def _load_client_error_context(path: str) -> Dict[str, Any]:
    try:
        return read_json_object(path, label="client error context")
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(
            f"[case][WARN] could not read structured client error context {path}: {exc}",
            file=sys.stderr,
        )
        return {}


def _annotate_timeout_placeholder_rows(
    rows: List[Dict[str, Any]],
    *,
    measurement_key,
    timeout_context: Dict[str, Any],
) -> None:
    first_missing_scale = measurement_key(rows[0])[0]
    try:
        triggering_scale = _format_scale_value(
            float(timeout_context.get("input_scale", first_missing_scale))
        )
    except (TypeError, ValueError):
        triggering_scale = first_missing_scale

    timeout_s = _timeout_context_float(
        timeout_context.get("request_timeout_s"),
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    request_id = str(timeout_context.get("request_id") or "unknown")
    request_phase = str(timeout_context.get("request_phase") or "unknown")
    repeat_idx = timeout_context.get("measurement_repeat_idx")
    attempted_key = None
    if request_phase == "measurement_warmup" and repeat_idx is not None:
        attempted_key = (triggering_scale, "1", int(repeat_idx))
    elif request_phase == "measurement_repeat" and repeat_idx is not None:
        attempted_key = (triggering_scale, "0", int(repeat_idx))

    context_source = (
        "client_timeout_sidecar"
        if timeout_context.get("input_scale") is not None
        else "inferred_from_first_missing_scale"
    )
    trigger_fields = (
        f"trigger_error=client_request_timeout; triggering_input_scale={triggering_scale}; "
        f"request_timeout_s={timeout_s:g}; triggering_request_latency_s>{timeout_s:g}; "
        f"trigger_phase={request_phase}; trigger_request_id={request_id}; "
        f"context_source={context_source}"
    )

    for row in rows:
        key = measurement_key(row)
        planned_scale = key[0]
        if attempted_key is not None and key == attempted_key:
            row["error"] = (
                "client_request_timeout: planned_request_attempted=true; "
                "measurement_row_completed=false; "
                f"planned_input_scale={planned_scale}; {trigger_fields}"
            )
            continue

        reason = (
            "triggering_scale_probe_timed_out"
            if planned_scale == triggering_scale
            else "skipped_after_prior_scale_timeout"
        )
        row["error"] = (
            "not_measured_after_timeout: planned_request_attempted=false; "
            "measurement_row_completed=false; "
            f"planned_input_scale={planned_scale}; reason={reason}; {trigger_fields}"
        )


def _write_startup_oom_pruned_case_csv(
    *,
    task_info: TaskInfo,
    output_dir: str,
    cpu: int,
    mem: int,
    gpu: str,
    reference_cpu: int,
    warmup: int,
    repeat: int,
    repeat_in_window: int,
    input_scales: Optional[str],
) -> str:
    case = ArtifactLayout.discover(output_dir).case(task_info.model_id, cpu, mem, gpu)
    case.csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv = str(case.csv)
    error = (
        "not_measured_after_startup_oom_pruning: "
        "planned_request_attempted=false; measurement_row_completed=false; "
        "reason=confirmed_startup_oom_at_reference_cpu; "
        f"reference_cpu_cores={reference_cpu}; reference_mem_cap_gb={mem}; "
        f"gpu_mode={normalize_gpu_mode(gpu)}; "
        "pruning_scope=container_startup_only; "
        "result_origin=inferred_not_measured"
    )
    _write_case_error_csv(
        task_info=task_info,
        out_csv=out_csv,
        cpu=cpu,
        mem=mem,
        gpu=gpu,
        warmup=warmup,
        repeat=repeat,
        repeat_in_window=repeat_in_window,
        input_scales=input_scales,
        error=error,
        result_origin="inferred_not_measured",
    )
    return out_csv


def run_matrix(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    output_dir: str,
    project_dir: str,
    batch_size: int = 1,
    warmup: int = 2,
    repeat: int = 5,
    repeat_in_window: int = DEFAULT_REPEAT_IN_WINDOW,
    repeat_window_seconds: float = DEFAULT_REPEAT_WINDOW_SECONDS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    sample_hz: float = 20.0,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS,
    idle_debug: bool = False,
    sniff_iface: str = "docker0",
    input_scales: Optional[str] = None,
    input_scale_plan_file: Optional[str] = None,
    compute_profile_plan_file: Optional[str] = None,
    execution_profile_plan_file: Optional[str] = None,
    progress_callback: Optional[Callable[[MatrixProgress], None]] = None,
    prune_startup_oom: bool = False,
    run_state=None,
    profiling_mode: str = "full",
    matrix_order: str = "seeded",
    matrix_seed: int = 0,
    dram_energy: str = "auto",
    cpuset_cpus: str = "",
    sampling_policy: str = "burst",
    raw_sample_limit: int = 100000,
) -> List[str]:
    """Run a frozen resource plan after independent readiness-only probes."""
    from acprof.cpu_affinity import normalize_cpu_set
    from acprof.host.matrix_plan import (
        MATRIX_PLAN_NAME,
        freeze_matrix_plan,
        load_matrix_plan,
        matrix_identity,
    )
    from acprof.host.startup_probe import PROBE_NAME, run_startup_probes, startup_oom_prefixes

    cpuset_cpus = normalize_cpu_set(cpuset_cpus)
    request_timeout_seconds = float(request_timeout_seconds)
    require_profiling_mode(profiling_mode)
    if request_timeout_seconds <= 0 or not math.isfinite(request_timeout_seconds):
        raise ValueError("request_timeout_seconds must be a finite value > 0")
    os.makedirs(output_dir, exist_ok=True)
    scales = resolve_input_scales(task_info.task_family, input_scales)
    identity = matrix_identity(task_info, image_info, cpu_list, mem_list, gpu_list, scales,
                               order=matrix_order, seed=matrix_seed, prune=prune_startup_oom,
                               input_plan_file=input_scale_plan_file, cpuset_cpus=cpuset_cpus)
    layout = ArtifactLayout.discover(output_dir)
    plan_path = layout.path(MATRIX_PLAN_NAME)
    if plan_path.exists():
        plan = load_matrix_plan(plan_path, identity)
        if prune_startup_oom:
            evidence = read_json_object(
                layout.path(PROBE_NAME), label="startup probe report"
            )
            if (evidence.get("schema_version") != 2 or evidence.get("status") != "complete"
                    or evidence.get("identity") != identity
                    or startup_oom_prefixes(evidence) != plan["startup_oom_prefixes"]):
                raise ValueError("matrix plan startup probe evidence mismatch")
    else:
        if run_state is not None and run_state.data.get("cases"):
            raise ValueError("missing frozen matrix plan for existing cases; start a new experiment")
        prefixes = {}
        if prune_startup_oom:
            evidence = run_startup_probes(output_dir, identity, task_info, image_info,
                                          request_timeout_seconds=request_timeout_seconds)
            prefixes = startup_oom_prefixes(evidence)
        plan = freeze_matrix_plan(plan_path, identity, prefixes)
    if run_state is not None:
        run_state.bind_matrix_plan(plan)
    print(f"[matrix] Frozen plan: {plan_path}; order={matrix_order}, seed={matrix_seed}, hash={plan['plan_sha256']}")
    result_csvs = []
    total = len(plan["cases"])
    for current, case in enumerate(plan["cases"], 1):
        cpu, mem, gpu = case["cpu_cores"], case["mem_cap_gb"], case["gpu_mode"]
        case_name = f"case_{model_token(task_info.model_id)}_{cpu}c_{mem}g_{gpu}"
        emit_event("case_selected", case_name, current=current, total=total, cpu=cpu, mem=mem, gpu=gpu)
        print(f"[matrix] Case {current}/{total}: CPU={cpu}, MEM={mem}GB, GPU={gpu}")
        filename = f"result_case_{model_token(task_info.model_id)}_{cpu}c_{mem}g_{gpu}.csv"
        cached_case = run_state.prepare_case(filename, cpu, mem, gpu) if run_state else None
        if cached_case:
            csv_path = cached_case
            print(f"[resume] 已完成，复用 case：{filename}")
            emit_event("case_skipped", case_name, status="completed")
        elif case["result_origin"] == "inferred_not_measured":
            csv_path = _write_startup_oom_pruned_case_csv(
                task_info=task_info, output_dir=output_dir, cpu=cpu, mem=mem, gpu=gpu,
                reference_cpu=min(cpu_list), warmup=warmup, repeat=repeat,
                repeat_in_window=repeat_in_window, input_scales=input_scales)
            emit_event("case_skipped", case_name, status="oom_pruned")
        else:
            csv_path = run_single_case(
                CaseRequest(
                    task_info=task_info, cpu=cpu, mem=mem, gpu=gpu, image_info=image_info,
                    output_dir=output_dir, project_dir=project_dir, cpuset_cpus=cpuset_cpus,
                ),
                workload=CaseWorkload(
                    batch_size=batch_size, warmup=warmup, repeat=repeat,
                    repeat_in_window=repeat_in_window, repeat_window_seconds=repeat_window_seconds,
                    request_timeout_seconds=request_timeout_seconds,
                ),
                measurement=CaseMeasurement(
                    sample_hz=sample_hz, idle_seconds=idle_seconds,
                    idle_cooldown_seconds=idle_cooldown_seconds, idle_debug=idle_debug,
                    sniff_iface=sniff_iface, profiling_mode=profiling_mode, dram_energy=dram_energy,
                    sampling_policy=sampling_policy, raw_sample_limit=raw_sample_limit,
                ),
                plans=CasePlans(
                    input_scales=input_scales, input_scale_plan_file=input_scale_plan_file,
                    compute_profile_plan_file=compute_profile_plan_file,
                    execution_profile_plan_file=execution_profile_plan_file,
                    input_scale_order=case["input_scales"],
                ),
                run_state=run_state,
            )
        if run_state is not None and csv_path and not cached_case:
            run_state.finish_case(csv_path, cpu, mem, gpu)
        if csv_path:
            result_csvs.append(csv_path)
        if progress_callback is not None:
            try:
                progress_callback(MatrixProgress(current, total, cpu, mem, gpu, csv_path or None))
            except Exception as exc:
                print(f"[progress][WARN] Progress callback failed: {type(exc).__name__}", file=sys.stderr)
    return result_csvs


def merge_all_csvs(csv_paths: List[str], output_path: str, *, expected=None) -> None:
    """Validate all cases before atomically publishing the final result CSV."""
    from acprof.result_csv import merge_result_csvs

    row_count = merge_result_csvs(csv_paths, output_path, expected=expected)
    print(f"[merge] Final CSV: {output_path} ({row_count} rows)")
