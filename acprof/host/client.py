"""AC-Prof Client - workload generation, latency measurement, energy monitoring.

Runs on HOST (not inside container). Generalized from example-code/client.py.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from typing import IO, Any, Literal

import requests

from acprof.artifact_layout import ArtifactLayout, case_sidecar
from acprof.artifacts import atomic_write, loads_finite_json
from acprof.capabilities import measurement_requested
from acprof.config import (
    CLIENT_REQUEST_TIMEOUT_EXIT_CODE,
    CSV_FIELDS,
    IDLE_DIAG_DIRNAME,
)
from acprof.host.measurement import publication as client_publication
from acprof.host.measurement.config import ClientConfig
from acprof.host.measurement.errors import EnergyAbort, MIPSAbort, RequestTimeoutAbort
from acprof.host.measurement.execution import WindowExecutor
from acprof.host.measurement.metrics import (
    _fmt_float,
    _prepared_body_size_bytes,
    to_float_or_nan as _to_float_or_nan,
)
from acprof.host.measurement.monitors import MonitorCleanupError
from acprof.host.measurement.requests import canonical_task_param
from acprof.host.measurement.types import MonitorDependencies, WindowSpec
from acprof.host.profiling.compute_plan import (
    load_compute_profile_plan as _load_compute_profile_plan,
)
from acprof.host.profiling.execution_plan import (
    ExecutionProfileResolver,
    load_execution_profile_plan as _load_execution_profile_plan,
)
from acprof.window_records import WindowPublicationContext


def _local_proxy_options(url: str) -> dict:
    from urllib.parse import urlsplit
    # Keep the existing loopback bypass scoped to inference requests. Never
    # change the user's proxy environment or affect later Hub requests.
    return {"proxies": {"http": "", "https": "", "all": ""}} if urlsplit(url).hostname in {
        "localhost", "127.0.0.1", "::1"} else {}


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _is_file_empty(path: str) -> bool:
    try:
        return (not os.path.exists(path)) or os.path.getsize(path) == 0
    except Exception:
        return True

def _request_phase_context(request_id: str) -> dict[str, Any]:
    auto_match = re.search(r"_auto_warmup(?P<request_idx>\d+)$", request_id)
    if auto_match:
        return {
            "request_phase": "auto_repeat_window_warmup",
            "request_index_in_window": int(auto_match.group("request_idx")),
        }

    measurement_match = re.search(
        r"_(?P<phase>[wr])(?P<repeat_idx>\d+):(?P<request_idx>\d+)$",
        request_id,
    )
    if measurement_match:
        phase = measurement_match.group("phase")
        return {
            "request_phase": (
                "measurement_warmup" if phase == "w" else "measurement_repeat"
            ),
            "measurement_repeat_idx": int(measurement_match.group("repeat_idx")),
            "request_index_in_window": int(measurement_match.group("request_idx")),
        }

    return {"request_phase": "unknown"}

def _parse_effective_input_scale(resp: dict[str, Any]) -> float | None:
    if not isinstance(resp, dict):
        return None
    value = resp.get("effective_input_scale")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _generic_scale_label(scale_value: float) -> str:
    return f"scale{float(scale_value):g}"


class ClientRunner:
    """Own one configuration, request state and monitor dependencies per execution."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self._proxy_options = _local_proxy_options(config.base_url)
        from acprof.platform import detect_environment
        self.collection_environment = detect_environment()
        self.first_predict_app_s = float("nan")
        self.input_scale_entries: list[dict[str, Any]] = []
        self.runtime_failures: list[dict[str, Any]] = []
        self.use_energy = config.gpu_mode == "on"
        self.energy_mod = self._monitor_module("energy_nvml") if self.use_energy else None
        self.cpu_energy_mod = self._monitor_module("energy_cpu")
        self.resource_usage_mod = self._monitor_module("resource_usage")
        self.perf_mips_mod = self._monitor_module("perf_mips")

    @staticmethod
    def _monitor_module(name):
        from importlib import import_module
        try:
            return import_module(f"acprof.monitors.{name}")
        except ImportError as exc:
            print(f"[WARN] monitor {name} unavailable: {exc}", file=sys.stderr)
            return None

    def _sniff_groups_path(self, csv_path: str) -> str:
        return self.config.sniff_groups_path or str(case_sidecar(csv_path, "sniff_groups"))

    def _idle_diag_path(self, csv_path: str) -> str:
        if self.config.idle_diag_path:
            return self.config.idle_diag_path
        csv_dir = os.path.dirname(csv_path)
        csv_name = os.path.basename(csv_path)
        return os.path.join(
            csv_dir,
            IDLE_DIAG_DIRNAME,
            f"{csv_name}.idle_diag.jsonl",
        )

    def _cold_start_row_metrics(self) -> dict[str, str]:
        result = {
            "cold_start_started_at": self.config.cold_start_started_at or "nan",
            "cold_start_ready_at": self.config.cold_start_ready_at or "nan",
            "cold_start_container_launch_s": self.config.cold_start_container_launch_s or "nan",
            "cold_start_server_setup_s": self.config.cold_start_server_setup_s or "nan",
            "cold_start_cuda_init_s": self.config.cold_start_cuda_init_s or "nan",
            "cold_start_model_load_s": self.config.cold_start_model_load_s or "nan",
            "cold_start_ready_wait_s": self.config.cold_start_ready_wait_s or "nan",
            "cold_start_first_predict_app_s": _fmt_float(self.first_predict_app_s),
            "cold_start_s": self.config.cold_start_s or "nan",
        }
        return dict.fromkeys(result, "nan") if self.config.runtime_kind == "process" else result

    def _write_client_error_sidecar(self, exc: RequestTimeoutAbort) -> None:
        payload = {
            "schema_version": 1,
            "error_type": "client_request_timeout",
            "message": str(exc),
            "input_scale": exc.input_scale,
            "request_id": exc.request_id,
            "request_timeout_s": exc.timeout_s,
            "measurement_completed": False,
            "timeout_semantics": "connect_or_read_inactivity",
            **_request_phase_context(exc.request_id),
        }
        from acprof.failures import Failure
        payload["failure"] = Failure("request", "request_timeout", str(exc), self.config.gpu_mode,
            os.getenv("ACPROF_RUNTIME_PROFILE", ""), "higher_budget", {
                "timeout_seconds": exc.timeout_s, "request_phase": payload["request_phase"],
                "request_id": exc.request_id, "input_scale": exc.input_scale,
                "model_loaded": True, "service_alive": None,
                "timeout_scope": "connect_or_read_inactivity"}).to_dict()
        self.runtime_failures.append(payload["failure"])
        if not self.config.client_error_path:
            return
        def write(stream) -> None:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")

        atomic_write(self.config.client_error_path, write)

    def _prepare_repeat_window(self,
        scale_value: float,
        scale_label: str,
        payload_override: dict[str, Any] | None,
    ) -> int:
        if self.config.repeat_in_window > 0:
            return self.config.repeat_in_window
        if self.config.repeat_window_seconds <= 0.0:
            raise EnergyAbort(
                f"invalid REPEAT_WINDOW_SECONDS={self.config.repeat_window_seconds!r}; expected a positive value"
            )
        if self.config.auto_warmup_requests < 0:
            raise EnergyAbort(
                "invalid AUTO_WARMUP_REQUESTS="
                f"{self.config.auto_warmup_requests!r}; expected >= 0"
            )

        for idx in range(self.config.auto_warmup_requests):
            req_id = f"{self.config.case_name}_{scale_label}_auto_warmup{idx}"
            self._one_request(scale_value, req_id=req_id, payload_override=payload_override)

        print(
            "[client] auto repeat-window "
            f"scale={scale_value:g} warmup_requests={self.config.auto_warmup_requests} "
            f"target_window_s={self.config.repeat_window_seconds:.3f}",
            flush=True,
        )
        return 1

    def _one_request(self, scale_value: float, req_id: str, payload_override: dict[str, Any] | None = None) -> dict[str, Any]:
        if payload_override is None:
            raise RuntimeError("input scale plan entry has no payload; the plan must be self-contained")
        payload = payload_override

        headers = {
            "Connection": "close",
            "X-Req-Id": req_id,
        }

        t0 = time.perf_counter()
        try:
            r = requests.post(
                self.config.base_url + self.config.endpoint,
                json=payload,
                headers=headers,
                timeout=(self.config.request_timeout_seconds, self.config.request_timeout_seconds),
                **self._proxy_options,
            )
        except requests.exceptions.Timeout as exc:
            raise RequestTimeoutAbort(
                "inference request connect/read inactivity timeout: "
                f"{self.config.request_timeout_seconds:g}s per phase (not a total deadline) "
                f"(input_scale={scale_value:g}, req_id={req_id})",
                input_scale=float(scale_value),
                request_id=req_id,
                timeout_s=float(self.config.request_timeout_seconds),
            ) from exc
        t1 = time.perf_counter()
        if r.status_code >= 400:
            try:
                response = loads_finite_json(r.text)
                detail = response.get("error", "")
            except Exception:
                response = {}
                detail = r.text[:500]
            if isinstance(response.get("failure"), dict):
                from dataclasses import replace

                from acprof.failures import Failure, RuntimeFailure
                failure = Failure(**response["failure"])
                raise RuntimeFailure(replace(failure, evidence={**failure.evidence, "input_scale": float(scale_value),
                    "request_id": req_id, "request_phase": _request_phase_context(req_id)["request_phase"]}))
            raise RuntimeError(f"HTTP {r.status_code}: {detail or r.reason}")
        resp = loads_finite_json(r.text)
        request_latency_s = t1 - t0
        if not math.isfinite(self.first_predict_app_s):
            self.first_predict_app_s = request_latency_s
        return {
            "latency_app_s": request_latency_s,
            "resp": resp,
            "effective_input_scale": _parse_effective_input_scale(resp),
            "request_payload_bytes": _prepared_body_size_bytes(r),
            "output_length": _to_float_or_nan(resp.get("output_length")),
            "output_token_count": _to_float_or_nan(resp.get("output_token_count")),
            "task_param": canonical_task_param(payload),
            "workload_contract": resp.get("workload_contract"),
        }

    def _load_input_scale_entries(self) -> list[dict[str, Any]]:
        if self.config.input_scale_plan_file:
            from acprof.artifacts import read_input_scale_plan
            plan = read_input_scale_plan(self.config.input_scale_plan_file)

            entries = plan.get("entries")
            if not isinstance(entries, list) or not entries:
                raise RuntimeError(f"invalid input scale plan file: {self.config.input_scale_plan_file}")

            loaded_entries: list[dict[str, Any]] = []
            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise RuntimeError(
                        f"invalid input scale plan entry at index {idx}: {entry!r}"
                    )

                raw_scale = entry.get("input_scale")
                payload = entry.get("payload")
                if raw_scale is None or not isinstance(payload, dict):
                    raise RuntimeError(
                        f"input scale plan entry missing input_scale/payload at index {idx}"
                    )

                scale_value = float(raw_scale)
                scale_label = str(
                    entry.get("scale_label") or _generic_scale_label(scale_value)
                )
                input_metadata = entry.get("input_metadata", {})
                if not isinstance(input_metadata, dict):
                    raise ValueError(f"invalid input_metadata at input scale plan entry {idx}")
                loaded_entries.append({
                    "input_scale": scale_value,
                    "scale_label": scale_label,
                    "payload": payload,
                    "input_metadata": input_metadata,
                    "workload": plan.get("workload", {}),
                })

            scale_order = self.config.input_scale_order
            if scale_order:
                order = json.loads(scale_order)
                available = {entry["input_scale"]: entry for entry in loaded_entries}
                if (not isinstance(order, list) or len(order) != len(available)
                        or len(set(order)) != len(order) or set(order) != set(available)):
                    raise ValueError("frozen matrix input-scale order does not match input plan")
                loaded_entries = [available[scale] for scale in order]
            return loaded_entries

        raise ValueError("INPUT_SCALE_PLAN_FILE is required; generate a schema v2 input plan first")

    def main(self) -> None:
        from acprof.host.preflight import require_collection_host
        require_collection_host(profiling_mode=self.config.profiling_mode,
                                dram_energy=self.config.dram_energy)
        from acprof.artifacts import read_static_metadata
        from acprof.latency_slo import latency_slo_threshold
        slow_latency_threshold_s = latency_slo_threshold(
            read_static_metadata(ArtifactLayout.from_csv(self.config.out_csv).root)
        )
        if not self.input_scale_entries:
            self.input_scale_entries = self._load_input_scale_entries()
        self.first_predict_app_s = float("nan")

        if self.config.runtime_kind == "docker" and self.config.profiling_mode == "basic" and self.resource_usage_mod is None:
            raise RuntimeError("basic profiling requires the container CPU and memory collector")
        if self.config.runtime_kind == "docker" and self.config.gpu_mode == "on" and not self.config.gpu_device_uuid:
            raise RuntimeError("GPU_DEVICE_UUID is required: launch the client through the AC-Prof orchestrator")

        if measurement_requested(self.config.profiling_mode, "gpu_power", gpu=self.use_energy) and self.energy_mod is None:
            raise EnergyAbort(
                "GPU energy monitoring is required for gpu_mode=on but NVML/pynvml is unavailable. "
                "Install nvidia-ml-py, verify NVIDIA driver access, or rerun with --gpus off."
            )
        if measurement_requested(self.config.profiling_mode, "cpu_instructions") and self.config.use_mips and self.perf_mips_mod is None:
            raise MIPSAbort(
                "MIPS profiling is enabled but perf_mips.py could not be imported."
            )

        compute_profile_plan = _load_compute_profile_plan(self.config.compute_profile_plan_file)
        execution_profiles = ExecutionProfileResolver(_load_execution_profile_plan(
            self.config.execution_profile_plan_file
        ))
        resource_factory = self.resource_usage_mod.ResourceUsageMonitor if self.resource_usage_mod is not None else None
        if self.config.runtime_kind == "process":
            from functools import partial

            from acprof.monitors.process_usage import ProcessUsageMonitor
            resource_factory = partial(ProcessUsageMonitor, pid=self.config.process_pid,
                                       start_ticks=self.config.process_start_ticks)
        windows = WindowExecutor(
            self.config,
            MonitorDependencies(
                gpu=self.energy_mod.GPUEnergyMonitor if self.energy_mod is not None else None,
                cpu=self.cpu_energy_mod.CPUEnergyMonitor if self.cpu_energy_mod is not None else None,
                resource=resource_factory,
                instructions=self.perf_mips_mod.PerfMIPSMonitor if self.perf_mips_mod is not None else None,
            ),
            one_request=self._one_request,
            cold_start_metrics=self._cold_start_row_metrics,
            environment_class=self.collection_environment.environment,
            compute_profile_plan=compute_profile_plan,
            execution_profiles=execution_profiles,
            use_energy=self.use_energy,
            runtime_failures=self.runtime_failures,
            slow_latency_threshold_s=slow_latency_threshold_s,
        )
        need_header = _is_file_empty(self.config.out_csv)
        publication = WindowPublicationContext(
            run_id=self.config.run_id, case_id=self.config.case_id,
            case_attempt_id=self.config.case_attempt_id,
            records_dir=self.config.window_records_dir, telemetry_dir=self.config.telemetry_dir,
        ) if self.config.window_records_dir else None
        fieldnames = CSV_FIELDS
        if not need_header:
            with open(self.config.out_csv, newline="", encoding="utf-8-sig") as existing:
                fieldnames = next(csv.reader(existing))
            if len(fieldnames) != len(CSV_FIELDS) or set(fieldnames) != set(CSV_FIELDS):
                raise RuntimeError(f"existing CSV columns do not match current fields: {self.config.out_csv}; use a new output file")
        sniff_groups_path = self._sniff_groups_path(self.config.out_csv)
        if not need_header:
            client_publication.reconcile_sniff_group_sidecar(
                self.config.out_csv,
                sniff_groups_path,
            )
        sidecar_mode: Literal["w", "a"] = "w" if need_header else "a"
        diag_context: IO[Any] | nullcontext[None]
        if self.config.idle_debug:
            diag_path = self._idle_diag_path(self.config.out_csv)
            os.makedirs(os.path.dirname(diag_path) or ".", exist_ok=True)
            diag_context = open(diag_path, sidecar_mode, encoding="utf-8")
        else:
            diag_context = nullcontext(None)
        with open(self.config.out_csv, "a", newline="", encoding="utf-8") as f, open(
            sniff_groups_path,
            sidecar_mode,
            encoding="utf-8",
        ) as sidecar_f, open(
            case_sidecar(self.config.out_csv, "requests"), sidecar_mode, encoding="utf-8"
        ) as requests_f, diag_context as diag_f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_MINIMAL,
            )
            if need_header:
                writer.writeheader()
                f.flush()
                os.fsync(f.fileno())

            # /ready check
            ready_error: Exception | None = None
            try:
                rr = requests.get(self.config.base_url + "/ready", timeout=60, headers={"Connection": "close"},
                                  **self._proxy_options)
            except requests.exceptions.RequestException as exc:
                ready_error = exc
            else:
                if rr.status_code >= 400:
                    ready_error = RuntimeError(
                        f"/ready HTTP {rr.status_code}: {rr.text[:200]}"
                    )
            if ready_error is not None:
                row = {k: "nan" for k in CSV_FIELDS}
                row.update({
                    "cpu_cores": self.config.cpu_cores,
                    "mem_cap_gb": self.config.mem_cap_gb,
                    "gpu_mode": self.config.gpu_mode,
                    **self._cold_start_row_metrics(),
                    "status": "error",
                    "error": f"ready_failed: {ready_error!r}",
                })
                client_publication._append_row(writer, row, f, sidecar_f, "")
                return

            for scale_entry in self.input_scale_entries:
                scale_val = float(scale_entry["input_scale"])
                payload_override = scale_entry.get("payload")
                repeat_request_limit = self._prepare_repeat_window(
                    scale_val,
                    str(scale_entry["scale_label"]),
                    payload_override,
                )
                for idx in range(self.config.warmup + self.config.repeat):
                    warmup_flag = 1 if idx < self.config.warmup else 0
                    repeat_idx = idx if warmup_flag else (idx - self.config.warmup)

                    result = windows.execute(
                        WindowSpec(scale_entry, repeat_request_limit, warmup_flag, repeat_idx),
                        requests_f,
                    )
                    client_publication._append_row(
                        writer,
                        result.row,
                        f,
                        sidecar_f,
                        result.sniff_group_id,
                        diag_f=diag_f,
                        idle_diag_record=result.idle_diag_record,
                        publication=publication,
                        telemetry_evidence=result.telemetry_evidence,
                        request_record=result.request_record,
                    )

    def run_cli(self) -> None:
        from acprof.failures import FAILURE_PREFIX, RuntimeFailure
        try:
            self.main()
        except RuntimeFailure as exc:
            self.runtime_failures.append(exc.failure.to_dict())
            raise SystemExit(1) from None
        except RequestTimeoutAbort as exc:
            try:
                self._write_client_error_sidecar(exc)
            except OSError as sidecar_exc:
                print(
                    "[case][WARN] failed to persist structured timeout context: "
                    f"{sidecar_exc}",
                    file=sys.stderr,
                )
            print(f"[case][ERROR] {exc}", file=sys.stderr)
            raise SystemExit(CLIENT_REQUEST_TIMEOUT_EXIT_CODE) from None
        except MonitorCleanupError as exc:
            print(f"[monitor][ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        except EnergyAbort as exc:
            print(f"[energy][ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        except MIPSAbort as exc:
            message = str(exc)
            if message.startswith("[mips][ERROR]"):
                print(message, file=sys.stderr)
            else:
                print(f"[mips][ERROR] {message}", file=sys.stderr)
            exit_code = getattr(self.perf_mips_mod, "MIPS_EXIT_CODE", 8) if self.perf_mips_mod else 8
            raise SystemExit(exit_code) from None
        finally:
            if self.runtime_failures:
                from acprof.artifacts import atomic_write_json
                atomic_write_json(case_sidecar(self.config.out_csv, "runtime_failures"),
                                  {"schema_version": 1, "failures": self.runtime_failures})
                for failure in self.runtime_failures:
                    print(FAILURE_PREFIX + json.dumps(failure, ensure_ascii=False), file=sys.stderr)


def main(config: ClientConfig | None = None) -> None:
    config = ClientConfig.from_env() if config is None else config
    config.validate()
    logging.getLogger(__name__).debug("client configuration: pipeline_tag=%s", config.pipeline_tag)
    ClientRunner(config).run_cli()


if __name__ == "__main__":
    main()
