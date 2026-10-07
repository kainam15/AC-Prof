"""在资源矩阵之前验证接口；验证容器退出后才开始正式采集。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, TypedDict

from acprof.artifacts import atomic_write, loads_finite_json
from acprof.container.runtime_validate import RESULT_PREFIX, STAGE_PREFIX
from acprof.failures import Failure, RuntimeFailure, failure_from_exception
from acprof.host.command import run_command
from acprof.host.container_lifecycle import (
    ContainerCleanupError,
    container_owner_labels,
    owned_container_id,
    recover_abandoned_containers,
    remove_owned_container,
)
from acprof.host.gpu_device import gpu_docker_args
from acprof.runtime_settings import runtime_docker_env_args

if TYPE_CHECKING:
    from acprof.host.detect import TaskInfo
    from acprof.host.input_plan import PlannedInputScales
    from acprof.host.runtime_images import ImageInfo


class RuntimeValidationReport(TypedDict, total=False):
    schema_version: int
    status: str
    platform: dict
    collection_tier: str
    comparability_class: str
    environment_class: str
    image_id: str
    build_fingerprint: str
    input_scale: float
    payload_sha256: str
    scope: str
    devices: dict[str, dict[str, Any]]
    mode: str
    profiler_validation: str
    cleanup_status: str
    cleanup_error: dict

_LOG = logging.getLogger(__name__)


class RuntimeValidationError(RuntimeFailure, RuntimeError):
    """A persisted runtime failure, including an inconclusive timeout."""


def validate_runtime(
    *, task_info: TaskInfo, image_info: ImageInfo, planned: PlannedInputScales, cpu_list: list[int],
    mem_list: list[int], gpu_list: list[str], output_dir: str,
    timeout_seconds: float = 300.0,
) -> RuntimeValidationReport:
    if (not gpu_list or any(value not in {"off", "on"} for value in gpu_list) or not cpu_list or not mem_list
            or any(type(value) is not int or value <= 0 for value in [*cpu_list, *mem_list])
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("runtime validation requires devices and positive resource/timeout limits")
    if not getattr(image_info, "runtime_environment", {}):
        raise ValueError("镜像缺少 runtime_environment；请使用当前版本重新构建运行环境")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_info.tag):
        raise ValueError("runtime validation requires an immutable image ID")
    from acprof.artifacts import atomic_write_json, read_input_scale_plan
    from acprof.host.container_state import inspect_container_state
    from acprof.host.env_utils import hf_offline_docker_env_args
    from acprof.host.model_store import acquire_mount, retain_mount_for_cleanup_debt

    if planned.plan_file is None:
        raise ValueError("runtime validation requires an input plan file")
    plan = read_input_scale_plan(planned.plan_file)
    entry = min(plan["entries"], key=lambda item: float(item["input_scale"]))
    encoded = json.dumps(entry["payload"], ensure_ascii=False).encode()
    from acprof.platform import detect_environment
    report: RuntimeValidationReport = {
        **detect_environment().metadata(),
        "schema_version": 1, "status": "running", "image_id": image_info.tag,
        "build_fingerprint": image_info.runtime_environment["build_fingerprint"],
        "input_scale": entry["input_scale"], "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "scope": "isolated_minimum_scale_predict_and_postprocess_before_measurement",
        "devices": {},
        "mode": "full",  # Existing report protocol; execution has no mode switch.
        "profiler_validation": "separate_profiler_plans; inference_success_does_not_prove_profiler_support",
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    from acprof.artifact_layout import ArtifactLayout
    layout = ArtifactLayout.discover(root)
    failure = None
    structured_failure = None
    cleanup_failure = None
    owner = container_owner_labels()
    try:
        recover_abandoned_containers(owner, run_command)
    except ContainerCleanupError as exc:
        report.update({"status": "error", "cleanup_status": "incomplete", "cleanup_error": exc.to_dict()})
        atomic_write_json(layout.path("runtime_validation.json"), report)
        raise
    labels = [part for key, value in owner.items() for part in ("--label", f"{key}={value}")]
    with tempfile.TemporaryDirectory(prefix="acprof-runtime-validation-") as temporary:
        payload = Path(temporary) / "payload.json"
        payload.write_bytes(encoded)
        for device_mode in dict.fromkeys(gpu_list):
            device_result: dict[str, Any]
            stages: list[dict[str, Any]]
            name = "acprof-validate-" + uuid.uuid4().hex[:16]
            cidfile = Path(temporary) / f"{device_mode}.cid"
            model_store_mount = acquire_mount(image_info.runtime_environment)
            try:
                command = [
                    "docker", "run", "--name", name, "--cidfile", str(cidfile), *labels, "--network", "none",
                    "--read-only", "--cap-drop=ALL", "--security-opt", "no-new-privileges", "--pids-limit=256",
                    "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
                    f"--cpus={max(cpu_list)}", f"--memory={max(mem_list)}g",
                    "-v", f"{payload}:/validation-input.json:ro",
                    "-e", f"MODEL_ID={task_info.model_id}",
                    "-e", f"MODEL_REVISION={task_info.model_revision}",
                    "-e", f"TASK_FAMILY={task_info.task_family}",
                    "-e", f"TASK_TYPE={task_info.pipeline_tag}",
                    "-e", f"RUNTIME_BACKEND={task_info.runtime_backend}",
                    "-e", f"USE_GPU={int(device_mode == 'on')}",
                    "-e", f"TORCH_NUM_THREADS={max(cpu_list)}",
                    "-e", f"ACPROF_REQUEST_TIMEOUT_S={timeout_seconds:g}",
                    *runtime_docker_env_args(),
                    *hf_offline_docker_env_args(),
                    *model_store_mount.args,
                    "-e", "HF_MODULES_CACHE=/tmp/hf-modules", "-e", "XDG_CACHE_HOME=/tmp/cache",
                    "-e", "PYTHONDONTWRITEBYTECODE=1",
                ]
                if device_mode == "on":
                    command += gpu_docker_args()
                command += ["--entrypoint", "python", image_info.tag, "-m", "acprof.container.runtime_validate", "/validation-input.json"]
            except BaseException:
                model_store_mount.close()
                raise
            print(f"[runtime-check] {device_mode}: 正在验证模型运行（独立容器）", flush=True)
            log = ""
            run_error: BaseException | None = None
            try:
                result = run_command(command, capture_output=True, text=True, timeout=timeout_seconds)
                log = (result.stdout or "") + "\n" + (result.stderr or "")
                records = [line[len(RESULT_PREFIX):] for line in (result.stdout or "").splitlines() if line.startswith(RESULT_PREFIX)]
                identifier = owned_container_id(cidfile)
                if not identifier and result.returncode == 0:
                    raise ContainerCleanupError("", [{"operation": "run", "detail": "Docker returned without a cidfile"}])
                state = (inspect_container_state(identifier) or {}) if identifier else {}
                if state.get("OOMKilled"):
                    device_result = {"status": "resource_limit", "error": "validation_container_oom", "mem_cap_gb": max(mem_list)}
                    device_result["failure"] = Failure("runtime_validation", "resource_limit", "validation_container_oom",
                        device_mode, task_info.runtime_profile_id, "higher_budget",
                        {"docker_state": state, "mem_cap_gb": max(mem_list), "measured_oom": True}).to_dict()
                elif records:
                    device_result = loads_finite_json(records[-1])
                    if not isinstance(device_result, dict) or device_result.get("status") not in {"ok", "error"}:
                        raise ValueError("invalid runtime validation response")
                    if result.returncode and device_result.get("status") == "ok":
                        device_result = {"status": "error", "error": f"container exit {result.returncode}"}
                    if device_result.get("status") == "ok":
                        required = {
                            "load", "preprocess", "predict", "postprocess", "validate_output"}
                        observed = {item.get("stage") for item in device_result.get("stages", [])
                                    if isinstance(item, dict) and item.get("status") == "verified"}
                        if not required <= observed or any(
                            device_result.get("validation", {}).get(layer, {}).get("status") != "verified"
                            for layer in ("protocol", "task")
                        ):
                            raise ValueError("incomplete runtime validation response")
                else:
                    device_result = {"status": "error", "error": log[-4000:] or f"container exit {result.returncode}"}
                from acprof.quality import cli_exit_quality
                device_result.setdefault("quality_checks", []).extend(cli_exit_quality(result.returncode, source=name))
                if device_result.get("failure"):
                    value = Failure(**device_result["failure"])
                    if value.reason_code == "request_timeout":
                        from dataclasses import replace
                        stages = device_result.get("stages", [])
                        value = replace(value, evidence={
                            "timeout_seconds": None, "request_phase": device_result.get("failed_stage", "unknown"),
                            "model_loaded": True if any(s.get("stage") == "load" and s.get("status") == "verified" for s in stages) else None,
                            "service_alive": state.get("Running"), **value.evidence,
                            "request_id": name, "input_scale": entry["input_scale"]})
                        device_result["status"] = "inconclusive"
                    device_result["failure"] = value.to_dict()
            except ContainerCleanupError as exc:
                cleanup_failure = exc
                device_result = {"status": "error", "error": str(exc)}
            except subprocess.TimeoutExpired as exc:
                run_error = exc
                _LOG.debug("runtime validation timeout: mode=%s timeout_s=%s", device_mode, timeout_seconds)
                def decoded(value):
                    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")

                log = decoded(exc.stdout) + "\n" + decoded(exc.stderr)
                stages = []
                for line in log.splitlines():
                    if line.startswith(STAGE_PREFIX):
                        try:
                            record = loads_finite_json(line[len(STAGE_PREFIX):])
                            if isinstance(record, dict):
                                stages.append(record)
                        except ValueError:
                            pass
                try:
                    identifier = owned_container_id(cidfile)
                except ContainerCleanupError as cid_error:
                    cleanup_failure = cid_error
                    identifier = ""
                if not identifier and cleanup_failure is None:
                    cleanup_failure = ContainerCleanupError("", [{"operation": "read_cidfile",
                        "detail": "no immutable ID after Docker timeout; container absence is unconfirmed"}])
                if cleanup_failure is not None:
                    cleanup_failure.run_error = exc
                state = (inspect_container_state(identifier) or {}) if identifier else {}
                detail = f"runtime_validation_timeout ({timeout_seconds:g}s); compatibility budget exhausted"
                value = Failure("runtime_validation", "compatibility_budget_exhausted", detail, device_mode,
                    task_info.runtime_profile_id, "higher_budget", {
                        "timeout_seconds": timeout_seconds, "request_phase": stages[-1].get("stage", "unknown") if stages else "unknown",
                        "request_id": name, "input_scale": entry["input_scale"],
                        "model_loaded": True if any(s.get("stage") == "load" and s.get("status") == "verified" for s in stages) else None,
                        "service_alive": state.get("Running"), "timeout_scope": "compatibility_budget", "stages": stages})
                device_result = {"status": "inconclusive", "error": detail, "failure": value.to_dict()}
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                run_error = exc
                _LOG.debug("runtime validation failed: mode=%s error_type=%s", device_mode, type(exc).__name__)
                device_result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            finally:
                try:
                    identifier = owned_container_id(cidfile)
                    if identifier:
                        remove_owned_container(identifier, run_command)
                except ContainerCleanupError as exc:
                    exc.run_error = exc.run_error or run_error
                    cleanup_failure = exc
                if cleanup_failure is None:
                    model_store_mount.close()
                else:
                    retain_mount_for_cleanup_debt(model_store_mount)
            if device_result["status"] == "error" and "failure" not in device_result:
                device_result["failure"] = failure_from_exception(
                    RuntimeError(device_result.get("error", "runtime validation failed")),
                    stage=device_result.get("failed_stage", "runtime_validation"), device=device_mode,
                    runtime_profile=task_info.runtime_profile_id).to_dict()
            log_path = (layout.path("logs") if layout.layout_version == 2 else root) / f"runtime_validation_{device_mode}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            def write_log(stream: TextIO) -> None:
                stream.write(log)

            atomic_write(log_path, write_log)
            report["devices"][device_mode] = device_result
            if cleanup_failure is not None:
                if cleanup_failure.run_error is None and device_result.get("error"):
                    cleanup_failure.run_error = RuntimeError(device_result["error"])
                device_result["cleanup_error"] = cleanup_failure.to_dict()
                report["cleanup_status"] = "incomplete"
                break
            if device_result["status"] in {"error", "inconclusive"}:
                failure = f"{device_mode}: {device_result.get('error', 'runtime validation failed')}"
                structured_failure = Failure(**device_result["failure"])
                break
            print(f"[runtime-check] {device_mode}: {device_result['status']}", flush=True)
    report["status"] = "error" if failure or cleanup_failure else (
        "ok" if all(item["status"] == "ok" for item in report["devices"].values()) else "resource_limited"
    )
    if cleanup_failure is None and any(item["status"] == "inconclusive" for item in report["devices"].values()):
        report["status"] = "inconclusive"
    atomic_write_json(layout.path("runtime_validation.json"), report)
    if getattr(task_info, "model_resolution", {}):
        from acprof.model_contract import record_runtime_validation, write_model_resolution
        record_runtime_validation(task_info, report)
        write_model_resolution(task_info, root)
    if cleanup_failure is not None:
        raise cleanup_failure
    if failure:
        raise RuntimeValidationError(structured_failure)
    return report
