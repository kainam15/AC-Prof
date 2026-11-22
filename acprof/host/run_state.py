"""主实验状态、目录互斥与 case 恢复；所有操作均在测量窗口之外。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import socket
import stat
import sys
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread, current_thread, local
from typing import TYPE_CHECKING, TypeGuard
from uuid import uuid4

from acprof.artifacts import (
    atomic_write_json,
    file_sha256,
    read_json_object,
    sync_directory_chain,
)
from acprof.experiments.arguments import flatten_run_options
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.execution_conditions import measurement_environment
from acprof.host.run_state_types import (
    AttemptState,
    CaseState,
    PlannedSnapshot,
    RunStateSnapshot,
    RuntimeSnapshot,
)
from acprof.messages import Message, join_messages, message
from acprof.platform import detect_environment, is_linux
from acprof.results.csv import (
    ResultValidationError,
    expected_measurements,
    publish_merged_cases,
    read_result_csv,
    read_result_csv_snapshot,
)
from acprof.results.layout import ArtifactLayout
from acprof.results.measurement_plan import (
    RecordedMeasurementOptions,
    RecordedMeasurementPlan,
    recorded_scales,
)
from acprof.results.metadata import frozen_input_plan_reference
from acprof.source_identity import measurement_sources, source_fingerprint
from acprof.system.cleanup import cleanup_on_exit, format_exception_notes

if TYPE_CHECKING:
    from acprof.result_store.writer import ResultWriter

RUN_STATE_NAME = "run_state.json"
RESULT_LOCK_NAME = ".acprof-result.lock"
MAX_RUN_STATE_BYTES = 4 * 1024 * 1024
# Native Linux only. Never derive this machine-wide per-user namespace from TMPDIR.
# Test runners inject an isolated directory in-process, not via a production env option.
MEASUREMENT_LOCK_ROOT = Path("/tmp")
# A Linux abstract address has no pathname to unlink. Keep its identity stable
# across checkouts, outputs, sessions and environment variables.
MEASUREMENT_LOCK_NAMESPACE = b"acprof.measurement"


class RunStateError(RuntimeError):
    pass


def _validate_lock_descriptor(descriptor: int, *, label: str) -> None:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise OSError(f"{label} lock is not a regular file owned only by this user")


def check_result_directory_idle(directory: str | Path) -> None:
    """Fail closed when an existing result-directory lock is active or malformed."""
    import fcntl

    path = ArtifactLayout.discover(directory).path(RESULT_LOCK_NAME)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RunStateError(f"无法检查结果目录锁；锁文件不可用：{path} ({exc})") from exc
    with cleanup_on_exit(lambda: os.close(descriptor), description=f"result lock probe {path}"):
        try:
            _validate_lock_descriptor(descriptor, label="result")
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunStateError(message("原实验仍有进程写入，请等待其退出后再恢复。")) from exc
        except OSError as exc:
            raise RunStateError(f"无法检查结果目录锁；锁文件不可用：{path} ({exc})") from exc


def _finite_json_number(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(message("run_state.json 包含非有限数值"))
    return value


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def host_identity(project_dir: str | Path) -> dict:
    source_hash = source_fingerprint(project_dir, measurement_sources(project_dir),
                                     scope="measurement-source-v2")
    packages = sorted((dist.metadata["Name"] or "", dist.version)
                      for dist in importlib.metadata.distributions())
    machine_id = Path("/etc/machine-id")
    return {
        **detect_environment().metadata(),
        "machine": platform.machine(), "kernel": platform.release(), "hostname": platform.node(),
        "machine_id_sha256": file_sha256(machine_id) if machine_id.is_file() else None,
        "python": platform.python_version(), "source_sha256": source_hash,
        "packages_sha256": hashlib.sha256(json.dumps(packages).encode()).hexdigest(),
    }


def run_options(args) -> dict:
    from acprof.host.gpu_device import selected_gpu_device
    ignored = {"resume", "rebuild_image", "output_dir", "notify", "runtime_selection", "process_gpu", "collection_plan",
               "inference_config", "confirm_inference"}
    options = {name: value for name, value in flatten_run_options(vars(args)).items() if name not in ignored}
    # Freeze even default policies: they affect both timing and retained evidence.
    options.setdefault("sampling_policy", "burst")
    options.setdefault("raw_sample_limit", 100000)
    for name in ("tasks", "task_run_id", "model_variant"):
        if not options.get(name):
            options.pop(name, None)
    from acprof.hub.repository import model_source
    source = model_source(options.get("model_source"))
    if source == "huggingface":
        options.pop("model_source", None)  # Preserve historical HF resume options.
    else:
        options["model_source"] = source
    if options.get("revision") is None:
        options.pop("revision", None)
    if options.get("input_scale_policy", "auto") == "auto":
        options.pop("input_scale_policy", None)
    if options.get("runtime") == "process":
        from acprof.host.runtimes.resources import inherited_resources
        # Allocation and device visibility can change within the same VM/container.
        # Refuse to append windows collected under different inherited conditions.
        options["process_allocation"] = inherited_resources()
        options["process_gpu"] = getattr(args, "process_gpu", {})
        options["nvidia_visible_devices"] = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    device = selected_gpu_device() if options.get("runtime") != "process" else {}
    if device:
        options["gpu_device"] = device["uuid"]
    elif options.get("gpu_device") is None:
        options.pop("gpu_device", None)
    from acprof.cpu_affinity import normalize_cpu_set
    cpuset = normalize_cpu_set(options.get("cpuset_cpus", ""))
    if cpuset:
        options["cpuset_cpus"] = cpuset
    else:
        options.pop("cpuset_cpus", None)
    for name in ("cpus", "mems"):
        options[name] = ",".join(part.strip().lower() for part in options[name].split(",") if part.strip())
    options["gpus"] = ",".join(normalize_gpu_mode(part) for part in options["gpus"].split(",") if part.strip())
    # New options only enter pre-execution identity when explicitly requested.
    # Absent defaults retain the old serialized options; no observed values enter here.
    options["measurement_environment"] = measurement_environment()
    if options.get("workload_spec"):
        path = Path(options["workload_spec"]).expanduser().resolve()
        options["workload_spec"] = str(path)
        options["workload_spec_sha256"] = file_sha256(path)
    if options.get("model_spec"):
        path = Path(options["model_spec"]).expanduser().resolve()
        options["model_spec"] = str(path)
        options["model_spec_sha256"] = file_sha256(path)
    else:
        options.pop("model_spec", None)
    return options


def _state_text(value: object, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise RunStateError(f"run_state.json 字段 {field} 无效：需要{'字符串' if empty else '非空字符串'}")
    return value


def _state_path(layout: ArtifactLayout, value: object, field: str, *, empty: bool = False) -> None:
    path = _state_text(value, field, empty=empty)
    if path:
        try:
            layout.contained(path)
        except ValueError as exc:
            raise RunStateError(f"run_state.json 字段 {field} 无效：{exc}") from exc


def _state_reference(layout: ArtifactLayout, value: object, field: str) -> None:
    if not isinstance(value, dict):
        raise RunStateError(f"run_state.json 字段 {field} 无效：需要产物引用")
    _state_path(layout, value.get("path"), f"{field}.path")
    checksum = _state_text(value.get("sha256"), f"{field}.sha256")
    if re.fullmatch(r"[a-f0-9]{64}", checksum) is None:
        raise RunStateError(f"run_state.json 字段 {field}.sha256 无效")
    if "size" in value and (type(value["size"]) is not int or value["size"] < 0):
        raise RunStateError(f"run_state.json 字段 {field}.size 无效：需要非负整数")


def _validate_cleanup_process(layout: ArtifactLayout, evidence: object, field: str) -> None:
    if not isinstance(evidence, dict):
        raise RunStateError(f"run_state.json 字段 {field} 无效：需要进程所有权对象")
    if not evidence:
        return  # Historical empty cleanup evidence is an absent optional record.
    for name in ("owner", "workload"):
        identity = evidence.get(name, {})
        if not isinstance(identity, dict) or (name == "workload" and not identity):
            raise RunStateError(f"run_state.json 字段 {field}.{name} 无效：需要进程身份对象")
        if identity:
            for key, minimum in (("pid", 1), ("start_ticks", 0)):
                if type(identity.get(key)) is not int or identity[key] < minimum:
                    raise RunStateError(f"run_state.json 字段 {field}.{name}.{key} 无效")
    source = _state_text(evidence.get("status_file"), f"{field}.status_file")
    path = Path(source).absolute()
    if path.parent != layout.path("logs").absolute() or not path.name.endswith(".process.json"):
        raise RunStateError(f"run_state.json 字段 {field}.status_file 无效：必须位于本实验 logs")
    _state_path(layout, path.relative_to(layout.root).as_posix(), f"{field}.status_file")


def _validate_case_state(layout: ArtifactLayout, record: object, field: str) -> None:
    if not isinstance(record, dict):
        raise RunStateError(f"原实验的 case 或产物记录无效。字段：{field}")
    if any(not isinstance(name, str) for name in record):
        raise RunStateError(f"run_state.json 字段 {field} 无效：字段名必须为字符串")
    status = _state_text(record.get("status"), f"{field}.status")
    if status not in {"running", "measured", "complete", "archived"}:
        raise RunStateError(f"run_state.json 字段 {field}.status 无效")
    for name, allowed in (
        ("measurement_status", {"running", "complete", "not_measured"}),
        ("processing_status", {"pending", "running", "failed", "complete"}),
        ("cleanup_status", {"pending", "failed", "complete"}),
    ):
        if name in record and _state_text(record[name], f"{field}.{name}") not in allowed:
            raise RunStateError(f"run_state.json 字段 {field}.{name} 无效")
    if "case_attempt_id" in record or record.get("measurement_status") == "complete":
        _state_text(record.get("case_attempt_id"), f"{field}.case_attempt_id")
    if "resources" in record or record.get("measurement_status") == "complete":
        resources = record.get("resources")
        if (not isinstance(resources, list) or len(resources) != 3
                or any(value is not None and (type(value) is not int or value <= 0)
                       for value in resources[:2])
                or not isinstance(resources[2], str) or resources[2] not in {"on", "off"}):
            raise RunStateError(f"run_state.json 字段 {field}.resources 无效")
    for name in ("row_count", "error_rows", "processing_attempts", "restored_windows"):
        if name in record and (type(record[name]) is not int or record[name] < 0):
            raise RunStateError(f"run_state.json 字段 {field}.{name} 无效：需要非负整数")
    if status == "complete":
        checksum = _state_text(record.get("sha256"), f"{field}.sha256")
        if re.fullmatch(r"[a-f0-9]{64}", checksum) is None:
            raise RunStateError(f"run_state.json 字段 {field}.sha256 无效")
    for name in ("measurement_checkpoint", "case_result"):
        if name in record:
            _state_reference(layout, record[name], f"{field}.{name}")
    if record.get("measurement_status") == "complete" and status != "complete":
        _state_reference(layout, record.get("measurement_checkpoint"), f"{field}.measurement_checkpoint")
    if "backup" in record:
        _state_path(layout, record["backup"], f"{field}.backup")
    if "cleanup_process" in record:
        _validate_cleanup_process(layout, record["cleanup_process"], f"{field}.cleanup_process")
    previous = record.get("previous_attempts", [])
    if not isinstance(previous, list):
        raise RunStateError(f"run_state.json 字段 {field}.previous_attempts 无效：需要列表")


def _validate_selected_tasks(task: dict) -> None:
    selected_tasks = task.get("selected_tasks", [])
    if not isinstance(selected_tasks, list):
        raise RunStateError("run_state.json 字段 runtime.task.selected_tasks 无效：需要列表")
    for index, selected in enumerate(selected_tasks):
        field = f"runtime.task.selected_tasks[{index}]"
        if not isinstance(selected, dict):
            raise RunStateError(f"run_state.json 字段 {field} 无效：需要任务对象")
        profile = selected.get("execution_profile", {})
        if not isinstance(profile, dict):
            raise RunStateError(f"run_state.json 字段 {field}.execution_profile 无效")
        signature = profile.get("signature", {})
        if not isinstance(signature, dict):
            raise RunStateError(f"run_state.json 字段 {field}.execution_profile.signature 无效")
        if "execution" not in signature:
            continue  # Match the existing legacy branch in validate_execution_identity.
        for name in ("task_id", "interface", "kind", "fingerprint"):
            _state_text(selected.get(name), f"{field}.{name}")
        _state_text(signature.get("backend"), f"{field}.execution_profile.signature.backend", empty=True)
        if not isinstance(signature["execution"], dict):
            raise RunStateError(f"run_state.json 字段 {field}.execution_profile.signature.execution 无效")
        config = selected.get("config")
        if not isinstance(config, dict):
            raise RunStateError(f"run_state.json 字段 {field}.config 无效")
        generation = config.get("generation", {})
        if not isinstance(generation, dict):
            raise RunStateError(f"run_state.json 字段 {field}.config.generation 无效")
        from acprof.tasks.groups import inference_signature
        try:
            inference_signature(selected, backend=signature["backend"])
        except (ValueError, TypeError, OverflowError) as exc:
            raise RunStateError(f"run_state.json 字段 {field}.config.generation 无效：{exc}") from exc


def _validate_measurement_options(options: object) -> RecordedMeasurementOptions:
    if not isinstance(options, dict):
        raise RunStateError(message("原实验缺少完整的参数或主机身份，无法续跑。"))
    if any(not isinstance(name, str) for name in options):
        raise RunStateError("run_state.json 字段 options 无效：字段名必须为字符串")
    try:
        return RecordedMeasurementOptions.parse(options)
    except ValueError as exc:
        raise RunStateError(f"run_state.json: {exc}") from exc


def _validate_scales(raw: object) -> tuple[int | float, ...]:
    try:
        return recorded_scales(raw)
    except ValueError as exc:
        raise RunStateError(f"run_state.json: {exc}") from exc


def input_plan_reference(
    layout: ArtifactLayout, plan: PlannedSnapshot, artifacts: Mapping[str, str],
) -> tuple[str, str | None]:
    """Resolve the plan path and reconcile the digests recorded for this experiment."""
    try:
        return frozen_input_plan_reference(layout, plan, artifacts)
    except ValueError as error:
        raise RunStateError(str(error)) from error


def _validate_planned_snapshot(plan: object, layout: ArtifactLayout) -> TypeGuard[PlannedSnapshot]:
    if not isinstance(plan, dict):
        raise RunStateError("run_state.json 字段 runtime.planned 无效：需要对象")
    if any(not isinstance(name, str) for name in plan):
        raise RunStateError("run_state.json 字段 runtime.planned 无效：字段名必须为字符串")
    _validate_scales(plan.get("scales"))
    _state_text(plan.get("source"), "runtime.planned.source", empty=True)
    if plan.get("plan_file") is not None:
        _state_path(layout, plan["plan_file"], "runtime.planned.plan_file", empty=True)
    if not isinstance(plan.get("workload", {}), dict):
        raise RunStateError("run_state.json 字段 runtime.planned.workload 无效")
    if "plan_sha256" in plan:
        digest = _state_text(plan["plan_sha256"], "runtime.planned.plan_sha256", empty=True)
        if digest and re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise RunStateError("run_state.json 字段 runtime.planned.plan_sha256 无效：需要 SHA256")
    return True


def _frozen_runtime_task(runtime: Mapping[str, object]):
    """Validate persisted values before any runtime restoration or recovery writes."""
    from acprof.host.configuration_transport import bound_inference_configuration
    from acprof.host.detect import TaskInfo

    try:
        task_snapshot = runtime["task"]
        if not isinstance(task_snapshot, dict):
            raise ValueError("runtime.task must be an object")
        task = TaskInfo(**task_snapshot)
        frozen = bound_inference_configuration(task)
        recorded = runtime.get("inference_configuration_sha256")
        if frozen:
            if recorded != frozen["identity_sha256"]:
                raise ValueError("runtime inference_configuration_sha256 differs from the frozen configuration")
            return task
        if recorded is not None:
            raise ValueError("runtime configuration identity has no frozen configuration")
    except (ValueError, TypeError, KeyError) as exc:
        raise RunStateError(f"run_state.json inference_configuration 无效：{exc}") from exc
    return None


def _validate_frozen_runtime_evidence(
    layout: ArtifactLayout, runtime: RuntimeSnapshot, artifacts: Mapping[str, str],
) -> tuple[str, str] | None:
    """Check the frozen evidence and return the verified input plan reference, if available."""
    from acprof.artifacts import read_input_scale_plan
    from acprof.host.configuration_transport import (
        verify_plan_configuration,
        verify_runtime_configuration,
    )

    task = _frozen_runtime_task(runtime)
    planned = runtime["planned"]
    if task is None and not (planned.get("plan_file") or planned.get("plan_sha256")):
        return None
    try:
        if task is not None and not (planned.get("plan_file") and planned.get("plan_sha256")):
            raise ValueError("frozen inference configuration requires its saved input plan and SHA256")
        plan_name, digest = input_plan_reference(layout, planned, artifacts)
        if not digest:
            return None
        plan = read_input_scale_plan(layout.contained(plan_name), expected_sha256=digest)
        if task is None:
            return plan_name, digest
        frozen = task.inference_configuration
        metadata = read_json_object(layout.path("static_meta.json"), label="static metadata")
        if metadata.get("inference_configuration") != frozen:
            raise ValueError("static_meta.json does not contain the frozen inference configuration")
        verify_plan_configuration(task, plan)
        validation = read_json_object(layout.path("runtime_validation.json"), label="runtime validation")
        if (validation.get("status") != "ok"
                or validation.get("inference_configuration_sha256") != frozen["identity_sha256"]):
            raise ValueError("independent runtime validation does not match the frozen configuration")
        devices = validation.get("devices")
        if not isinstance(devices, dict) or set(devices) != set(frozen["loading"]):
            raise ValueError("independent runtime validation does not cover the frozen devices")
        for device, report in devices.items():
            if not isinstance(report, dict) or report.get("status") != "ok":
                raise ValueError("frozen configuration has an unsuccessful runtime validation")
            verify_runtime_configuration(task, report, expected_device=device, require_consumers=True)
        if metadata.get("runtime_validation") != validation:
            raise ValueError("static_meta.json runtime validation differs from its independent report")
        return plan_name, digest
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RunStateError(f"冻结推理配置或输入计划证据不可恢复：{exc}") from exc


def _validate_runtime_state(layout: ArtifactLayout, runtime: object, options: object) -> None:
    from acprof.host.detect import TaskInfo
    from acprof.host.images.runtime import ImageInfo
    from acprof.host.input_plan import PlannedInputScales
    from acprof.host.runtimes.contracts import ProcessInfo

    if not isinstance(runtime, dict):
        raise RunStateError("run_state.json 字段 runtime 无效：需要冻结运行环境对象")
    if any(not isinstance(name, str) for name in runtime):
        raise RunStateError("run_state.json 字段 runtime 无效：字段名必须为字符串")
    kind = _state_text(runtime.get("kind", "docker"), "runtime.kind")
    if kind not in {"docker", "process"}:
        raise RunStateError("run_state.json 字段 runtime.kind 无效")
    for name in ("task", "image", "planned"):
        if not isinstance(runtime.get(name), dict):
            raise RunStateError(f"run_state.json 字段 runtime.{name} 无效：需要对象")
    # Use the pure dataclass shapes; restoring an image/process may have side effects.
    shapes = {"task": TaskInfo, "image": ProcessInfo if kind == "process" else ImageInfo,
              "planned": PlannedInputScales}
    for name, shape in shapes.items():
        if set(runtime[name]) - {item.name for item in fields(shape)}:
            raise RunStateError(f"run_state.json 字段 runtime.{name} 无效：包含未知构造字段")
    for name in ("model_id", "pipeline_tag", "task_family", "runtime_backend",
                 "library_name", "model_revision", "detection_method"):
        _state_text(runtime["task"].get(name), f"runtime.task.{name}", empty=True)
    task = runtime["task"]
    repository_files = task.get("repository_files", ())
    if (not isinstance(repository_files, (list, tuple)) or
            any(not isinstance(name, str) for name in repository_files)):
        raise RunStateError("run_state.json 字段 runtime.task.repository_files 无效：需要字符串序列")
    for name in ("model_config", "repository_metadata", "model_resolution", "model_spec", "inference_configuration"):
        if not isinstance(task.get(name, {}), dict):
            raise RunStateError(f"run_state.json 字段 runtime.task.{name} 无效：需要对象")
    resolution = task.get("model_resolution", {})
    for name in ("contract", "generated_spec"):
        if name in resolution and not isinstance(resolution[name], dict):
            raise RunStateError(f"run_state.json 字段 runtime.task.model_resolution.{name} 无效")
    if not isinstance(task.get("repository_metadata", {}).get("acprof_model.json", {}), dict):
        raise RunStateError("run_state.json 字段 runtime.task.repository_metadata.acprof_model.json 无效")
    pipelines = task.get("model_config", {}).get("custom_pipelines")
    if pipelines is not None and not isinstance(pipelines, dict):
        raise RunStateError("run_state.json 字段 runtime.task.model_config.custom_pipelines 无效")
    _validate_selected_tasks(runtime["task"])
    _frozen_runtime_task(runtime)
    image_fields = ("python", "code_root", "model_path", "environment_id") if kind == "process" else ("tag",)
    for name in image_fields:
        _state_text(runtime["image"].get(name), f"runtime.image.{name}")
    if not isinstance(runtime["image"].get("runtime_environment", {}), dict):
        raise RunStateError("run_state.json 字段 runtime.image.runtime_environment 无效")
    _validate_planned_snapshot(runtime["planned"], layout)
    for name in ("compute_plan", "execution_plan"):
        _state_path(layout, runtime.get(name), f"runtime.{name}", empty=True)
    _validate_measurement_options(options)


def _validate_run_state(data: object, layout: ArtifactLayout) -> TypeGuard[RunStateSnapshot]:
    """Validate durable state before any archival, projection recovery or new attempt."""
    path = layout.path(RUN_STATE_NAME)
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1):
        raise RunStateError(message("不支持的主实验状态格式：{0}", path))
    if any(not isinstance(name, str) for name in data):
        raise RunStateError("run_state.json 状态字段名必须为字符串")
    version = data.get("layout_version", 1)
    if type(version) is not int or version != layout.layout_version:
        raise RunStateError(message("恢复状态与产物布局不一致：{0}", path))
    _state_text(data.get("run_id"), "run_id")
    status = _state_text(data.get("status"), "status")
    if status not in {"preparing", "running", "complete", "interrupted", "failed"}:
        raise RunStateError("run_state.json 字段 status 无效")
    if not isinstance(data.get("options"), dict) or not isinstance(data.get("host"), dict):
        raise RunStateError(message("原实验缺少完整的参数或主机身份，无法续跑。"))
    for name in ("cases", "artifacts"):
        if not isinstance(data.get(name), dict):
            raise RunStateError(f"原实验的 case 或产物记录无效。字段：{name}")
    attempts = data.get("attempts")
    if not isinstance(attempts, list):
        raise RunStateError("run_state.json 字段 attempts 无效：需要列表")
    for index, attempt in enumerate(attempts):
        field = f"attempts[{index}]"
        if not isinstance(attempt, dict):
            raise RunStateError(f"run_state.json 字段 {field} 无效：需要对象")
        if any(not isinstance(name, str) for name in attempt):
            raise RunStateError(f"run_state.json 字段 {field} 无效：字段名必须为字符串")
        for name in ("attempt_id", "started_at"):
            _state_text(attempt.get(name), f"{field}.{name}")
        if type(attempt.get("pid")) is not int or attempt["pid"] < 1:
            raise RunStateError(f"run_state.json 字段 {field}.pid 无效：需要正整数")
        if type(attempt.get("resume")) is not bool:
            raise RunStateError(f"run_state.json 字段 {field}.resume 无效：需要布尔值")
        for name in ("ended_at", "outcome"):
            if name in attempt:
                _state_text(attempt[name], f"{field}.{name}")
        for name in ("preparation_backup", "preparation_evidence"):
            if name in attempt:
                _state_path(layout, attempt[name], f"{field}.{name}")
    if "runtime" not in data:
        if status in {"running", "complete"} or recovery_phase(layout.root, data) != "preparation":
            raise RunStateError(message("已有测量证据但缺少冻结运行环境，不能按准备失败重试。"))
    else:
        _validate_runtime_state(layout, data["runtime"], data["options"])
        planned = data["runtime"]["planned"]
        if planned.get("plan_sha256"):
            input_plan_reference(layout, planned, data["artifacts"])
    pending_cases = []
    for name, record in data["cases"].items():
        field = f"cases[{name!r}]"
        _state_path(layout, name, field)
        if layout.layout_version == 2:
            try:
                layout.case_from_csv(layout.contained(name))
            except ValueError as exc:
                raise RunStateError(f"run_state.json 字段 {field} 无效：{exc}") from exc
        pending_cases.append((record, field))
    while pending_cases:
        record, field = pending_cases.pop()
        _validate_case_state(layout, record, field)
        pending_cases.extend((item, f"{field}.previous_attempts[{index}]")
                             for index, item in enumerate(record.get("previous_attempts", [])))
    for name, checksum in data["artifacts"].items():
        field = f"artifacts[{name!r}]"
        _state_path(layout, name, field)
        if re.fullmatch(r"[a-f0-9]{64}", _state_text(checksum, field)) is None:
            raise RunStateError(f"run_state.json 字段 {field} 无效：需要 SHA256")
    return True


def load_run_state(directory: str | Path) -> RunStateSnapshot:
    try:
        layout = ArtifactLayout.discover(directory)
        path = layout.path(RUN_STATE_NAME)
        with path.open("rb") as stream:
            content = stream.read(MAX_RUN_STATE_BYTES + 1)
        if len(content) > MAX_RUN_STATE_BYTES:
            raise ValueError(message("run_state.json 超过 4 MiB 读取上限"))
        try:
            payload: object = json.loads(content.decode("utf-8"), parse_float=_finite_json_number,
                                 parse_constant=_finite_json_number)
        except json.JSONDecodeError as exc:
            raise ValueError(message("run_state.json 的 JSON 格式无效")) from exc
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        reason = exc.args[0] if exc.args and isinstance(exc.args[0], Message) else str(exc)
        raise RunStateError(message(
            "无法读取恢复状态 {0}；历史实验请使用新输出目录：{1}", directory, reason,
        )) from exc
    if _validate_run_state(payload, layout):
        return payload
    raise RunStateError("run_state.json 状态校验失败")


def _expected_plan(data: RunStateSnapshot) -> RecordedMeasurementPlan:
    return RecordedMeasurementPlan(
        options=_validate_measurement_options(data["options"]),
        scales=_validate_scales(data["runtime"]["planned"]["scales"]),
    )


def _expected(data: RunStateSnapshot, cpu=None, mem=None, gpu=None):
    plan = _expected_plan(data)
    options = plan.options
    return expected_measurements(
        (cpu,) if cpu is not None else options.cpus,
        (mem,) if mem is not None else options.mems,
        (gpu,) if gpu is not None else options.gpus,
        plan.scales, options.warmup, options.repeat,
    )


def recovery_phase(directory: str | Path, data: Mapping[str, object]) -> str:
    layout = ArtifactLayout.discover(directory)
    if data.get("status") == "complete":
        return "complete"
    has_rows = layout.result_csv.exists()
    if has_rows:
        try:
            manifest = read_json_object(layout.result_csv)
        except ValueError as exc:
            raise RunStateError(f"测量结果清单无效，不能按准备失败重试：{exc}") from exc
        has_rows = manifest.get("schema_version") != 3 or manifest.get("row_count", 0) > 0
    if (data.get("runtime") or data.get("cases") or has_rows
            or any(layout.root.glob("result_case_*.csv"))
            or any((layout.root / ".acprof/work/cases").glob("*/result.csv"))):
        return "measurement"
    return "preparation"


def validate_resume(directory: str | Path, data: object, *, project_dir: str | Path,
                    options: dict | None = None, repair: bool = False) -> str:
    """Read-only checks shared by recovery review and the locked writer.

    Image availability and live hardware preflight remain the runner's responsibility.
    Returning a phase never grants permission to mix changed measurement identities.
    """
    layout = ArtifactLayout.discover(directory)
    if not _validate_run_state(data, layout):
        raise RunStateError("run_state.json 状态校验失败")
    previous_options, previous_host = data["options"], data["host"]
    if not isinstance(previous_options, dict) or not isinstance(previous_host, dict):
        raise RunStateError(message("原实验缺少完整的参数或主机身份，无法续跑。"))
    issues = []
    if options is not None:
        previous = {"profiling_mode": "full", **flatten_run_options(previous_options)}
        current = {"profiling_mode": "full", **flatten_run_options(options)}
        changed = sorted(name for name in previous.keys() | current.keys()
                         if (name in previous) != (name in current) or previous.get(name) != current.get(name))
        if changed:
            issues.append(message("实验参数已变化：{0}", ", ".join(changed)))
    current_host = host_identity(project_dir)
    labels = {"source_sha256": "AC-Prof 采集源码", "packages_sha256": "Python 依赖",
              "python": "Python 版本"}
    for name in sorted(previous_host.keys() | current_host.keys()):
        if previous_host.get(name) != current_host.get(name):
            issues.append(message("{0}已变化", message(labels.get(name, "主机身份（{0}）"), name)))
    if issues:
        raise RunStateError(message("无法续跑；请使用新实验保留原结果。\n{0}", join_messages("\n", issues)))
    verified_plan = None
    if "runtime" in data:
        from acprof.host.detect import TaskInfo
        from acprof.tasks.interfaces import validate_execution_identity
        task_snapshot = data["runtime"]["task"]
        if not isinstance(task_snapshot, dict):
            raise RunStateError("run_state.json 字段 runtime.task 无效：需要对象")
        try:
            validate_execution_identity(TaskInfo(**task_snapshot), restoring=True)
        except ValueError as exc:
            raise RunStateError(f"run_state.json 字段 runtime.task.selected_tasks 无效：{exc}") from exc
        verified_plan = _validate_frozen_runtime_evidence(layout, data["runtime"], data["artifacts"])
    phase = recovery_phase(directory, data)
    complete = phase == "complete"
    if repair:
        # This branch is used only by the locked runner, after run identity checks.
        from acprof.host.measurement.window_recovery import repair_result_projections
        repair_result_projections(layout, data)
    else:
        from acprof.host.measurement.window_recovery import check_result_identity
        check_result_identity(layout, data["run_id"])
    # Completed CSVs may have documented posthoc changes; preserve that contract.
    if complete:
        read_result_csv(layout.result_csv, expected=_expected_plan(data))
    for name, digest in data.get("artifacts", {}).items():
        if verified_plan == (name, digest):
            continue
        if complete and Path(name).name not in {"matrix_plan.json", "startup_oom_pruning.json"}:
            continue
        path = layout.contained(name)
        if not path.is_file() or file_sha256(path) != digest:
            raise RunStateError(message("恢复产物缺失或已改变：{0}；请恢复原文件或创建新实验。", path))
    if not complete:
        for name, record in data.get("cases", {}).items():
            if record.get("status") == "complete":
                path = layout.contained(name)
                if not path.is_file() or file_sha256(path) != record.get("sha256"):
                    raise RunStateError(message("已完成 case 的 CSV 缺失或改变：{0}", path))
            elif record.get("measurement_status") == "complete":
                from acprof.host.measurement.checkpoint import read_measurement_checkpoint
                case = layout.case_from_csv(layout.contained(name))
                cpu, mem, gpu = record["resources"]
                try:
                    read_measurement_checkpoint(case, record["measurement_checkpoint"],
                                                run_id=data["run_id"], attempt_id=record["case_attempt_id"],
                                                expected=_expected(data, cpu, mem, gpu))
                except (OSError, ValueError, KeyError) as exc:
                    raise RunStateError(f"已完成测量的原始证据无效：{case.case_id}: {exc}") from exc
    return phase


class ResultDirectoryLock:
    """Advisory directory lock released on process exit, including Windows static exports."""
    def __init__(self, directory: str | Path, *, new: bool = False):
        layout = ArtifactLayout.for_new_run(directory) if new else ArtifactLayout.discover(directory)
        self.path = layout.path(RESULT_LOCK_NAME)
        self._resources: ExitStack | None = None

    def _acquire_file(self, resources: ExitStack, *, label: str) -> int:
        if sys.platform == "win32":
            import msvcrt

            # Create exclusively, then open an existing entry without truncation.
            try:
                stream = self.path.open("x+b")
            except FileExistsError:
                stream = self.path.open("r+b")
            resources.enter_context(stream)
            descriptor = stream.fileno()
            info = os.fstat(descriptor)
            entry = self.path.lstat()
            if (not stat.S_ISREG(entry.st_mode) or info.st_nlink != 1 or not os.path.samestat(info, entry)
                    or getattr(entry, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise OSError(f"{label} lock is not a single regular file")
            stream.seek(0)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            resources.enter_context(cleanup_on_exit(lambda: msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1),
                                                    description=f"{label} lock {self.path}"))
            return descriptor
        import fcntl

        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        resources.enter_context(cleanup_on_exit(lambda: os.close(descriptor),
                                                description=f"{label} lock descriptor {self.path}"))
        _validate_lock_descriptor(descriptor, label=label)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor

    def __enter__(self):
        if self._resources is not None:
            raise RunStateError("结果目录锁已由此对象持有")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with ExitStack() as resources:
                self._acquire_file(resources, label="result")
                self._resources = resources.pop_all()
        except OSError as exc:
            raise RunStateError(
                "无法取得结果目录锁；另一个采集或补采进程可能正在使用结果目录，"
                f"或锁文件不可用：{self.path} ({format_exception_notes(exc)})"
            ) from exc
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        resources, self._resources = self._resources, None
        if resources is not None:
            return resources.__exit__(exc_type, exc_value, traceback)
        return False


class MeasurementLock(ResultDirectoryLock):
    """同机同用户的实验串行化，防止跨结果目录共享端口及采样资源。"""
    def __init__(self):
        self.path = MEASUREMENT_LOCK_ROOT / f"acprof-measurement-{os.getuid()}.lock"
        self._resources = None
        self._descriptors: tuple[int, ...] = ()
        self._owner: tuple[int, Thread] | None = None

    @property
    def inherited_descriptors(self) -> tuple[int, ...]:
        """Transfer both guards to an independent child via Popen(pass_fds=...)."""
        if self._resources is None:
            raise RunStateError("Measurement lock must be held before descriptor transfer")
        return self._descriptors

    def __enter__(self):
        if self._resources is not None:
            raise RunStateError("本机测量锁已由此对象持有")
        try:
            with ExitStack() as resources:
                guard = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                resources.enter_context(cleanup_on_exit(guard.close, description="measurement kernel guard"))
                guard.bind(b"\0" + MEASUREMENT_LOCK_NAMESPACE + b"." + str(os.getuid()).encode("ascii"))
                descriptor = self._acquire_file(resources, label="measurement")
                opened = os.fstat(descriptor)
                named = self.path.stat(follow_symlinks=False)
                if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                    raise OSError("measurement lock pathname changed during acquisition")
                self._descriptors = (guard.fileno(), descriptor)
                self._owner = (os.getpid(), current_thread())
                self._resources = resources.pop_all()
        except OSError as error:
            raise RunStateError(
                "无法取得本机测量锁；同一用户可能已有 AC-Prof 采集或补采运行，"
                f"或锁文件不可用：{self.path} ({format_exception_notes(error)})"
            ) from error
        return self


@contextmanager
def measurement_lock_scope(held: MeasurementLock | None = None) -> Iterator[MeasurementLock]:
    """Acquire a lock or borrow one from the same synchronous call chain."""
    if held is None:
        with MeasurementLock() as acquired:
            yield acquired
    else:
        if (not isinstance(held, MeasurementLock) or held._resources is None
                or held._owner != (os.getpid(), current_thread())):
            raise RunStateError("Cannot borrow a measurement lock outside its active owner scope")
        yield held


class _StateFactoryContext(local):
    binding: tuple[Path, MeasurementLock] | None = None


_state_factory_context = _StateFactoryContext()


@contextmanager
def state_factory_measurement_scope(directory: str | Path, held: MeasurementLock | None) -> Iterator[None]:
    """Lend this synchronous factory the collection lock for its requested result directory."""
    previous = _state_factory_context.binding
    _state_factory_context.binding = (Path(directory).resolve(), held) if held is not None else None
    try:
        yield
    finally:
        _state_factory_context.binding = previous


def offline_analysis_lock() -> AbstractContextManager:
    """Exclude collection for offline work on Linux, including WSL and containers."""
    return MeasurementLock() if is_linux() else nullcontext()


class RunState:
    data: RunStateSnapshot

    def __init__(self, directory: str | Path, options: dict, *, resume: bool, project_dir: str,
                 preparation_artifacts: dict[str, str] | None = None):
        self.directory = Path(directory).resolve()
        self._matrix_publication: tuple[tuple[int, Thread], ResultWriter, ExitStack] | None = None
        self.layout = ArtifactLayout.discover(self.directory) if resume else ArtifactLayout.for_new_run(self.directory)
        self.path = self.layout.path(RUN_STATE_NAME)
        self.lock = ResultDirectoryLock(self.directory, new=not resume)
        binding = _state_factory_context.binding
        held = binding[1] if binding is not None and binding[0] == self.directory else None
        with ExitStack() as locks:
            self.measurement_lock = locks.enter_context(measurement_lock_scope(held))
            locks.enter_context(self.lock)
            attempt: AttemptState = {"attempt_id": uuid4().hex, "started_at": utc_now(),
                                     "pid": os.getpid(), "resume": resume}
            if resume:
                self.data = load_run_state(self.directory)
                phase = validate_resume(self.directory, self.data, project_dir=project_dir, options=options, repair=True)
                if options.get("runtime") == "process":
                    from acprof.host.runtimes.process_control import verify_process_cleanup
                    verify_process_cleanup(self.layout.path("logs"))
                if phase == "preparation":
                    attempt["preparation_backup"] = self._archive_preparation()
                    self.data["status"] = "preparing"
            else:
                prepared = preparation_artifacts or {}
                allowed_prepared = {str(self.layout.path(name).relative_to(self.directory))
                                    for name in ("auto_report.json", "model_resolution.json")}
                if set(prepared) - allowed_prepared:
                    raise RunStateError("自动准备产物包含不允许的文件")
                for name, digest in prepared.items():
                    path = self.layout.contained(name)
                    if not path.is_file() or file_sha256(path) != digest:
                        raise RunStateError(f"自动准备产物缺失或已改变：{path}")
                try:
                    self.layout.initialize(allowed_files=prepared)
                except ValueError as exc:
                    raise RunStateError(str(exc)) from exc
                result_identity: object = read_json_object(self.layout.result_csv)["run_id"]
                self.data = {"schema_version": 1, "layout_version": 2,
                             "run_id": _state_text(result_identity, "run_id"), "created_at": utc_now(),
                             "status": "preparing", "options": options, "host": host_identity(project_dir),
                             "cases": {}, "artifacts": {}, "attempts": []}
            self.data["attempts"].append(attempt)
            self.save()
            self._locks = locks.pop_all()

    @property
    def ready(self) -> bool:
        return "runtime" in self.data

    @property
    def complete(self) -> bool:
        return self.data.get("status") == "complete"

    @property
    def run_id(self) -> str:
        return self.data["run_id"]

    @property
    def has_cases(self) -> bool:
        return bool(self.data.get("cases"))

    def case_paths(self) -> tuple[Path, ...]:
        return tuple(self.artifact_path(name) for name in self.data["cases"])

    def case_record(self, path: str) -> CaseState:
        key = str(Path(path).relative_to(self.directory))
        return self.data["cases"][key].copy()

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        atomic_write_json(self.path, self.data)

    def preparation_evidence(self):
        """Register this attempt without changing the frozen measurement artifacts."""
        from acprof.host.preparation.preparation_evidence import PreparationEvidence
        attempt = self.data["attempts"][-1]
        evidence = PreparationEvidence(self.layout, attempt["attempt_id"])
        attempt["preparation_evidence"] = evidence.relative_summary_path
        self.save()
        return evidence

    def artifact_path(self, relative: str) -> Path:
        try:
            return self.layout.contained(relative)
        except ValueError as exc:
            raise RunStateError(str(exc)) from exc

    def _archive_preparation(self) -> str:
        """Copy preparation evidence under the writer lock before any retry overwrites it."""
        names = ("run_state.json", "auto_report.json", "model_resolution.json",
                 "interface_validation.json", "runtime_validation.json", "static_meta.json",
                 "capability_report.json", "collection_history.json", "input_scale_plan.json",
                 "compute_profile_plan.json", "execution_profile_plan.json")
        sources = [self.layout.path(name) for name in names]
        for name in ("logs", "compute_profiles", "execution_profiles"):
            sources.extend(self.layout.path(name).rglob("*"))
        backup = self.layout.path("interrupted_cases").parent / "preparation_attempts" / uuid4().hex
        backup_directories: set[Path] = set()
        for source in sources:
            source = self.artifact_path(str(source.relative_to(self.directory)))
            if not source.exists():
                continue
            if source.is_dir():
                continue
            if not source.is_file():
                raise RunStateError(message("不支持的准备产物类型：{0}", source))
            target = backup / source.relative_to(self.directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            backup_directories.add(target.parent)
        for directory in sorted(backup_directories, key=lambda path: len(path.parts), reverse=True):
            sync_directory_chain(directory, self.directory)
        return str(backup.relative_to(self.directory))

    def bind_matrix_plan(self, plan: dict) -> None:
        from acprof.host.matrix_plan import MATRIX_PLAN_NAME
        from acprof.host.startup_probe import PROBE_NAME
        previous = self.data.get("matrix_plan_sha256")
        if previous is not None and previous != plan["plan_sha256"]:
            raise RunStateError("冻结 matrix plan 身份已改变")
        names = [MATRIX_PLAN_NAME]
        if plan["identity"]["prune_startup_oom"]:
            names.append(PROBE_NAME)
        for name in names:
            name = str(self.layout.path(name).relative_to(self.directory))
            checksum = file_sha256(self.artifact_path(name))
            recorded = self.data["artifacts"].get(name)
            if recorded is not None and recorded != checksum:
                raise RunStateError(f"冻结计划或 probe 证据已改变：{name}")
            self.data["artifacts"][name] = checksum
        self.data["matrix_plan_sha256"] = plan["plan_sha256"]
        self.save()

    def bind_runtime(self, task, image, planned, compute_plan: str, execution_plan: str) -> None:
        from acprof.host.configuration_transport import bound_inference_configuration

        frozen = bound_inference_configuration(task)
        plan = asdict(planned)
        if plan.get("plan_file"):
            plan["plan_file"] = str(Path(plan["plan_file"]).resolve().relative_to(self.directory))
        if not _validate_planned_snapshot(plan, self.layout):
            raise RunStateError("run_state.json 输入计划校验失败")
        from acprof.host.runtimes.contracts import runtime_kind
        snapshot: RuntimeSnapshot = {
            "kind": runtime_kind(image),
            "task": {key: value for key, value in asdict(task).items() if key != "repository_sources"},
            "image": asdict(image), "planned": plan,
            "compute_plan": str(Path(compute_plan).resolve().relative_to(self.directory)) if compute_plan else "",
            "execution_plan": str(Path(execution_plan).resolve().relative_to(self.directory)) if execution_plan else "",
        }
        if frozen:
            snapshot["inference_configuration_sha256"] = frozen["identity_sha256"]
        verified_plan = _validate_frozen_runtime_evidence(self.layout, snapshot, self.data["artifacts"])
        paths = [str(self.layout.path(name).relative_to(self.directory))
                 for name in ("static_meta.json", "collection_history.json")]
        paths.extend(name for name in (snapshot["compute_plan"], snapshot["execution_plan"]) if name)
        artifacts = {}
        if verified_plan:
            plan_name, digest = verified_plan
            artifacts[plan_name] = digest
        else:
            paths.append(plan.get("plan_file") or str(self.layout.path("input_scale_plan.json").relative_to(self.directory)))
        for name in paths:
            path = self.artifact_path(name)
            if path.is_file():
                artifacts[name] = file_sha256(path)
        self.data["artifacts"].update(artifacts)
        self.data["runtime"] = snapshot
        self.data["status"] = "running"
        self.save()

    def restore_runtime(self):
        from acprof.host.detect import TaskInfo
        from acprof.host.input_plan import PlannedInputScales
        from acprof.host.runtimes.service import execution_runtime
        from acprof.tasks.interfaces import validate_execution_identity
        snapshot = self.data["runtime"]
        task_snapshot = snapshot["task"]
        if not isinstance(task_snapshot, dict):
            raise RunStateError("run_state.json 字段 runtime.task 无效：需要对象")
        task = TaskInfo(**task_snapshot)
        # Reject changed interface identity before touching the saved runtime artifact.
        validate_execution_identity(task, restoring=True)
        verified_plan = _validate_frozen_runtime_evidence(self.layout, snapshot, self.data["artifacts"])
        image_snapshot = snapshot["image"]
        if not isinstance(image_snapshot, dict):
            raise RunStateError("run_state.json 字段 runtime.image 无效：需要对象")
        image = execution_runtime(snapshot.get("kind", "docker")).restore(
            image_snapshot, output_dir=self.directory,
        )
        plan = snapshot["planned"]
        plan_file, digest = verified_plan or (plan.get("plan_file"), plan.get("plan_sha256", ""))
        if plan_file:
            plan_file = str(self.artifact_path(plan_file))
        workload = plan.get("workload", {})
        if not isinstance(workload, dict):
            raise RunStateError("run_state.json 字段 runtime.planned.workload 无效")
        restored_plan = PlannedInputScales(
            scales=plan["scales"], source=plan["source"], plan_file=plan_file,
            workload=workload, plan_sha256=digest,
        )
        return (task, image, restored_plan,
                str(self.artifact_path(snapshot["compute_plan"])) if snapshot["compute_plan"] else "",
                str(self.artifact_path(snapshot["execution_plan"])) if snapshot["execution_plan"] else "")

    def expected(self, cpu=None, mem=None, gpu=None):
        return _expected(self.data, cpu, mem, gpu)

    def expected_plan(self) -> RecordedMeasurementPlan:
        """Return the recorded axes and window counts for whole-run validation."""
        return _expected_plan(self.data)

    def prepare_case(self, filename: str, cpu: int | None, mem: int | None, gpu: str) -> str | None:
        from acprof.host.case_recovery import prepare_case
        return prepare_case(self, filename, cpu, mem, gpu)

    def case_attempt_id(self, path: str) -> str:
        record = self.data["cases"][str(Path(path).relative_to(self.directory))]
        return record["case_attempt_id"]

    def bind_case_container(self, path: str, container_id: str) -> None:
        """Record cleanup ownership before acquisition, including crash recovery."""
        if not re.fullmatch(r"[a-f0-9]{64}", container_id):
            raise RunStateError("Case container must have an immutable Docker ID")
        record = self.data["cases"][str(Path(path).relative_to(self.directory))]
        record.update({"cleanup_container_id": container_id, "cleanup_status": "pending"})
        self.save()

    def bind_case_process(self, path: str, evidence: dict) -> None:
        record = self.data["cases"][str(Path(path).relative_to(self.directory))]
        record.update({"cleanup_process": evidence, "cleanup_status": "pending"})
        self.save()

    def checkpoint_measurement(self, path: str, cpu: int | None, mem: int | None, gpu: str, *, processing: dict,
                               container_id: str) -> None:
        from acprof.host.measurement.checkpoint import publish_measurement_checkpoint
        key = str(Path(path).relative_to(self.directory))
        record = self.data["cases"][key]
        case = self.layout.case_from_csv(path)
        from acprof.host.measurement.window_recovery import archived_captures
        reference = publish_measurement_checkpoint(
            case, run_id=self.data["run_id"],
            attempt_id=record["case_attempt_id"], processing=processing,
            capture_segments=archived_captures(case, record),
            expected=self.expected(cpu, mem, gpu),
        )
        record.update({"status": "measured", "measurement_status": "complete", "measured_at": utc_now(),
                       "processing_status": "pending", "measurement_checkpoint": reference,
                       "cleanup_status": "pending", "cleanup_container_id": container_id})
        self.save()

    def processing_checkpoint(self, path: str, cpu: int | None, mem: int | None, gpu: str) -> dict | None:
        from acprof.host.measurement.checkpoint import read_measurement_checkpoint
        record = self.data["cases"].get(str(Path(path).relative_to(self.directory)))
        if record is None or record.get("measurement_status") != "complete":
            return None
        try:
            checkpoint = read_measurement_checkpoint(
                self.layout.case_from_csv(path), record["measurement_checkpoint"],
                run_id=self.data["run_id"], attempt_id=record["case_attempt_id"],
                expected=self.expected(cpu, mem, gpu),
            )
        except (OSError, ValueError, KeyError) as exc:
            raise RunStateError(f"已完成测量的原始证据无效：{exc}") from exc
        return checkpoint

    def recover_processing_checkpoint(
        self, path: str, cpu: int | None, mem: int | None, gpu: str,
    ) -> dict | None:
        from acprof.host.case_recovery import recover_processing_checkpoint
        return recover_processing_checkpoint(self, path, cpu, mem, gpu)

    def record_case_processing(self, path: str, status: str) -> None:
        if status not in {"running", "failed", "complete"}:
            raise ValueError(f"Invalid case processing status: {status}")
        record = self.data["cases"][str(Path(path).relative_to(self.directory))]
        record["processing_status"] = status
        if status == "running":
            record["processing_attempts"] = record.get("processing_attempts", 0) + 1
        self.save()
        if status == "failed":
            from acprof.host.measurement.window_recovery import mark_packet_processing_failed
            mark_packet_processing_failed(self.layout, self.layout.case_from_csv(path).case_id)

    def record_case_cleanup(self, path: str, status: str) -> None:
        if status not in {"complete", "failed"}:
            raise ValueError(f"Invalid case cleanup status: {status}")
        record = self.data["cases"][str(Path(path).relative_to(self.directory))]
        record["cleanup_status"] = status
        self.save()

    def finish_case(self, path: str, cpu: int | None, mem: int | None, gpu: str) -> None:
        from acprof.host.case_recovery import publish_case_result
        from acprof.host.measurement.window_recovery import is_incremental

        case = self.layout.case_from_csv(path)
        key = str(case.csv.relative_to(self.directory))
        record = self.data["cases"].get(key)
        if record is None:
            record = {"status": "complete"}
        expected = self.expected(cpu, mem, gpu)
        if is_incremental(self.layout):
            evidence = []
            checkpoint_reference = record.get("measurement_checkpoint")
            if checkpoint_reference:
                if not isinstance(checkpoint_reference, dict):
                    raise RunStateError("run_state.json 字段 measurement_checkpoint 无效：需要产物引用")
                evidence.append({name: checkpoint_reference[name] for name in ("path", "sha256")})
            snapshot = publish_case_result(case, expected, run_id=self.run_id, evidence=evidence)
            record["case_result"] = {"path": snapshot.path.relative_to(self.directory).as_posix(),
                                     "sha256": snapshot.sha256}
        else:
            snapshot = read_result_csv_snapshot(case.csv, expected=expected)
            case.retain_requests()
        self.data["cases"][key] = {
            **record, "processing_status": "complete",
            "measurement_status": "complete" if record.get("measurement_status") == "complete" else "not_measured",
            "status": "complete", "completed_at": utc_now(), "sha256": snapshot.sha256,
            "row_count": len(snapshot.rows), "error_rows": sum(row.get("status") == "error" for row in snapshot.rows),
        }
        self.save()

    @contextmanager
    def merge_cases(self, paths: list[str], final_csv: str) -> Iterator[int]:
        """Keep one verified writer for consolidation, capability checks and freeze."""
        if Path(final_csv).absolute() != self.layout.result_csv:
            raise RunStateError("最终结果不属于正在提交的实验目录")
        case_sha256s = {}
        for path in paths:
            csv_path = self.layout.case_from_csv(path).csv
            record = self.data["cases"].get(str(csv_path.relative_to(self.directory)))
            if record is not None and record["status"] == "complete":
                case_sha256s[csv_path] = _state_text(record.get("sha256"), "case.sha256")
        with ExitStack() as lease:
            count, writer = publish_merged_cases(paths, final_csv, scope=lease, expected=self.expected_plan(),
                                                 expected_run_id=self.run_id, case_sha256s=case_sha256s)
            if writer is not None:
                self._matrix_publication = (os.getpid(), current_thread()), writer, lease
            try:
                yield count
            finally:
                self._matrix_publication = None

    def finish(self, final_csv: str) -> None:
        from acprof.host.measurement.window_recovery import is_incremental
        if self._matrix_publication is not None and self._matrix_publication[0] == (os.getpid(), current_thread()):
            from acprof.result_store.sources import verify_source_files
            _, rows = read_result_csv(final_csv, expected=self.expected_plan())
            _, writer, lease = self._matrix_publication
            if writer.layout.result_csv != Path(final_csv).absolute() or writer.expected_run_id != self.run_id:
                raise RunStateError("最终结果与正在提交的实验身份不一致")
            verify_source_files(writer.layout, writer.manifest, writer=writer)
            writer.freeze()
            digest = file_sha256(writer.layout.result_csv)
            lease.close()
            self._matrix_publication = None
        else:
            snapshot = read_result_csv_snapshot(final_csv, expected=self.expected_plan())
            rows, digest = snapshot.rows, snapshot.sha256
            if is_incremental(self.layout):
                from acprof.result_store.writer import ResultWriter
                if Path(final_csv).absolute() != self.layout.result_csv:
                    raise RunStateError("最终结果不属于正在提交的实验目录")
                with ResultWriter(self.directory, expected_run_id=self.run_id) as writer:
                    if file_sha256(writer.layout.result_csv) != digest:
                        raise ResultValidationError(f"result source changed: {final_csv}")
                    writer.freeze()
                    digest = file_sha256(writer.layout.result_csv)
        self.data.update({"status": "complete", "completed_at": utc_now(), "result_sha256": digest,
                          "row_count": len(rows), "outcome": "partial" if any(row.get("status") == "error" for row in rows) else "ok"})
        self.save()

    def close(self, outcome: str = "interrupted") -> None:
        with cleanup_on_exit(self._locks.close, description=f"run locks {self.directory}"):
            if not self.complete:
                self.data["status"] = outcome
            self.data["attempts"][-1]["ended_at"] = utc_now()
            self.data["attempts"][-1]["outcome"] = self.data["status"]
            self.save()
