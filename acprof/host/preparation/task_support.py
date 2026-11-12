"""Check known collection gaps before image preparation or measurement.

Task detection identifies a model; it does not establish collection support.
This check uses local metadata only and does not load models or runtime libraries.
"""

from __future__ import annotations

from dataclasses import replace

from acprof.extensions import CATALOG, select_extension
from acprof.failures import Failure, RuntimeFailure
from acprof.host.detect import TaskInfo


class TaskSupportError(RuntimeFailure):
    """Collection preflight failed before image preparation or measurement."""

    def __init__(self, failure: Failure | str):
        super().__init__(failure if isinstance(failure, Failure) else
                         Failure("preflight", "runtime_task_unsupported", failure))


def _read_dependency_source(task_info: TaskInfo, name: str) -> str:
    from acprof.host.detect import read_model_source

    if name not in task_info.repository_sources:
        task_info.repository_sources[name] = read_model_source(
            task_info.model_id, name, task_info.model_revision,
        )
    return task_info.repository_sources[name]


def require_task_support(task_info: TaskInfo, *, batch_size: int = 1, devices=()) -> None:
    """Record one actionable static decision without claiming inference success."""
    from acprof.models.inspection.decision import (
        clear_preflight_failure,
        record_resolution_failure,
    )

    clear_preflight_failure(task_info)
    upstream_failure = task_info.model_resolution.get("failure")
    input_status = task_info.model_resolution.get("status")
    if not upstream_failure:
        task_info.model_resolution.pop("compatibility", None)
        task_info.model_resolution.pop("precision", None)
    try:
        _require_task_support(task_info, batch_size=batch_size, devices=devices)
    except RuntimeFailure as exc:
        failure = exc.failure
        if not upstream_failure:
            failure = replace(failure, evidence={**failure.evidence, "preflight_input_status": input_status})
        failure = record_resolution_failure(task_info, failure)
        raise TaskSupportError(failure) from exc


def _require_task_support(task_info: TaskInfo, *, batch_size: int, devices) -> None:
    """Reject known gaps and invalid task routing, without loading a checkpoint."""
    task_info.model_resolution.update(model_id=task_info.model_id, model_revision=task_info.model_revision)
    if task_info.model_resolution.get("failure"):
        raise TaskSupportError(Failure(**task_info.model_resolution["failure"]))
    if task_info.metadata_errors:
        raise TaskSupportError(Failure("resolution", "metadata_invalid", "\n".join([
            f"[model-metadata][ERROR] Cannot resolve model metadata: {task_info.model_id}",
            "  元数据读取失败，尚不能判断模型接口是否兼容。",
            "  原因：" + "; ".join(task_info.metadata_errors),
            "  请检查 HF_ENDPOINT、网络、仓库访问权限及 JSON 内容后重试。",
            "  本次未进入镜像准备、资源矩阵或推理测量。",
        ]), retryability="after_configuration", evidence={"metadata_errors": list(task_info.metadata_errors)}))
    from acprof.models.resolution import require_resolved_candidate
    from acprof.models.spec import custom_code_files
    try:
        require_resolved_candidate(task_info)
        custom_code_files(task_info.model_config or {})
    except RuntimeFailure as exc:
        raise TaskSupportError(exc.failure) from exc
    except ValueError as exc:
        status = task_info.model_resolution.get("status")
        reason_code = {
            "selection_required": "task_selection_required",
            "ambiguous": "evidence_conflict",
        }.get(status, "model_contract_required")
        if task_info.model_resolution.get("semantics", {}).get("status") == "conflict":
            reason_code = "evidence_conflict"
        raise TaskSupportError(Failure(
            "resolution", reason_code, f"[model-resolution][ERROR] {exc}\n  未进入镜像准备或正式测量。",
            retryability="after_configuration", evidence={"task": task_info.pipeline_tag},
        )) from exc
    task = task_info.pipeline_tag
    expected_family = CATALOG.task_families.get(task)
    reason = None
    if expected_family is None:
        reason = "当前项目尚未登记该任务类型的采集适配。"
    if reason is None and task_info.task_family != expected_family:
        reason = (
            f"任务 {task} 对应任务族 {expected_family}，"
            f"当前选择的任务族 {task_info.task_family} 与之不匹配。"
        )
    if reason is None:
        try:
            extension = select_extension(task_info)
            if batch_size != 1 and extension.execution.get("batch") == "unsupported":
                reason = f"{expected_family} 采集每个请求使用一个样本；请设置 --batch-size 1。"
        except RuntimeFailure as exc:
            task_info.model_resolution["failure"] = exc.failure.to_dict()
            raise TaskSupportError(exc.failure) from exc
        except ValueError as exc:
            reason = str(exc)
    if reason is None:
        from acprof.models.resolution import resolve_model_interface
        from acprof.runtime.profiles import select_runtime_profile

        try:
            task_info.model_resolution = resolve_model_interface(task_info)
            profile = select_runtime_profile(task_info)
            task_info.runtime_profile_id, task_info.model_adapter = profile.profile_id, profile.adapter
            task_info.model_resolution["runtime_profile"] = profile.profile_id
            from acprof.runtime.dependencies import dependency_preflight
            dependency_preflight(
                task_info, profile, read_source=lambda name: _read_dependency_source(task_info, name),
            )
            from acprof.models.support import supports_transformers_task
            from acprof.precision import remote_code_allowed, resolve_precision
            from acprof.runtime.profiles import locked_transformers_version
            extension = select_extension(task_info)
            version = locked_transformers_version(profile.environment) or ""
            config = task_info.model_config or {}
            remote_required = config.get("custom_pipelines") or (config.get("auto_map") and version and
                supports_transformers_task(version, task, config.get("model_type", ""), config) is not True)
            if remote_required and not remote_code_allowed(profile, extension):
                raise RuntimeFailure(Failure("preflight", "remote_code_disallowed",
                    "Model requires remote code but its registered runtime/extension policy forbids it",
                    runtime_profile=profile.profile_id, retryability="after_configuration",
                    evidence={"trust_remote_code": False, "code_files": task_info.model_resolution.get("code_files", [])}))
            task_info.model_resolution["trust_remote_code"] = remote_code_allowed(profile, extension)
            task_info.model_resolution["requested_devices"] = list(devices)
            native_backend = task_info.runtime_backend in {"transformers_model", "transformers_pipeline"}
            custom = bool(config.get("auto_map") or config.get("custom_pipelines"))
            support = (supports_transformers_task(version, task, config.get("model_type", ""), config)
                       if native_backend and version and not custom else None)
            declared = not native_backend or extension.adapter != "family-default"
            interface_status = "supported" if declared or support is True else "unsupported" if support is False else "unknown"
            layout_status = task_info.model_resolution.get("artifact_layout", {}).get("status", "unknown")
            overall_status = interface_status
            if interface_status == "supported" and layout_status == "unknown":
                overall_status = "unknown"
            task_info.model_resolution["compatibility"] = {
                "status": overall_status,
                "interface_status": interface_status,
                "artifact_status": layout_status,
                "evidence_level": "static",
                "source": "registered_extension" if declared else "locked_transformers_mapping",
                "runtime_profile": profile.profile_id,
                "transformers_version": version or None,
                "checkpoint_verified": False,
            }
            if overall_status == "unknown":
                layout = task_info.model_resolution.get("artifact_layout", {})
                task_info.model_resolution["compatibility"].update(
                    detail=layout.get("detail") if layout_status == "unknown" else
                    "Architecture support is not established by the registered runtime metadata.",
                    next_action="Review the pinned artifacts and registered backend capability; "
                    "static evidence does not replace a separately authorized interface probe.",
                )
            task_info.model_resolution["precision"] = {
                device: resolve_precision(profile, extension, device=device, task=task,
                    model_type=config.get("model_type", ""), version=version)
                for device in devices
            }
        except RuntimeFailure as exc:
            task_info.model_resolution["failure"] = exc.failure.to_dict()
            if exc.failure.reason_code == "model_contract_required":
                task_info.model_resolution["status"] = "needs_configuration"
            raise TaskSupportError(exc.failure) from exc
        except ValueError as exc:
            reason = str(exc)
        else:
            return

    lines = [
        f"[task-support][ERROR] Unsupported collection task: {task}",
        "  当前项目暂不支持此任务或所选任务族的采集，已停止本次任务。",
        f"  模型：{task_info.model_id}",
        f"  任务：{task}；任务族：{task_info.task_family}；后端：{task_info.runtime_backend}",
        f"  原因：{reason}",
        "  解决办法：",
        "    1. 如需立即采集，请选择已适配任务的模型；支持范围见 README.md。",
        "    2. 如需继续使用此任务，请等待支持该任务的项目版本，"
        "或按 README.md 的扩展说明补齐输入、推理、输出与指标适配并验证。",
        "    3. 若任务识别有误，请核对模型页后用 --task / --task-family / --backend 纠正；"
        "TUI 可在高级配置的“识别覆盖”中设置。仅在模型确实支持目标任务时使用覆盖。",
    ]
    lines.append("  本次未进入资源矩阵或推理测量，不会生成新的测量 CSV；已有测量结果保留。")
    failure = Failure("preflight", "runtime_task_unsupported", "\n".join(lines),
                      runtime_profile=task_info.runtime_profile_id,
                      evidence={"task": task})
    task_info.model_resolution["failure"] = failure.to_dict()
    raise TaskSupportError(failure)
