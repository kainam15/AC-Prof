"""任务 IO 与精度由扩展声明产生；返回副本供静态元数据补充。"""
from __future__ import annotations

from typing import Any

from acprof.extensions import CATALOG, UnsupportedExtensionError
from acprof.host.detect import TaskInfo


def _model_io_formats(task_info: TaskInfo) -> tuple[dict[str, Any], dict[str, Any]]:
    formats = CATALOG.describe(task_info).io_format
    if not {"input", "output"} <= formats.keys():
        raise UnsupportedExtensionError(f"IO format is not declared for {task_info.pipeline_tag}")
    return formats["input"], formats["output"]


def _inference_precision_by_device(task_info: TaskInfo) -> dict[str, str]:
    policy = task_info.model_resolution.get("precision", {})
    if policy:
        return {"cpu" if device in {"cpu", "off"} else "gpu": value["dtype"] for device, value in policy.items()}
    if task_info.runtime_backend in {"onnxruntime", "torchscript", "skops"}:
        return CATALOG.describe(task_info).precision
    from acprof.failures import RuntimeFailure
    from acprof.precision import resolve_precision
    from acprof.runtime_profiles import _transformers_version, select_runtime_profile
    profile = select_runtime_profile(task_info)
    extension = CATALOG.describe(task_info).declaration
    values = {}
    for device in ("cpu", "gpu"):
        try:
            values[device] = resolve_precision(profile, extension, device=device, task=task_info.pipeline_tag,
                model_type=task_info.model_config.get("model_type", ""), version=_transformers_version(profile.environment) or "")["dtype"]
        except RuntimeFailure:
            values[device] = "unsupported"
    return values
