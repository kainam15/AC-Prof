"""Pure model file path and checkpoint index validation shared by Host and containers."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import PurePosixPath


class ModelFilesError(ValueError):
    """模型文件声明不完整或不合法；重试下载不能修复这个错误。"""


def safe_path(name: str) -> str:
    if not isinstance(name, str):
        raise ModelFilesError(f"invalid model file path: {name!r}")
    path = PurePosixPath(name)
    if (not name or name == "." or "\0" in name or path.is_absolute()
            or ".." in path.parts or "\\" in name or str(path) != name):
        raise ModelFilesError(f"invalid model file path: {name!r}")
    return name


def resolve_checkpoint(names: set[str], read_json: Callable, prefix: str = "", *, diffusion: bool = False) -> dict | None:
    stems = [("diffusion_pytorch_model", "safetensors"), ("diffusion_pytorch_model", "bin")] if diffusion else [
        ("model", "safetensors"), ("pytorch_model", "bin"),
    ]
    for stem, extension in stems:
        base = prefix + stem + "." + extension
        if base in names:
            return {"component": prefix.rstrip("/") or ".", "format": extension, "variant": None, "files": [base]}
        index = base + ".index.json"
        if index in names:
            data = read_json(index)
            weight_map = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ModelFilesError(f"invalid or empty weight_map: {index}")
            if not all(isinstance(value, str) for value in weight_map.values()):
                raise ModelFilesError(f"invalid shard filenames: {index}")
            shards = {prefix + safe_path(value) for value in weight_map.values()}
            missing = shards - names
            if missing:
                raise ModelFilesError(f"missing checkpoint shards for {index}: {', '.join(sorted(missing))}")
            return {"component": prefix.rstrip("/") or ".", "format": extension, "variant": None,
                    "files": [index, *sorted(shards)]}
    return None
