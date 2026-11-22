"""Pure model file path and checkpoint index validation shared by Host and containers."""
from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from pathlib import PurePosixPath
from typing import Any, TypedDict


class Checkpoint(TypedDict):
    """Files selected for one checkpoint component, without loading weights."""

    component: str
    format: str
    variant: str | None
    files: list[str]


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


def uses_sentence_transformer_modules(backend: str, files: set[str]) -> bool:
    """Only the selected sentence encoder loads its checkpoint directories from modules.json."""
    return backend == "sentence_transformers" and "modules.json" in files


def validate_variant(variant: str) -> str:
    if isinstance(variant, str) and variant.endswith("/") and len(variant) <= 256:
        folder = safe_path(variant[:-1])
        if all(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", part) for part in folder.split("/")):
            return variant
    if not isinstance(variant, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", variant):
        raise ModelFilesError("invalid Transformers checkpoint variant")
    return variant


def _checkpoint_choices(prefix: str, *, diffusion: bool, variant: str = "") -> Iterator[tuple[str, str, str]]:
    stems = [("diffusion_pytorch_model", "safetensors"), ("diffusion_pytorch_model", "bin")] if diffusion else [
        ("model", "safetensors"), ("pytorch_model", "bin"),
    ]
    suffix = "." + validate_variant(variant) if variant else ""
    for stem, extension in stems:
        base = prefix + stem
        yield (f"{base}{suffix}.{extension}", f"{base}.{extension}.index{suffix}.json", extension)


_NAMED_CHECKPOINT = re.compile(
    r"(?:model\.([A-Za-z0-9_-]{1,64})\.safetensors|pytorch_model\.([A-Za-z0-9_-]{1,64})\.bin"
    r"|(?:model\.safetensors|pytorch_model\.bin)\.index\.([A-Za-z0-9_-]{1,64})\.json)")


def named_checkpoint_file(name: str) -> bool:
    return bool(_NAMED_CHECKPOINT.fullmatch(name))


def checkpoint_variants(names: set[str]) -> list[str]:
    """Only offer root filenames understood by the registered Transformers loader."""
    found = set()
    for name in names:
        match = _NAMED_CHECKPOINT.fullmatch(name)
        if match:
            variant = next(value for value in match.groups() if value)
            if not re.search(r"-[0-9]+-of-[0-9]+$", variant):
                found.add(variant)
    return sorted(found)


def has_default_checkpoint(names: set[str], prefix: str = "") -> bool:
    return any(base in names or index in names for base, index, _ in _checkpoint_choices(prefix, diffusion=False))


def checkpoint_directories(names: set[str]) -> list[str]:
    """Find separate pretrained roots, excluding components owned by a compound model."""
    compound = {path.rpartition("/")[0] for path in names
                if path.rsplit("/", 1)[-1] in {"model_index.json", "modules.json"}}
    result = []
    for path in sorted(names):
        if not path.endswith("/config.json"):
            continue
        prefix = path[:-len("config.json")]
        if (coreml_artifact(path) or "" in compound or any(prefix.startswith(root + "/") for root in compound)
                or prefix + "adapter_config.json" in names or not has_default_checkpoint(names, prefix)):
            continue
        try:
            validate_variant(prefix)
        except ModelFilesError:
            continue
        result.append(prefix)
    return result


def checkpoint_index(names: set[str], prefix: str = "", *, diffusion: bool = False, variant: str = "") -> str | None:
    """Return only the index selected by the same priority as the weight loader."""
    for base, index, _ in _checkpoint_choices(prefix, diffusion=diffusion, variant=variant):
        if base in names:
            return None
        if index in names:
            return index
    return None


def resolve_checkpoint(
    names: set[str], read_json: Callable[[str], Any], prefix: str = "", *, diffusion: bool = False, variant: str = "",
) -> Checkpoint | None:
    for base, index, extension in _checkpoint_choices(prefix, diffusion=diffusion, variant=variant):
        if base in names:
            return {"component": prefix.rstrip("/") or ".", "format": extension, "variant": variant or None, "files": [base]}
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
            return {"component": prefix.rstrip("/") or ".", "format": extension, "variant": variant or None,
                    "files": [index, *sorted(shards)]}
    return None


def coreml_artifact(path: str) -> str | None:
    parts = path.split("/")
    for index, part in enumerate(parts):
        if part.lower().endswith((".mlpackage", ".mlmodelc")):
            return "/".join(parts[:index + 1])
    return path if parts[-1].lower().endswith(".mlmodel") else None
