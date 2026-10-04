"""Pinned repository planning and file verification for the host Model Store."""
from __future__ import annotations

import fnmatch
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

from acprof.container.model_files import (
    ModelFilesError,
    plan_download,
    resolve_checkpoint,
    seal_plan,
)
from acprof.hf_transport import configure_hf_transport

CACHE_DIR = None


def _native_model_types() -> set[str]:
    if importlib.util.find_spec("transformers") is None:
        return set()
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    return set(CONFIG_MAPPING_NAMES)


def _library_versions() -> dict[str, str]:
    result = {}
    for name in ("huggingface-hub", "transformers", "diffusers", "sentence-transformers", "chronos-forecasting"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _prepare_repository_plan(endpoint: str, model_id: str, revision: str, *, dependency: dict | None = None,
                             cache_dir: str | None = None, task=None, native_types=None, library_versions=None,
                             source: str = "huggingface") -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    from acprof.model_repository import (
        model_source,
        modelscope_file,
        modelscope_info,
        repository_context,
    )

    source = model_source(source)
    if source == "modelscope":
        info = modelscope_info(model_id, revision)
    else:
        configure_hf_transport()
        info = HfApi(endpoint=endpoint).model_info(model_id, revision=revision, files_metadata=True)
    if len(revision) == 40 and info.sha != revision:
        raise ModelFilesError("Hub response does not match the requested model commit")
    files = {}
    for item in info.siblings:
        lfs = getattr(item, "lfs", None)
        lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files[item.rfilename] = {
            "size": getattr(item, "size", None), "blob_id": getattr(item, "blob_id", None),
            "lfs_sha256": lfs_sha,
        }

    def read_json(name: str):
        if files[name].get("size") is None or files[name]["size"] > 4 * 1024 * 1024:
            raise ModelFilesError(f"metadata size is unknown or exceeds 4 MiB: {name}")
        path = (modelscope_file(model_id, name, info.sha, Path(cache_dir).parent, sha256=files[name]["lfs_sha256"])
                if source == "modelscope" else hf_hub_download(model_id, name, revision=info.sha,
                    cache_dir=cache_dir or CACHE_DIR, endpoint=endpoint))
        try:
            with Path(path).open("rb") as stream:
                data = stream.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ModelFilesError(f"metadata exceeds 4 MiB: {name}")
            return json.loads(data)
        except (ValueError, UnicodeError) as exc:
            raise ModelFilesError(f"invalid model metadata: {name}") from exc

    excluded = []
    if dependency is not None and dependency.get("allow_patterns"):
        selected = {name: record for name, record in files.items()
                    if any(fnmatch.fnmatchcase(name, pattern) for pattern in dependency["allow_patterns"])}
        excluded = sorted(set(files) - set(selected))
        files = selected
        if not files:
            raise ModelFilesError(f"dependency patterns select no files: {model_id}")
    if dependency is not None:
        # Explicit declarations also need completeness checks before weights or builds.
        # Metadata-only dependencies have no checkpoint and remain valid.
        resolve_checkpoint(set(files), read_json)
    plan = plan_download(
        model_id=model_id, revision=info.sha, family=(task.task_family if task else os.getenv("TASK_FAMILY", "")) if dependency is None else "dependency",
        backend=task.runtime_backend if task else os.getenv("RUNTIME_BACKEND", ""), files=files, read_json=read_json,
        policy=(task.model_download_policy if task else os.getenv("MODEL_DOWNLOAD_POLICY", "auto")) if dependency is None else "full",
        adapter=task.model_adapter if task else os.getenv("MODEL_ADAPTER", "family-default"),
        native_model_types=_native_model_types() if native_types is None else native_types,
        library_versions=_library_versions() if library_versions is None else library_versions,
    )
    if dependency is not None:
        plan.update(reason="declared_dependency", excluded_files=excluded)
    plan["endpoint"] = endpoint
    plan["source"] = source
    plan["repository_context"] = repository_context(info)
    plan["requested_revision"] = revision
    return seal_plan(plan)



def verify_download(target: str | Path, plan: dict) -> dict:
    """构建期间检查完整性并记录实际 SHA256；不会进入 server 启动路径。"""
    root = Path(target)
    hashes = {}
    for item in plan["files"]:
        path = root / item["path"]
        if not path.is_file():
            raise ModelFilesError(f"downloaded snapshot is missing: {item['path']}")
        stat = path.stat()
        if item.get("size") is not None and stat.st_size != item["size"]:
            raise ModelFilesError(f"model file size mismatch: {item['path']}")
        key = (stat.st_dev, stat.st_ino)
        if key not in hashes:
            sha256 = hashlib.sha256()
            git_sha1 = hashlib.sha1(f"blob {stat.st_size}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    sha256.update(chunk)
                    git_sha1.update(chunk)
            hashes[key] = (sha256.hexdigest(), git_sha1.hexdigest())
        sha256_hex, git_hex = hashes[key]
        if item.get("sha256") and item["sha256"] != sha256_hex:
            raise ModelFilesError(f"stored model file SHA256 mismatch: {item['path']}")
        if item.get("lfs_sha256"):
            if sha256_hex != item["lfs_sha256"]:
                raise ModelFilesError(f"model file SHA256 mismatch: {item['path']}")
        elif item.get("blob_id") and git_hex != item["blob_id"]:
            raise ModelFilesError(f"model file Git blob mismatch: {item['path']}")
        item.update(size=stat.st_size, sha256=sha256_hex)
    plan["selected_bytes"] = sum(item["size"] for item in plan["files"])
    if plan.get("dependencies"):
        plan["total_selected_bytes"] = plan["selected_bytes"] + sum(item["download"]["selected_bytes"] for item in plan["dependencies"])
    plan["verification"] = "sha256"
    return seal_plan(plan)



def main(argv=None) -> None:
    raise SystemExit("Container downloads are retired; use AC-Prof preparation with the host Model Store and --max-download")


if __name__ == "__main__":
    main()
