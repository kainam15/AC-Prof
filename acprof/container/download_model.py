"""Pinned repository planning and file verification for the host Model Store."""
from __future__ import annotations

import fnmatch
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import stat
from pathlib import Path

from acprof.container.model_files import (
    ModelFilesError,
    plan_download,
    resolve_checkpoint,
    seal_plan,
    validate_plan,
)
from acprof.hub.huggingface.auth import hf_token_for_url
from acprof.hub.huggingface.transport import configure_hf_transport

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

    from acprof.hub.repository import (
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
        info = HfApi(endpoint=endpoint, token=hf_token_for_url(endpoint)).model_info(model_id, revision=revision, files_metadata=True)
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
        size = files[name].get("size")
        if type(size) is not int or size < 0 or size > 4 * 1024 * 1024:
            raise ModelFilesError(f"metadata size is unknown or exceeds 4 MiB: {name}")
        if source == "modelscope":
            verified_sha = files[name].get("lfs_sha256")
            if not isinstance(verified_sha, str) or not verified_sha:
                raise ModelFilesError(f"ModelScope metadata lacks a verified SHA256: {name}")
            if cache_dir is None:
                raise ModelFilesError("ModelScope metadata requires an explicit model cache directory")
            path = modelscope_file(model_id, name, info.sha, Path(cache_dir).parent,
                                   sha256=verified_sha)
        else:
            path = hf_hub_download(model_id, name, revision=info.sha,
                                   cache_dir=cache_dir or CACHE_DIR, endpoint=endpoint, token=hf_token_for_url(endpoint))
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



_FileIdentity = tuple[int, int, int, int, int]


def _file_identity(info: os.stat_result) -> _FileIdentity:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_model_file(path: Path):
    # Snapshot symlinks are intentional; a FIFO must not wait for a writer.
    return open(path, "rb", opener=lambda name, flags: os.open(name, flags | getattr(os, "O_NONBLOCK", 0)))


def _require_file_identity(path: Path, expected: _FileIdentity) -> None:
    try:
        # Use fstat consistently: path stat and handle stat can expose different
        # ctime semantics on Windows. Reopening also detects replaced symlinks.
        with _open_model_file(path) as stream:
            current = os.fstat(stream.fileno())
            if not stat.S_ISREG(current.st_mode) or _file_identity(current) != expected:
                raise ModelFilesError(f"model file changed during verification: {path.name}")
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise ModelFilesError(f"model file changed during verification: {path.name}") from exc


def _hash_model_file(path: Path, item: dict,
                     hashes: dict[_FileIdentity, tuple[str, str]]) -> tuple[_FileIdentity, str, str]:
    # Inspect the same opened target whose bytes supply the hashes.
    try:
        stream = _open_model_file(path)
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise ModelFilesError(f"downloaded snapshot is missing or not a regular file: {item['path']}") from exc
    with stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ModelFilesError(f"model payload must be a regular file: {item['path']}")
        if item.get("size") is not None and info.st_size != item["size"]:
            raise ModelFilesError(f"model file size mismatch: {item['path']}")
        identity = _file_identity(info)
        if identity not in hashes:
            sha256 = hashlib.sha256()
            git_sha1 = hashlib.sha1(f"blob {info.st_size}\0".encode())
            remaining = info.st_size
            while chunk := stream.read(min(4 * 1024 * 1024, remaining + 1)):
                remaining -= len(chunk)
                if remaining < 0:
                    raise ModelFilesError(f"model file changed during verification: {item['path']}")
                sha256.update(chunk)
                git_sha1.update(chunk)
            if remaining:
                raise ModelFilesError(f"model file changed during verification: {item['path']}")
            digests = sha256.hexdigest(), git_sha1.hexdigest()
        else:
            digests = hashes[identity]
        if _file_identity(os.fstat(stream.fileno())) != identity:
            raise ModelFilesError(f"model file changed during verification: {item['path']}")
    hashes[identity] = digests
    return identity, *digests


def verify_download(target: str | Path, plan: dict) -> dict:
    """Record hashes only after every file verifies; failures leave the plan intact."""
    validate_plan(plan)
    root = Path(target)
    hashes: dict[_FileIdentity, tuple[str, str]] = {}
    observations = []
    verified_files = []
    for item in plan["files"]:
        path = root / item["path"]
        identity, sha256_hex, git_hex = _hash_model_file(path, item, hashes)
        if item.get("sha256") and item["sha256"] != sha256_hex:
            raise ModelFilesError(f"stored model file SHA256 mismatch: {item['path']}")
        if item.get("lfs_sha256"):
            if sha256_hex != item["lfs_sha256"]:
                raise ModelFilesError(f"model file SHA256 mismatch: {item['path']}")
        elif item.get("blob_id") and git_hex != item["blob_id"]:
            raise ModelFilesError(f"model file Git blob mismatch: {item['path']}")
        observations.append((path, identity))
        verified_files.append({**item, "size": identity[2], "sha256": sha256_hex})
    # An earlier file may have changed while a later large payload was hashed.
    # Recheck metadata only; stable hardlinks/symlinks still share one hash pass.
    for path, identity in observations:
        _require_file_identity(path, identity)
    verified = {**plan, "files": verified_files,
                "selected_bytes": sum(item["size"] for item in verified_files), "verification": "sha256"}
    if "dependencies" in plan or "total_selected_bytes" in plan:
        verified["total_selected_bytes"] = verified["selected_bytes"] + sum(
            item["download"]["selected_bytes"] for item in plan.get("dependencies", []))
    plan.update(seal_plan(verified))
    return plan



def main(argv=None) -> None:
    raise SystemExit("Container downloads are retired; use AC-Prof preparation with the host Model Store and --max-download")


if __name__ == "__main__":
    main()
