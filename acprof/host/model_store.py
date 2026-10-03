"""One host Hub cache, immutable per-plan views, read-only runtime mounts.

No model/framework code is loaded here. Entry views contain symlinks only;
CPU/GPU builds share bytes and retain independent pinned dependency refs.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Event

from acprof.container.model_files import PLAN_FILENAME, seal_plan, validate_plan
from acprof.dependency_locks import content_digest
from acprof.hf_endpoints import hf_endpoints
from acprof.hf_transport import configure_hf_transport, observed_hf_sources
from acprof.model_spec import task_model_spec
from acprof.network_policy import (
    DownloadPolicyError,
    DownloadSource,
    enforce_download_budget,
    parse_bytes,
    require_source_transition,
)

ENTRY_METADATA_MAX_BYTES = 4 * 1024 * 1024
_LEASES: dict[str, object] = {}


class ModelStoreCancelled(RuntimeError):
    """The caller cancelled preparation or lock waiting before a GC mutation."""


def _check_cancelled(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ModelStoreCancelled("Model Store operation cancelled")


def store_root() -> Path:
    return Path(os.environ.get("ACPROF_MODEL_STORE", "").strip() or Path.home() / ".cache/acprof/model-store").expanduser().resolve()


def _recorded_root(record: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if os.environ.get("ACPROF_MODEL_STORE", "").strip():
        return store_root()
    return Path(record["host_path"]).expanduser().resolve() if record.get("host_path") else store_root()


def repository_plans(plan: dict):
    return [plan, *(entry["download"] for entry in plan.get("dependencies", []))]


def entry_key(task) -> str:
    # Environment/device/profile deliberately excluded: weights are shared.
    import hashlib

    from acprof.installation import resource_root
    return content_digest({"schema": 1, "planner": hashlib.sha256((resource_root() / "acprof/container/model_files.py").read_bytes()).hexdigest(),
        "model_types": hashlib.sha256((resource_root() / "acprof/container/compat/transformers_model_types.json").read_bytes()).hexdigest(),
        "model": task.model_id, "revision": task.model_revision,
        "policy": task.model_download_policy, "family": task.task_family, "backend": task.runtime_backend,
        "adapter": task.model_adapter, "dependencies": task_model_spec(task).get("dependencies", [])})


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def store_lock(root: Path, *, cancel: Event | None = None, on_wait: Callable[[], None] | None = None):
    _check_cancelled(cancel)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        if cancel is None and on_wait is None:
            fcntl.flock(stream, fcntl.LOCK_EX)
        else:
            while True:
                _check_cancelled(cancel)
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if on_wait is not None:
                        on_wait()
                        on_wait = None
                    if cancel is not None:
                        cancel.wait(0.1)
                    else:
                        time.sleep(0.1)
        _check_cancelled(cancel)
        yield


def read_entry(key: str, root: Path | None = None) -> dict | None:
    root = root or store_root()
    if not re.fullmatch(r"[a-f0-9]{64}", key):
        raise ValueError("invalid Model Store entry ID")
    try:
        with ExitStack() as stack:
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            stack.callback(os.close, directory)
            # Anchor each component to its opened parent so a concurrent rename
            # cannot redirect a metadata read through a replacement symlink.
            for name in ("entries", key):
                directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                stack.callback(os.close, directory)
            descriptor = os.open(PLAN_FILENAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            stack.callback(os.close, descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("Model Store metadata path must be a regular file")
            if metadata.st_size > ENTRY_METADATA_MAX_BYTES:
                raise ValueError("Model Store entry metadata exceeds the 4 MiB limit")
            stream = stack.enter_context(os.fdopen(descriptor, "rb", closefd=False))
            data = stream.read(ENTRY_METADATA_MAX_BYTES + 1)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("Invalid Model Store metadata path: symlink or non-directory component") from exc
        raise
    if len(data) > ENTRY_METADATA_MAX_BYTES:
        raise ValueError("Model Store entry metadata exceeds the 4 MiB limit")
    plan = json.loads(data.decode("utf-8"))
    validate_plan(plan)
    if plan.get("verification") != "sha256":
        raise ValueError("Model Store entry is not verified")
    return plan


def plan_model(task, root: Path | None = None) -> dict:
    from acprof.container.download_model import _prepare_repository_plan
    from acprof.dependency_locks import package_versions, read_python_lock
    from acprof.installation import resource_root
    from acprof.runtime_profiles import select_runtime_profile
    profile = select_runtime_profile(task)
    versions = package_versions(read_python_lock(resource_root() / profile.environment.requirements_lock))
    catalog_path = resource_root() / "acprof/container/compat/transformers_model_types.json"
    catalog = json.loads(catalog_path.read_text())["versions"]
    native_types = set(catalog.get(versions.get("transformers"), {}).get("model_types", []))
    root = root or store_root()
    cached = read_entry(entry_key(task), root)
    if cached:
        return cached
    endpoints = hf_endpoints()
    for index, endpoint in enumerate(endpoints):
        if index:
            require_source_transition(endpoints[index - 1], endpoint)
        try:
            plan = _prepare_repository_plan(endpoint, task.model_id, task.model_revision,
                                            cache_dir=str(root / "hf"), task=task,
                                            native_types=native_types, library_versions=versions)
            dependencies = task_model_spec(task).get("dependencies", [])
            if dependencies:
                plan["dependencies"] = [{**item, "download": _prepare_repository_plan(
                    endpoint, item["repo_id"], item["revision"], dependency=item,
                    cache_dir=str(root / "hf"), task=task,
                    native_types=native_types, library_versions=versions)} for item in dependencies]
                sizes = [p["selected_bytes"] for p in repository_plans(plan)]
                plan["total_selected_bytes"] = sum(sizes) if all(type(n) is int for n in sizes) else None
            return seal_plan(plan)
        except (DownloadPolicyError, ValueError):
            raise
        except Exception:
            if index == len(endpoints) - 1:
                raise
    raise AssertionError("empty endpoint policy")


def cached_file(root: Path, plan: dict, record: dict) -> Path | None:
    from huggingface_hub import try_to_load_from_cache
    result = try_to_load_from_cache(plan["model_id"], record["path"], revision=plan["model_revision"],
                                   cache_dir=str(root / "hf"))
    if not isinstance(result, str):
        return None
    path = Path(result)
    if path.is_file() and (record.get("size") is None or path.stat().st_size == record["size"]):
        return path
    return None


def model_sources(plan: dict, root: Path | None = None) -> list[DownloadSource]:
    root = root or store_root()
    result = []
    for repo in repository_plans(plan):
        cached = 0
        missing = []
        for record in repo["files"]:
            path = cached_file(root, repo, record)
            if path:
                cached += path.stat().st_size
            else:
                missing.append(record.get("size"))
        expected = sum(missing) if all(type(n) is int for n in missing) else None
        result.append(DownloadSource("model", repo["endpoint"], expected, cached,
            "hit" if not missing else "partial" if cached else "miss",
            detail=f"{repo['model_id']}@{repo['model_revision']}; total_bytes={repo['selected_bytes']}"))
    return result


def disk_report(root: Path | None = None, *, cancel: Event | None = None) -> dict:
    _check_cancelled(cancel)
    root = root or store_root()
    existing = root
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    used = 0
    for path in root.rglob("*"):
        _check_cancelled(cancel)
        try:
            if path.is_file() and not path.is_symlink():
                used += path.stat().st_size
        except FileNotFoundError:
            continue  # Unlocked capacity summaries may overlap another task's cleanup.
    models = []
    for path in sorted((root / "entries").glob(f"*/{PLAN_FILENAME}")):
        _check_cancelled(cancel)
        try:
            plan = read_entry(path.parent.name, root)
            if plan is None:
                continue
            stamp = path.parent / "last-used"
            models.append({"entry_id": path.parent.name, "model_id": plan["model_id"],
                "model_revision": plan["model_revision"], "model_artifact_bytes": plan.get("total_selected_bytes", plan["selected_bytes"]),
                "last_used": stamp.stat().st_mtime if stamp.exists() else path.stat().st_mtime})
        except FileNotFoundError:
            continue
    return {"path": str(root), "total_bytes": used, "free_bytes": free, "models": models,
            "capacity_bytes": parse_bytes(os.environ.get("ACPROF_MODEL_STORE_MAX"))}


def require_space(plan: dict, root: Path | None = None) -> dict:
    root = root or store_root()
    sources = model_sources(plan, root)
    sizes = [source.estimated_bytes for source in sources]
    if any(n is None for n in sizes):
        raise DownloadPolicyError("Model Store 文件大小未知；无法核验剩余空间，尚未下载权重")
    needed = sum(sizes)
    report = disk_report(root)
    # Headroom covers directory metadata and small plan/index files, not another weights copy.
    reserve = 64 * 1024 * 1024
    report.update(required_bytes=needed, remaining_bytes=report["free_bytes"] - needed,
                  reclaimable_bytes=prune_candidates(root)["reclaimable_bytes"])
    if needed + reserve > report["free_bytes"]:
        raise DownloadPolicyError(f"Model Store 磁盘不足，尚未下载：{report}")
    if report["capacity_bytes"] is not None and report["total_bytes"] + needed > report["capacity_bytes"]:
        raise DownloadPolicyError(f"Model Store 容量上限不足，尚未下载；请先显式 prune：{report}")
    return report


def prepare_model(task, plan: dict, root: Path | None = None, *, planned_download_bytes: int | None = None) -> dict:
    from huggingface_hub import snapshot_download

    from acprof.container.download_model import verify_download
    root = root or store_root()
    key = entry_key(task)
    configure_hf_transport()
    with store_lock(root):
        require_space(plan, root)
        sources = model_sources(plan, root)
        needed = sum(source.estimated_bytes for source in sources)
        enforce_download_budget({"expected_download_bytes": needed}, os.environ.get("ACPROF_MAX_DOWNLOAD"))
        if planned_download_bytes is not None and needed > planned_download_bytes:
            raise DownloadPolicyError("Model Store 缓存自预检后已变化；尚未下载权重，请重新预检")
        existing = read_entry(key, root)
        if existing:
            plan = existing
        targets = []
        escape = {"[": "[[]", "*": "[*]", "?": "[?]"}
        for repo in repository_plans(plan):
            observed_hf_sources(reset=True)
            target = snapshot_download(repo_id=repo["model_id"], revision=repo["model_revision"],
                cache_dir=str(root / "hf"), endpoint=repo["endpoint"],
                allow_patterns=["".join(escape.get(c, c) for c in f["path"]) for f in repo["files"]],
                local_files_only=existing is not None or all(cached_file(root, repo, f) for f in repo["files"]))
            # Verification happens before any runtime or formal measurement starts.
            verify_download(target, repo)
            if not existing:
                repo["actual_source_hosts"] = observed_hf_sources()
                seal_plan(repo)
            targets.append(Path(target))
        if plan.get("dependencies"):
            plan["total_selected_bytes"] = sum(repo["selected_bytes"] for repo in repository_plans(plan))
        seal_plan(plan)
        entries = root / "entries"
        entries.mkdir(exist_ok=True)
        destination = entries / key
        if not destination.exists():
            with tempfile.TemporaryDirectory(prefix=".preparing-", dir=entries) as directory:
                temporary = Path(directory)
                for repo, source in zip(repository_plans(plan), targets):
                    cache = temporary / "hf" / ("models--" + repo["model_id"].replace("/", "--"))
                    snapshot = cache / "snapshots" / repo["model_revision"]
                    for record in repo["files"]:
                        link = snapshot / record["path"]
                        link.parent.mkdir(parents=True, exist_ok=True)
                        blob = (source / record["path"]).resolve()
                        if not blob.is_relative_to(root):
                            raise ValueError("Model Store snapshot escaped its store root")
                        link.symlink_to(os.path.relpath(blob, link.parent))
                    (cache / "refs").mkdir(exist_ok=True)
                    (cache / "refs/main").write_text(repo["model_revision"])
                _json(temporary / PLAN_FILENAME, plan)
                # Rename preserves relative symlink depth.
                temporary.rename(destination)
        (destination / "last-used").touch()
    print("[network-download] " + json.dumps({"category": "model",
        "verified_new_payload_bytes": sum(source.estimated_bytes for source in sources),
        "cache_savings_bytes": sum(source.cached_bytes for source in sources),
        "wire_bytes": None, "endpoint": plan["endpoint"]}), flush=True)
    return {"schema_version": 1, "entry_id": key, "plan_sha256": plan["plan_sha256"],
            "model_artifact_bytes": plan.get("total_selected_bytes", plan["selected_bytes"]),
            "model_download": plan}


def mount_args(manifest: dict, root: Path | None = None) -> list[str]:
    record = manifest.get("model_store")
    if not record:
        return []  # Historical immutable baked images retain their original semantics.
    root = _recorded_root(record, root)
    key = record["entry_id"]
    with store_lock(root):
        plan = read_entry(key, root)
        if not plan or plan["plan_sha256"] != record["plan_sha256"]:
            raise RuntimeError("Model Store 缺失或计划 SHA256 不匹配；请先重新准备，禁止在线 fallback")
        lease_key = str(root / "entries" / key)
        if lease_key not in _LEASES:
            lock = (root / (key + ".lease")).open("a")
            fcntl.flock(lock, fcntl.LOCK_SH)
            _LEASES[lease_key] = lock
        (root / "entries" / key / "last-used").touch()
    cache = f"/models/entries/{key}/hf"
    snapshot = f"{cache}/models--{plan['model_id'].replace('/', '--')}/snapshots/{plan['model_revision']}"
    return ["--mount", f"type=bind,src={root},dst=/models,readonly", "-e", f"HF_HUB_CACHE={cache}",
            "-e", f"TRANSFORMERS_CACHE={cache}", "-e", "HF_MODULES_CACHE=/tmp/acprof-hf-modules",
            "-e", f"MODEL_LOCAL_PATH={snapshot}"]


def verify_entry(manifest: dict, root: Path | None = None) -> None:
    from acprof.container.download_model import verify_download
    record = manifest.get("model_store")
    if not record:
        return
    root = _recorded_root(record, root)
    mount_args(manifest, root)  # Acquire a lease before reading the snapshot.
    plan = read_entry(record["entry_id"], root)
    if plan is None:
        raise RuntimeError("Model Store entry disappeared after acquiring its lease")
    for repo in repository_plans(plan):
        snapshot = root / "entries" / record["entry_id"] / "hf" / ("models--" + repo["model_id"].replace("/", "--")) / "snapshots" / repo["model_revision"]
        verify_download(snapshot, repo)


def prune_candidates(root: Path | None = None, keep: set[str] | None = None, *,
                     target_bytes: str | int | None = None, approved_entries: set[str] | None = None,
                     cancel: Event | None = None) -> dict:
    """Scan entry/blob references once, then decrement references in LRU order."""
    root = root or store_root()
    keep = keep or set()
    target = parse_bytes(target_bytes)
    candidates, entry_blobs = [], {}
    references: Counter[Path] = Counter()
    for path in (root / "entries").glob(f"*/{PLAN_FILENAME}"):
        _check_cancelled(cancel)
        key = path.parent.name
        leased = False
        try:
            with (root / (key + ".lease")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            leased = True
        links = set()
        for link in (path.parent / "hf").rglob("*"):
            _check_cancelled(cancel)
            if link.is_symlink():
                links.add(link.resolve())
        entry_blobs[key] = links
        references.update(links)
        if not leased and key not in keep and (approved_entries is None or key in approved_entries):
            stamp = path.parent / "last-used"
            candidates.append((stamp.stat().st_mtime if stamp.exists() else 0, key))
    blobs = {}
    for path in (root / "hf").glob("models--*/blobs/*"):
        _check_cancelled(cancel)
        if path.is_file() and not path.is_symlink():
            blobs[path.resolve()] = (path, path.stat().st_size)
    unused = {blob for blob in blobs if not references[blob]}
    reclaimed = sum(blobs[blob][1] for blob in unused)
    used = disk_report(root, cancel=cancel)["total_bytes"] if target is not None else 0
    selected = []
    for _, key in sorted(candidates):
        _check_cancelled(cancel)
        if target is not None and used - reclaimed <= target:
            break
        selected.append(key)
        for blob in entry_blobs[key]:
            references[blob] -= 1
            if references[blob] == 0 and blob in blobs:
                unused.add(blob)
                reclaimed += blobs[blob][1]
    return {"entries": selected, "reclaimable_bytes": reclaimed,
            "blobs": [blobs[blob][0] for blob in sorted(unused)]}


def prune_store(*, root: Path | None = None, apply: bool = False, keep: set[str] | None = None,
                target_bytes: str | int | None = None, approved_entries: set[str] | None = None,
                cancel: Event | None = None, on_wait: Callable[[], None] | None = None) -> dict:
    root = root or store_root()
    with store_lock(root, cancel=cancel, on_wait=on_wait):
        # Preview is advisory. Every application rechecks leases and shared blobs
        # under the same lock used when publishing entries or acquiring leases.
        result = prune_candidates(root, keep, target_bytes=target_bytes, approved_entries=approved_entries, cancel=cancel)
        _check_cancelled(cancel)
        if apply:
            # Once deletion begins, finish before releasing the lock/UI busy state.
            for key in result["entries"]:
                shutil.rmtree(root / "entries" / key)
            for path in result["blobs"]:
                path.unlink()
            for path in (root / "hf").rglob("*"):
                if path.is_symlink() and not path.exists():
                    path.unlink()
        return {k: v for k, v in result.items() if k != "blobs"}
