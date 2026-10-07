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
from urllib.parse import urlsplit

import acprof.artifacts as artifact_io
from acprof.container.model_files import PLAN_FILENAME, seal_plan, validate_plan
from acprof.dependency_locks import content_digest
from acprof.hf_download import try_hf_endpoints
from acprof.hf_endpoints import hf_endpoints
from acprof.hf_transport import (
    configure_hf_transport,
    observed_hf_provenance,
    observed_hf_sources,
)
from acprof.model_spec import task_model_spec
from acprof.network_policy import (
    DownloadPolicyError,
    DownloadSource,
    enforce_download_budget,
    parse_bytes,
)

ENTRY_METADATA_MAX_BYTES = 4 * 1024 * 1024
class ModelStoreMount:
    """One explicitly owned runtime mount lease.

    The shared flock must stay alive for exactly as long as a consumer may
    access the read-only Model Store bind mount. Closing is idempotent so
    cleanup paths can hand ownership to a RunningContainer safely.
    """

    def __init__(self, args: list[str], lock=None):
        self.args = args
        self._lock = lock

    def close(self) -> None:
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()


_RETAINED_MOUNTS: list[ModelStoreMount] = []


def retain_mount_for_cleanup_debt(mount: ModelStoreMount) -> None:
    """Keep a lease alive when container absence could not be proven."""
    _RETAINED_MOUNTS.append(mount)


class ModelStoreCancelled(RuntimeError):
    """The caller cancelled preparation or lock waiting before a GC mutation."""


def _check_cancelled(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ModelStoreCancelled("Model Store operation cancelled")


def _open_lock_file(path: Path, *, label: str):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise OSError(f"{label} lock is not a regular file owned only by this user")
        return os.fdopen(descriptor, "a+")
    except BaseException:
        os.close(descriptor)
        raise


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
    source = getattr(task, "model_source", "huggingface")
    return content_digest({"schema": 1, **({"source": source} if source != "huggingface" else {}),
        "planner": hashlib.sha256((resource_root() / "acprof/container/model_files.py").read_bytes()).hexdigest(),
        "model_types": hashlib.sha256((resource_root() / "acprof/container/compat/transformers_model_types.json").read_bytes()).hexdigest(),
        "model": task.model_id, "revision": task.model_revision,
        "policy": task.model_download_policy, "family": task.task_family, "backend": task.runtime_backend,
        "adapter": task.model_adapter, "dependencies": task_model_spec(task).get("dependencies", [])})


def _json(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    artifact_io.atomic_write(path, lambda stream: stream.write(payload))


def _sync_directory_tree(root: Path) -> None:
    """Make private staged directory entries durable before publishing the root."""
    directories = [root, *(path for path in root.rglob("*") if path.is_dir() and not path.is_symlink())]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        artifact_io.sync_directory(directory)


@contextmanager
def store_lock(root: Path, *, cancel: Event | None = None, on_wait: Callable[[], None] | None = None):
    _check_cancelled(cancel)
    root.mkdir(parents=True, exist_ok=True)
    with _open_lock_file(root / ".lock", label="Model Store") as stream:
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
    plan = artifact_io.loads_finite_json(data.decode("utf-8"))
    validate_plan(plan)
    if plan.get("verification") != "sha256":
        raise ValueError("Model Store entry is not verified")
    return plan


def _cached_repositories(model_id: str, revision: str | None, source: str):
    """Read bounded, verified entry metadata before attempting any network I/O."""
    root = store_root()
    entries = root / "entries"
    if not entries.is_dir():
        return
    candidates = []
    for entry in entries.iterdir():
        if not re.fullmatch(r"[a-f0-9]{64}", entry.name):
            continue
        try:
            metadata = entry.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            continue
        candidates.append((metadata.st_mtime, entry))
    for _, entry in sorted(candidates, key=lambda item: item[0], reverse=True):
        plan = read_entry(entry.name, root)
        if plan is None:
            continue
        for repo in repository_plans(plan):
            if repo["model_id"] != model_id or repo.get("source", "huggingface") != source:
                continue
            if revision != repo["model_revision"]:
                default_ref = "master" if source == "modelscope" else "main"
                if ("requested_revision" not in repo
                        or (revision or default_ref) != (repo["requested_revision"] or default_ref)):
                    continue
            snapshot = entry / "hf" / ("models--" + model_id.replace("/", "--")) / "snapshots" / repo["model_revision"]
            if all((snapshot / f["path"]).is_file() and (snapshot / f["path"]).stat().st_size == f["size"]
                   for f in repo["files"]):
                yield repo, snapshot


def cached_metadata(model_id: str, name: str, revision: str | None, *, source: str,
                    max_bytes: int = ENTRY_METADATA_MAX_BYTES) -> Path | None:
    import hashlib

    from acprof.container.model_files import ModelFilesError, safe_path
    name = safe_path(name)
    for plan, snapshot in _cached_repositories(model_id, revision, source):
        record = next((item for item in plan["files"] if item["path"] == name), None)
        if record is None:
            continue
        if record["size"] > max_bytes:
            raise ModelFilesError(f"cached metadata exceeds {max_bytes} bytes: {name}")
        path = snapshot / name
        with path.open("rb") as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ModelFilesError(f"cached metadata exceeds {max_bytes} bytes: {name}")
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ModelFilesError(f"cached model metadata SHA256 mismatch: {name}")
        return path
    return None


def cached_model_info(model_id: str, revision: str | None, *, source: str):
    from types import SimpleNamespace

    from acprof.container.model_files import ModelFilesError, safe_path
    for plan, _ in _cached_repositories(model_id, revision, source):
        metadata = plan.get("repository_context", {})
        # Historical entries remain valid runtime snapshots, but their selected
        # payload files and resolved task are not complete original Hub evidence.
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            continue
        files = metadata.get("repository_files")
        if (not isinstance(files, list) or not files
                or any(not isinstance(metadata.get(key), dict) for key in ("config", "transformers_info", "card_data"))
                or not isinstance(metadata.get("tags"), list)
                or not {"pipeline_tag", "library_name", "safetensors"}.issubset(metadata)):
            raise ModelFilesError("invalid cached repository metadata")
        files = [safe_path(name) for name in files]
        if not {item["path"] for item in plan["files"]}.issubset(files):
            raise ModelFilesError("cached repository inventory omits selected model files")
        return SimpleNamespace(sha=plan["model_revision"],
            siblings=[SimpleNamespace(rfilename=name) for name in files],
            **{key: metadata[key] for key in ("pipeline_tag", "library_name", "config",
                                            "transformers_info", "card_data", "tags", "safetensors")})
    return None


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
    key = entry_key(task)
    cached = read_entry(key, root)
    if cached:
        return cached
    # Planning metadata has no entry reference yet, so serialize cache misses with GC.
    with store_lock(root):
        # Another planner may have published the same entry while this caller waited.
        cached = read_entry(key, root)
        if cached:
            return cached
        from acprof.model_repository import MODELSCOPE_ENDPOINT
        source = getattr(task, "model_source", "huggingface")
        endpoints = [MODELSCOPE_ENDPOINT] if source == "modelscope" else hf_endpoints()
        def prepare(endpoint):
            plan = _prepare_repository_plan(endpoint, task.model_id, task.model_revision,
                                                cache_dir=str(root / "hf"), task=task,
                                                native_types=native_types, library_versions=versions, source=source)
            dependencies = task_model_spec(task).get("dependencies", [])
            if dependencies:
                plan["dependencies"] = []
                for item in dependencies:
                    if source != "huggingface" and item["repo_id"] == task.model_id:
                        raise ValueError("Main model and HF dependency share an offline loader ID but differ in source")
                    def prepare_dependency(hub):
                        return _prepare_repository_plan(hub, item["repo_id"], item["revision"], dependency=item,
                            cache_dir=str(root / "hf"), task=task, native_types=native_types, library_versions=versions)
                    dependency = (try_hf_endpoints(item["repo_id"], prepare_dependency)
                                  if source == "modelscope" else prepare_dependency(endpoint))
                    plan["dependencies"].append({**item, "download": dependency})
                sizes = [p["selected_bytes"] for p in repository_plans(plan)]
                plan["total_selected_bytes"] = sum(sizes) if all(type(n) is int for n in sizes) else None
            plan["requested_revision"] = getattr(task, "requested_revision", None)
            return seal_plan(plan)
        return prepare(MODELSCOPE_ENDPOINT) if source == "modelscope" else try_hf_endpoints(task.model_id, prepare, endpoints=endpoints)


def cached_file(root: Path, plan: dict, record: dict) -> Path | None:
    if plan.get("source", "huggingface") == "modelscope":
        from acprof.model_repository import modelscope_snapshot
        path = modelscope_snapshot(root, plan["model_id"], plan["model_revision"]) / record["path"]
        return path if path.is_file() and path.stat().st_size == record.get("size") else None
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


def probe_model_download(plan: dict, root: Path | None = None) -> list[dict]:
    """HEAD a missing artifact through storage; no GET and no availability cache."""
    from urllib.parse import quote

    from huggingface_hub.utils import build_hf_headers, get_session

    root = root or store_root()
    reports = []
    for repo in repository_plans(plan):
        missing = [item for item in repo["files"] if cached_file(root, repo, item) is None]
        if not missing or repo.get("source", "huggingface") != "huggingface":
            continue
        configure_hf_transport()
        sample = max(missing, key=lambda item: item.get("size") or 0)
        def probe(endpoint):
            url = f"{endpoint}/{repo['model_id']}/resolve/{repo['model_revision']}/{quote(sample['path'])}"
            with get_session().stream("HEAD", url, headers=build_hf_headers(), follow_redirects=True, timeout=8) as response:
                response.raise_for_status()
            return endpoint
        endpoint = try_hf_endpoints(repo["model_id"], probe,
                                   endpoints=list(dict.fromkeys([repo["endpoint"], *hf_endpoints()])))
        reports.append({"model_id": repo["model_id"], "method": "HEAD", "sample_file": sample["path"],
                        "scope": "one missing file; not a permanent availability verdict",
                        "transport": observed_hf_provenance()})
        repo["endpoint"] = endpoint
        seal_plan(repo)
    seal_plan(plan)
    return reports


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
            offline = existing is not None or all(cached_file(root, repo, f) for f in repo["files"])
            source = repo.get("source", "huggingface")
            def download(endpoint):
                if source == "modelscope":
                    from acprof.model_repository import modelscope_file, modelscope_snapshot
                    if not offline:
                        for record in repo["files"]:
                            modelscope_file(repo["model_id"], record["path"], repo["model_revision"], root,
                                            sha256=record["lfs_sha256"])
                    return str(modelscope_snapshot(root, repo["model_id"], repo["model_revision"])), endpoint
                target = snapshot_download(repo_id=repo["model_id"], revision=repo["model_revision"],
                    cache_dir=str(root / "hf"), endpoint=endpoint,
                    allow_patterns=["".join(escape.get(c, c) for c in f["path"]) for f in repo["files"]],
                    local_files_only=offline)
                return target, endpoint
            if offline or source == "modelscope":
                target, endpoint = download(repo["endpoint"])
            else:
                choices = list(dict.fromkeys([repo["endpoint"], *hf_endpoints()]))
                target, endpoint = try_hf_endpoints(repo["model_id"], download, endpoints=choices)
            # Verification happens before any runtime or formal measurement starts.
            verify_download(target, repo)
            if not existing:
                repo["source"] = source
                repo["endpoint"] = endpoint
                repo["actual_source_hosts"] = [] if offline or source == "modelscope" else observed_hf_sources()
                repo["download_provenance"] = ({"schema_version": 1, "cache_hit": True,
                    "public_egress": "externally-managed"} if offline else {"schema_version": 1,
                    "public_egress": "externally-managed", "requests": [], "observation": "SDK redirects unobserved"}
                    if source == "modelscope" else observed_hf_provenance())
                events = repo["download_provenance"].get("requests", [])
                payloads = [event for event in events if event["method"] == "GET" and event["status"] in {200, 206}
                            and "/api/" not in urlsplit(event["url"]).path]
                repo["download_provenance"]["final_endpoint_type"] = ("cache" if offline else
                    payloads[-1]["endpoint_type"] if payloads else "unknown")
                repo["download_provenance"]["hub_endpoint"] = endpoint
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
                    revision = repo["model_revision"]
                    artifact_io.atomic_write(cache / "refs/main", lambda stream: stream.write(revision))
                _json(temporary / PLAN_FILENAME, plan)
                # Persist every staged directory edge before publishing the tree root.
                _sync_directory_tree(temporary)
                # Rename preserves relative symlink depth; syncing entries makes publication durable.
                temporary.rename(destination)
                artifact_io.sync_directory(entries)
        (destination / "last-used").touch()
    print("[network-download] " + json.dumps({"category": "model",
        "verified_new_payload_bytes": sum(source.estimated_bytes for source in sources),
        "cache_savings_bytes": sum(source.cached_bytes for source in sources),
        "wire_bytes": None, "endpoint": plan["endpoint"]}), flush=True)
    return {"schema_version": 1, "entry_id": key, "plan_sha256": plan["plan_sha256"],
            "model_artifact_bytes": plan.get("total_selected_bytes", plan["selected_bytes"]),
            "model_download": plan}


def _require_entry_plan(record: dict, root: Path) -> tuple[str, dict]:
    key = record["entry_id"]
    plan = read_entry(key, root)
    if not plan or plan["plan_sha256"] != record["plan_sha256"]:
        raise RuntimeError("Model Store 缺失或计划 SHA256 不匹配；请先重新准备，禁止在线 fallback")
    return key, plan


def require_entry(manifest: dict, root: Path | None = None) -> dict | None:
    """Validate a recorded Model Store entry without pinning it against GC."""
    record = manifest.get("model_store")
    if not record:
        return None
    root = _recorded_root(record, root)
    with store_lock(root):
        _key, plan = _require_entry_plan(record, root)
    return plan


def _runtime_mount_args(root: Path, key: str, plan: dict) -> list[str]:
    cache = f"/models/entries/{key}/hf"
    snapshot = f"{cache}/models--{plan['model_id'].replace('/', '--')}/snapshots/{plan['model_revision']}"
    return ["--mount", f"type=bind,src={root},dst=/models,readonly", "-e", f"HF_HUB_CACHE={cache}",
            "-e", f"TRANSFORMERS_CACHE={cache}", "-e", "HF_MODULES_CACHE=/tmp/acprof-hf-modules",
            "-e", f"MODEL_LOCAL_PATH={snapshot}"]


def acquire_mount(manifest: dict, root: Path | None = None) -> ModelStoreMount:
    """Validate a Model Store entry and hold its GC lease until close()."""
    record = manifest.get("model_store")
    if not record:
        return ModelStoreMount([])
    root = _recorded_root(record, root)
    with store_lock(root):
        key, plan = _require_entry_plan(record, root)
        lock = _open_lock_file(root / (key + ".lease"), label="Model Store lease")
        try:
            fcntl.flock(lock, fcntl.LOCK_SH)
            (root / "entries" / key / "last-used").touch()
            args = _runtime_mount_args(root, key, plan)
        except BaseException:
            lock.close()
            raise
    return ModelStoreMount(args, lock)


def verify_entry(manifest: dict, root: Path | None = None) -> None:
    from acprof.container.download_model import verify_download
    record = manifest.get("model_store")
    if not record:
        return
    root = _recorded_root(record, root)
    lock = None
    with store_lock(root):
        key, plan = _require_entry_plan(record, root)
        lock = _open_lock_file(root / (key + ".lease"), label="Model Store lease")
        try:
            fcntl.flock(lock, fcntl.LOCK_SH)
            (root / "entries" / key / "last-used").touch()
        except BaseException:
            lock.close()
            raise
    try:
        for repo in repository_plans(plan):
            snapshot = root / "entries" / key / "hf" / ("models--" + repo["model_id"].replace("/", "--")) / "snapshots" / repo["model_revision"]
            verify_download(snapshot, repo)
    finally:
        lock.close()


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
            with _open_lock_file(root / (key + ".lease"), label="Model Store lease") as lock:
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
    from itertools import chain
    for path in chain((root / "hf").glob("models--*/blobs/*"),
                      (root / "modelscope").glob("models--*/snapshots/*/**/*")):
        _check_cancelled(cancel)
        # Production GC scans run under store_lock, shared with prepare_model, so any
        # Hugging Face .incomplete blob visible here is residue from an interrupted download.
        if path.is_file() and not path.is_symlink() and not path.name.endswith(".lock"):
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
