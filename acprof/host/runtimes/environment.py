"""Prepare immutable user-owned Python environments before any inference window."""
from __future__ import annotations

import fcntl
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

from acprof.artifacts import atomic_write_json, file_sha256, read_json_object
from acprof.dependency_locks import content_digest, package_versions, read_python_lock
from acprof.host.command import run_command
from acprof.host.runtimes.contracts import ProcessInfo
from acprof.runtime_profiles import select_runtime_profile
from acprof.source_identity import service_sources, source_fingerprint


def host_abi() -> dict:
    return {"system": platform.system(), "machine": platform.machine(), "libc": list(platform.libc_ver())}


def installed_packages(python: str) -> dict:
    result = run_command([python, "-I", "-c",
        "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))"],
        check=True, timeout=30)
    return json.loads(result.stdout)


def prepare_environment(task, project_dir: str | Path) -> tuple[Path, dict]:
    profile = select_runtime_profile(task)
    if profile.environment is None:
        raise ValueError("selected profile has no dependency lock")
    declaration = profile.environment
    project = Path(project_dir)
    abi = host_abi()
    if abi["system"] != "Linux" or abi["machine"] not in {"x86_64", "amd64"}:
        raise ValueError("Process dependency locks currently target Linux x86_64")
    lock = project / declaration.requirements_lock
    records = read_python_lock(lock)
    identity = {"schema_version": 1, "kind": "process", "host_abi": abi,
                "python": declaration.platform.python_version, "artifacts": records,
                "service_sha256": source_fingerprint(project, service_sources(project), scope="process-service-v1")}
    key = content_digest(identity)
    root = Path(os.environ.get("ACPROF_PROCESS_STORE", "~/.cache/acprof/process-environments")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / key
    with (root / (key + ".lock")).open("a+b") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        if target.exists():
            manifest = read_json_object(target / "manifest.json", label="Process environment")
            if manifest["identity"] != identity:
                raise ValueError("cached Process environment identity changed")
            verify_environment(target, manifest)
            return target, manifest
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("Process preparation requires uv; install uv before starting collection")
        with tempfile.TemporaryDirectory(prefix=".preparing-", dir=root) as directory:
            staging = Path(directory) / "environment"
            run_command([uv, "venv", "--relocatable", "--python", declaration.platform.python_version,
                         str(staging / "venv")], check=True, timeout=300)
            python = str(staging / "venv/bin/python")
            run_command([uv, "pip", "sync", "--python", python, "--link-mode", "copy", "--require-hashes", str(lock)],
                        check=True, timeout=3600)
            actual = installed_packages(python)
            expected = package_versions(records)
            if actual != expected:
                raise ValueError("prepared Process package set differs from the complete lock")
            code = staging / "code"
            for path in service_sources(project):
                destination = code / path.relative_to(project)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
            manifest = {"identity": identity, "environment_id": key, "packages": actual,
                        "interpreter_sha256": file_sha256(python),
                        "system_packages": "host_managed; validated by import and inference",
                        "isolation": "python_environment; not a security or network sandbox"}
            atomic_write_json(staging / "manifest.json", manifest)
            staging.rename(target)
    return target, manifest


def verify_environment(root: Path, manifest: dict) -> None:
    identity = manifest["identity"]
    if content_digest(identity) != manifest["environment_id"] or identity["host_abi"] != host_abi():
        raise ValueError("Process environment identity or host ABI changed")
    if installed_packages(str(root / "venv/bin/python")) != manifest["packages"]:
        raise ValueError("Process environment packages changed; refuse reuse")
    if file_sha256(root / "venv/bin/python") != manifest["interpreter_sha256"]:
        raise ValueError("Process interpreter changed; refuse reuse")
    code = root / "code"
    if source_fingerprint(code, service_sources(code), scope="process-service-v1") != identity["service_sha256"]:
        raise ValueError("Process environment service code changed; refuse reuse")


def prepare_process(task, project_dir, output_dir, *, rebuild=False, timeout_seconds=300) -> ProcessInfo:
    if rebuild:
        raise ValueError("--rebuild-image applies to Docker; Process environments are immutable")
    from acprof.host.model_store import plan_model, prepare_model, store_root, verify_entry
    from acprof.host.runtimes.validation import probe_interface
    root, environment = prepare_environment(task, project_dir)
    probe_interface(task, root, environment, output_dir, timeout_seconds=timeout_seconds)
    plan = plan_model(task)
    record = {**prepare_model(task, plan), "host_path": str(store_root())}
    task.model_store = record
    manifest = {**environment, "model_store": record, "model_download": record["model_download"],
                "runtime_kind": "process", "logs_dir": str(Path(output_dir).absolute() / "logs"),
                "build_fingerprint": content_digest({"environment": environment["environment_id"],
                                                     "model": record["plan_sha256"]})}
    verify_entry(manifest)
    snapshot = (store_root() / "entries" / record["entry_id"] / "hf" /
                ("models--" + task.model_id.replace("/", "--")) / "snapshots" / task.model_revision)
    return ProcessInfo(str(root / "venv/bin/python"), str(root / "code"), str(snapshot),
                       manifest["build_fingerprint"], manifest)


def restore_process(snapshot: dict) -> ProcessInfo:
    from acprof.host.model_store import verify_entry
    info = ProcessInfo(**snapshot)
    root = Path(info.python).parent.parent.parent
    verify_environment(root, info.runtime_environment)
    verify_entry(info.runtime_environment)
    expected = content_digest({"environment": info.runtime_environment["environment_id"],
                               "model": info.runtime_environment["model_store"]["plan_sha256"]})
    if expected != info.environment_id or not Path(info.model_path).is_dir():
        raise ValueError("recorded Process/model identity changed")
    return info
