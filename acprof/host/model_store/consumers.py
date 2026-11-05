"""Durable Docker consumer records complement process-scoped Model Store locks."""
from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from acprof.artifacts import atomic_write_json, loads_finite_json
from acprof.host.container_lifecycle import (
    LIFECYCLE_LABEL,
    OWNER_PREFIX,
    container_owner_has_exited,
    container_owner_labels,
    owner_containers_absent,
)
from acprof.host.model_store.paths import StorePaths, store_paths
from acprof.models.evidence import content_digest

_PREFIX = ".docker-consumer-"
_NAME = re.compile(r"\.docker-consumer-([a-f0-9]{32})\.json")
_OWNER_KEYS = {LIFECYCLE_LABEL, *(OWNER_PREFIX + key for key in ("host", "uid", "boot", "pid", "start"))}
_MAX_BYTES = 4096


@dataclass(frozen=True)
class DockerConsumer:
    root: Path
    entry: Path
    name: str
    record: dict


def create_consumer(root: Path, key: str) -> DockerConsumer:
    """Publish before launch while the caller holds the store lock and entry flock."""
    identity = uuid4().hex
    entry = Path("entries") / key
    name = f"{_PREFIX}{identity}.json"
    record = {"schema_version": 1, "consumer_id": identity, "entry_id": key,
              "owner_labels": container_owner_labels()}
    record["record_sha256"] = content_digest(record)
    with store_paths(root, lambda: None) as paths, paths.directory(entry) as descriptor:
        paths.check_root()
        # Keep the existing atomic publication contract anchored to this directory.
        # If publication fails after link/fsync, the valid record stays conservative.
        atomic_write_json(Path(f"/proc/self/fd/{descriptor}") / name, record, overwrite=False)
        paths.check_root()
        return DockerConsumer(paths.root, entry, name, record)


def _read_record(descriptor: int, name: str, key: str) -> dict:
    match = _NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid Model Store Docker consumer filename: {name}")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=descriptor)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError("Model Store Docker consumer must be a regular file owned only by this user")
        raw = stream.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError("Model Store Docker consumer exceeds 4096 bytes")
    record = loads_finite_json(raw.decode("utf-8"))
    if (not isinstance(record, dict)
            or set(record) != {"schema_version", "consumer_id", "entry_id", "owner_labels", "record_sha256"}
            or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["consumer_id"] != match[1] or record["entry_id"] != key):
        raise ValueError("Invalid Model Store Docker consumer identity or schema")
    if record["record_sha256"] != content_digest({key: value for key, value in record.items() if key != "record_sha256"}):
        raise ValueError("Model Store Docker consumer integrity check failed")
    owner = record["owner_labels"]
    if (not isinstance(owner, dict) or set(owner) != _OWNER_KEYS
            or not all(isinstance(value, str) and value for value in owner.values())
            or owner[LIFECYCLE_LABEL] != "1"
            or not all(owner[OWNER_PREFIX + field].isdecimal() for field in ("uid", "pid", "start"))
            or int(owner[OWNER_PREFIX + "pid"]) <= 0):
        raise ValueError("Invalid Model Store Docker consumer owner")
    return record


def remove_consumer(consumer: DockerConsumer) -> None:
    """Clear only this mount's record after its caller has confirmed cleanup."""
    with store_paths(consumer.root, lambda: None) as paths, paths.directory(consumer.entry) as descriptor:
        paths.check_root()
        try:
            record = _read_record(descriptor, consumer.name, consumer.entry.name)
        except FileNotFoundError:
            os.fsync(descriptor)  # Retry a successful unlink whose directory fsync failed.
            return
        if record != consumer.record:
            raise ValueError("Model Store Docker consumer changed before release")
        os.unlink(consumer.name, dir_fd=descriptor)
        os.fsync(descriptor)
        paths.check_root()


class DockerConsumerInspector:
    """Read-only, invocation-local evidence for GC and explicit view rebuilding."""

    def __init__(self) -> None:
        self._current: dict[str, str] | None = None
        self._docker_verified = False
        self._protected: dict[tuple[tuple[str, str], ...], bool] = {}

    def protects(self, paths: StorePaths, entry: Path, *,
                 children: list[tuple[Path, os.stat_result]] | None = None) -> bool:
        records = []
        entries = paths.scan(entry, missing_ok=True) if children is None else children
        for relative, _info in entries:
            paths.check_cancelled()
            if relative.name.startswith(_PREFIX):
                with paths.directory(entry) as descriptor:
                    records.append(_read_record(descriptor, relative.name, entry.name))
        return any(self._owner_protected(record["owner_labels"], paths.check_cancelled) for record in records)

    def _owner_protected(self, owner: dict[str, str], check_cancelled: Callable[[], None]) -> bool:
        check_cancelled()
        if self._current is None:
            self._current = container_owner_labels()
        if not container_owner_has_exited(owner, self._current):
            return True
        identity = tuple(sorted(owner.items()))
        if identity not in self._protected:
            from acprof.host.preflight import require_native_docker
            from acprof.system.command import run_command
            if not self._docker_verified:
                try:
                    require_native_docker()
                except SystemExit as exc:
                    raise RuntimeError("Cannot verify Model Store Docker consumers against the local daemon") from exc
                self._docker_verified = True
            check_cancelled()
            self._protected[identity] = not owner_containers_absent(owner, run_command)
        return self._protected[identity]
