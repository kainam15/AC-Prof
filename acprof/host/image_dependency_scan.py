"""Read dependency evidence from an immutable Docker image on demand.

Only AC-Prof platform/runtime images with recognized build history may be
inspected. Records are tied to a Docker daemon and immutable image ID.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from acprof.host.image_management import DockerConnection, ManagedImage, _run

SCAN_VERSION = 1
MAX_RECORD_BYTES = 2 * 1024 * 1024
PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+:-]{0,199}$")


@dataclass(frozen=True)
class DependencySnapshot:
    python: tuple[tuple[str, str], ...]
    system: tuple[tuple[str, str], ...]
    source: str


def _packages(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or len(value) > 20000:
        raise ValueError("dependency package map missing or oversized")
    pairs = []
    for name, version in value.items():
        if (not isinstance(name, str) or not PACKAGE_NAME.fullmatch(name)
                or not isinstance(version, str) or not 0 < len(version) < 256
                or any(ord(char) < 32 for char in version)):
            raise ValueError("invalid package name or version")
        pairs.append((name, version))
    return tuple(sorted(pairs))


def _cache_path(daemon_id: str, image_id: str) -> Path:
    # Hash daemon IDs so filesystem paths do not expose arbitrary Docker names.
    root = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return root / "acprof" / "image-dependencies" / f"v{SCAN_VERSION}" / hashlib.sha256(daemon_id.encode()).hexdigest() / (image_id.removeprefix("sha256:") + ".json")


def _read_cache(daemon_id: str, image_id: str) -> DependencySnapshot | None:
    path = _cache_path(daemon_id, image_id)
    try:
        if not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
            return None
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            return None
        if (value.get("version") != SCAN_VERSION or value.get("daemon") != daemon_id
                or value.get("image") != image_id or value.get("source") not in {"manifest", "scan"}):
            return None
        return DependencySnapshot(_packages(value["python"]), _packages(value["system"]), value["source"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cache(daemon_id: str, image_id: str, snapshot: DependencySnapshot) -> None:
    path = _cache_path(daemon_id, image_id)
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = json.dumps({"version": SCAN_VERSION, "daemon": daemon_id, "image": image_id,
                           "source": snapshot.source,
                           "python": dict(snapshot.python), "system": dict(snapshot.system)}, sort_keys=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pending-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, path)
    except OSError:
        # Cache is optional: a read-only filesystem must not hide valid evidence.
        pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _docker_read(connection: DockerConnection, image_id: str, entrypoint: str,
                 args: list[str], *, timeout: int = 20) -> str:
    command = [*connection.arguments, "run", "--rm", "--pull", "never",
               "--network", "none", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "64",
               "--memory", "512m", "--cpus", "1", "--user", "65534:65534",
               "--entrypoint", entrypoint, image_id, *args]
    value = _run(command, timeout=timeout)
    if len(value.encode()) > MAX_RECORD_BYTES:
        raise ValueError("image dependency output oversized")
    return value


# -I prevents PYTHONPATH and user site; -S avoids sitecustomize execution.
# Scan site-packages explicitly so -S does not hide installed distributions.
SCAN_CODE = """import importlib.metadata as m, site, subprocess, json
paths = site.getsitepackages()
python = {d.metadata['Name'].lower().replace('_', '-'): d.version
          for d in m.distributions(path=paths)}
lines = subprocess.check_output(['dpkg-query', '-W',
    '-f=${Package}:${Architecture}\\t${Version}\\t${db:Status-Status}\\n'],
    text=True, timeout=10).splitlines()
system = {name: version for line in lines
          for name, version, state in [line.split('\\t')]
          if state == 'installed'}
print(json.dumps({'python': python, 'system': system}))
"""


def cached_snapshot(daemon_id: str, image_id: str) -> DependencySnapshot | None:
    """Return only persisted evidence; safe during periodic inventory refresh."""
    return _read_cache(daemon_id, image_id)


def read_snapshot(connection: DockerConnection, daemon_id: str,
                  item: ManagedImage) -> tuple[DependencySnapshot | None, str]:
    if item.kind not in {"base", "runtime"} or not item.acprof or item.dependency_stage != item.kind:
        return None, "构建步骤无法核验，未运行镜像内程序。"
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", item.image_id):
        return None, "镜像 ID 无效。"
    cached = _read_cache(daemon_id, item.image_id)
    if cached is not None:
        return cached, ""
    kind = "platform" if item.kind == "base" else "environment"
    reason = ""
    try:
        raw = _docker_read(connection, item.image_id, "cat", [f"/opt/acprof/{kind}-manifest.json"])
        manifest = json.loads(raw)
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                or manifest.get("platform_id") != item.platform_id
                or manifest.get("platform_build_fingerprint") != item.platform_key
                or (item.kind == "runtime" and (
                    manifest.get("environment_id") != item.environment_id
                    or manifest.get("environment_build_fingerprint") != item.environment_key))):
            raise ValueError("manifest identity mismatches immutable image labels")
        snapshot = DependencySnapshot(_packages(manifest["packages"]),
                                      _packages(manifest["system_packages"]), "manifest")
        _write_cache(daemon_id, item.image_id, snapshot)
        return snapshot, ""
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        reason = f"构建记录不可用或身份不匹配（{type(error).__name__}）。"
    # Only run the isolated interpreter for recognized, project-managed images.
    try:
        raw = _docker_read(connection, item.image_id, "python", ["-I", "-S", "-B", "-c", SCAN_CODE], timeout=30)
        output = json.loads(raw)
        snapshot = DependencySnapshot(_packages(output["python"]), _packages(output["system"]), "scan")
        _write_cache(daemon_id, item.image_id, snapshot)
        return snapshot, ""
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        return None, reason + f" 实际扫描失败（{type(error).__name__}）。"
