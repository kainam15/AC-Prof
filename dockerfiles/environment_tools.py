"""构建时安装与核验锁；由 BuildKit bind mount 使用，不留在依赖镜像中。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlsplit

from dependency_locks import (
    content_digest,
    normalized_name,
    package_versions,
    read_python_lock,
    read_system_lock,
    require_exact_packages,
    require_parent_subset,
    system_lock_identity,
)
from network_policy import DEPENDENCY_USER_AGENT, require_source_transition


class PolicyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_source_transition(req.full_url, newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_transfer(entry, category, actual_bytes, source_host, cache_status):
    destination = Path("/opt/acprof/download-sources.jsonl")
    with destination.open("a") as stream:
        stream.write(json.dumps({"category": category, "url": entry["url"], "sha256": entry["sha256"],
            "actual_source_host": source_host, "actual_payload_bytes": actual_bytes,
            "wire_bytes": None, "cache_status": cache_status}) + "\n")


def cached_artifact(entry, path, category):
    """Content-addressed payload cache with guarded redirects and no source fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and _digest(path) == entry["sha256"]:
        sidecar = path.with_suffix(path.suffix + ".source.json")
        origin = json.loads(sidecar.read_text()).get("actual_source_host") if sidecar.is_file() else None
        _record_transfer(entry, category, 0, origin, "hit")
        print(f"[dependency-cache] {category} hit name={entry['name']} saved_bytes={path.stat().st_size}", flush=True)
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    opener = urllib.request.build_opener(PolicyRedirectHandler())
    request = urllib.request.Request(entry["url"], headers={"User-Agent": DEPENDENCY_USER_AGENT})
    try:
        with opener.open(request, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
            actual_source = urlsplit(source.geturl()).hostname
        if _digest(temporary) != entry["sha256"]:
            raise ValueError(f"artifact SHA256 mismatch: {entry['name']}")
        temporary.replace(path)
        path.with_suffix(path.suffix + ".source.json").write_text(json.dumps({"actual_source_host": actual_source}) + "\n")
        _record_transfer(entry, category, path.stat().st_size, actual_source, "miss")
        print(f"[dependency-cache] {category} miss name={entry['name']} actual_bytes={path.stat().st_size} source={actual_source}", flush=True)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def installed_packages() -> dict[str, str]:
    result = {}
    for dist in importlib.metadata.distributions():
        name = normalized_name(dist.metadata["Name"])
        if name in result:
            raise ValueError(f"duplicate installed Python distribution: {name}")
        result[name] = dist.version
    return dict(sorted(result.items()))


def system_packages() -> dict[str, str]:
    output = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${Package}:${Architecture}\t${Version}\t${db:Status-Status}\n"], text=True,
    )
    return {name: version for line in output.splitlines()
            for name, version, status in [line.split("\t")] if status == "installed"}


def install_system(path: Path) -> None:
    lock = read_system_lock(path)
    cache = Path("/root/.cache/acprof/debs")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        files = []
        for entry in lock["artifacts"]:
            cached = cache / (entry["sha256"] + ".deb")
            cached_artifact(entry, cached, "debian")
            # APT 缓存名包含版本 epoch（例如 1%3a），上游 URL 的文件名可能省略它。
            version = entry["version"].replace(":", "%3a")
            destination = Path(directory) / f"{entry['name']}_{version}_{entry['architecture']}.deb"
            shutil.copyfile(cached, destination)
            files.append(str(destination))
        subprocess.run(["apt-get", "-o", f"Dir::Cache::archives={directory}",
                        "install", "-y", "--no-install-recommends", "--no-download", *files],
                       check=True, env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
    require_exact_packages(lock["packages"], system_packages(), kind="system")
    for existing in Path("/etc/apt/sources.list.d").glob("*"):
        existing.unlink()
    Path("/etc/apt/sources.list").write_text("".join(
        f"deb [check-valid-until=no signed-by=/usr/share/keyrings/debian-archive-keyring.pgp] {source['url']} {suite} main\n"
        for source in lock["sources"] for suite in source["suites"]
    ))


def install_python(kind: str) -> None:
    lock = read_python_lock("/opt/acprof/requirements.lock")
    if kind == "environment":
        parent = read_python_lock("/opt/acprof/platform-requirements.lock")
        require_parent_subset(parent, lock)
        parent_names = set(package_versions(parent))
        delta = [entry for entry in lock if entry["name"] not in parent_names]
    else:
        current = installed_packages()
        delta = [entry for entry in lock if current.get(entry["name"]) != entry["version"]]
    if delta:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "install.txt"
            lines = []
            for entry in delta:
                filename = unquote(urlsplit(entry["url"]).path.rsplit("/", 1)[-1])
                cached = cached_artifact(entry, Path("/root/.cache/pip/acprof-artifacts") / entry["sha256"] / filename, "python")
                lines.append(f"{entry['name']} @ {cached.as_uri()} --hash=sha256:{entry['sha256']}")
            path.write_text("\n".join(lines) + "\n")
            subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                            "--no-index", "--no-deps", "--require-hashes", "--only-binary=:all:",
                            "--progress-bar", "off", "-r", str(path)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    require_exact_packages(package_versions(lock), installed_packages())


def write_manifest(kind: str) -> None:
    expected = json.loads(Path("/opt/acprof/expectation.json").read_text())
    if platform.python_version() != expected["python_version"] or platform.machine() != "x86_64":
        raise ValueError("actual Python/architecture differs from platform definition")
    lock = read_python_lock("/opt/acprof/requirements.lock")
    system = read_system_lock("/opt/acprof/system.lock")
    packages = installed_packages()
    require_exact_packages(package_versions(lock), packages)
    actual_system = system_packages()
    require_exact_packages(system["packages"], actual_system, kind="system")
    manifest = {**expected, "schema_version": 1, "packages": packages, "system_packages": actual_system,
                "system_lock_sha256": content_digest(system_lock_identity(system)),
                "dependency_lock_sha256": hashlib.sha256(Path("/opt/acprof/requirements.lock").read_bytes()).hexdigest(),
                "resolved_packages_sha256": hashlib.sha256(json.dumps(packages, sort_keys=True).encode()).hexdigest()}
    sources = Path("/opt/acprof/download-sources.jsonl")
    manifest["build_download_sources"] = [json.loads(line) for line in sources.read_text().splitlines()] if sources.exists() else []
    Path(f"/opt/acprof/{kind}-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    if kind == "platform":
        shutil.copyfile("/opt/acprof/requirements.lock", "/opt/acprof/platform-requirements.lock")


if __name__ == "__main__":
    if sys.argv[1] == "system":
        install_system(Path(sys.argv[2]))
    else:
        install_python(sys.argv[1])
        write_manifest(sys.argv[1])
