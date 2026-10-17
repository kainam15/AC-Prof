"""构建时安装与核验锁；由 BuildKit bind mount 使用，不留在依赖镜像中。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import URLError
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
from network_policy import (
    DEPENDENCY_USER_AGENT,
    dependency_artifact_url,
    is_direct_artifact_mirror,
    require_direct_artifact_mirror_redirect,
    require_source_transition,
)


class PolicyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_direct_artifact_mirror_redirect(req.full_url, newurl)
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


def _retryable_download_error(error):
    # HTTP/certificate/policy/hash errors are not transient transport failures.
    if isinstance(error, URLError):
        error = error.reason
    return isinstance(error, (TimeoutError, ConnectionError, ssl.SSLEOFError, IncompleteRead))


def cached_artifact(entry, path, category):
    """Download an exact SHA256 artifact; retain verified-origin partial bytes for HTTP Range resume.

    Cache mounts survive interrupted Docker builds. A partial is never treated as
    a finished artifact, and a resumed body is accepted only with a matching
    Content-Range. Final publication still requires the complete SHA256.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and _digest(path) == entry["sha256"]:
        sidecar = path.with_suffix(path.suffix + ".source.json")
        origin = json.loads(sidecar.read_text()).get("actual_source_host") if sidecar.is_file() else None
        _record_transfer(entry, category, 0, origin, "hit")
        print(f"[dependency-cache] {category} hit name={entry['name']} saved_bytes={path.stat().st_size}", flush=True)
        return path

    source_url = dependency_artifact_url(category, entry["url"])
    direct_mirror = is_direct_artifact_mirror(source_url)
    temporary = path.with_suffix(path.suffix + ".part")
    partial_meta = temporary.with_suffix(temporary.suffix + ".source.json")

    def discard_partial():
        temporary.unlink(missing_ok=True)
        partial_meta.unlink(missing_ok=True)

    try:
        metadata = json.loads(partial_meta.read_text()) if partial_meta.is_file() else {}
    except (OSError, ValueError, TypeError):
        metadata = {}
    if not (temporary.is_file() and metadata.get("url") == source_url
            and metadata.get("sha256") == entry["sha256"]):
        if temporary.is_file() and metadata.get("sha256") == entry["sha256"] and metadata.get("url") != source_url:
            print("[dependency-cache] source changed; discarding incompatible partial (complete hashes remain cached)",
                  flush=True)
        discard_partial()
        metadata = {}
    elif _digest(temporary) == entry["sha256"]:
        origin = metadata.get("actual_source_host")
        temporary.replace(path)
        partial_meta.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".source.json").write_text(
            json.dumps({"actual_source_host": origin}) + "\n"
        )
        _record_transfer(entry, category, 0, origin, "hit")
        return path

    # A selected domestic mirror must not consume the configured HTTP(S) VPN proxy.
    # No silent fallback to the official URL (and therefore no surprise VPN bytes).
    handlers = [urllib.request.ProxyHandler({})] if direct_mirror else []
    opener = urllib.request.build_opener(*handlers, PolicyRedirectHandler())
    actual_source = None
    try:
        for attempt in range(1, 4):
            offset = temporary.stat().st_size if temporary.is_file() else 0
            headers = {"User-Agent": DEPENDENCY_USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(source_url, headers=headers)
            print(
                f"[dependency-cache] {category} fetching name={entry['name']} "
                f"resume_bytes={offset} source={urlsplit(source_url).hostname} "
                f"transport={'direct-mirror' if direct_mirror else 'system-network'}",
                flush=True,
            )
            try:
                with opener.open(request, timeout=60) as source:
                    actual_source = urlsplit(
                        getattr(source, "geturl", lambda: source_url)()
                    ).hostname
                    if direct_mirror and actual_source != urlsplit(source_url).hostname:
                        raise ValueError("PyPI mirror response changed host; refusing external fallback")
                    status = getattr(source, "status", 200)
                    response_headers = getattr(source, "headers", {}) or {}
                    match = None
                    if offset and status == 206:
                        content_range = response_headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if match is None or int(match[1]) != offset:
                            raise ValueError("HTTP resume Content-Range does not match cached offset")
                        if metadata.get("actual_source_host") not in (None, actual_source):
                            raise ValueError("HTTP resume origin changed; cannot preserve source provenance")
                        mode = "ab"
                    elif status == 200:
                        mode = "wb"  # A server may ignore Range; never append a full response.
                        offset = 0
                    else:
                        raise ValueError(f"unexpected artifact HTTP status: {status}")
                    partial_meta.write_text(json.dumps({
                        "url": source_url, "sha256": entry["sha256"],
                        "actual_source_host": actual_source,
                    }) + "\n")
                    metadata = json.loads(partial_meta.read_text())
                    expected = response_headers.get("Content-Length")
                    expected_bytes = int(expected) if expected and str(expected).isdigit() else None
                    received = 0
                    next_progress = 16 * 1024 * 1024
                    with temporary.open(mode) as target:
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            target.write(block)
                            received += len(block)
                            if received >= next_progress:
                                print(
                                    f"[dependency-cache] {category} progress name={entry['name']} "
                                    f"cached_bytes={offset + received}",
                                    flush=True,
                                )
                                next_progress += 16 * 1024 * 1024
                    if expected_bytes is not None and received != expected_bytes:
                        raise IncompleteRead(b"", expected_bytes - received)
                    if match is not None and temporary.stat().st_size != int(match[3]):
                        raise IncompleteRead(b"", int(match[3]) - temporary.stat().st_size)
            except (OSError, IncompleteRead) as exc:
                if attempt == 3 or not _retryable_download_error(exc):
                    raise
                delay = 2 ** (attempt - 1)
                reason = type(exc.reason if isinstance(exc, URLError) else exc).__name__
                print(
                    f"[dependency-cache] {category} retry name={entry['name']} "
                    f"attempt={attempt}/3 source={urlsplit(source_url).hostname} "
                    f"reason={reason} retry_in_s={delay}",
                    flush=True,
                )
                time.sleep(delay)
            else:
                break
        if _digest(temporary) != entry["sha256"]:
            raise ValueError(f"artifact SHA256 mismatch: {entry['name']}")
        temporary.replace(path)
        path.with_suffix(path.suffix + ".source.json").write_text(
            json.dumps({"actual_source_host": actual_source}) + "\n"
        )
        _record_transfer(entry, category, path.stat().st_size, actual_source, "miss")
        print(
            f"[dependency-cache] {category} miss name={entry['name']} "
            f"actual_bytes={path.stat().st_size} source={actual_source}",
            flush=True,
        )
        return path
    finally:
        error = sys.exc_info()[1]
        if (error is None or not _retryable_download_error(error)
                or not temporary.is_file() or temporary.stat().st_size == 0):
            discard_partial()


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
