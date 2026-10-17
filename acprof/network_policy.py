"""Download source policy; no measurement, framework imports, or network at import."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Mapping
from urllib.parse import urlsplit
from urllib.request import proxy_bypass_environment

DEPENDENCY_USER_AGENT = "acprof-dependency-downloader/1.0"

TENCENT_PYPI_ARTIFACT_HOST = "mirrors.cloud.tencent.com"
TENCENT_PYPI_ARTIFACT_PREFIX = f"https://{TENCENT_PYPI_ARTIFACT_HOST}/pypi"


def configured_pypi_artifact_mirror(environ: Mapping[str, str] | None = None) -> str:
    """Validate the one approved optional mirror setting even before a download."""
    env = os.environ if environ is None else environ
    mode = env.get("ACPROF_PYPI_ARTIFACT_MIRROR", "").strip().lower()
    if mode not in {"", "tencent"}:
        raise DownloadPolicyError("ACPROF_PYPI_ARTIFACT_MIRROR must be empty or tencent")
    return mode


def dependency_artifact_url(category: str, locked_url: str, environ: Mapping[str, str] | None = None) -> str:
    """Opt-in transport mirror for PyPI's immutable /packages/ artifacts only.

    Version and digest remain those from the official lock; final SHA256 is
    checked by the caller. Unknown modes fail closed rather than using a proxy.
    """
    mode = configured_pypi_artifact_mirror(environ)
    if mode != "tencent" or category != "python":
        return locked_url
    parsed = urlsplit(locked_url)
    if parsed.hostname != "files.pythonhosted.org":
        return locked_url  # PyTorch dedicated index and other registries are unchanged.
    if (parsed.scheme != "https" or parsed.netloc != "files.pythonhosted.org"
            or not parsed.path.startswith("/packages/") or ".." in parsed.path.split("/")
            or parsed.query or parsed.fragment):
        raise DownloadPolicyError("cannot mirror an unexpected PyPI artifact URL")
    return TENCENT_PYPI_ARTIFACT_PREFIX + parsed.path


def is_direct_artifact_mirror(url: str, environ: Mapping[str, str] | None = None) -> bool:
    return (configured_pypi_artifact_mirror(environ) == "tencent"
            and urlsplit(url).scheme == "https"
            and urlsplit(url).netloc == TENCENT_PYPI_ARTIFACT_HOST
            and urlsplit(url).path.startswith("/pypi/packages/"))


def require_direct_artifact_mirror_redirect(previous: str, current: str) -> None:
    if urlsplit(previous).hostname == TENCENT_PYPI_ARTIFACT_HOST:
        parsed = urlsplit(current)
        if parsed.scheme != "https" or parsed.netloc != TENCENT_PYPI_ARTIFACT_HOST:
            raise DownloadPolicyError("PyPI mirror redirected outside its HTTPS host; refusing proxy fallback")



class DownloadPolicyError(ValueError):
    """A forbidden source or unbounded download must stop before payload transfer."""


def parse_bytes(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)?", str(value).strip(), re.I)
    if not match:
        raise ValueError("下载容量格式无效；例如 5GB、5GiB 或 0（禁止下载）")
    unit = (match[2] or "B").upper()
    power = {"B": 0, "K": 1, "M": 2, "G": 3, "T": 4}[unit[0]]
    result = Decimal(match[1]) * ((1024 if "I" in unit else 1000) ** power)
    if result != int(result):
        raise ValueError("下载容量必须是整数 bytes")
    return int(result)


def source_route(url: str, environ: Mapping[str, str] | None = None) -> str:
    """Application socket configuration only, never a claim about VPN/egress.

    Match standard proxy environment precedence without modifying the process
    or exposing credentials. Transparent proxies are outside our observation.
    """
    env = os.environ if environ is None else environ
    proxies = {key.lower()[:-6]: value for key, value in env.items()
               if key.lower().endswith("_proxy") and value}
    if "REQUEST_METHOD" in env:
        proxies.pop("http", None)
    for key, value in env.items():
        if key.endswith("_proxy"):
            if value:
                proxies[key[:-6]] = value
            else:
                proxies.pop(key[:-6], None)
    parsed = urlsplit(url)
    if proxy_bypass_environment(parsed.netloc, proxies):
        return "direct-socket"
    return "explicit-proxy" if proxies.get(parsed.scheme) or proxies.get("all") else "direct-socket"


def require_source_transition(previous: str, current: str) -> None:
    if previous == current:
        return
    print(f"[network] endpoint fallback: {urlsplit(previous).hostname} -> {urlsplit(current).hostname}; system-network", flush=True)


@dataclass
class DownloadSource:
    category: str
    url: str
    estimated_bytes: int | None
    cached_bytes: int = 0
    cache_status: str = "unknown"
    cacheable: bool = True
    actual_bytes: int | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {**asdict(self), "host": urlsplit(self.url).hostname,
                "route": ("externally-managed" if self.category == "oci" else
                          "direct-socket" if self.category == "python" and is_direct_artifact_mirror(self.url)
                          else source_route(self.url)),
                "public_egress": "unknown"}


def summarize_downloads(sources: list[DownloadSource]) -> dict:
    def total(items):
        values = [item.estimated_bytes for item in items]
        return sum(values) if all(type(value) is int and value >= 0 for value in values) else None
    return {
        "schema_version": 2,
        "expected_download_bytes": total(sources),
        "sources": [s.to_dict() for s in sources],
        "network": "system-network",
        "public_egress": "externally-managed",
        "route_evidence": "application proxy configuration only; public egress unknown",
    }


def enforce_download_budget(plan: dict, maximum: str | int | None) -> None:
    limit = parse_bytes(maximum)
    expected = plan["expected_download_bytes"]
    if limit is not None and (expected is None or expected > limit):
        raise DownloadPolicyError(f"下载尚未开始：expected_download_bytes={expected}, max_download={limit}；未知大小也不能证明符合预算")
