"""Download source policy; no measurement, framework imports, or network at import."""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal
from urllib import request as urllib_request
from urllib.parse import urlsplit

DEPENDENCY_USER_AGENT = "acprof-dependency-downloader/1.0"

TENCENT_PYPI_ARTIFACT_HOST = "mirrors.cloud.tencent.com"
TENCENT_PYPI_ARTIFACT_PREFIX = f"https://{TENCENT_PYPI_ARTIFACT_HOST}/pypi"
ALIYUN_PYTORCH_ARTIFACT_PREFIX = "https://mirrors.aliyun.com/pytorch-wheels"
TENCENT_DEBIAN_ARTIFACT_PREFIX = f"https://{TENCENT_PYPI_ARTIFACT_HOST}/debian"

_MIRROR_OPTIONS = {
    "ACPROF_PYPI_ARTIFACT_MIRROR": {"", "tencent"},
    "ACPROF_PYTORCH_ARTIFACT_MIRROR": {"", "aliyun"},
    "ACPROF_DEBIAN_ARTIFACT_MIRROR": {"", "tencent"},
}


def configured_artifact_mirrors(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Validate all download options before starting an expensive build."""
    env = os.environ if environ is None else environ
    result = {}
    for key, values in _MIRROR_OPTIONS.items():
        mode = env.get(key, "").strip().lower()
        if mode not in values:
            raise DownloadPolicyError(f"{key} has an unsupported mirror setting")
        result[key] = mode
    return result


def dependency_artifact_url(category: str, locked_url: str, environ: Mapping[str, str] | None = None) -> str:
    """Opt-in transport remapping: locked URLs and SHA256 remain immutable."""
    settings = configured_artifact_mirrors(environ)
    parsed = urlsplit(locked_url)
    if category == "python" and parsed.hostname == "files.pythonhosted.org" and settings[
        "ACPROF_PYPI_ARTIFACT_MIRROR"
    ] == "tencent":
        if (parsed.scheme != "https" or parsed.netloc != "files.pythonhosted.org"
                or not parsed.path.startswith("/packages/") or ".." in parsed.path.split("/")
                or parsed.query or parsed.fragment):
            raise DownloadPolicyError("cannot mirror an unexpected PyPI artifact URL")
        return TENCENT_PYPI_ARTIFACT_PREFIX + parsed.path
    if category == "python" and parsed.hostname == "download-r2.pytorch.org" and settings[
        "ACPROF_PYTORCH_ARTIFACT_MIRROR"
    ] == "aliyun":
        if (parsed.scheme != "https" or parsed.netloc != "download-r2.pytorch.org"
                or not parsed.path.startswith("/whl/") or ".." in parsed.path.split("/")
                or parsed.query or parsed.fragment):
            raise DownloadPolicyError("cannot mirror an unexpected PyTorch artifact URL")
        return ALIYUN_PYTORCH_ARTIFACT_PREFIX + "/" + parsed.path[len("/whl/"):]
    if category == "debian" and parsed.hostname == "snapshot.debian.org" and settings[
        "ACPROF_DEBIAN_ARTIFACT_MIRROR"
    ] == "tencent":
        if (parsed.scheme != "https" or parsed.netloc != "snapshot.debian.org"
                or not re.fullmatch(r"/archive/debian(?:-security)?/[0-9]{8}T[0-9]{6}Z/pool/.+\.deb",
                                    parsed.path)
                or ".." in parsed.path.split("/") or parsed.query or parsed.fragment):
            raise DownloadPolicyError("cannot mirror an unexpected Debian snapshot URL")
        return TENCENT_DEBIAN_ARTIFACT_PREFIX + "/pool/" + parsed.path.split("/pool/", 1)[1]
    return locked_url


def is_direct_artifact_mirror(url: str, environ: Mapping[str, str] | None = None) -> bool:
    settings = configured_artifact_mirrors(environ)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.query or parsed.fragment:
        return False
    return (
        (settings["ACPROF_PYPI_ARTIFACT_MIRROR"] == "tencent"
         and parsed.netloc == TENCENT_PYPI_ARTIFACT_HOST
         and parsed.path.startswith("/pypi/packages/"))
        or (settings["ACPROF_PYTORCH_ARTIFACT_MIRROR"] == "aliyun"
            and parsed.netloc == "mirrors.aliyun.com"
            and parsed.path.startswith("/pytorch-wheels/"))
        or (settings["ACPROF_DEBIAN_ARTIFACT_MIRROR"] == "tencent"
            and parsed.netloc == TENCENT_PYPI_ARTIFACT_HOST
            and parsed.path.startswith("/debian/pool/"))
    )


def require_direct_artifact_mirror_redirect(previous: str, current: str) -> None:
    parsed = urlsplit(previous)
    if ((parsed.netloc == TENCENT_PYPI_ARTIFACT_HOST
         and (parsed.path.startswith("/pypi/packages/") or parsed.path.startswith("/debian/pool/")))
            or (parsed.netloc == "mirrors.aliyun.com" and parsed.path.startswith("/pytorch-wheels/"))):
        nxt = urlsplit(current)
        if nxt.scheme != "https" or nxt.netloc != parsed.netloc:
            raise DownloadPolicyError("direct artifact mirror redirected outside HTTPS mirror host")


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
    if getattr(urllib_request, "proxy_bypass_environment")(parsed.netloc, proxies):
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
                          "direct-socket" if is_direct_artifact_mirror(self.url)
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
