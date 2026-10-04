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
                "route": "externally-managed" if self.category == "oci" else source_route(self.url),
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
