"""Download source policy; no measurement, framework imports, or network at import."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Mapping
from urllib.parse import urlsplit

DIRECT_HOSTS = frozenset({"hf-mirror.com", "docker.m.daocloud.io"})
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


def direct_hosts(environ: Mapping[str, str] | None = None) -> set[str]:
    env = os.environ if environ is None else environ
    extra = env.get("ACPROF_DIRECT_HOSTS", "").split(",")
    hosts = set(DIRECT_HOSTS)
    for host in extra:
        host = host.strip().lower()
        if not host:
            continue
        if not re.fullmatch(r"[a-z0-9.-]+", host):
            raise DownloadPolicyError("ACPROF_DIRECT_HOSTS requires exact hostnames")
        hosts.add(host)
    return hosts


def source_route(url: str, environ: Mapping[str, str] | None = None) -> str:
    """Conservative routing expectation; unknown hosts are potentially proxied."""
    return "DIRECT" if urlsplit(url).hostname in direct_hosts(environ) else "PROXY"


def require_source_transition(previous: str, current: str, *, allow_proxy: bool | None = None) -> None:
    if previous == current:
        return
    old, new = source_route(previous), source_route(current)
    print(f"[network] source fallback: {urlsplit(previous).hostname} ({old}) -> {urlsplit(current).hostname} ({new})", flush=True)
    allowed = os.environ.get("ACPROF_ALLOW_PROXY_FALLBACK") == "1" if allow_proxy is None else allow_proxy
    if old == "DIRECT" and new == "PROXY" and not allowed:
        raise DownloadPolicyError("DIRECT -> PROXY fallback 已停止；请显式选择源，或设置 ACPROF_ALLOW_PROXY_FALLBACK=1 后重试")


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
        return {**asdict(self), "host": urlsplit(self.url).hostname, "route": source_route(self.url)}


def summarize_downloads(sources: list[DownloadSource]) -> dict:
    def total(items):
        values = [item.estimated_bytes for item in items]
        return sum(values) if all(type(value) is int and value >= 0 for value in values) else None
    return {
        "schema_version": 1,
        "expected_download_bytes": total(sources),
        "direct_download_bytes": total([s for s in sources if source_route(s.url) == "DIRECT"]),
        "proxy_download_bytes": total([s for s in sources if source_route(s.url) == "PROXY"]),
        "sources": [s.to_dict() for s in sources],
        "route_evidence": "configured expectation; upstream VPN routing must be verified separately",
    }


def enforce_download_budget(plan: dict, maximum: str | int | None) -> None:
    limit = parse_bytes(maximum)
    expected = plan["expected_download_bytes"]
    if limit is not None and (expected is None or expected > limit):
        raise DownloadPolicyError(f"下载尚未开始：expected_download_bytes={expected}, max_download={limit}；未知大小也不能证明符合预算")
