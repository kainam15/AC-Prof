"""Shared, explicit Hugging Face Hub endpoint policy (standard library only)."""
from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlsplit

HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"
HF_OFFICIAL_ENDPOINT = "https://huggingface.co"
HF_OFFICIAL_ENDPOINTS = (HF_OFFICIAL_ENDPOINT, "https://www.huggingface.co", "https://hf.co")
HF_DOWNLOAD_MODES = ("auto", "mirror-only", "mirror-preferred", "official")


def hf_origin(url: str) -> tuple[str, str, int] | None:
    """HTTP origin identity; malformed or credential-bearing URLs have none."""
    try:
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port
        if (parsed.scheme not in {"http", "https"} or not host
                or parsed.username is not None or parsed.password is not None):
            return None
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        return parsed.scheme, host, port
    except ValueError:
        return None


def hf_storage_host(url: str) -> bool:
    """Organization-controlled storage namespaces, excluding Hub hostnames."""
    origin = hf_origin(url)
    if origin is None:
        return False
    host = origin[1]
    return host not in {"huggingface.co", "www.huggingface.co", "hf.co"} and any(
        host.endswith("." + domain) for domain in ("hf.co", "huggingface.co")
    )


def hf_download_mode(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    mode = env.get("HF_DOWNLOAD_MODE", "").strip() or "auto"
    if mode not in HF_DOWNLOAD_MODES:
        raise ValueError("HF_DOWNLOAD_MODE must be auto/mirror-only/mirror-preferred/official")
    return mode


def normalize_hf_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    try:
        parsed = urlsplit(endpoint)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment
                 and not any(char.isspace() or char in ",;" for char in endpoint))
        parsed.port  # Validate the port without including the supplied URL in errors.
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("HF endpoint must be an HTTP(S) URL without credentials, query or fragment")
    return endpoint


def hf_endpoints(environ: Mapping[str, str] | None = None) -> list[str]:
    """Hub entry order only; this policy never restricts trusted storage hosts.

    Legacy CLI modes remain available for advanced use. TUI settings migrate to
    auto; mirror-only restricts *entry selection*, not legitimate redirects.
    """
    env = os.environ if environ is None else environ
    mode = hf_download_mode(env)
    if mode == "official":
        return [HF_OFFICIAL_ENDPOINT]
    primary = (env.get("HF_ENDPOINT", "").strip()
               or env.get("HF_HUB_ENDPOINT", "").strip() or HF_DEFAULT_ENDPOINT)
    endpoints = [normalize_hf_endpoint(primary)]
    fallback = env.get("HF_FALLBACK_ENDPOINTS", "").strip()
    if mode == "mirror-only" and fallback:
        raise ValueError("mirror-only does not permit HF_FALLBACK_ENDPOINTS; remove it or explicitly select mirror-preferred")
    if mode == "mirror-only" and urlsplit(endpoints[0]).hostname in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError("mirror-only requires a mirror endpoint; use HF_DOWNLOAD_MODE=official for the official Hub")
    for value in fallback.replace(";", ",").split(","):
        if value.strip():
            endpoint = normalize_hf_endpoint(value)
            if endpoint not in endpoints:
                endpoints.append(endpoint)
    if mode in {"auto", "mirror-preferred"} and HF_OFFICIAL_ENDPOINT not in endpoints:
        endpoints.append(HF_OFFICIAL_ENDPOINT)
    return endpoints
