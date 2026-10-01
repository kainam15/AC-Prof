"""Shared, explicit Hugging Face Hub endpoint policy (standard library only)."""
from __future__ import annotations

import os
from typing import Mapping
from urllib.parse import urlsplit

HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"
HF_OFFICIAL_ENDPOINT = "https://huggingface.co"
HF_DOWNLOAD_MODES = ("mirror-only", "mirror-preferred", "official")


def hf_download_mode(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    mode = env.get("HF_DOWNLOAD_MODE", "").strip() or "mirror-only"
    if mode not in HF_DOWNLOAD_MODES:
        raise ValueError("HF_DOWNLOAD_MODE must be mirror-only/mirror-preferred/official")
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
    """Choose sources explicitly; mirror-only never admits the official Hub."""
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
    if mode == "mirror-preferred" and HF_OFFICIAL_ENDPOINT not in endpoints:
        endpoints.append(HF_OFFICIAL_ENDPOINT)
    return endpoints
