"""Guard Hub HTTP transports, including every redirect, before network I/O.

Uses public client factories for Hub 0.x (requests) and 1.x (httpx).
Xet/hf_transfer bypass these factories and are always disabled here.
"""
from __future__ import annotations

import os
import threading
from urllib.parse import urlsplit

from acprof.hf_endpoints import hf_download_mode, hf_endpoints
from acprof.network_policy import DownloadPolicyError, require_source_transition

_LOCK = threading.Lock()
_INSTALLED = False
_SOURCES: set[str] = set()


def observed_hf_sources(*, reset: bool = False) -> list[str]:
    with _LOCK:
        result = sorted(_SOURCES)
        if reset:
            _SOURCES.clear()
        return result


def _response_hook(response):
    if response.status_code == 200 and response.request.method == "GET" and "/api/" not in response.request.url.path:
        with _LOCK:
            _SOURCES.add(urlsplit(str(response.request.url)).hostname)


def check_hf_url(url: str) -> None:
    parsed = urlsplit(str(url))
    endpoints = hf_endpoints()
    primary = urlsplit(endpoints[0])
    if parsed.scheme not in {"https", "http"} or parsed.username or parsed.password:
        raise DownloadPolicyError("HF request has an invalid URL")
    if parsed.hostname == primary.hostname and parsed.port == primary.port and parsed.scheme == primary.scheme:
        return
    if hf_download_mode() == "mirror-only":
        raise DownloadPolicyError(f"mirror-only blocked non-allowed domain before request: {parsed.hostname}")
    if hf_download_mode() == "mirror-preferred":
        require_source_transition(endpoints[0], str(url))


def _request_hook(request):
    check_hf_url(str(request.url))


def _httpx_factory():
    import httpx
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, write=60.0),
                        event_hooks={"request": [_request_hook], "response": [_response_hook]})


def _requests_factory():
    import requests
    from requests.adapters import HTTPAdapter

    class GuardedAdapter(HTTPAdapter):
        def send(self, request, **kwargs):
            check_hf_url(request.url)
            response = super().send(request, **kwargs)
            if response.status_code == 200 and request.method == "GET" and "/api/" not in urlsplit(request.url).path:
                with _LOCK:
                    _SOURCES.add(urlsplit(request.url).hostname)
            return response

    session = requests.Session()
    session.mount("https://", GuardedAdapter())
    session.mount("http://", GuardedAdapter())
    return session


def configure_hf_transport() -> None:
    global _INSTALLED
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import huggingface_hub as hub
    from huggingface_hub import constants
    # Constants may have been imported before CLI/.env configuration.
    constants.HF_HUB_DISABLE_XET = True
    constants.HF_HUB_ENABLE_HF_TRANSFER = False
    with _LOCK:
        if _INSTALLED:
            return
        if hasattr(hub, "set_client_factory"):
            hub.set_client_factory(_httpx_factory)
        else:
            hub.configure_http_backend(backend_factory=_requests_factory)
        _INSTALLED = True
