"""Guard Hub HTTP transports, including every redirect, before network I/O.

Uses public client factories for Hub 0.x (requests) and 1.x (httpx).
Xet/hf_transfer bypass these factories and are always disabled here.
"""
from __future__ import annotations

import os
import threading
from collections import deque
from urllib.parse import urljoin, urlsplit, urlunsplit

from acprof.hf_endpoints import HF_OFFICIAL_ENDPOINT, hf_endpoints
from acprof.network_policy import DownloadPolicyError, source_route

_LOCK = threading.Lock()
_INSTALLED = False
_SOURCES: set[str] = set()
_EVENTS: deque[dict] = deque(maxlen=256)
_EVENT_COUNT = 0
_FAILED_ATTEMPTS: list[dict] = []
# Organization-controlled DNS namespaces, not a list of CDN machines/regions.
# Generic S3/CloudFront suffixes are deliberately NOT trusted.
_STORAGE_NAMESPACES = ("hf.co", "huggingface.co")
_OFFICIAL_HUB_ENDPOINTS = (HF_OFFICIAL_ENDPOINT, "https://www.huggingface.co", "https://hf.co")
_OFFICIAL_HUB_HOSTS = frozenset(urlsplit(endpoint).hostname for endpoint in _OFFICIAL_HUB_ENDPOINTS)


def _validated_urlsplit(url: str):
    """Parse once and reject malformed authorities before policy checks."""
    try:
        parsed = urlsplit(str(url))
        parsed.port
    except ValueError:
        return None
    return parsed


class UntrustedHfEndpointError(DownloadPolicyError):
    def __init__(self, url: str):
        parsed = _validated_urlsplit(url)
        self.host = parsed.hostname if parsed is not None and parsed.hostname else "invalid"
        self.stage = "redirect"
        super().__init__(f"Untrusted HF endpoint rejected before request: {self.host}")


def safe_url(url: str) -> str:
    """Provenance excludes signed query strings, credentials and fragments."""
    parsed = _validated_urlsplit(url)
    if parsed is None:
        return "invalid-url"
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    authority = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))


def endpoint_type(url: str) -> str:
    parsed = _validated_urlsplit(url)
    if parsed is None:
        return "unknown"
    if parsed.hostname in _OFFICIAL_HUB_HOSTS:
        return "official-hub"
    if any(parsed.hostname == urlsplit(value).hostname for value in hf_endpoints()):
        return "hub-entry"
    host = parsed.hostname or ""
    if any(host == domain or host.endswith("." + domain) for domain in _STORAGE_NAMESPACES):
        return "storage"
    return "unknown"


def observed_hf_provenance(*, reset: bool = False) -> dict:
    global _EVENT_COUNT
    with _LOCK:
        result = {"schema_version": 1, "requests": list(_EVENTS),
                  "truncated": _EVENT_COUNT > len(_EVENTS), "public_egress": "externally-managed",
                  "failed_attempts": list(_FAILED_ATTEMPTS)}
        if reset:
            _EVENTS.clear()
            _SOURCES.clear()
            _EVENT_COUNT = 0
            _FAILED_ATTEMPTS.clear()
        return result


def record_failed_attempts(attempts: list[dict]) -> None:
    with _LOCK:
        _FAILED_ATTEMPTS[:] = attempts


def observed_hf_sources(*, reset: bool = False) -> list[str]:
    with _LOCK:
        result = sorted(_SOURCES)
        if reset:
            _SOURCES.clear()
        return result


def _response_hook(response):
    global _EVENT_COUNT
    url = str(response.request.url)
    event = {"url": safe_url(url), "host": urlsplit(url).hostname,
             "endpoint_type": endpoint_type(url), "method": response.request.method,
             "status": response.status_code, "application_route": source_route(url)}
    location = response.headers.get("location")
    if location and response.status_code in {301, 302, 303, 307, 308}:
        target = urljoin(url, location)
        event["redirect_to"] = safe_url(target)
        event["redirect_endpoint_type"] = endpoint_type(target)
    with _LOCK:
        _EVENTS.append(event)
        _EVENT_COUNT += 1
        if response.status_code in {200, 206} and response.request.method == "GET" and "/api/" not in urlsplit(url).path:
            _SOURCES.add(urlsplit(url).hostname)
    # Validate even when the SDK consumes Location without following it yet.
    if location and response.status_code in {301, 302, 303, 307, 308}:
        target = urljoin(url, location)
        # An explicitly configured HTTP entry permits direct use, not a secure
        # Hub response silently downgrading its redirect to plaintext.
        if urlsplit(url).scheme == "https" and urlsplit(target).scheme == "http":
            raise UntrustedHfEndpointError(target)
        check_hf_url(target)


def check_hf_url(url: str) -> None:
    parsed = _validated_urlsplit(url)
    if parsed is None or parsed.username or parsed.password or not parsed.hostname:
        raise UntrustedHfEndpointError(url)
    default_ports = {"https": 443, "http": 80}
    port = parsed.port if parsed.port is not None else default_ports.get(parsed.scheme)
    for endpoint in [*hf_endpoints(), *_OFFICIAL_HUB_ENDPOINTS]:
        hub = urlsplit(endpoint)
        hub_port = hub.port if hub.port is not None else default_ports.get(hub.scheme)
        if (parsed.scheme, parsed.hostname, port) == (hub.scheme, hub.hostname, hub_port):
            return
    if parsed.scheme == "https" and parsed.port in {None, 443} and endpoint_type(url) == "storage":
        return
    raise UntrustedHfEndpointError(url)


def _request_hook(request):
    import huggingface_hub as hub
    from huggingface_hub import constants
    from huggingface_hub.errors import OfflineModeIsEnabled

    # Custom client factories replace the SDK hooks/adapters that enforce offline
    # mode. Preserve that contract before either HTTP backend reaches a socket.
    is_offline_mode = getattr(hub, "is_offline_mode", None)
    offline = is_offline_mode() if is_offline_mode is not None else constants.HF_HUB_OFFLINE
    if offline:
        raise OfflineModeIsEnabled("HF_HUB_OFFLINE is enabled; no HTTP request was sent")
    check_hf_url(str(request.url))
    if endpoint_type(str(request.url)) == "storage":
        # SDK versions differ in how they carry auth across absolute redirects.
        request.headers.pop("authorization", None)
        request.headers.pop("cookie", None)


def _httpx_factory(*, transport=None):
    import httpx

    class HubClient(httpx.Client):
        def send(self, request, **kwargs):
            # Hub 1.x versions before upstream #4739 only follow relative HEAD
            # redirects and discard their query. Follow Hub HEADs here, stopping
            # at storage so the SDK can read x-linked-etag/size on the redirect.
            for _ in range(self.max_redirects + 1):
                response = super().send(request, **kwargs)
                location = response.headers.get("location")
                if request.method != "HEAD" or not response.is_redirect or not location:
                    return response
                target = urljoin(str(response.url), location)
                if endpoint_type(target) not in {"hub-entry", "official-hub"}:
                    return response
                headers = dict(request.headers)
                if urlsplit(str(request.url)).netloc != urlsplit(target).netloc:
                    headers.pop("authorization", None)
                    headers.pop("cookie", None)
                headers.pop("host", None)
                response.close()
                request = self.build_request("HEAD", target, headers=headers)
            raise httpx.TooManyRedirects("HF Hub HEAD redirect loop", request=request)

    return HubClient(transport=transport, follow_redirects=True, timeout=httpx.Timeout(30.0, write=60.0),
                        event_hooks={"request": [_request_hook], "response": [_response_hook]})


def _requests_factory():
    import requests
    from requests.adapters import HTTPAdapter

    class GuardedAdapter(HTTPAdapter):
        def send(self, request, **kwargs):
            _request_hook(request)
            response = super().send(request, **kwargs)
            _response_hook(response)
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
