"""Ephemeral, bounded connectivity evidence; never writes downloaded artifacts."""
from __future__ import annotations

import os
import re
import socket
import ssl
import time
from dataclasses import dataclass
from urllib.error import URLError

from acprof.hf_auth import hf_token_for_url
from acprof.hf_download import exception_chain
from acprof.hf_transport import UntrustedHfEndpointError, configure_hf_transport, safe_url
from acprof.network_policy import DownloadPolicyError, parse_bytes

PROBE_MAX_BYTES = 256 * 1024
# HTTPX's HTTP/1.1 backend reads at most 64 KiB per raw iteration. Reserve a
# complete read before advancing: chunk boundaries need not align with Range.
PROBE_READ_BYTES = 64 * 1024


@dataclass
class ProbeBudget:
    limit: int = PROBE_MAX_BYTES
    received: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.received)


def probe_allowance(model_bytes: int | None, total_bytes: int | None) -> int:
    """Reserve the complete planned download before spending any probe bytes."""
    allowance = PROBE_MAX_BYTES
    for name, expected in (("ACPROF_MODEL_MAX_DOWNLOAD", model_bytes), ("ACPROF_MAX_DOWNLOAD", total_bytes)):
        limit = parse_bytes(os.environ.get(name))
        if limit is not None:
            allowance = min(allowance, 0 if expected is None else max(0, limit - expected))
    return allowance


def failure_reason(exc: BaseException, *, phase: str) -> str:
    import httpx
    from huggingface_hub.errors import OfflineModeIsEnabled

    chain = list(exception_chain(exc))
    chain += [item.reason for item in chain if isinstance(item, URLError) and isinstance(item.reason, BaseException)]
    if any(isinstance(item, OfflineModeIsEnabled) for item in chain):
        return "offline"
    if any(isinstance(item, UntrustedHfEndpointError) for item in chain):
        return "untrusted-host"
    if any(isinstance(item, ssl.SSLError) for item in chain):
        return "tls"
    if any(isinstance(item, socket.gaierror) for item in chain):
        return "dns"
    if any(isinstance(item, httpx.ConnectTimeout) for item in chain):
        return "connect-timeout"
    if any(isinstance(item, (TimeoutError, httpx.TimeoutException)) for item in chain):
        return "read-timeout" if phase == "GET" else "timeout"
    if any(isinstance(item, httpx.TooManyRedirects) for item in chain):
        return "redirect-loop"
    for item in chain:
        response = getattr(item, "response", None)
        status = getattr(response, "status_code", None) or getattr(item, "code", None)
        if isinstance(status, int):
            return f"http-{status}"
    if any(isinstance(item, DownloadPolicyError) for item in chain):
        return "source-policy"
    if any(isinstance(item, (httpx.RemoteProtocolError, httpx.ReadError, ConnectionResetError)) for item in chain):
        return "transfer-interrupted"
    return "connection"


def probe_record(category: str, url: str, method: str) -> dict:
    return {"category": category, "source": safe_url(url), "method": method,
            "status": "unverified", "final_host": None, "http_status": None,
            "received_bytes": 0, "elapsed_seconds": 0.0, "reason": None}


def probe_hf_file(url: str, budget: ProbeBudget, *, timeout: float = 8, total_timeout: float = 20) -> tuple[dict, Exception | None]:
    """Use the SDK client/auth/hooks, closing redirects without draining bodies."""
    import httpx
    from huggingface_hub.errors import OfflineModeIsEnabled
    from huggingface_hub.utils import build_hf_headers, get_session

    result = probe_record("model", url, "HEAD")
    result["scope"] = "one missing file; bounded payload only; not a complete download guarantee"
    started = time.monotonic()
    deadline = started + total_timeout
    initial_bytes = budget.received
    phase = "HEAD"
    configure_hf_transport()
    client = get_session()

    def remaining_timeout() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("probe deadline exceeded")
        return min(timeout, remaining)

    def open_response(method, headers):
        request = client.build_request(method, url, headers=headers, timeout=remaining_timeout())
        request.extensions["acprof_probe"] = True
        for _ in range(client.max_redirects + 1):
            request.extensions["timeout"] = dict.fromkeys(("connect", "read", "write", "pool"), remaining_timeout())
            response = client.send(request, stream=True, follow_redirects=False)
            result.update(final_host=response.url.host, http_status=response.status_code)
            if not response.is_redirect:
                return response
            next_request = response.next_request
            response.close()
            if next_request is None:
                raise httpx.RemoteProtocolError("redirect without a target", request=request)
            request = next_request
        raise httpx.TooManyRedirects("probe redirect loop", request=request)

    try:
        headers = {**build_hf_headers(token=hf_token_for_url(url)), "Accept-Encoding": "identity"}
        response = open_response("HEAD", headers)
        try:
            response.raise_for_status()
            result["head_status"] = response.status_code
            result["head_final_host"] = response.url.host
        finally:
            response.close()
        if budget.remaining < PROBE_READ_BYTES:
            result["reason"] = "probe-budget"
            return result, None
        phase = "GET"
        result.update(http_status=None, final_host=None)
        result["method"] = "HEAD+GET"
        limit = budget.remaining
        result["byte_limit"] = limit
        response = open_response("GET", {**headers, "Range": f"bytes=0-{limit - 1}"})
        try:
            response.raise_for_status()
            if response.status_code not in {200, 206} or response.headers.get("content-encoding", "identity") != "identity":
                raise httpx.RemoteProtocolError("unexpected probe response", request=response.request)
            if response.status_code == 206:
                match = re.fullmatch(r"bytes 0-(\d+)/(\d+|\*)", response.headers.get("content-range", ""))
                if match is None or int(match[1]) >= limit:
                    raise httpx.RemoteProtocolError("invalid probe Content-Range", request=response.request)
            result["range_supported"] = response.status_code == 206
            stream = response.iter_raw()
            while budget.remaining >= PROBE_READ_BYTES:
                response.request.extensions["timeout"]["read"] = remaining_timeout()
                try:
                    chunk = next(stream)
                except StopIteration:
                    break
                budget.received += len(chunk)
            if budget.received == initial_bytes:
                raise httpx.RemoteProtocolError("empty model payload", request=response.request)
            result["status"] = "passed"
        finally:
            response.close()
        return result, None
    except (httpx.HTTPError, OSError, DownloadPolicyError, OfflineModeIsEnabled) as exc:
        result.update(status="failed", reason=failure_reason(exc, phase=phase))
        return result, exc
    finally:
        result["received_bytes"] = budget.received - initial_bytes
        result["elapsed_seconds"] = round(time.monotonic() - started, 4)
