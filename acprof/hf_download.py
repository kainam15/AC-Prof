"""Automatic Hub entry retries and structured, credential-free diagnostics."""
from __future__ import annotations

import socket
import ssl

from acprof.hf_endpoints import hf_endpoints
from acprof.hf_transport import (
    UntrustedHfEndpointError,
    endpoint_type,
    observed_hf_provenance,
    record_failed_attempts,
    safe_url,
)
from acprof.network_policy import require_source_transition


def exception_chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _model_failure(exc: BaseException) -> bool:
    from huggingface_hub import errors

    # Hub 1.x split remote file errors from local cache misses. The latter must
    # never become an automatic endpoint retry.
    remote_entry_error = getattr(errors, "RemoteEntryNotFoundError", errors.EntryNotFoundError)
    kinds = (errors.RepositoryNotFoundError, errors.RevisionNotFoundError,
             remote_entry_error, errors.BadRequestError, errors.DisabledRepoError)
    return any(isinstance(cause, kinds) for cause in exception_chain(exc))


def network_failure(exc: BaseException) -> bool:
    import httpx
    import requests
    from huggingface_hub.errors import (
        FileMetadataError,
        HfHubHTTPError,
        OfflineModeIsEnabled,
    )
    if _model_failure(exc) or any(isinstance(cause, OfflineModeIsEnabled) for cause in exception_chain(exc)):
        return False
    return any(isinstance(cause, (httpx.HTTPError, requests.RequestException, ConnectionError,
                                  TimeoutError, socket.gaierror, ssl.SSLError, HfHubHTTPError,
                                  FileMetadataError, UntrustedHfEndpointError))
               for cause in exception_chain(exc))


def failure_record(exc: BaseException, endpoint: str) -> dict:
    from urllib.parse import urlsplit

    import httpx
    import requests

    chain = list(exception_chain(exc))
    url = endpoint
    status = None
    for cause in chain:
        response = getattr(cause, "response", None)
        try:
            request = getattr(cause, "request", None)
        except RuntimeError:
            request = None  # Some SDK errors are constructed without a request.
        if request is not None:
            url = str(request.url)
        elif response is not None:
            try:
                url = str(response.url)
            except RuntimeError:
                pass  # The selected entry remains the only observed address.
        if response is not None:
            status = response.status_code
    kind = endpoint_type(url)
    stage = "storage" if kind == "storage" else "Hub"
    reason = "entry"
    if any(isinstance(c, UntrustedHfEndpointError) for c in chain):
        stage, reason = "redirect", "untrusted-host"
    elif any(isinstance(c, (ssl.SSLError, requests.exceptions.SSLError)) for c in chain):
        reason = "TLS"
    elif any(isinstance(c, socket.gaierror) for c in chain):
        reason = "DNS"
    elif any(isinstance(c, (TimeoutError, httpx.TimeoutException, requests.exceptions.Timeout)) for c in chain):
        reason = "timeout"
    elif any(isinstance(c, (httpx.TooManyRedirects, requests.TooManyRedirects)) for c in chain):
        stage, reason = "redirect", "redirect-loop"
    elif status is not None:
        reason = "HTTP"
    elif stage == "storage":
        reason = "storage-connection"
    host = next((c.host for c in chain if isinstance(c, UntrustedHfEndpointError)), urlsplit(url).hostname)
    return {"endpoint": safe_url(endpoint), "stage": stage, "reason": reason,
            "host": host or "unknown", "http_status": status,
            "exception_type": type(exc).__name__, "transport": observed_hf_provenance()}


class HfDownloadError(OSError):
    """All configured HF entries failed; source identity has not changed."""

    def __init__(self, model_id: str, attempts: list[dict]):
        self.model_id = model_id
        self.attempts = attempts
        last = attempts[-1]
        self.stage = last["stage"]
        super().__init__(f"无法获取 Hugging Face 模型 {model_id}；"
                         f"{last['stage']} / {last['reason']} / {last['host']}。"
                         "请配置自己的系统 VPN/代理后重试，或显式选择 ModelScope 模型。")

    def to_dict(self) -> dict:
        return {"source": "huggingface", "model_id": self.model_id,
                "attempts": self.attempts, "can_switch_source": True}


def try_hf_endpoints(model_id: str, operation, *, endpoints: list[str] | None = None):
    """Try alternate Hub entries while preserving terminal model error types."""
    choices = endpoints or hf_endpoints()
    attempts = []
    for index, endpoint in enumerate(choices):
        if index:
            require_source_transition(choices[index - 1], endpoint)
        observed_hf_provenance(reset=True)
        try:
            result = operation(endpoint)
            record_failed_attempts(attempts)
            return result
        except Exception as exc:
            model_failure = _model_failure(exc)
            if not model_failure and not network_failure(exc):
                raise
            attempts.append(failure_record(exc, endpoint))
            if index == len(choices) - 1:
                record_failed_attempts(attempts)
                # A mirror may be stale or deny access while the official Hub
                # succeeds. If every entry fails, retain the final actionable
                # repository/revision/file/access error for CLI/TUI consumers.
                if model_failure:
                    raise
                raise HfDownloadError(model_id, attempts) from exc
    raise AssertionError("empty HF endpoint list")
