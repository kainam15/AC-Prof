"""Exercise the real Hub HTTP boundary without transferring model weights."""
import os
from unittest.mock import patch

import httpx
import pytest

from acprof.hf_endpoints import hf_download_mode, hf_endpoints
from acprof.hf_transport import _request_hook, _response_hook, safe_url
from acprof.host.env_utils import load_project_env, save_project_env
from acprof.network_policy import DownloadPolicyError, source_route


@pytest.mark.parametrize(("url", "expected"), [
    ("https://user:secret@example.com:8443/path?q=1#frag", "https://example.com:8443/path"),
    ("https://example.com/path?q=1#frag", "https://example.com/path"),
    ("http://[2001:db8::1]:8080/file?sig=x", "http://[2001:db8::1]:8080/file"),
    ("https://example.com:bad/file?signature=secret", "invalid-url"),
    ("https://example.com:99999/file?signature=secret", "invalid-url"),
])
def test_safe_url_preserves_authority_without_secrets(url, expected):
    assert safe_url(url) == expected


def test_offline_hub_api_blocks_network_before_endpoint_fallback(monkeypatch):
    import huggingface_hub as hub
    from huggingface_hub import constants
    from huggingface_hub.errors import OfflineModeIsEnabled

    from acprof.hf_download import try_hf_endpoints
    from acprof.hf_transport import _httpx_factory

    monkeypatch.setenv("HF_DOWNLOAD_MODE", "auto")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.delenv("HF_FALLBACK_ENDPOINTS", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    visited, attempted = [], []

    def respond(request):
        visited.append(request.url.host)
        return httpx.Response(200, json={"id": "demo/model", "sha": "a" * 40, "siblings": []})

    def lookup(endpoint):
        attempted.append(endpoint)
        return hub.HfApi(endpoint=endpoint, token=False).model_info("demo/model")

    hub.set_client_factory(lambda: _httpx_factory(transport=httpx.MockTransport(respond)))
    try:
        with pytest.raises(OfflineModeIsEnabled):
            try_hf_endpoints("demo/model", lookup)
    finally:
        hub.set_client_factory(_httpx_factory)
    assert attempted == ["https://hf-mirror.com"]
    assert visited == []


def test_offline_requests_transport_blocks_before_adapter(monkeypatch):
    from huggingface_hub import constants
    from huggingface_hub.errors import OfflineModeIsEnabled

    from acprof.hf_transport import _requests_factory

    monkeypatch.setenv("HF_DOWNLOAD_MODE", "official")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    with patch("requests.adapters.HTTPAdapter.send", side_effect=AssertionError("offline adapter reached")) as send:
        with _requests_factory() as client, pytest.raises(OfflineModeIsEnabled):
            client.head("https://huggingface.co/demo/model/resolve/main/config.json")
    send.assert_not_called()


def test_offline_sdk_uses_cached_branch_without_transport(tmp_path, monkeypatch):
    import huggingface_hub as hub
    from huggingface_hub import constants

    from acprof.hf_transport import _httpx_factory

    monkeypatch.setenv("HF_DOWNLOAD_MODE", "official")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    commit = "d" * 40
    repo = tmp_path / "models--demo--model"
    cached = repo / "snapshots" / commit / "config.json"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"{}")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit)

    def respond(request):
        raise AssertionError(f"offline cache lookup attempted {request.method}")

    hub.set_client_factory(lambda: _httpx_factory(transport=httpx.MockTransport(respond)))
    try:
        assert hub.hf_hub_download("demo/model", "config.json", revision="main",
            cache_dir=tmp_path, endpoint="https://huggingface.co", token=False) == str(cached)
    finally:
        hub.set_client_factory(_httpx_factory)


@pytest.mark.parametrize("error_name,status", [
    ("RepositoryNotFoundError", 401), ("GatedRepoError", 403),
    ("RevisionNotFoundError", 404), ("RemoteEntryNotFoundError", 404),
    ("BadRequestError", 400), ("DisabledRepoError", 403),
])
def test_hub_semantic_errors_preserve_type_after_entry_fallback(error_name, status, monkeypatch):
    from huggingface_hub import errors

    from acprof.hf_download import network_failure, try_hf_endpoints

    monkeypatch.setenv("HF_DOWNLOAD_MODE", "auto")
    choices = ["https://hf-mirror.com", "https://huggingface.co"]
    error_type = getattr(errors, error_name)
    failures = [error_type("fixture model error", response=httpx.Response(status,
        request=httpx.Request("GET", endpoint + "/api/models/demo/model"))) for endpoint in choices]
    attempted = []

    def lookup(endpoint):
        attempted.append(endpoint)
        raise failures[choices.index(endpoint)]

    with pytest.raises(error_type) as raised:
        try_hf_endpoints("demo/model", lookup, endpoints=choices)
    assert raised.value is failures[-1]
    assert attempted == choices
    assert not network_failure(failures[-1])


def test_mirror_revision_failure_can_fall_back_to_successful_official_entry(monkeypatch):
    from huggingface_hub.errors import RevisionNotFoundError

    from acprof.hf_download import try_hf_endpoints
    from acprof.hf_transport import observed_hf_provenance

    monkeypatch.setenv("HF_DOWNLOAD_MODE", "auto")
    attempted = []

    def lookup(endpoint):
        attempted.append(endpoint)
        if endpoint == "https://hf-mirror.com":
            raise RevisionNotFoundError("mirror has not synced the revision", response=httpx.Response(404,
                request=httpx.Request("GET", endpoint + "/api/models/demo/model/revision/latest")))
        return "resolved-on-official"

    assert try_hf_endpoints("demo/model", lookup,
        endpoints=["https://hf-mirror.com", "https://huggingface.co"]) == "resolved-on-official"
    assert attempted == ["https://hf-mirror.com", "https://huggingface.co"]
    assert observed_hf_provenance()["failed_attempts"][0]["exception_type"] == "RevisionNotFoundError"


def test_auto_tries_domestic_entry_then_official():
    assert hf_download_mode({}) == "auto"
    assert hf_endpoints({}) == ["https://hf-mirror.com", "https://huggingface.co"]


@pytest.mark.parametrize("host", ["cas-bridge.xethub.hf.co", "us.aws.cdn.hf.co", "future-region.cdn.hf.co", "www.huggingface.co"])
@pytest.mark.parametrize("mode", ["auto", "mirror-only"])
def test_mirror_trusted_hub_or_storage_redirect_is_allowed(host, mode):
    visited = []

    def respond(request):
        visited.append(request.url.host)
        if request.url.host == "hf-mirror.com":
            return httpx.Response(302, headers={"location": f"https://{host}/weight?signature=secret"})
        return httpx.Response(200, content=b"ok")

    with patch.dict(os.environ, {"HF_DOWNLOAD_MODE": mode, "HF_ENDPOINT": "https://hf-mirror.com",
                                 "HF_FALLBACK_ENDPOINTS": ""}):
        with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True,
                          event_hooks={"request": [_request_hook], "response": [_response_hook]}) as client:
            assert client.get("https://hf-mirror.com/model/resolve/main/weight").content == b"ok"
    assert visited == ["hf-mirror.com", host]


@pytest.mark.parametrize("target", ["https://evil.example/a", "https://cdn.hf.co.evil.example/a",
                                    "https://evil-hf.co/a", "http://cdn.hf.co/a",
                                    "https://huggingface.co:0/a", "https://huggingface.co:bad/a?signature=secret",
                                    "https://huggingface.co:99999/a?signature=secret"])
def test_unknown_redirect_never_reaches_network(target):
    visited = []

    def respond(request):
        visited.append(request.url.host)
        return httpx.Response(302, headers={"location": target})

    with patch.dict(os.environ, {"HF_DOWNLOAD_MODE": "official"}):
        with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True,
                          event_hooks={"request": [_request_hook], "response": [_response_hook]}) as client:
            with pytest.raises(DownloadPolicyError):
                client.get("https://huggingface.co/model/resolve/main/weight")
    assert visited == ["huggingface.co"]


@pytest.mark.parametrize("backend", ["httpx", "requests"])
def test_https_redirect_cannot_downgrade_to_configured_http_entry(backend, monkeypatch):
    from urllib.parse import urlsplit

    from acprof.hf_transport import UntrustedHfEndpointError, _httpx_factory, _requests_factory

    endpoint = "http://fixture.hub.invalid"
    official = "https://huggingface.co/demo/model/resolve/main/config.json"
    monkeypatch.setenv("HF_DOWNLOAD_MODE", "auto")
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    monkeypatch.delenv("HF_FALLBACK_ENDPOINTS", raising=False)
    visited = []

    def respond(request):
        visited.append(str(request.url))
        redirected = urlsplit(str(request.url)).scheme == "https"
        headers = {"location": endpoint + "/config.json"} if redirected else {}
        if backend == "httpx":
            return httpx.Response(302 if redirected else 200, headers=headers, content=b"{}")
        import requests

        response = requests.Response()
        response.status_code = 302 if redirected else 200
        response.url = request.url
        response.request = request
        response.headers.update(headers)
        response._content = b"{}"
        return response

    def check(client):
        with pytest.raises(UntrustedHfEndpointError):
            client.get(official)
        assert visited == [official]
        # Direct use of an explicitly configured HTTP entry remains supported.
        assert client.get(endpoint + "/config.json").status_code == 200
        assert visited == [official, endpoint + "/config.json"]

    if backend == "httpx":
        with _httpx_factory(transport=httpx.MockTransport(respond)) as client:
            check(client)
    else:
        with patch("requests.adapters.HTTPAdapter.send", autospec=True,
                   side_effect=lambda _adapter, request, **_kwargs: respond(request)):
            with _requests_factory() as client:
                check(client)


def test_application_proxy_status_does_not_infer_public_egress():
    assert source_route("https://huggingface.co", {}) == "direct-socket"
    assert source_route("https://hf-mirror.com", {"https_proxy": "http://proxy.example:8080"}) == "explicit-proxy"
    assert source_route("https://hf-mirror.com", {"HTTPS_PROXY": "http://proxy.example:8080",
                                                 "NO_PROXY": "hf-mirror.com"}) == "direct-socket"


def test_retired_project_proxy_values_do_not_replace_system_environment(tmp_path):
    original = 'HTTP_PROXY="http://old.example:8080"\nhttps_proxy="http://old.example:8080"\nNO_PROXY="*"\n'
    (tmp_path / ".env.local").write_text(original)
    env = {"HTTPS_PROXY": "http://system.example:8080", "no_proxy": "localhost"}
    load_project_env(tmp_path, environ=env)
    assert env == {"HTTPS_PROXY": "http://system.example:8080", "no_proxy": "localhost"}
    save_project_env(tmp_path, {"HF_TOKEN": ""}, environ=env)
    assert (tmp_path / ".env.local").read_text().startswith(original)
    assert "HTTP_PROXY" not in env


def test_automatic_access_check_uses_model_store_without_network():
    from acprof.host.automation import check_repository_access
    with patch("acprof.host.model_store.cached_model_info", return_value=object()), patch(
            "huggingface_hub.HfApi.auth_check", side_effect=AssertionError("cache hit made a network request")):
        evidence = check_repository_access("demo/model")
    assert evidence["scope"] == "local_model_store_only"


def test_real_hub_sdk_follows_mirror_hub_storage_chain(tmp_path, monkeypatch):
    import hashlib

    import huggingface_hub as hub

    from acprof.hf_transport import _httpx_factory, observed_hf_provenance
    monkeypatch.setenv("HF_DOWNLOAD_MODE", "auto")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.delenv("HF_FALLBACK_ENDPOINTS", raising=False)
    visited = []
    commit, content = "a" * 40, b"{}"
    etag = hashlib.sha256(content).hexdigest()

    def respond(request):
        visited.append((request.method, str(request.url), request.headers.get("authorization")))
        if request.url.host == "hf-mirror.com":
            return httpx.Response(308, headers={"location": f"https://huggingface.co/demo/model/resolve/{commit}/config.json?etag=preserved"})
        if request.url.host == "huggingface.co":
            assert request.url.params["etag"] == "preserved"
            return httpx.Response(302, headers={"location": "https://us.aws.cdn.hf.co/file?signature=secret",
                "x-repo-commit": commit, "x-linked-etag": etag, "x-linked-size": "2"})
        assert request.url.host == "us.aws.cdn.hf.co"
        return httpx.Response(200, content=content, headers={"content-length": "2"})

    observed_hf_provenance(reset=True)
    hub.set_client_factory(lambda: _httpx_factory(transport=httpx.MockTransport(respond)))
    try:
        path = hub.hf_hub_download("demo/model", "config.json", revision=commit,
            cache_dir=tmp_path, endpoint="https://hf-mirror.com", token="hf_test_token")
        assert __import__("pathlib").Path(path).read_bytes() == content
        assert [item[0] for item in visited] == ["HEAD", "HEAD", "GET"]
        assert visited[-1][2] is None
        evidence = observed_hf_provenance()
        assert evidence["requests"][-1]["endpoint_type"] == "storage"
        assert "signature" not in str(evidence) and "hf_test_token" not in str(evidence)
    finally:
        hub.set_client_factory(_httpx_factory)


@pytest.mark.parametrize("reason", ["DNS", "TLS", "timeout", "HTTP"])
def test_failure_stage_and_final_storage_host_are_preserved(reason, monkeypatch):
    import socket
    import ssl

    from acprof.hf_download import HfDownloadError, try_hf_endpoints
    monkeypatch.setenv("HF_DOWNLOAD_MODE", "auto")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.delenv("HF_FALLBACK_ENDPOINTS", raising=False)
    attempts = []

    def download(endpoint):
        attempts.append(endpoint)
        request = httpx.Request("GET", "https://us.aws.cdn.hf.co/file?secret=not-in-provenance")
        if reason == "HTTP":
            response = httpx.Response(503, request=request)
            response.raise_for_status()
        if reason == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        try:
            raise (socket.gaierror(-2, "DNS failure") if reason == "DNS" else ssl.SSLCertVerificationError("bad cert"))
        except OSError as exc:
            raise httpx.ConnectError("connection failed", request=request) from exc

    with pytest.raises(HfDownloadError) as error:
        try_hf_endpoints("demo/model", download)
    assert attempts == ["https://hf-mirror.com", "https://huggingface.co"]
    assert error.value.attempts[-1]["stage"] == "storage"
    assert error.value.attempts[-1]["reason"] == reason
    assert error.value.attempts[-1]["host"] == "us.aws.cdn.hf.co"
    assert error.value.to_dict()["can_switch_source"]
    assert "not-in-provenance" not in str(error.value.to_dict())


def test_light_probe_follows_storage_with_head_only_and_cache_skips_all_io(tmp_path, monkeypatch):
    import huggingface_hub as hub

    from acprof.hf_transport import _httpx_factory
    from acprof.host.model_store import probe_model_download
    monkeypatch.setenv("HF_DOWNLOAD_MODE", "auto")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    visited = []

    def respond(request):
        visited.append((request.method, request.url.host))
        if request.url.host == "hf-mirror.com":
            return httpx.Response(302, headers={"location": "https://cas-bridge.xethub.hf.co/large"})
        return httpx.Response(200, headers={"content-length": "100000000000"})

    plan = {"model_id": "demo/model", "model_revision": "a" * 40, "endpoint": "https://hf-mirror.com",
            "files": [{"path": "model.safetensors", "size": 100_000_000_000}]}
    hub.set_client_factory(lambda: _httpx_factory(transport=httpx.MockTransport(respond)))
    try:
        with patch("acprof.host.model_store.configure_hf_transport"):
            report = probe_model_download(plan, tmp_path)
            assert report[0]["method"] == "HEAD"
            assert visited == [("HEAD", "hf-mirror.com"), ("HEAD", "cas-bridge.xethub.hf.co")]
            visited.clear()
            with patch("acprof.host.model_store.cached_file", return_value=tmp_path / "cached"):
                assert probe_model_download(plan, tmp_path) == []
            assert visited == []
    finally:
        hub.set_client_factory(_httpx_factory)


def test_socket_bypass_is_local_and_does_not_change_proxy_environment():
    from acprof.host.client import _local_proxy_options
    before = dict(os.environ)
    assert _local_proxy_options("http://127.0.0.1:8080")["proxies"]["all"] == ""
    assert _local_proxy_options("https://huggingface.co") == {}
    assert dict(os.environ) == before


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_system_http_proxy_is_used_without_changing_environment(scheme):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from acprof.hf_transport import _httpx_factory
    requests = []
    class Proxy(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(("GET", self.path))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
        def do_CONNECT(self):
            requests.append(("CONNECT", self.path))
            self.send_error(502)  # Prove proxy use without tunneling to the internet.
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"{scheme}://fixture.hub.invalid"
    env = {"HF_ENDPOINT": endpoint, f"{scheme.upper()}_PROXY": f"http://127.0.0.1:{server.server_port}"}
    try:
        with patch.dict(os.environ, env, clear=True), _httpx_factory() as client:
            if scheme == "https":
                with pytest.raises(httpx.ProxyError):
                    client.get(endpoint)
            else:
                assert client.get(endpoint).json() == {}
            assert dict(os.environ) == env
            assert source_route(endpoint) == "explicit-proxy"
        assert requests == ([("CONNECT", "fixture.hub.invalid:443")] if scheme == "https" else
                            [("GET", "http://fixture.hub.invalid/")])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_real_hub_sdk_resumes_partial_file_and_keeps_content_identity(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path

    import huggingface_hub as hub

    from acprof.hf_transport import _httpx_factory
    monkeypatch.setenv("HF_DOWNLOAD_MODE", "official")
    content, commit = b"abcdef", "c" * 40
    etag = hashlib.sha256(content).hexdigest()
    partial = tmp_path / "models--demo--model" / "blobs" / (etag + ".incomplete")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(content[:2])
    requests = []
    def respond(request):
        requests.append(request)
        if request.method == "HEAD":
            return httpx.Response(302, headers={"location": "https://us.aws.cdn.hf.co/file",
                "x-repo-commit": commit, "x-linked-etag": etag, "x-linked-size": "6"})
        assert request.headers["range"] == "bytes=2-"
        return httpx.Response(206, content=content[2:], headers={"content-length": "4", "content-range": "bytes 2-5/6"})
    hub.set_client_factory(lambda: _httpx_factory(transport=httpx.MockTransport(respond)))
    try:
        path = hub.hf_hub_download("demo/model", "model.safetensors", revision=commit, cache_dir=tmp_path,
                                   endpoint="https://huggingface.co")
        assert Path(path).read_bytes() == content
        assert not partial.exists()
    finally:
        hub.set_client_factory(_httpx_factory)
