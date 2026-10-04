import os
import tempfile
from unittest.mock import patch

import pytest

from acprof.hf_endpoints import hf_endpoints
from acprof.hf_transport import _request_hook, configure_hf_transport
from acprof.network_policy import DownloadPolicyError, enforce_download_budget, parse_bytes


def test_default_is_auto():
    assert (hf_endpoints({})) == (["https://hf-mirror.com", "https://huggingface.co"])

def test_mirror_only_rejects_official_fallback():
    with pytest.raises(ValueError, match="mirror-only"):
        hf_endpoints({"HF_DOWNLOAD_MODE": "mirror-only",
                      "HF_FALLBACK_ENDPOINTS": "https://huggingface.co"})

def test_official_is_explicit():
    assert (hf_endpoints({"HF_DOWNLOAD_MODE": "official"})) == (["https://huggingface.co"])

def test_mirror_preferred_requires_explicit_proxy_fallback_opt_in():
    assert (hf_endpoints({"HF_DOWNLOAD_MODE": "mirror-preferred"})) == (["https://hf-mirror.com", "https://huggingface.co"])

@pytest.mark.parametrize('download_case', range(2), ids=['hub.hf_hub_download', 'hub.snapshot_download'])
@pytest.mark.parametrize('target', ('https://unknown.example/file', 'https://hf-mirror.com.evil.example/file'))
def test_actual_hub_downloads_block_redirect_before_transport(download_case, target):
    import httpx
    import huggingface_hub as hub
    if not hasattr(hub, "set_client_factory"):
        pytest.skip("httpx Hub integration; requests transport is exercised separately")
    download = tuple((hub.hf_hub_download, hub.snapshot_download))[download_case]
    with tempfile.TemporaryDirectory() as cache:
        visited = []

        def respond(request):
            visited.append(request.url.host)
            if "/api/models/" in request.url.path:
                return httpx.Response(200, json={"id": "example/model", "sha": "a" * 40,
                    "siblings": [{"rfilename": "config.json"}]})
            return httpx.Response(302, headers={"location": target, "x-repo-commit": "a" * 40,
                "x-linked-etag": "b" * 64, "x-linked-size": "2", "content-length": "2"})

        with patch.dict(os.environ, {"HF_DOWNLOAD_MODE": "mirror-only", "HF_ENDPOINT": "https://hf-mirror.com",
                                     "HF_FALLBACK_ENDPOINTS": ""}):
            configure_hf_transport()
            hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(respond),
                follow_redirects=True, event_hooks={"request": [_request_hook]}))
            try:
                kwargs = {"repo_id": "example/model", "revision": "a" * 40,
                          "cache_dir": cache, "endpoint": "https://hf-mirror.com"}
                if download is hub.hf_hub_download:
                    kwargs["filename"] = "config.json"
                with pytest.raises(DownloadPolicyError):
                    download(**kwargs)
                assert (visited)
                assert (set(visited)) == ({"hf-mirror.com"})
            finally:
                from acprof.hf_transport import _httpx_factory
                hub.set_client_factory(_httpx_factory)

def test_xet_is_disabled_even_if_imported_before_configuration():
    from huggingface_hub import constants
    with patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0"}), patch.object(constants, "HF_HUB_DISABLE_XET", False):
        configure_hf_transport()
        assert (os.environ["HF_HUB_DISABLE_XET"]) == ("1")
        assert (constants.HF_HUB_DISABLE_XET)

def test_requests_transport_checks_redirect_before_network():
    import requests

    from acprof.hf_transport import _requests_factory

    def respond(_adapter, request, **kwargs):
        response = requests.Response()
        response.status_code = 302
        response.url = request.url
        response.request = request
        response._content = b""
        response.headers["Location"] = "https://untrusted.example/file"
        return response

    with patch.dict(os.environ, {"HF_DOWNLOAD_MODE": "mirror-only", "HF_ENDPOINT": "https://hf-mirror.com",
                                 "HF_FALLBACK_ENDPOINTS": ""}), patch("requests.adapters.HTTPAdapter.send", autospec=True, side_effect=respond) as transport:
        with _requests_factory() as session, pytest.raises(DownloadPolicyError):
            session.get("https://hf-mirror.com/resolve/file")
        assert (transport.call_count) == (1)

def test_budget_rejects_unknown_and_oversize_before_download():
    assert (parse_bytes("5GB")) == (5_000_000_000)
    assert (parse_bytes("5GiB")) == (5 * 1024 ** 3)
    for estimate in (None, 5000000001):
        with pytest.raises(DownloadPolicyError):
            enforce_download_budget({"expected_download_bytes": estimate}, "5GB")
    enforce_download_budget({"expected_download_bytes": 0}, "0")
