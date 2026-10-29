"""Credential origin boundaries, with real SDK headers and guarded transports."""
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import pytest

from acprof.hub.huggingface.transport import (
    _httpx_factory,
    _requests_factory,
    observed_hf_provenance,
)

MIRROR = "https://hf-mirror.com"
OFFICIAL = "https://huggingface.co"
TOKEN = "hf_test_only_origin_token"


@pytest.fixture(autouse=True)
def isolated_hf_auth(tmp_path, monkeypatch):
    from huggingface_hub import constants

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
                "ACPROF_HF_AUTH_ORIGINS", "HF_FALLBACK_ENDPOINTS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_ENDPOINT", MIRROR)
    monkeypatch.setenv("HF_DOWNLOAD_MODE", "auto")
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tmp_path / "token"))
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_TELEMETRY", True)
    observed_hf_provenance(reset=True)


@contextmanager
def guarded_client(backend, respond):
    if backend == "httpx":
        with _httpx_factory(transport=httpx.MockTransport(respond)) as client:
            yield client
    else:
        import requests

        def send(_adapter, request, **_kwargs):
            payload = respond(request)
            response = requests.Response()
            response.status_code = payload.status_code
            response.url = request.url
            response.request = request
            response.headers.update(payload.headers)
            response._content = payload.content
            return response

        with patch("requests.adapters.HTTPAdapter.send", autospec=True, side_effect=send):
            with _requests_factory() as client:
                yield client


@pytest.mark.parametrize("backend", ["httpx", "requests"])
@pytest.mark.parametrize("credential_source", ["environment", "login"])
@pytest.mark.parametrize(("target", "opt_in", "authenticated"), [
    (MIRROR, "", False),
    (MIRROR, MIRROR, True),
    (MIRROR, "https://hf-mirror.com:443", True),
    (MIRROR, "https://hf-mirror.com:8443", False),
    (MIRROR, "http://hf-mirror.com", False),
    (OFFICIAL, "", True),
    ("https://us.aws.cdn.hf.co", "https://us.aws.cdn.hf.co", False),
])
def test_final_send_boundary_filters_sdk_implicit_credentials(
    backend, credential_source, target, opt_in, authenticated, monkeypatch, tmp_path,
):
    from huggingface_hub.utils import build_hf_headers

    if credential_source == "environment":
        monkeypatch.setenv("HF_TOKEN", TOKEN)
    else:
        (tmp_path / "token").write_text(TOKEN)
    monkeypatch.setenv("ACPROF_HF_AUTH_ORIGINS", opt_in)
    headers = {**build_hf_headers(), "Cookie": "session=test-only-cookie"}
    assert headers["authorization"] == f"Bearer {TOKEN}"
    sent = []

    def respond(request):
        sent.append((str(request.url), dict(request.headers)))
        return httpx.Response(200, content=b"{}")

    with guarded_client(backend, respond) as client:
        assert client.get(target + "/file?signature=test-only-signature", headers=headers).status_code == 200
    sent_headers = {key.lower(): value for key, value in sent[0][1].items()}
    assert sent_headers.get("authorization") == (f"Bearer {TOKEN}" if authenticated else None)
    assert ("cookie" in sent_headers) is authenticated
    assert "signature=test-only-signature" in sent[0][0]
    provenance = str(observed_hf_provenance())
    assert TOKEN not in provenance
    assert "test-only-cookie" not in provenance
    assert "test-only-signature" not in provenance


@pytest.mark.parametrize("backend", ["httpx", "requests"])
def test_redirect_cannot_move_official_credentials_to_anonymous_mirror(backend, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    sent = []

    def respond(request):
        sent.append((str(request.url), request.headers.get("authorization")))
        if str(request.url).startswith(OFFICIAL):
            return httpx.Response(302, headers={"location": MIRROR + "/file"})
        return httpx.Response(200, content=b"{}")

    with guarded_client(backend, respond) as client:
        assert client.get(OFFICIAL + "/file", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert sent == [(OFFICIAL + "/file", f"Bearer {TOKEN}"), (MIRROR + "/file", None)]


def test_bootstrap_does_not_promote_local_login_to_environment(tmp_path, monkeypatch):
    import os

    from acprof.host.env_utils import bootstrap_project_env

    (tmp_path / "token").write_text(TOKEN)
    bootstrap_project_env(tmp_path)
    configured = {key: os.environ.get(key) for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    assert configured == {"HF_TOKEN": None, "HUGGING_FACE_HUB_TOKEN": None}


@pytest.mark.parametrize(("primary", "alias", "source"), [
    (" configured ", "alias", "HF_TOKEN"),
    ("configured", "", "HF_TOKEN"),
    (" \t ", " configured ", "HUGGING_FACE_HUB_TOKEN"),
    ("", "", "local-login"),
])
def test_credential_source_and_environment_are_preserved(primary, alias, source, monkeypatch, tmp_path):
    import os

    from acprof.hub.huggingface.auth import resolve_hf_credential

    monkeypatch.setenv("HF_TOKEN", primary)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", alias)
    (tmp_path / "token").write_text(TOKEN)
    credential = resolve_hf_credential()
    assert credential.source == source
    assert credential.token == (TOKEN if source == "local-login" else "configured")
    assert credential.token not in repr(credential)
    assert os.environ["HF_TOKEN"] == primary
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == alias


def test_anonymous_origin_never_reads_login_and_missing_login_is_anonymous():
    from acprof.hub.huggingface.auth import hf_token_for_url, resolve_hf_credential

    with patch("huggingface_hub.utils.get_token", side_effect=AssertionError("login read")):
        assert hf_token_for_url(MIRROR) is False
    credential = resolve_hf_credential()
    assert credential.token is None
    assert credential.source == "none"
    assert hf_token_for_url(OFFICIAL) is False


def test_unreadable_login_is_reported_instead_of_silently_disabling_auth():
    from acprof.hub.huggingface.auth import hf_token_for_url

    with patch("huggingface_hub.utils.get_token", side_effect=OSError("unreadable login")):
        with pytest.raises(OSError, match="unreadable login"):
            hf_token_for_url(OFFICIAL)


def test_disable_implicit_token_avoids_login_lookup(monkeypatch):
    from acprof.hub.huggingface.auth import hf_token_for_url

    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    with patch("huggingface_hub.utils.get_token", side_effect=AssertionError("login read")):
        assert hf_token_for_url(OFFICIAL) is False


@pytest.mark.parametrize("value", [
    "https://user:secret@hf-mirror.com", "https://hf-mirror.com/path",
    "https://hf-mirror.com?token=secret", "https://hf-mirror.com#secret",
    "https://hf-mirror.com:bad", "https://hf-mirror.com:99999", "*", "hf-mirror.com",
])
def test_invalid_auth_origin_fails_without_echoing_secrets(value, monkeypatch):
    from acprof.hub.huggingface.auth import hf_token_for_url

    monkeypatch.setenv("ACPROF_HF_AUTH_ORIGINS", value)
    with pytest.raises(ValueError, match="ACPROF_HF_AUTH_ORIGINS") as error:
        hf_token_for_url(OFFICIAL)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("credential_source", ["environment", "login"])
def test_repository_access_reselects_credentials_for_official_fallback(monkeypatch, tmp_path, credential_source):
    import huggingface_hub as hub

    from acprof.host.automation import check_repository_access
    from acprof.hub.huggingface.transport import configure_hf_transport

    if credential_source == "environment":
        monkeypatch.setenv("HF_TOKEN", TOKEN)
    else:
        (tmp_path / "token").write_text(TOKEN)
    configure_hf_transport()
    sent = []

    def respond(request):
        sent.append((request.url.host, request.headers.get("authorization")))
        if request.url.host == "hf-mirror.com":
            return httpx.Response(401, headers={"X-Error-Code": "RepoNotFound"})
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        return httpx.Response(200, content=b"{}")

    hub.set_client_factory(lambda: _httpx_factory(transport=httpx.MockTransport(respond)))
    try:
        with patch("acprof.host.model_store.cached_model_info", return_value=None), patch(
            "huggingface_hub.utils._headers.get_token", side_effect=AssertionError("SDK implicit lookup"),
        ):
            report = check_repository_access("demo/private-model")
        assert report["status"] == "accessible"
        assert TOKEN not in str(report) + str(observed_hf_provenance())
        assert sent == [("hf-mirror.com", None), ("huggingface.co", f"Bearer {TOKEN}")]
    finally:
        hub.set_client_factory(_httpx_factory)


@pytest.mark.parametrize(("endpoint", "opt_in", "expected_auth"), [
    (MIRROR, "", None), (MIRROR, MIRROR, f"Bearer {TOKEN}"),
    (OFFICIAL, "", f"Bearer {TOKEN}"),
])
@pytest.mark.parametrize("operation", ["metadata", "snapshot", "probe"])
def test_download_callers_disable_sdk_lookup_and_apply_origin_policy(
    tmp_path, monkeypatch, endpoint, opt_in, expected_auth, operation,
):
    import hashlib
    from types import SimpleNamespace

    import huggingface_hub as hub
    from huggingface_hub import constants

    from acprof.container.model_files import plan_download, seal_plan
    from acprof.host import model_store
    from acprof.host.detect import _download_metadata
    from acprof.host.download_probe import ProbeBudget, probe_hf_file
    from acprof.hub.huggingface.transport import configure_hf_transport

    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    monkeypatch.setenv("ACPROF_HF_AUTH_ORIGINS", opt_in)
    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "cache"))
    configure_hf_transport()
    content, commit = b"{}", "a" * 40
    digest = hashlib.sha256(content).hexdigest()
    sent = []

    def respond(request):
        sent.append((request.method, str(request.url), request.headers.get("authorization")))
        assert request.headers.get("authorization") == expected_auth
        if "/api/models/" in request.url.path:
            if "/tree/" in request.url.path:
                return httpx.Response(200, json=[{"type": "file", "path": "config.json", "size": 2,
                                                  "oid": "c" * 40}])
            return httpx.Response(200, json={"id": "demo/model", "sha": commit,
                                             "siblings": [{"rfilename": "config.json"}]})
        return httpx.Response(200, headers={"x-repo-commit": commit, "etag": digest, "content-length": "2"},
                              stream=httpx.ByteStream(b"" if request.method == "HEAD" else content))

    hub.set_client_factory(lambda: _httpx_factory(transport=httpx.MockTransport(respond)))
    try:
        with patch("huggingface_hub.utils._headers.get_token", side_effect=AssertionError("SDK implicit lookup")):
            if operation == "metadata":
                assert _download_metadata("demo/model", "config.json", commit).content == content
            elif operation == "snapshot":
                task = SimpleNamespace(model_id="demo/model", model_revision=commit,
                    task_family="nlp", runtime_backend="transformers_pipeline", model_download_policy="full",
                    model_adapter="family-default", model_spec={}, model_resolution={})
                plan = plan_download(model_id="demo/model", revision=commit, family="nlp",
                    backend="transformers_pipeline", policy="full", read_json=lambda _: {},
                    files={"config.json": {"size": 2, "lfs_sha256": digest}})
                plan["endpoint"] = endpoint
                seal_plan(plan)
                record = model_store.prepare_model(task, plan, tmp_path / "store")
                assert record["entry_id"]
            else:
                report, error = probe_hf_file(endpoint + "/demo/model/resolve/main/config.json", ProbeBudget())
                assert error is None
                assert report["status"] == "passed"
        assert sent
        assert {row[2] for row in sent} == {expected_auth}
    finally:
        hub.set_client_factory(_httpx_factory)


def test_lookup_diagnostics_redact_credentials_and_signed_urls():
    from acprof.host.detect import _format_failure
    from acprof.host.model_errors import model_lookup_error

    error = ValueError(f"Authorization: Bearer {TOKEN}\nhttps://huggingface.co/file?signature=test-only-signature")
    diagnostic = _format_failure(error)
    for result in (diagnostic, str(model_lookup_error("demo/model", error).to_dict()),
                   str(model_lookup_error("demo/model", error, diagnostics=[str(error)]).to_dict())):
        assert TOKEN not in result
        assert "test-only-signature" not in result
