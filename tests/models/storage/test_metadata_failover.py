"""A model lookup must reuse the Hub entry that just recovered its metadata."""
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from acprof.host.detect import detect_task
from acprof.hub.huggingface.download import hf_endpoint_scope, try_hf_endpoints
from acprof.hub.huggingface.transport import (
    _httpx_factory,
    configure_hf_transport,
    observed_hf_provenance,
)

SHA = "a" * 40
MODEL = "fixture/metadata-failover"
MIRROR = "hf-mirror.com"
OFFICIAL = "huggingface.co"


@pytest.fixture
def metadata_hub(tmp_path, monkeypatch):
    import huggingface_hub as hub
    from huggingface_hub import constants
    from huggingface_hub.utils import _http

    monkeypatch.setenv("HF_DOWNLOAD_MODE", "auto")
    monkeypatch.setenv("HF_ENDPOINT", f"https://{MIRROR}")
    monkeypatch.setenv("HF_FALLBACK_ENDPOINTS", "")
    monkeypatch.setenv("HF_TOKEN", "hf_fixture_metadata_token")
    monkeypatch.setenv("ACPROF_HF_AUTH_ORIGINS", "")
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "huggingface")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr("acprof.host.model_store.store.cached_model_info", lambda *a, **kw: None)
    monkeypatch.setattr("acprof.host.model_store.store.cached_metadata", lambda *a, **kw: None)
    state = SimpleNamespace(visited=[], sleeps=[], fail="api", recovered=False)
    monkeypatch.setattr(_http.time, "sleep", state.sleeps.append)
    files = {
        "config.json": {"model_type": "bert", "architectures": ["BertForMaskedLM"]},
        "tokenizer_config.json": {"tokenizer_class": "BertTokenizer"},
        "preprocessor_config.json": {"fixture": True},
    }

    def respond(request):
        host, path = request.url.host, request.url.path
        state.visited.append((request.method, host, path))
        assert request.headers.get("authorization") == (
            "Bearer hf_fixture_metadata_token" if host == OFFICIAL else None
        )
        assert not request.url.params
        is_api = path.startswith("/api/models/")
        if host == MIRROR and not state.recovered and (state.fail == "api" or not is_api):
            raise httpx.ConnectError("fixture mirror TLS failure", request=request)
        if is_api:
            return httpx.Response(200, json={
                "id": MODEL, "sha": SHA, "pipeline_tag": "fill-mask",
                "library_name": "transformers", "config": files["config.json"],
                "siblings": [{"rfilename": name} for name in files],
            })
        assert f"/resolve/{SHA}/" in path
        payload = json.dumps(files[path.rsplit("/", 1)[-1]]).encode()
        return httpx.Response(200, headers={
            "x-repo-commit": SHA, "etag": hashlib.sha256(payload).hexdigest(),
            "content-length": str(len(payload)),
        }, content=payload if request.method == "GET" else b"")

    configure_hf_transport()
    hub.set_client_factory(lambda: _httpx_factory(transport=httpx.MockTransport(respond)))
    try:
        yield state
    finally:
        hub.set_client_factory(_httpx_factory)


def test_api_failover_is_reused_for_every_metadata_file(metadata_hub):
    info = detect_task(MODEL, revision=SHA)

    assert info.model_revision == SHA
    assert info.pipeline_tag == "fill-mask"
    assert not info.metadata_errors
    assert set(info.repository_metadata) == {
        "config.json", "tokenizer_config.json", "preprocessor_config.json",
    }
    mirror = [request for request in metadata_hub.visited if request[1] == MIRROR]
    assert mirror == [("GET", MIRROR, f"/api/models/{MODEL}/revision/{SHA}")]
    assert metadata_hub.sleeps == []
    failures = observed_hf_provenance()["failed_attempts"]
    assert len(failures) == 1
    assert failures[0]["host"] == MIRROR


def test_file_failover_is_reused_by_later_metadata_files(metadata_hub):
    metadata_hub.fail = "file"

    info = detect_task(MODEL, revision=SHA)

    assert not info.metadata_errors
    mirror_files = [path for _, host, path in metadata_hub.visited
                    if host == MIRROR and "/resolve/" in path]
    assert mirror_files
    assert set(mirror_files) == {f"/{MODEL}/resolve/{SHA}/config.json"}
    assert set(info.repository_metadata) == {
        "config.json", "tokenizer_config.json", "preprocessor_config.json",
    }


def test_next_detection_retries_the_configured_primary(metadata_hub):
    detect_task(MODEL, revision=SHA)
    metadata_hub.visited.clear()
    metadata_hub.recovered = True

    info = detect_task(MODEL, revision=SHA)

    assert info.model_revision == SHA
    assert metadata_hub.visited[0] == ("GET", MIRROR, f"/api/models/{MODEL}/revision/{SHA}")
    assert observed_hf_provenance()["failed_attempts"] == []


def test_failed_detection_releases_its_endpoint_selection(metadata_hub, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("fixture declaration conflict")

    with monkeypatch.context() as patcher:
        patcher.setattr("acprof.models.candidates.apply_model_candidate", reject)
        with pytest.raises(ValueError, match="fixture declaration conflict"):
            detect_task(MODEL, revision=SHA)
    metadata_hub.visited.clear()
    metadata_hub.recovered = True

    assert detect_task(MODEL, revision=SHA).model_revision == SHA
    assert metadata_hub.visited[0][1] == MIRROR


@pytest.mark.parametrize("different", ["model", "endpoints"])
def test_selection_is_isolated_by_model_and_endpoint_configuration(different):
    seen = []
    choices = [f"https://{MIRROR}", f"https://{OFFICIAL}"]

    def unavailable(endpoint):
        if endpoint == choices[0]:
            raise ConnectionError("fixture mirror unavailable")
        return "first lookup"

    with hf_endpoint_scope():
        assert try_hf_endpoints(MODEL, unavailable, endpoints=choices) == "first lookup"
        second_model = MODEL + "-dependency" if different == "model" else MODEL
        second_choices = choices if different == "model" else ["https://other.example", choices[-1]]
        try_hf_endpoints(second_model, seen.append, endpoints=second_choices)

    assert seen == [second_choices[0]]


def test_nested_lookup_restores_the_outer_successful_fallback():
    choices = [f"https://{MIRROR}", f"https://{OFFICIAL}"]
    seen = []

    def lookup(endpoint):
        seen.append(endpoint)
        if endpoint == choices[0]:
            raise ConnectionError("fixture mirror unavailable")
        return "metadata"

    with hf_endpoint_scope():
        try_hf_endpoints(MODEL, lookup, endpoints=choices)
        with hf_endpoint_scope():
            try_hf_endpoints(MODEL, lookup, endpoints=choices)
        try_hf_endpoints(MODEL, lookup, endpoints=choices)
    try_hf_endpoints(MODEL, lookup, endpoints=choices)

    assert seen == [*choices, *choices, choices[-1], *choices]


def test_selected_endpoint_keeps_the_final_file_error_and_prior_diagnostics():
    from huggingface_hub.errors import RemoteEntryNotFoundError

    choices = [f"https://{MIRROR}", f"https://{OFFICIAL}"]
    seen = []
    error = RemoteEntryNotFoundError("fixture file missing", response=httpx.Response(
        404, request=httpx.Request("HEAD", choices[-1] + "/missing.json"),
    ))

    def first(endpoint):
        if endpoint == choices[0]:
            raise ConnectionError("fixture mirror unavailable")
        return "available config"

    def missing(endpoint):
        seen.append(endpoint)
        raise error

    with hf_endpoint_scope():
        try_hf_endpoints(MODEL, first, endpoints=choices)
        with pytest.raises(RemoteEntryNotFoundError) as raised:
            try_hf_endpoints(MODEL, missing, endpoints=choices)

    assert raised.value is error
    assert seen == [choices[-1]]
    assert [attempt["exception_type"] for attempt in observed_hf_provenance()["failed_attempts"]] == [
        "ConnectionError", "RemoteEntryNotFoundError",
    ]
