"""Static inspection must reject oversized Hub files before downloading bodies."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from acprof.host import detect

SHA = "a" * 40
METADATA_LIMIT = 1024 * 1024
SOURCE_LIMIT = 256 * 1024


@pytest.fixture
def hub_http(tmp_path, monkeypatch):
    import huggingface_hub as hub
    from huggingface_hub import constants

    from acprof.hf_transport import _httpx_factory, _request_hook, configure_hf_transport

    monkeypatch.setenv("HF_DOWNLOAD_MODE", "mirror-only")
    monkeypatch.setenv("HF_ENDPOINT", "https://metadata.example")
    monkeypatch.setenv("HF_FALLBACK_ENDPOINTS", "")
    cache = tmp_path / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    configure_hf_transport()

    def install(handler):
        hub.set_client_factory(lambda: httpx.Client(
            transport=httpx.MockTransport(handler), follow_redirects=True,
            event_hooks={"request": [_request_hook]},
        ))
        return cache

    try:
        yield install
    finally:
        hub.set_client_factory(_httpx_factory)


@pytest.mark.parametrize("filename,limit", [
    ("config.json", METADATA_LIMIT),
    ("model_index.json", METADATA_LIMIT),
    ("pipeline.py", SOURCE_LIMIT),
])
def test_oversized_static_files_never_download_a_body(hub_http, filename, limit):
    visited = []

    def respond(request):
        visited.append(request.method)
        return httpx.Response(200, headers={
            "x-repo-commit": SHA, "etag": "b" * 64, "content-length": str(limit + 1),
        }, content=b"{}" if request.method == "GET" else b"")

    cache = hub_http(respond)
    with pytest.raises((ValueError, OSError)):
        if filename.endswith(".py"):
            detect.read_model_source("fixture/model", filename, SHA)
        else:
            detect._download_metadata("fixture/model", filename, SHA)
    assert visited == ["HEAD"]
    assert not list(cache.rglob("*.incomplete"))


def test_pinned_download_reuses_its_cache_without_network(hub_http):
    payload = b'{"architectures":["BertForMaskedLM"]}'
    visited = []

    def respond(request):
        visited.append((request.method, request.url.path))
        return httpx.Response(200, headers={
            "x-repo-commit": SHA, "etag": "b" * 64, "content-length": str(len(payload)),
        }, content=payload if request.method == "GET" else b"")

    hub_http(respond)
    path = detect._download_metadata("fixture/model", "config.json", SHA)
    assert Path(path).read_bytes() == payload
    assert visited[0] == ("HEAD", f"/fixture/model/resolve/{SHA}/config.json")
    assert all(f"/resolve/{SHA}/" in url for _, url in visited)
    initial_requests = list(visited)
    assert detect._download_metadata("fixture/model", "config.json", SHA) == path
    assert visited == initial_requests


@pytest.mark.parametrize("size", [None, True, -1, "10", METADATA_LIMIT + 1])
def test_unknown_or_invalid_preflight_size_stops_before_download(size):
    info = SimpleNamespace(file_size=size, commit_hash=SHA)
    with patch("huggingface_hub.hf_hub_download", return_value=info) as download:
        with pytest.raises(ValueError, match="size"):
            detect._download_metadata("fixture/model", "config.json", SHA)
    assert download.call_count == 1
    assert download.call_args.kwargs["dry_run"] is True


@pytest.mark.parametrize("commit", ["main", "b" * 40, None])
def test_preflight_cannot_change_a_requested_commit(commit):
    info = SimpleNamespace(file_size=2, commit_hash=commit)
    with patch("huggingface_hub.hf_hub_download", return_value=info) as download:
        with pytest.raises(ValueError, match="commit|revision"):
            detect._download_metadata("fixture/model", "config.json", SHA)
    assert download.call_count == 1


def test_download_rechecks_local_size_after_preflight(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b" " * (METADATA_LIMIT + 1))
    info = SimpleNamespace(file_size=2, commit_hash=SHA)
    with patch("huggingface_hub.hf_hub_download", side_effect=[info, str(path)]):
        with pytest.raises(ValueError, match="size|exceeds"):
            detect._download_metadata("fixture/model", "config.json", SHA)


def test_cached_oversized_file_is_rejected_without_network(hub_http):
    visited = []

    def respond(request):
        visited.append(request.method)
        raise AssertionError("a pinned cache hit must not contact the Hub")

    cache = hub_http(respond)
    path = cache / "models--fixture--model" / "snapshots" / SHA / "config.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b" " * (METADATA_LIMIT + 1))
    with pytest.raises(ValueError, match="size|exceeds"):
        detect._download_metadata("fixture/model", "config.json", SHA)
    assert visited == []


def test_cached_config_fallback_still_works_offline(hub_http):
    from huggingface_hub.errors import OfflineModeIsEnabled

    def respond(request):
        raise OfflineModeIsEnabled("fixture offline")

    cache = hub_http(respond)
    root = cache / "models--fixture--model"
    path = root / "snapshots" / SHA / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"architectures": ["BertForMaskedLM"]}), encoding="utf-8")
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(SHA, encoding="utf-8")
    info = detect._detect_from_config("fixture/model")
    assert info is not None
    assert info.model_revision == SHA
    assert info.pipeline_tag == "fill-mask"


@pytest.mark.parametrize("reader", ["repository", "config_fallback", "diffusers"])
def test_static_readers_bound_content_when_the_cached_file_changes(tmp_path, reader):
    path = tmp_path / "snapshots" / SHA / "config.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b" " * METADATA_LIMIT + (
        b'{"architectures":["BertForMaskedLM"],"_class_name":"StableDiffusionPipeline"}'
    ))
    diagnostics = []
    with patch("acprof.host.detect._download_metadata", return_value=str(path)):
        if reader == "repository":
            metadata = detect._repository_metadata(
                "fixture/model", SHA,
                SimpleNamespace(siblings=[SimpleNamespace(rfilename="config.json")]),
            )
            assert metadata["repository_metadata"] == {}
            diagnostics = list(metadata["metadata_errors"])
        elif reader == "config_fallback":
            assert detect._detect_from_config("fixture/model", diagnostics) is None
        else:
            assert detect._diffusers_task_from_index("fixture/model", SHA, diagnostics) is None
    assert "exceeds" in ";".join(diagnostics)


def test_endpoint_fallback_keeps_the_requested_commit(monkeypatch):
    from huggingface_hub.errors import FileMetadataError, LocalEntryNotFoundError

    monkeypatch.setenv("HF_DOWNLOAD_MODE", "mirror-preferred")
    monkeypatch.setenv("HF_ENDPOINT", "https://metadata.example")
    monkeypatch.setenv("HF_FALLBACK_ENDPOINTS", "https://huggingface.co")
    monkeypatch.setenv("ACPROF_ALLOW_PROXY_FALLBACK", "1")
    error = LocalEntryNotFoundError("fixture metadata failure")
    error.__cause__ = FileMetadataError("missing headers")
    first = SimpleNamespace(file_size=2, commit_hash=SHA)
    changed = SimpleNamespace(file_size=2, commit_hash="b" * 40)
    with patch("huggingface_hub.hf_hub_download", side_effect=[first, error, changed]) as download:
        with pytest.raises(ValueError, match="different model revision"):
            detect._download_metadata("fixture/model", "config.json", SHA)
    calls = download.call_args_list
    assert len(calls) == 3
    assert all(call.kwargs["revision"] == SHA for call in calls)
    assert calls[-1].kwargs["endpoint"] == "https://huggingface.co"
    assert calls[-1].kwargs["dry_run"] is True


def test_fresh_default_config_fallback_remains_available_offline(hub_http):
    from huggingface_hub.errors import OfflineModeIsEnabled

    payload = b'{"architectures":["BertForMaskedLM"]}'
    offline = False

    def respond(request):
        if offline:
            raise OfflineModeIsEnabled("fixture offline")
        return httpx.Response(200, headers={
            "x-repo-commit": SHA, "etag": "b" * 64, "content-length": str(len(payload)),
        }, content=payload if request.method == "GET" else b"")

    hub_http(respond)
    first = detect._detect_from_config("fixture/model")
    assert first is not None
    offline = True
    second = detect._detect_from_config("fixture/model")
    assert second is not None
    assert second.model_revision == first.model_revision == SHA
    assert second.pipeline_tag == "fill-mask"


def test_unpinned_fallback_rejects_oversized_content_before_parsing(tmp_path):
    path = tmp_path / "snapshots" / SHA / "config.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b" " * METADATA_LIMIT + b'{"architectures":["BertForMaskedLM"]}')
    diagnostics = []
    with patch("huggingface_hub.hf_hub_download", return_value=str(path)) as download, patch(
        "acprof.host.detect._read_json_metadata",
    ) as parse:
        assert detect._detect_from_config("fixture/model", diagnostics) is None
    assert "exceeds" in ";".join(diagnostics)
    assert download.call_count == 1
    assert "dry_run" not in download.call_args.kwargs
    parse.assert_not_called()
