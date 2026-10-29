"""Static inspection must reject oversized Hub files before downloading bodies."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from acprof.host import detect
from acprof.model_repository import MetadataFile, read_metadata_file

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


@pytest.mark.parametrize(("filename", "limit"), [
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
    download = detect.read_model_source if filename.endswith(".py") else detect._download_metadata
    with pytest.raises((ValueError, OSError)):
        download("fixture/model", filename, SHA)
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
    metadata = detect._download_metadata("fixture/model", "config.json", SHA)
    assert metadata.content == payload
    assert metadata.path.parent.name == SHA
    assert visited[0] == ("HEAD", f"/fixture/model/resolve/{SHA}/config.json")
    assert all(f"/resolve/{SHA}/" in url for _, url in visited)
    initial_requests = list(visited)
    assert detect._download_metadata("fixture/model", "config.json", SHA) == metadata
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
def test_static_readers_reject_oversized_metadata_buffers(tmp_path, reader):
    path = tmp_path / "snapshots" / SHA / "config.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b" " * METADATA_LIMIT + (
        b'{"architectures":["BertForMaskedLM"],"_class_name":"StableDiffusionPipeline"}'
    ))
    diagnostics = []
    with patch("acprof.host.detect._download_metadata", return_value=MetadataFile(path, path.read_bytes())):
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


def test_diffusers_reads_downloaded_content_even_if_snapshot_is_removed(hub_http):
    payload = b'{"_class_name":"StableDiffusionPipeline"}'

    def respond(request):
        return httpx.Response(200, headers={
            "x-repo-commit": SHA, "etag": "b" * 64, "content-length": str(len(payload)),
        }, content=payload if request.method == "GET" else b"")

    hub_http(respond)
    download = detect._download_metadata

    def remove_after_download(*args, **kwargs):
        metadata = download(*args, **kwargs)
        metadata.path.unlink()
        return metadata

    diagnostics = []
    with patch("acprof.host.detect._download_metadata", side_effect=remove_after_download):
        assert detect._diffusers_task_from_index("fixture/model", SHA, diagnostics) == "text-to-image"
    assert diagnostics == []


@pytest.mark.parametrize(("directory", "commit"), [("snapshots", "main"), ("blobs", SHA), ("snapshots", "b" * 40)])
def test_config_fallback_keeps_snapshot_identity_checks(tmp_path, directory, commit):
    path = tmp_path / directory / commit / "config.json"
    path.parent.mkdir(parents=True)
    content = b'{"architectures":["BertForMaskedLM"]}'
    path.write_bytes(content)
    diagnostics = []
    preflight = SimpleNamespace(file_size=len(content), commit_hash=SHA)
    with patch("huggingface_hub.hf_hub_download", side_effect=[preflight, str(path)]):
        assert detect._detect_from_config("fixture/model", diagnostics, requested_revision=SHA) is None
    assert any("pinned snapshot" in error or "different model revision" in error for error in diagnostics)


@pytest.mark.parametrize("replacement", [b" " * 9, b" "])
def test_metadata_checks_bytes_if_file_changes_after_fstat(tmp_path, replacement):
    path = tmp_path / "config.json"
    path.write_bytes(b"{}")
    fstat = os.fstat

    def change_after_stat(descriptor):
        info = fstat(descriptor)
        path.write_bytes(replacement)
        return info

    with patch("acprof.model_repository.os.fstat", side_effect=change_after_stat):
        with pytest.raises(ValueError, match="exceeds|size"):
            read_metadata_file(path, max_bytes=8, expected_size=2)


@pytest.mark.parametrize("content", [b"", b"{}"])
def test_metadata_accepts_exact_byte_limit_and_keeps_original_snapshot_path(tmp_path, content):
    blob = tmp_path / "blob"
    blob.write_bytes(content)
    link = tmp_path / "config.json"
    link.symlink_to(blob)
    metadata = read_metadata_file(link, max_bytes=len(content), expected_size=len(content))
    blob.unlink()
    assert metadata.path == link
    assert metadata.content == content


@pytest.mark.parametrize("limit", [-1, True])
def test_invalid_metadata_read_limit_cannot_disable_bounds(tmp_path, limit):
    with pytest.raises(ValueError, match="byte limit"):
        read_metadata_file(tmp_path / "not-opened", max_bytes=limit)


def test_metadata_rejects_fifo_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / "config.json"
    os.mkfifo(path)
    result = subprocess.run([sys.executable, "-c",
        "from acprof.model_repository import read_metadata_file; "
        "import sys; read_metadata_file(sys.argv[1], max_bytes=1024)", str(path)],
        capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0
    assert "metadata must be a regular file" in result.stderr
