"""Wrapped repository failures retain their public classification and fallback policy."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from acprof.host.detect import detect_task
from acprof.host.preparation.model_errors import ModelLookupError, model_lookup_error
from tests.support.model_lookup_fixtures import hf_lookup_environment as hf_lookup_environment

MODEL = "fixture/lookup-context"
SHA = "a" * 40


def http_failure(kind, status):
    response = httpx.Response(status, request=httpx.Request("GET", f"https://huggingface.co/api/models/{MODEL}"))
    return kind("fixture repository failure", response=response)


def implicit_wrapper(failure):
    try:
        try:
            raise failure
        except Exception:
            # Exercise SDK-style implicit wrapping, without an explicit cause.
            raise RuntimeError("fixture lookup wrapper")  # noqa: B904
    except RuntimeError as wrapped:
        return wrapped


@pytest.fixture
def cached_config(tmp_path, monkeypatch, hf_lookup_environment):
    path = tmp_path / "snapshots" / SHA / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model_type": "bert", "architectures": ["BertForMaskedLM"]}))

    def download(*, repo_id, filename, revision, dry_run=False, **_kwargs):
        assert (repo_id, filename, revision) == (MODEL, "config.json", SHA)
        if dry_run:
            return SimpleNamespace(commit_hash=SHA, file_size=path.stat().st_size)
        return str(path)

    sdk_download = Mock(side_effect=download)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", sdk_download)
    return sdk_download


@pytest.mark.parametrize(("kind", "status", "reason"), [
    (GatedRepoError, 403, "access_denied"),
    (RepositoryNotFoundError, 404, "repository_unavailable"),
    (RevisionNotFoundError, 404, "revision_not_found"),
])
@pytest.mark.parametrize("override", [None, "fill-mask"])
def test_implicit_terminal_failure_cannot_use_cached_config_or_manual_task(
    cached_config, monkeypatch, kind, status, reason, override,
):
    failure = implicit_wrapper(http_failure(kind, status))
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", Mock(side_effect=failure))

    with pytest.raises(ModelLookupError) as caught:
        detect_task(MODEL, revision=SHA, override_tag=override)

    assert caught.value.reason_code == reason
    assert caught.value.blocks_fallback
    assert not caught.value.retryable
    assert caught.value.revision == SHA
    cached_config.assert_not_called()


@pytest.mark.parametrize(("failure", "reason", "retryable"), [
    (httpx.ReadTimeout("fixture timeout"), "network_error", True),
    (httpx.ConnectError("fixture connection"), "network_error", True),
    (OfflineModeIsEnabled("fixture offline"), "offline", False),
    (http_failure(HfHubHTTPError, 503), "hub_unavailable", True),
])
def test_implicit_failures_keep_network_service_and_offline_diagnostics(
    cached_config, monkeypatch, failure, reason, retryable,
):
    wrapped = implicit_wrapper(failure)
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", Mock(side_effect=wrapped))
    cached_config.side_effect = wrapped

    with pytest.raises(ModelLookupError) as caught:
        detect_task(MODEL, revision=SHA)

    error = caught.value
    assert error.reason_code == reason
    assert error.retryable is retryable
    assert not error.blocks_fallback
    if reason != "offline":
        assert error.download_error is not None
        assert error.download_error["source"] == "huggingface"
        assert error.download_error["attempts"]


def test_implicit_network_failure_can_recover_from_pinned_config(cached_config, monkeypatch):
    failure = implicit_wrapper(httpx.ReadTimeout("fixture timeout"))
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", Mock(side_effect=failure))

    task = detect_task(MODEL, revision=SHA)

    assert (task.pipeline_tag, task.model_revision, task.detection_method) == ("fill-mask", SHA, "config_infer")
    assert cached_config.called


def test_explicit_cause_keeps_precedence_over_incidental_context():
    failure = implicit_wrapper(http_failure(GatedRepoError, 403))
    failure.__cause__ = httpx.ReadTimeout("actual explicit cause")

    error = model_lookup_error(MODEL, failure)

    assert error.reason_code == "network_error"
    assert error.retryable
    assert not error.blocks_fallback


def test_cyclic_unknown_context_remains_an_unclassified_failure():
    failure = RuntimeError("fixture unknown failure")
    failure.__context__ = failure

    error = model_lookup_error(MODEL, failure)

    assert error.reason_code == "lookup_failed"
    assert not error.retryable
    assert "fixture unknown failure" in error.detail
