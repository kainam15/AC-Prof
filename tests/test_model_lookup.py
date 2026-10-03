"""Model lookup failures retain actionable causes without terminating callers."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from acprof.cli.main import main
from acprof.host.detect import detect_task
from acprof.host.model_errors import ModelLookupError


def hub_error(kind, status):
    response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/models/asdf"))
    return kind("fixture Hub failure", response=response)
def test_invalid_id_is_rejected_before_network_even_with_manual_task():
    with patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(
        sha="a" * 40, pipeline_tag="text-generation", library_name="transformers",
    )) as hub, patch(
        "acprof.host.detect._download_metadata",
    ) as download, pytest.raises(ModelLookupError) as caught:
        detect_task("invalid model/id", override_tag="text-generation")
    assert (caught.value.reason_code) == ("invalid_model_id")
    assert ("invalid model/id") in (str(caught.value))
    hub.assert_not_called()
    download.assert_not_called()

@pytest.mark.parametrize('override', (None, 'text-generation'))
@pytest.mark.parametrize('failure_case', range(4), ids=["(hub_error(RepositoryNotFoundError, 401), 'repository_unavailable')", "(hub_error(RepositoryNotFoundError, 404), 'repository_unavailable')", "(hub_error(GatedRepoError, 403), 'access_denied')", "(hub_error(RevisionNotFoundError, 404), 'revision_not_found')"])
def test_permanent_lookup_errors_cannot_be_bypassed_by_manual_task(override, failure_case):
    (failure, reason) = tuple(((hub_error(RepositoryNotFoundError, 401), 'repository_unavailable'), (hub_error(RepositoryNotFoundError, 404), 'repository_unavailable'), (hub_error(GatedRepoError, 403), 'access_denied'), (hub_error(RevisionNotFoundError, 404), 'revision_not_found')))[failure_case]
    with patch("huggingface_hub.HfApi.model_info", side_effect=failure), patch(
        "acprof.host.detect._download_metadata",
    ) as download, pytest.raises(ModelLookupError) as caught:
        detect_task("asdf", revision="missing-revision", override_tag=override)
    error = caught.value
    assert (error.reason_code) == (reason)
    assert not (error.retryable)
    assert (error.__cause__) is (failure)
    assert ("asdf") in (str(error))
    assert ("--task-family") not in (str(error))
    download.assert_not_called()

@pytest.mark.parametrize('failure_case', range(6), ids=["(httpx.ReadTimeout('fixture timeout'), 'network_error', True)", "(httpx.ConnectError('fixture connection failure'), 'network_error', True)", "(OfflineModeIsEnabled('fixture offline'), 'offline', False)", "(hub_error(HfHubHTTPError, 429), 'hub_unavailable', True)", "(hub_error(HfHubHTTPError, 503), 'hub_unavailable', True)", "(RuntimeError('model not found timeout'), 'lookup_failed', False)"])
def test_network_and_service_errors_remain_distinct_from_missing_models(failure_case):
    (failure, reason, retryable) = tuple(((httpx.ReadTimeout('fixture timeout'), 'network_error', True), (httpx.ConnectError('fixture connection failure'), 'network_error', True), (OfflineModeIsEnabled('fixture offline'), 'offline', False), (hub_error(HfHubHTTPError, 429), 'hub_unavailable', True), (hub_error(HfHubHTTPError, 503), 'hub_unavailable', True), (RuntimeError('model not found timeout'), 'lookup_failed', False)))[failure_case]
    with patch("huggingface_hub.HfApi.model_info", side_effect=failure), patch(
        "acprof.host.detect._download_metadata", side_effect=failure,
    ), pytest.raises(ModelLookupError) as caught:
        detect_task("asdf")
    error = caught.value
    assert (error.reason_code) == (reason)
    assert (error.retryable) == (retryable)
    assert (error.__cause__) is (failure)
    assert (type(failure).__name__) in (error.detail)

def test_network_failure_can_still_use_pinned_cached_config():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "snapshots", "a" * 40, "config.json")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"architectures": ["BertForMaskedLM"]}))
        with patch("huggingface_hub.HfApi.model_info", side_effect=httpx.ReadTimeout("fixture timeout")), patch(
            "acprof.host.detect._download_metadata", return_value=str(path),
        ):
            task = detect_task("asdf")
    assert (task.model_revision) == ("a" * 40)
    assert (task.pipeline_tag) == ("fill-mask")

def test_accessible_model_without_task_is_not_reported_as_missing():
    hub = SimpleNamespace(sha="a" * 40, pipeline_tag=None, library_name="transformers")
    with patch("huggingface_hub.HfApi.model_info", return_value=hub):
        task = detect_task("asdf")
    assert (task.model_id) == ("asdf")
    assert (task.pipeline_tag) == ("unknown")

def test_inspect_cli_reports_lookup_failure_without_traceback_or_probe():
    stderr = io.StringIO()
    with patch("acprof.host.env_utils.bootstrap_project_env"), patch(
        "huggingface_hub.HfApi.model_info", side_effect=hub_error(RepositoryNotFoundError, 404),
    ), patch("acprof.host.detect._download_metadata", side_effect=OSError("missing")), patch(
        "acprof.host.model_inspection.validate_model_runtime",
    ) as probe, contextlib.redirect_stderr(stderr):
        result = main(["inspect", "asdf", "--probe-interface"])
    assert (result) == (1)
    assert ("asdf") in (stderr.getvalue())
    assert ("未找到") in (stderr.getvalue())
    assert ("Traceback") not in (stderr.getvalue())
    assert ("--task-family") not in (stderr.getvalue())
    probe.assert_not_called()
