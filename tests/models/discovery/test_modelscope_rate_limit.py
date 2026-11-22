"""Provider response codes retain their meaning in static inspection and coverage."""
import json
from unittest.mock import patch

import pytest
from modelscope_hub.errors import APIError, raise_for_status
from requests import Response

from acprof.cli.main import main
from acprof.failures import RuntimeFailure
from acprof.host.automation import check_repository_access
from acprof.host.detect import detect_task
from acprof.hub.repository import MetadataFile


@pytest.mark.parametrize(("status", "code", "reason", "retryability"), [
    (403, 10013201001, "metadata_unavailable", "transient"),
    (429, None, "metadata_unavailable", "transient"),
    (403, 1001, "access_denied", "after_configuration"),
    (401, 10013201001, "access_denied", "after_configuration"),
])
def test_source_rejection_preserved_across_public_entries(tmp_path, monkeypatch, capsys,
                                                         status, code, reason, retryability):
    response = Response()
    response.status_code = status
    # Same message for throttling and denial: only structured evidence distinguishes them.
    response._content = json.dumps({"Code": code, "Message": "操作过于频繁，请稍后再试!", "Success": False}).encode()
    with pytest.raises(APIError) as caught:
        raise_for_status(response)

    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "modelscope")
    sample = tmp_path / "sample.json"
    model = {"source": "modelscope", "model_id": "fixture/model", "revision": "a" * 40, "weight": 1}
    sample.write_text(json.dumps({"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
                                 "models": [model]}))
    out = tmp_path / "report"
    with patch("acprof.hub.repository.modelscope_info", side_effect=caught.value) as lookup, patch(
            "acprof.host.env_utils.bootstrap_project_env"):
        assert main(["inspect", model["model_id"], "--revision", model["revision"], "--model-source", "modelscope"]) == 1
        text = capsys.readouterr().err
        assert ("MODELSCOPE_API_TOKEN" in text) == (reason == "access_denied")
        assert lookup.call_count == 1
        assert main(["coverage", "run", str(sample), "--output-dir", str(out)]) == 0
        assert lookup.call_count == 2
        row = json.loads((out / "coverage.json").read_bytes())["rows"][0]
        assert row["failed_stage"] == "resolution"
        assert row["reason_code"] == row["failure"]["reason_code"] == reason
        assert row["failure"]["retryability"] == retryability
        lookup_reason = "hub_unavailable" if retryability == "transient" else "access_denied"
        assert row["failure"]["evidence"]["lookup_reason"] == lookup_reason
        assert ("MODELSCOPE_API_TOKEN" in row["next_action"]) == (reason == "access_denied")
        assert row["runtime_status"] == "not_requested"

        with pytest.raises(RuntimeFailure) as access:
            check_repository_access(model["model_id"], source="modelscope", revision=model["revision"])
        assert lookup.call_count == 3
        assert access.value.failure.stage == "access"
        assert access.value.failure.reason_code == reason
        assert access.value.failure.retryability == retryability


def test_throttled_repository_can_use_matching_cached_config_without_network(tmp_path, monkeypatch):
    from modelscope_hub.errors import RateLimitError

    sha = "c" * 40
    model_id = "fixture/model"
    config = MetadataFile(tmp_path / "snapshots" / sha / "config.json", b'{"architectures":["BertForMaskedLM"]}')
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "modelscope")
    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path / "store"))

    def cached(repo_id, name, revision, *, source, max_bytes):
        assert (repo_id, revision, source) == (model_id, sha, "modelscope")
        return config if name == "config.json" else None

    with patch("acprof.hub.repository.modelscope_info", side_effect=RateLimitError("throttled")) as lookup, patch(
            "acprof.host.model_store.store.cached_metadata", side_effect=cached):
        task = detect_task(model_id, revision=sha)
    assert (task.model_source, task.model_revision, task.pipeline_tag) == ("modelscope", sha, "fill-mask")
    lookup.assert_called_once()
