"""Exercise static metadata errors through the offline public inspection and coverage commands."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from huggingface_hub.errors import (
    BadRequestError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
)
from modelscope_hub.errors import NotExistError, PermissionDeniedError, RequestTimeoutError

from acprof.cli.main import main
from acprof.host import detect
from acprof.hub.repository import MetadataFile
from acprof.testing.model_corpus import _offline_repositories, load_corpus

CORPUS = Path(__file__).resolve().parents[2] / "fixtures/model_corpus.json"
HF = "hf-bert-base-uncased"
MS = "ms-qwen3-06b"
FILENAME = "tokenizer_config.json"
REQUEST = httpx.Request("GET", "https://huggingface.co")


@contextmanager
def frozen_metadata(case_id, changes, *, without_inventory=False):
    manifest, content, _ = load_corpus(CORPUS)
    case = next(item for item in manifest["models"] if item["case_id"] == case_id)
    if without_inventory:
        case["repository_files"] = []
    with _offline_repositories(manifest, content), patch("acprof.host.env_utils.bootstrap_project_env"):
        frozen = detect._download_metadata

        def read(model, name, revision):
            if name not in changes:
                return frozen(model, name, revision)
            value = changes[name]
            if isinstance(value, Exception):
                raise value
            return MetadataFile(Path(name), value)

        with patch("acprof.host.detect._download_metadata", side_effect=read):
            yield case


def inspect(case, output):
    return main(["inspect", case["model_id"], "--model-source", case["source"], "--revision", case["revision"],
                 "--task", case["hub"]["pipeline_tag"], "--output-dir", str(output)])


@pytest.mark.parametrize(("case_id", "value", "code", "retry", "lookup"), [
    (HF, LocalEntryNotFoundError("uncached entry"), "metadata_unavailable", "after_configuration", "local_cache_miss"),
    (MS, RequestTimeoutError("timed out: https://modelscope.cn/config?token=redaction-canary"), "metadata_unavailable", "transient", "network_error"),
    (MS, NotExistError("not found or private"), "metadata_unavailable", "after_configuration", "repository_unavailable"),
    (MS, PermissionDeniedError("forbidden"), "access_denied", "after_configuration", "access_denied"),
    (HF, RemoteEntryNotFoundError("entry missing", response=httpx.Response(404, request=REQUEST)),
     "artifact_incomplete", "after_configuration", "entry_not_found"),
    (HF, BadRequestError("bad request", response=httpx.Response(400, request=REQUEST)),
     "metadata_unavailable", "after_configuration", "lookup_failed"),
    (HF, b"{bad", "metadata_invalid", "not_retryable", None),
], ids=["hf-cache-miss", "ms-timeout", "ms-ambiguous-404", "ms-denied", "hf-remote-missing", "bad-request", "invalid-json"])
def test_inspect_distinguishes_pinned_metadata_failures(tmp_path, case_id, value, code, retry, lookup):
    from acprof.tui.run.preparation import failure_views
    from acprof.tui.ui.view_models import reason_presentation

    output = tmp_path / "inspect"
    with frozen_metadata(case_id, {FILENAME: value}) as case:
        assert inspect(case, output) == 2
    result = json.loads((output / "model_resolution.json").read_bytes())
    assert result["status"] == "blocked"
    assert "redaction-canary" not in json.dumps(result)
    failure = result["failure"]
    assert (failure["reason_code"], failure["retryability"]) == (code, retry)
    diagnostic, = failure["evidence"]["diagnostics"]
    assert diagnostic["file"] == FILENAME
    if lookup:
        assert diagnostic["lookup_reason"] == lookup
        assert "raw_sha256" not in diagnostic
    else:
        assert diagnostic["exception_type"] == "JSONDecodeError"
        assert diagnostic["raw_sha256"] == hashlib.sha256(b"{bad").hexdigest()
    candidates = result["configuration_options"]["backend_candidates"]
    assert candidates
    assert all(not option["selectable"] for option in candidates)
    for task in result["configuration_options"]["task_options"]:
        assert task["source_discovery"]["diagnostics"] == [diagnostic]
    if lookup:
        assert all(option["compatibility"]["status"] == "unknown" for option in candidates)
    assert failure_views({"failure": failure}, "resolution")[1].problem == reason_presentation(code).problem
    if code == "metadata_unavailable":
        report = tmp_path / "report"
        assert main(["coverage", "report", str(output), "--output-dir", str(report)]) == 0
        row, = json.loads((report / "coverage.json").read_bytes())["rows"]
        assert (row["status"], row["failure"]["reason_code"], row["revision"]) == ("inconclusive", code, case["revision"])


@pytest.mark.parametrize("case_id", [HF, MS])
def test_inspect_intact_snapshot_still_resolves(tmp_path, case_id):
    with frozen_metadata(case_id, {}) as case:
        assert inspect(case, tmp_path) == 0
    resolution = json.loads((tmp_path / "model_resolution.json").read_bytes())
    assert resolution["compatibility"]["status"] == "supported"


def test_malformed_content_preserves_concurrent_read_error(tmp_path):
    changes = {"config.json": RequestTimeoutError("timed out"), FILENAME: b"{bad"}
    with frozen_metadata(MS, changes) as case:
        assert inspect(case, tmp_path) == 2
    failure = json.loads((tmp_path / "model_resolution.json").read_bytes())["failure"]
    assert failure["reason_code"] == "metadata_invalid"
    assert {item["file"]: item["reason_code"] for item in failure["evidence"]["diagnostics"]} == {
        "config.json": "metadata_unavailable", FILENAME: "metadata_invalid"}


def test_coverage_keeps_metadata_unavailable_for_retry_and_tui(tmp_path):
    from acprof.tui.models.candidates import coverage_candidates, model_choice
    from acprof.tui.ui.view_models import reason_presentation

    sample = tmp_path / "sample.json"
    output = tmp_path / "coverage"
    with frozen_metadata(MS, {FILENAME: RequestTimeoutError("timed out")}) as case:
        sample.write_text(json.dumps({"schema_version": 1, "sampling": "fixed", "weight_basis": "unit",
                                      "models": [case]}))
        assert main(["coverage", "run", str(sample), "--output-dir", str(output)]) == 0
    path = output / "coverage.json"
    row, = json.loads(path.read_bytes())["rows"]
    assert (row["failure"]["reason_code"], row["resolved"], row["supported"], row["runtime_status"]) == (
        "metadata_unavailable", False, False, "not_requested")
    candidate, = coverage_candidates(path)
    callout = model_choice(candidate).callout
    assert callout.problem != reason_presentation("unknown").problem
    assert "metadata_unavailable" in callout.detail
    with frozen_metadata(MS, {}):
        assert main(["coverage", "run", str(sample), "--output-dir", str(output), "--resume",
                     "--retry-reason", "metadata_unavailable"]) == 0
    recovered, = json.loads(path.read_bytes())["rows"]
    assert recovered["supported"]
    assert recovered["attempt_id"] != row["attempt_id"]


def test_coverage_denied_while_reading_a_listed_file(tmp_path):
    sample = tmp_path / "sample.json"
    output = tmp_path / "coverage"
    with frozen_metadata(MS, {FILENAME: PermissionDeniedError("forbidden")}) as case:
        sample.write_text(json.dumps({"schema_version": 1, "sampling": "fixed", "weight_basis": "unit",
                                      "models": [case]}))
        assert main(["coverage", "run", str(sample), "--output-dir", str(output)]) == 0
    row, = json.loads((output / "coverage.json").read_bytes())["rows"]
    assert (row["resolution_status"], row["failure"]["reason_code"]) == ("access_denied", "access_denied")
    assert not row["supported"]


def test_pinned_snapshot_without_file_inventory_keeps_unknown_layout(tmp_path):
    with frozen_metadata(HF, {}, without_inventory=True) as case:
        assert inspect(case, tmp_path) == 0
    result = json.loads((tmp_path / "model_resolution.json").read_bytes())
    assert result["artifact_layout"]["status"] == "unknown"
    assert result["compatibility"]["status"] == "unknown"
    assert "failure" not in result
