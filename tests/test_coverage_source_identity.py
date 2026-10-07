"""Coverage identity is source + repository + revision, never process defaults."""
import csv
import json
import os
from contextlib import nullcontext
from contextvars import Context
from unittest.mock import Mock, call

import pytest
from test_resolution_decisions import candidate

from acprof import model_repository
from acprof.analysis.compatibility import report_results
from acprof.host.model_coverage import run_sample, snapshot_sample, validate_sample

REVISION = "a" * 40
DEPENDENCY_REVISION = "b" * 40


def item(**changes):
    return {"model_id": "example/model", "revision": REVISION, "weight": 1, **changes}


def sample(*models):
    return {"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform", "models": list(models)}


@pytest.fixture
def isolated(monkeypatch):
    seen = []

    def detect(model_id, *, revision):
        source = model_repository.model_source()
        seen.append((source, model_id, revision))
        task = candidate(tag="text-generation")
        task.model_id, task.model_revision, task.model_source = model_id, revision, source
        return task

    monkeypatch.setattr("acprof.host.detect.detect_task", detect)
    monkeypatch.setattr("acprof.host.run_state.host_identity", lambda *args: {"machine_id_sha256": "fixture"})
    monkeypatch.setattr("acprof.host.run_state.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.host.task_support.require_task_support", Mock())
    monkeypatch.setattr("acprof.model_contract.write_model_resolution", Mock())
    access = Mock()
    runtime = Mock(return_value={"status": "ok", "devices": {}})
    monkeypatch.setattr("acprof.host.automation.check_repository_access", access)
    monkeypatch.setattr("acprof.host.model_inspection.validate_model_runtime", runtime)
    return seen, access, runtime


@pytest.mark.parametrize("source", [None, "huggingface", "modelscope"])
def test_sample_source_overrides_ambient_source_and_reaches_access_checks(tmp_path, monkeypatch, isolated, source):
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "modelscope" if source != "modelscope" else "huggingface")
    ambient = os.environ["ACPROF_MODEL_SOURCE"]
    model = item(**({} if source is None else {"source": source}))
    report = run_sample(sample(model), tmp_path / "report", probe="full")
    expected = source or "huggingface"
    seen, access, runtime = isolated
    assert seen == [(expected, model["model_id"], REVISION)]
    access.assert_called_once_with(model["model_id"], source=expected, revision=REVISION)
    runtime.assert_called_once()
    assert report["rows"][0]["source"] == expected
    assert report["rows"][0]["runtime_status"] == "ok"
    assert os.environ["ACPROF_MODEL_SOURCE"] == ambient
    assert model_repository.model_source() == ambient


def test_same_repo_and_revision_in_different_sources_stay_separate(tmp_path, isolated):
    root = tmp_path / "report"
    report = run_sample(sample(item(source="huggingface"), item(source="modelscope")), root)
    assert [row["source"] for row in report["rows"]] == ["huggingface", "modelscope"]
    with (root / "models.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["source"] for row in rows] == ["huggingface", "modelscope"]
    text = (root / "REPORT.md").read_text()
    assert "| Source |" in text and REVISION in text
    assert "huggingface" in text and "modelscope" in text


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
def test_duplicate_identity_is_rejected_within_one_source(source):
    with pytest.raises(ValueError, match="duplicate"):
        validate_sample(sample(item(source=source), item(source=source)))


@pytest.mark.parametrize("source", ["", "other", None, False, 7, [], {}])
def test_invalid_source_is_rejected_before_directory_or_network(tmp_path, monkeypatch, source):
    detect = Mock(side_effect=AssertionError("network must not be reached"))
    monkeypatch.setattr("acprof.host.detect.detect_task", detect)
    with pytest.raises(ValueError, match="source"):
        run_sample(sample(item(source=source)), tmp_path / "report")
    assert not (tmp_path / "report").exists()
    detect.assert_not_called()


def test_resolver_source_mismatch_is_not_counted_as_success(tmp_path, monkeypatch, isolated):
    task = candidate(tag="text-generation")
    task.model_source = "modelscope"
    monkeypatch.setattr("acprof.host.detect.detect_task", Mock(return_value=task))
    report = run_sample(sample(item(source="huggingface")), tmp_path / "report", probe="full")
    row = report["rows"][0]
    assert not row["resolved"] and not row["supported"]
    assert row["runtime_status"] == "failed"
    assert row["failed_stage"] == "resolution"
    assert row["source"] == "huggingface"
    isolated[1].assert_not_called()
    isolated[2].assert_not_called()


def test_modelscope_main_keeps_huggingface_dependency_and_its_own_revision(tmp_path, monkeypatch, isolated):
    monkeypatch.setattr("acprof.model_spec.task_model_spec", lambda task: {
        "dependencies": [{"repo_id": "example/dependency", "revision": DEPENDENCY_REVISION}],
    })
    run_sample(sample(item(source="modelscope")), tmp_path / "report", probe="full")
    assert isolated[1].call_args_list == [
        call("example/model", source="modelscope", revision=REVISION),
        call("example/dependency", source="huggingface", revision=DEPENDENCY_REVISION),
    ]


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_source_scope_restores_context_after_nested_failure(monkeypatch, error):
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "huggingface")
    with model_repository.model_source_scope("modelscope"):
        assert model_repository.model_source() == "modelscope"
        assert model_repository.model_source("huggingface") == "huggingface"
        assert Context().run(model_repository.model_source) == "huggingface"
        with pytest.raises(error):
            with model_repository.model_source_scope("huggingface"):
                raise error("fixture cancellation")
        assert model_repository.model_source() == "modelscope"
    assert model_repository.model_source() == "huggingface"
    assert os.environ["ACPROF_MODEL_SOURCE"] == "huggingface"


@pytest.mark.parametrize("replacement", [None, "huggingface"])
def test_resume_rejects_missing_or_conflicting_modelscope_row_source(tmp_path, isolated, replacement):
    manifest = sample(item(source="modelscope"))
    root = tmp_path / "report"
    run_sample(manifest, root)
    attempt = root / "attempts/attempt-000001/attempt.json"
    data = json.loads(attempt.read_text())
    if replacement is None:
        data["rows"][0].pop("source", None)
    else:
        data["rows"][0]["source"] = replacement
    attempt.write_text(json.dumps(data))
    before = attempt.read_bytes()
    with pytest.raises(ValueError, match="frozen sample"):
        run_sample(manifest, root, resume=True)
    assert attempt.read_bytes() == before
    assert not (root / "attempts/attempt-000002").exists()


def test_legacy_huggingface_attempt_is_not_rewritten_when_source_is_expanded(tmp_path, isolated):
    manifest = sample(item())
    root = tmp_path / "report"
    run_sample(manifest, root)
    attempt = root / "attempts/attempt-000001/attempt.json"
    data = json.loads(attempt.read_text())
    data["rows"][0].pop("source", None)
    attempt.write_text(json.dumps(data))
    before = attempt.read_bytes()
    report = run_sample(manifest, root, resume=True)
    assert report["rows"][0]["source"] == "huggingface"
    assert attempt.read_bytes() == before
    assert len(isolated[0]) == 1


@pytest.mark.parametrize("where", ["metadata", "resolution", "legacy"])
def test_recorded_report_preserves_source_without_changing_inputs(tmp_path, where):
    root = tmp_path / "result"
    root.mkdir()
    metadata = {"model_id": "example/model", "model_revision": REVISION}
    resolution = {"model_id": "example/model"}
    if where == "metadata":
        metadata["model_source"] = "modelscope"
    elif where == "resolution":
        resolution["provenance"] = {"sources": {"repository_snapshot": {"source": "modelscope"}}}
    (root / "static_meta.json").write_text(json.dumps(metadata))
    (root / "model_resolution.json").write_text(json.dumps(resolution))
    before = {path: path.read_bytes() for path in root.iterdir()}
    report = report_results([root], tmp_path / "output")
    assert report["rows"][0]["source"] == ("huggingface" if where == "legacy" else "modelscope")
    assert {path: path.read_bytes() for path in before} == before


def test_conflicting_recorded_sources_are_invalid_not_guessed(tmp_path):
    root = tmp_path / "result"
    root.mkdir()
    (root / "static_meta.json").write_text(json.dumps({"model_source": "huggingface"}))
    (root / "model_resolution.json").write_text(json.dumps({
        "provenance": {"sources": {"repository_snapshot": {"source": "modelscope"}}},
    }))
    report = report_results([root], tmp_path / "output")
    row = report["rows"][0]
    assert row["source"] == "unknown"
    assert row["failure"]["reason_code"] == "recorded_evidence_invalid"


@pytest.mark.parametrize("name", ["static_meta.json", "model_resolution.json"])
def test_unreadable_source_evidence_remains_unknown_instead_of_defaulting_to_hf(tmp_path, name):
    root = tmp_path / "result"
    root.mkdir()
    (root / "static_meta.json").write_text(json.dumps({"model_source": "huggingface"}))
    (root / "model_resolution.json").write_text(json.dumps({
        "provenance": {"sources": {"repository_snapshot": {"source": "huggingface"}}},
    }))
    (root / name).write_text("{broken")
    report = report_results([root], tmp_path / "output")
    assert report["rows"][0]["source"] == "unknown"
    assert report["rows"][0]["failure"]["reason_code"] == "recorded_evidence_invalid"
    assert (root / name).read_text() == "{broken"


def test_hub_snapshot_records_its_source_explicitly(monkeypatch):
    from types import SimpleNamespace
    model = SimpleNamespace(id="example/model", sha=REVISION, downloads=5,
                            pipeline_tag="text-generation", library_name="transformers")
    monkeypatch.setattr("huggingface_hub.HfApi.list_models", lambda *args, **kwargs: [model])
    result = snapshot_sample(["text-generation:transformers"], 1)
    assert result["models"][0]["source"] == "huggingface"
