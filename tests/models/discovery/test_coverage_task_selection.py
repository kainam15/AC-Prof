"""Frozen coverage rows select discovered workloads without changing the sample weight."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.cli.main import main
from acprof.host.detect import detect_task
from acprof.host.preparation.task_selection import select_tasks
from acprof.testing.model_corpus import _offline_repositories, load_corpus

CORPUS = Path(__file__).resolve().parents[2] / "fixtures/model_corpus.json"


@pytest.fixture
def coverage(tmp_path):
    manifest, content, _ = load_corpus(CORPUS)
    cases = {c["case_id"]: c for c in manifest["models"]}

    def run(case_id, name, *, changes=None, flags=()):
        case = cases[case_id]
        model = {key: case[key] for key in ("model_id", "source", "revision", "weight")}
        model.update(changes or {})
        sample = tmp_path / "sample.json"
        sample.write_text(json.dumps({"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform", "models": [model]}))
        root = tmp_path / name
        rc = main(["coverage", "run", str(sample), "--output-dir", str(root), *flags])
        return rc, json.loads((root / "coverage.json").read_bytes()) if (root / "coverage.json").is_file() else {}

    with _offline_repositories(manifest, content), patch("acprof.host.env_utils.bootstrap_project_env"), patch(
        "acprof.host.automation.check_repository_access",
    ), patch("acprof.host.preparation.model_inspection.validate_model_runtime", return_value={"status": "ok"}) as runtime:
        yield run, runtime, cases


def test_coverage_selects_one_discovered_t5_profile_or_semantic_task(coverage, tmp_path):
    run, runtime, cases = coverage
    case_id = "t5-author-config"
    case = cases[case_id]
    rc, unselected = run(case_id, "unselected", flags=("--validate-runtime",))
    assert rc == 0
    pending, = unselected["rows"]
    failure = pending["failure"]
    assert failure["reason_code"] == "task_selection_required"
    assert "task_id" in pending["next_action"]
    assert "translation_en_to_fr" in failure["evidence"]["semantic_task_ids"]
    assert len(failure["evidence"]["profile_ids"]) == 2
    assert all(task_id in pending["next_action"] for task_id in failure["evidence"]["profile_ids"])
    runtime.assert_not_called()

    for index, task_id in enumerate(["translation_en_to_fr", *failure["evidence"]["profile_ids"]]):
        output = f"chosen-{index}"
        changes = {"task_id": task_id, "semantic_reference": {"task": "text2text-generation", "source": "review"}}
        rc, report = run(case_id, output, changes=changes, flags=("--validate-runtime",))
        assert rc == 0
        row, = report["rows"]
        task = runtime.call_args.args[0]
        choice, = task.selected_tasks
        assert (task.model_revision, task.model_source) == (case["revision"], case["source"])
        assert (row["requested_task_id"], row["selected_task_id"], row["task_selection_source"]) == (
            task_id, choice["task_id"], "explicit")
        assert row["selected_task"] == "text2text-generation"
        assert row["runtime_status"] == "ok"
        assert row["semantic_correct"] is None
        assert report["summary"]["semantic_reviewed_weight"] == 0
        assert report["summary"]["sample_weight"] == case["weight"]
        resolution = json.loads(next((tmp_path / output).rglob("model_resolution.json")).read_bytes())
        assert resolution["selected_task"]["fingerprint"] == choice["fingerprint"]
        if index == 0:
            assert choice["execution_profile"]["representative_task_id"] == task_id
        else:
            assert choice["task_id"] == task_id
    assert runtime.call_count == 3

    rc, _ = run(case_id, "chosen-0", changes={**changes, "task_id": "summarization"}, flags=("--validate-runtime", "--resume"))
    assert rc == 2
    assert runtime.call_count == 3


@pytest.mark.parametrize("value", [None, 3, "", "  ", "all", "a,b"])
def test_invalid_manifest_selector_refused_before_output(coverage, value):
    run, runtime, _ = coverage
    rc, result = run("t5-author-config", "bad", changes={"task_id": value})
    assert rc == 2
    assert result == {}
    runtime.assert_not_called()


def test_unknown_and_preselected_ids_cannot_reach_runtime(coverage):
    run, runtime, cases = coverage
    case = cases["hf-bert-base-uncased"]
    task = detect_task(case["model_id"], revision=case["revision"])
    selected = select_tasks(task, SimpleNamespace(tasks="", task="", backend=""), single_profile=True)
    for index, original in enumerate((task, selected)):
        with patch("acprof.host.detect.detect_task", return_value=original):
            rc, report = run(case["case_id"], f"typo-{index}", changes={"task_id": "unknown-role"}, flags=("--validate-runtime",))
        assert rc == 0
        row, = report["rows"]
        assert row["reason_code"] == "task_selection_required"
        assert row["requested_task_id"] == "unknown-role"
        assert "selected_task_id" not in row
        assert not row["supported"]
        assert row["compatibility"]["status"] == "unknown"
    runtime.assert_not_called()


@pytest.mark.parametrize(("case_id", "reason"), [
    ("invalid-metadata", "metadata_invalid"), ("missing-shard", "artifact_incomplete"),
    ("gguf-incompatible", "artifact_layout_unsupported"), ("conflicting-hub", "evidence_conflict"),
])
def test_existing_static_blocker_takes_priority(coverage, tmp_path, case_id, reason):
    run, runtime, _ = coverage
    _, report = run(case_id, "blocked", changes={"task_id": "unrelated"}, flags=("--validate-runtime",))
    row, = report["rows"]
    assert row["reason_code"] == reason
    decision = json.loads(next((tmp_path / "blocked").rglob("model_resolution.json")).read_bytes())
    assert decision["failure"]["reason_code"] == row["reason_code"]
    assert decision["status"] == row["inspection_status"]
    assert row["requested_task_id"] == "unrelated"
    assert "selected_task_id" not in row
    runtime.assert_not_called()


def test_unbound_historical_attempt_cannot_satisfy_explicit_request(coverage, tmp_path):
    run, runtime, _ = coverage
    changes = {"task_id": "translation_en_to_fr"}
    rc, _ = run("t5-author-config", "chosen", changes=changes, flags=("--validate-runtime",))
    assert rc == 0
    journal = next((tmp_path / "chosen").rglob("attempt.json"))
    attempt = json.loads(journal.read_bytes())
    del attempt["rows"][0]["requested_task_id"]
    journal.write_text(json.dumps(attempt))
    rc, _ = run("t5-author-config", "chosen", changes=changes, flags=("--validate-runtime", "--resume"))
    assert rc == 2
    runtime.assert_called_once()


def test_preselected_translation_cannot_claim_a_different_exemplar_in_the_same_profile(coverage):
    run, runtime, cases = coverage
    case = cases["t5-author-config"]
    task = detect_task(case["model_id"], revision=case["revision"])
    args = SimpleNamespace(tasks="translation_en_to_de", task="", backend="")
    selected = select_tasks(task, args, single_profile=True)
    with patch("acprof.host.detect.detect_task", return_value=selected):
        rc, report = run(case["case_id"], "other-language", changes={"task_id": "translation_en_to_fr"},
                         flags=("--validate-runtime",))
    assert rc == 0
    row, = report["rows"]
    assert row["reason_code"] == "task_selection_required"
    assert row["failure"]["evidence"]["semantic_task_ids"] == ["translation_en_to_de"]
    assert "selected_task_id" not in row
    runtime.assert_not_called()
