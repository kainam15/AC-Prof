"""Frozen metadata replay exercises production decisions, never checkpoint execution."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def corpus(tmp_path):
    content = b'{"model_type":"bert","architectures":["BertForMaskedLM"]}\n'
    (tmp_path / "config.json").write_bytes(content)
    case = {
        "case_id": "bert", "source": "huggingface", "model_id": "fixture/bert",
        "revision": "a" * 40, "weight": 1, "fixture_kind": "synthetic",
        "description": "Reviewed native masked language model layout.",
        "hub": {"pipeline_tag": "fill-mask", "library_name": "transformers", "config": {}},
        "repository_files": ["config.json", "model.safetensors"],
        "metadata": {"config.json": {
            "path": "config.json", "sha256": hashlib.sha256(content).hexdigest(),
            "origin": "synthetic",
        }},
        "expected": {"inspection_status": "candidate", "compatibility_status": "supported",
                     "reason_code": None, "selected_task": "fill-mask"},
    }
    manifest = {"schema_version": 1, "corpus_version": 1,
                "sampling": "fixed synthetic regression scenarios; not Hub coverage",
                "weight_basis": "uniform scenarios", "models": [case]}
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def save_manifest(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_replay_uses_real_inspection_and_never_requests_runtime(corpus, tmp_path):
    from acprof.testing.model_corpus import replay_corpus

    path, _ = corpus
    with patch("acprof.host.preparation.model_inspection.validate_model_runtime",
               side_effect=AssertionError("runtime must not execute")), patch(
        "acprof.host.model_store.store.cached_model_info",
        side_effect=AssertionError("personal cache must not affect the corpus"),
    ):
        report = replay_corpus(path, tmp_path / "out")

    assert report["summary"] == {"cases": 1, "matched": 1, "mismatched": 0,
                                  "fixture_kinds": {"synthetic": 1}, "attention": []}
    assert report["rows"][0]["actual"]["selected_task"] == "fill-mask"
    assert report["rows"][0]["actual"]["compatibility_status"] == "supported"
    assert report["rows"][0]["actual"]["runtime_status"] == "not_requested"
    assert report["source_identity"]["source_sha256"]
    coverage = json.loads((tmp_path / "out/coverage.json").read_text())
    assert coverage["summary"]["weighted_runtime_success_rate"] is None
    assert json.loads((tmp_path / "out/corpus.json").read_text()) == report
    assert (tmp_path / "out/corpus.csv").is_file()
    assert "not checkpoint" in (tmp_path / "out/CORPUS.md").read_text()


def test_expected_mismatch_is_reported_and_cli_fails(corpus, tmp_path, capsys):
    from scripts.replay_model_corpus import main

    path, manifest = corpus
    manifest["models"][0]["expected"]["selected_task"] = "text-generation"
    save_manifest(path, manifest)

    assert main([str(path), "--output-dir", str(tmp_path / "out")]) == 1
    report = json.loads((tmp_path / "out/corpus.json").read_text())
    assert report["summary"]["mismatched"] == 1
    assert report["rows"][0]["differences"] == {
        "selected_task": {"expected": "text-generation", "actual": "fill-mask"}}
    assert "1 mismatch" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["hash", "escape", "duplicate", "revision", "expectation"])
def test_invalid_corpus_is_rejected_before_creating_output(corpus, tmp_path, damage):
    from acprof.testing.model_corpus import replay_corpus

    path, manifest = corpus
    case = manifest["models"][0]
    if damage == "hash":
        (tmp_path / "config.json").write_text("{}")
    elif damage == "escape":
        case["metadata"]["config.json"]["path"] = "../outside.json"
    elif damage == "duplicate":
        manifest["models"].append(copy.deepcopy(case))
    elif damage == "revision":
        case["revision"] = "main"
    else:
        case["expected"]["typo_status"] = "supported"
    save_manifest(path, manifest)

    with pytest.raises(ValueError, match={
        "hash": "SHA256", "escape": "path|unsafe", "duplicate": "duplicate",
        "revision": "commit SHA", "expectation": "decision fields",
    }[damage]):
        replay_corpus(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_source_identity_keeps_same_model_from_two_hubs_separate(corpus, tmp_path):
    from acprof.testing.model_corpus import replay_corpus

    path, manifest = corpus
    second = copy.deepcopy(manifest["models"][0])
    second.update(case_id="bert-modelscope", source="modelscope")
    manifest["models"].append(second)
    save_manifest(path, manifest)
    report = replay_corpus(path, tmp_path / "out")

    assert report["summary"]["matched"] == 2
    assert [row["source"] for row in report["rows"]] == ["huggingface", "modelscope"]


def test_baseline_keeps_before_and_after_decisions(corpus, tmp_path):
    from acprof.failures import Failure
    from acprof.host.preparation.task_support import TaskSupportError
    from acprof.models.inspection.decision import record_resolution_failure
    from acprof.testing.model_corpus import replay_corpus

    path, _ = corpus
    replay_corpus(path, tmp_path / "before")

    def reject(task, **_kwargs):
        failure = record_resolution_failure(task, Failure("preflight", "runtime_unavailable", "fixture regression"))
        raise TaskSupportError(failure)

    with patch("acprof.host.preparation.task_support.require_task_support", side_effect=reject):
        report = replay_corpus(path, tmp_path / "after", baseline=tmp_path / "before/corpus.json")

    assert report["summary"]["mismatched"] == 1
    change = report["changes"][0]
    assert change["case_id"] == "bert"
    assert change["fields"]["reason_code"] == {"before": None, "after": "runtime_unavailable"}
    assert report["summary"]["attention"][0]["reason_code"] == "runtime_unavailable"
    assert report["summary"]["attention"][0]["case_ids"] == ["bert"]


@pytest.mark.parametrize("damage", ["sample", "missing-row", "duplicate-row", "source", "runtime", "provenance"])
def test_incomparable_or_invalid_baseline_is_rejected(corpus, tmp_path, damage):
    from acprof.testing.model_corpus import replay_corpus

    path, manifest = corpus
    replay_corpus(path, tmp_path / "before")
    baseline = tmp_path / "before/corpus.json"
    old = json.loads(baseline.read_text())
    if damage == "sample":
        manifest["models"][0]["hub"]["pipeline_tag"] = "text-generation"
        save_manifest(path, manifest)
    elif damage == "missing-row":
        old["rows"].clear()
    elif damage == "duplicate-row":
        old["rows"].append(copy.deepcopy(old["rows"][0]))
    elif damage == "source":
        old["rows"][0]["source"] = "modelscope"
    elif damage == "runtime":
        old["rows"][0]["actual"]["runtime_status"] = "ok"
    else:
        old.pop("source_identity")
    baseline.write_text(json.dumps(old))

    with pytest.raises(ValueError, match="baseline|runtime evidence"):
        replay_corpus(path, tmp_path / "after", baseline=baseline)
    assert not (tmp_path / "after").exists()


def test_expected_rejections_count_as_matching_regressions(corpus, tmp_path):
    from acprof.testing.model_corpus import replay_corpus

    path, manifest = corpus
    case = manifest["models"][0]
    case["repository_files"] = ["config.json", "model.gguf"]
    case["expected"] = {"inspection_status": "blocked", "compatibility_status": "unknown",
                        "reason_code": "artifact_layout_unsupported"}
    save_manifest(path, manifest)
    report = replay_corpus(path, tmp_path / "out")
    assert report["summary"]["matched"] == 1
    assert report["summary"]["attention"][0]["reason_code"] == "artifact_layout_unsupported"
    assert report["rows"][0]["failure"]["detail"]


@pytest.mark.parametrize("operation", ["ipv4", "ipv6", "connect_ex", "dns"])
def test_network_attempt_is_an_invalid_replay_not_a_model_failure(corpus, tmp_path, operation):
    import errno
    import socket

    from acprof.testing.model_corpus import replay_corpus

    if operation == "ipv6":
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM):
                pass
        except OSError as exc:
            if exc.errno in {errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT}:
                pytest.skip("host kernel does not support IPv6 sockets")
            raise

    def unexpected_network(*_args, **_kwargs):
        if operation == "dns":
            return socket.getaddrinfo("example.invalid", 443)
        with socket.socket(socket.AF_INET6 if operation == "ipv6" else socket.AF_INET, socket.SOCK_STREAM) as stream:
            if operation == "connect_ex":
                return stream.connect_ex(("127.0.0.1", 9))
            stream.connect(("::1" if operation == "ipv6" else "127.0.0.1", 9))

    with patch("acprof.host.preparation.task_support.require_task_support", side_effect=unexpected_network):
        with pytest.raises(ValueError, match="offline replay"):
            replay_corpus(corpus[0], tmp_path / "out")
    assert not (tmp_path / "out/corpus.json").exists()


@pytest.mark.parametrize("operation", ["runtime", "unfrozen-metadata"])
def test_attempts_outside_static_snapshot_invalidate_replay(corpus, tmp_path, operation):
    from acprof.host import detect
    from acprof.host.preparation import model_inspection
    from acprof.testing.model_corpus import replay_corpus

    def unexpected_operation(*args, **kwargs):
        if operation == "runtime":
            return model_inspection.validate_model_runtime(*args, **kwargs)
        return detect._download_metadata("fixture/bert", "unfrozen.json", revision="a" * 40)

    with patch("acprof.host.preparation.task_support.require_task_support", side_effect=unexpected_operation):
        with pytest.raises(ValueError, match="invalid offline replay"):
            replay_corpus(corpus[0], tmp_path / "out")
    assert not (tmp_path / "out/corpus.json").exists()


def test_checked_in_corpus_matches_reviewed_expectations(tmp_path):
    from acprof.testing.model_corpus import replay_corpus

    report = replay_corpus(ROOT / "tests/fixtures/model_corpus.json", tmp_path / "out")
    assert report["summary"]["cases"] >= 10
    assert report["summary"]["mismatched"] == 0, [
        (row["case_id"], row["differences"]) for row in report["rows"] if row["differences"]]
    assert all(row["actual"]["runtime_status"] == "not_requested" for row in report["rows"])


def test_expectation_review_does_not_masquerade_as_a_behavior_change(corpus, tmp_path):
    from acprof.testing.model_corpus import replay_corpus

    path, manifest = corpus
    before = replay_corpus(path, tmp_path / "before")
    manifest["models"][0]["expected"]["selected_task"] = "text-generation"
    save_manifest(path, manifest)
    after = replay_corpus(path, tmp_path / "after", baseline=tmp_path / "before/corpus.json")

    assert after["changes"] == []
    assert after["summary"]["mismatched"] == 1
    assert after["expectations_sha256"] != before["expectations_sha256"]
    assert after["baseline"]["expectations_sha256"] == before["expectations_sha256"]


def test_fresh_process_replays_without_loading_inference_frameworks(corpus, tmp_path):
    script = """
import sys
from acprof.testing.model_corpus import replay_corpus
report = replay_corpus(sys.argv[1], sys.argv[2])
assert report['summary']['mismatched'] == 0
assert not {'torch', 'transformers', 'diffusers', 'onnxruntime', 'kokoro'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(corpus[0]), str(tmp_path / "out")],
        cwd=ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "MODELSCOPE_OFFLINE": "1"}, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_attention_ranks_repeated_blockers_without_hiding_expected_rejections(corpus, tmp_path):
    from acprof.testing.model_corpus import replay_corpus

    path, manifest = corpus
    original = manifest["models"][0]
    cases = []
    for name in ("z-gguf", "a-gguf", "bare-onnx"):
        case = copy.deepcopy(original)
        case.update(case_id=name, model_id=f"fixture/{name}")
        if name == "bare-onnx":
            case.update(hub={}, metadata={}, repository_files=["iris.onnx"], expected={
                "inspection_status": "needs_configuration", "compatibility_status": "unknown",
                "reason_code": "model_contract_required"})
        else:
            case.update(repository_files=["config.json", "model.gguf"], expected={
                "inspection_status": "blocked", "compatibility_status": "unknown",
                "reason_code": "artifact_layout_unsupported"})
        cases.append(case)
    manifest["models"] = cases
    save_manifest(path, manifest)
    report = replay_corpus(path, tmp_path / "out")

    assert report["summary"]["cases"] == report["summary"]["matched"] == 3
    assert [(group["reason_code"], group["count"], group["case_ids"])
            for group in report["summary"]["attention"]] == [
        ("artifact_layout_unsupported", 2, ["a-gguf", "z-gguf"]),
        ("model_contract_required", 1, ["bare-onnx"]),
    ]
