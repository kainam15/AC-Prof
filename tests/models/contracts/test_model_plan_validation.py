"""A matching digest does not make persisted download instructions valid."""
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container.download_model import verify_download
from acprof.container.model_files import ModelFilesError, plan_download, seal_plan, validate_plan
from acprof.host.model_store import store as model_store


@pytest.fixture
def plan():
    return plan_download(model_id="example/model", revision="a" * 40, family="nlp",
        backend="transformers_pipeline", policy="full", files={"config.json": {"size": 1}},
        read_json=lambda _: {})


@pytest.mark.parametrize(("field", "value"), [
    ("schema_version", True), ("schema_version", 1.0),
    ("model_id", None), ("model_id", 7), ("model_id", "../model"),
    ("model_id", "/model"), ("model_id", "a/b/c"), ("model_id", "a--b"),
    ("model_id", "a\\b"), ("model_id", "a\x00b"), ("model_id", "用户/model"),
    ("model_revision", None), ("model_revision", True), ("model_revision", "main"),
    ("model_revision", "a" * 39), ("model_revision", "/outside"),
    ("model_revision", "../../outside"),
    ("source", None), ("source", "unknown"), ("source", {}),
    ("selected_bytes", True), ("selected_bytes", 1.0), ("selected_bytes", -1),
    ("selected_bytes", None), ("selected_bytes", 2),
    ("total_selected_bytes", True), ("total_selected_bytes", 1.0),
    ("total_selected_bytes", -1), ("total_selected_bytes", 2),
])
def test_resealed_plan_rejects_invalid_identity_and_accounting(plan, field, value):
    plan[field] = value
    seal_plan(plan)
    with pytest.raises(ModelFilesError):
        validate_plan(plan)


@pytest.mark.parametrize("name", [".", "a\x00b", "../outside", "/outside", "a//b", "a/./b", "a\\b"])
def test_resealed_plan_rejects_invalid_file_path(plan, name):
    plan["files"][0]["path"] = name
    seal_plan(plan)
    with pytest.raises(ModelFilesError, match="path"):
        validate_plan(plan)


@pytest.mark.parametrize("size", [True, -1, 1.0, "1"])
@pytest.mark.parametrize("verified", [False, True])
def test_resealed_file_size_is_an_integer_byte_count(plan, size, verified):
    plan["files"][0].update(size=size, sha256=hashlib.sha256(b"x").hexdigest())
    plan["selected_bytes"] = size
    if verified:
        plan["verification"] = "sha256"
    seal_plan(plan)
    with pytest.raises(ModelFilesError):
        validate_plan(plan)


def test_verified_file_digest_must_be_a_string(plan):
    plan["verification"] = "sha256"
    plan["files"][0]["sha256"] = int("1" * 64)
    seal_plan(plan)
    with pytest.raises(ModelFilesError, match="hash"):
        validate_plan(plan)


@pytest.mark.parametrize("aggregate", ["absent", "present", "empty-dependencies"])
def test_unknown_size_stays_unknown_until_payload_verification(plan, tmp_path, aggregate):
    plan["files"][0].pop("size")
    plan["selected_bytes"] = None
    if aggregate != "absent":
        plan["total_selected_bytes"] = None
    if aggregate == "empty-dependencies":
        plan["dependencies"] = []
    seal_plan(plan)
    validate_plan(plan)
    (tmp_path / "config.json").write_bytes(b"{}")
    verified = verify_download(tmp_path, plan)
    assert verified["files"][0]["size"] == 2
    assert verified["selected_bytes"] == 2
    assert verified["files"][0]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    if aggregate != "absent":
        assert verified["total_selected_bytes"] == 2
    validate_plan(verified)


@pytest.mark.parametrize("source", [None, "huggingface", "modelscope"])
def test_valid_v1_plan_retains_optional_source_and_provenance(plan, source):
    if source is not None:
        plan["source"] = source
        plan["download_provenance"] = {"schema_version": 1, "observation": "unknown"}
        seal_plan(plan)
    original = copy.deepcopy(plan)
    validate_plan(plan)
    assert plan == original


def test_dependency_file_total_is_checked_even_when_parent_total_matches(plan):
    child = copy.deepcopy(plan)
    child.update(model_id="example/base", selected_bytes=0)
    seal_plan(child)
    plan["dependencies"] = [{"repo_id": "example/base", "revision": "a" * 40, "download": child}]
    plan["total_selected_bytes"] = 1
    seal_plan(plan)
    with pytest.raises(ModelFilesError, match="bytes"):
        validate_plan(plan)


@pytest.mark.parametrize("model_id", ["bert-base-uncased", "Org/Model_1.2"])
def test_planner_accepts_existing_repository_shapes(model_id):
    plan = plan_download(model_id=model_id, revision="b" * 40, family="nlp", backend="custom",
        policy="full", files={"中文 config.json": {"size": 0}}, read_json=lambda _: {})
    validate_plan(plan)
    assert plan["selected_bytes"] == 0


@pytest.mark.parametrize(("model_id", "revision", "name", "size"), [
    ("../model", "a" * 40, "config.json", 1),
    ("example/model", None, "config.json", 1),
    ("example/model", "a" * 40, ".", 1),
    ("example/model", "a" * 40, "a\x00b", 1),
    ("example/model", "a" * 40, "config.json", -1),
    ("example/model", "a" * 40, "config.json", True),
])
def test_planner_cannot_create_invalid_file_instructions(model_id, revision, name, size):
    with pytest.raises(ModelFilesError):
        plan_download(model_id=model_id, revision=revision, family="nlp", backend="custom",
            policy="full", files={name: {"size": size}}, read_json=lambda _: {})


@pytest.mark.parametrize(("field", "value"), [
    ("model_revision", "../../outside"), ("selected_bytes", 0), ("plan_sha256", "0" * 64),
])
def test_prepare_rejects_invalid_plan_before_cache_or_transport(plan, tmp_path, field, value):
    task = SimpleNamespace(model_id=plan["model_id"], model_revision=plan["model_revision"],
        task_family="nlp", runtime_backend="transformers_pipeline", model_adapter="family-default",
        model_download_policy="full", model_spec={}, model_resolution={})
    plan["endpoint"] = "https://huggingface.co"
    plan[field] = value
    if field != "plan_sha256":
        seal_plan(plan)
    root = tmp_path / "store"
    with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")) as download, patch.object(
            model_store, "configure_hf_transport") as transport:
        with pytest.raises(ModelFilesError):
            model_store.prepare_model(task, plan, root)
    download.assert_not_called()
    transport.assert_not_called()
    assert not root.exists()


def test_verification_rejects_escaping_file_before_reading_payload(plan, tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    plan["files"][0]["path"] = "../outside"
    seal_plan(plan)
    with patch("pathlib.Path.open", side_effect=AssertionError("unexpected payload read")):
        with pytest.raises(ModelFilesError, match="path"):
            verify_download(root, plan)
    assert outside.read_bytes() == b"x"


@pytest.mark.parametrize(("field", "value"), [
    ("model_id", "../model"), ("model_revision", "/outside"), ("selected_bytes", -1),
])
def test_persisted_resealed_plan_is_rejected_without_rewriting(plan, tmp_path, field, value):
    plan.update(verification="sha256")
    plan["files"][0]["sha256"] = hashlib.sha256(b"x").hexdigest()
    plan[field] = value
    seal_plan(plan)
    path = tmp_path / "entries" / ("a" * 64) / "model_download_plan.json"
    path.parent.mkdir(parents=True)
    payload = json.dumps(plan).encode()
    path.write_bytes(payload)
    with pytest.raises(ModelFilesError):
        model_store.read_entry("a" * 64, tmp_path)
    assert path.read_bytes() == payload
