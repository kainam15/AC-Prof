"""A valid plan must still belong to the task that will consume its entry."""
import copy
import hashlib
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from acprof.container.model_files import ModelFilesError, plan_download, seal_plan
from acprof.host import model_store
from acprof.host.detect import TaskInfo

DIFFERENCES = {
    "source": "modelscope", "model_id": "example/other", "model_revision": "b" * 40,
    "requested_policy": "auto", "task_family": "multimodal", "backend": "diffusers", "adapter": "custom",
}


@pytest.fixture
def task():
    value = TaskInfo(model_id="example/model", model_revision="a" * 40, pipeline_tag="feature-extraction",
        task_family="nlp", runtime_backend="transformers_pipeline", library_name="transformers", detection_method="manual")
    value.model_download_policy = "full"
    return value


@pytest.fixture
def plan(task):
    result = plan_download(model_id=task.model_id, revision=task.model_revision, family="nlp",
        backend="transformers_pipeline", policy="full", files={"config.json": {"size": 2}}, read_json=lambda _: {})
    result["endpoint"] = "https://huggingface.co"
    return seal_plan(result)


def _mismatch(plan, field):
    result = copy.deepcopy(plan)
    if field == "dependencies":
        child = copy.deepcopy(plan)
        child["model_id"] = "example/base"
        seal_plan(child)
        result["dependencies"] = [{"repo_id": "example/base", "revision": "a" * 40, "download": child}]
        result["total_selected_bytes"] = 4
    else:
        result[field] = DIFFERENCES[field]
    return seal_plan(result)


def _publish(root, key, plan):
    plan = copy.deepcopy(plan)
    for repo in reversed(model_store.repository_plans(plan)):
        repo["verification"] = "sha256"
        repo["files"][0]["sha256"] = hashlib.sha256(b"{}").hexdigest()
        seal_plan(repo)
    path = root / "entries" / key / "model_download_plan.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan))
    return path


@pytest.mark.parametrize("field", [*DIFFERENCES, "dependencies"])
def test_preparation_binds_plan_to_task_before_any_cache_or_transport(task, plan, tmp_path, field):
    wrong = _mismatch(plan, field)
    root = tmp_path / "store"
    with patch.object(model_store, "configure_hf_transport") as transport, patch(
            "huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")) as hf, patch(
            "acprof.hub.repository.modelscope_file", side_effect=AssertionError("unexpected download")) as ms:
        with pytest.raises(ModelFilesError, match=field):
            model_store.prepare_model(task, wrong, root)
    transport.assert_not_called()
    hf.assert_not_called()
    ms.assert_not_called()
    assert not root.exists()


@pytest.mark.parametrize("field", [*DIFFERENCES, "dependencies"])
def test_cached_plan_must_match_the_task_without_network_fallback(task, plan, tmp_path, field):
    path = _publish(tmp_path, model_store.entry_key(task), _mismatch(plan, field))
    before = path.read_bytes()
    with patch("acprof.container.download_model._prepare_repository_plan", side_effect=AssertionError("unexpected metadata")) as lookup:
        with pytest.raises(ModelFilesError, match=field):
            model_store.plan_model(task, tmp_path)
    lookup.assert_not_called()
    assert path.read_bytes() == before


def test_cached_plan_appearing_while_waiting_for_lock_is_also_bound_to_task(task, plan, tmp_path):
    root = tmp_path / "store"
    original_lock = model_store.store_lock
    published = []

    @contextmanager
    def publish_under_lock(directory):
        with original_lock(directory):
            published.append(_publish(root, model_store.entry_key(task), _mismatch(plan, "model_id")))
            yield

    with patch.object(model_store, "store_lock", side_effect=publish_under_lock), patch(
            "acprof.container.download_model._prepare_repository_plan", side_effect=AssertionError("unexpected metadata")) as lookup:
        with pytest.raises(ModelFilesError, match="model_id"):
            model_store.plan_model(task, root)
    lookup.assert_not_called()
    assert len(published) == 1
    assert json.loads(published[0].read_text())["model_id"] == "example/other"


def test_preparation_rejects_wrong_existing_entry_before_budget_or_view_work(task, plan, tmp_path):
    path = _publish(tmp_path, model_store.entry_key(task), _mismatch(plan, "model_revision"))
    before = path.read_bytes()
    with patch.object(model_store, "view_complete", side_effect=AssertionError("unexpected view check")) as view, patch.object(
            model_store, "require_space", side_effect=AssertionError("unexpected budget check")) as budget:
        with pytest.raises(ModelFilesError, match="model_revision"):
            model_store.prepare_model(task, plan, tmp_path)
    view.assert_not_called()
    budget.assert_not_called()
    assert path.read_bytes() == before


def test_dependency_file_patterns_are_part_of_the_request(task, plan, tmp_path):
    plan = _mismatch(plan, "dependencies")
    task.model_spec = {"dependencies": [{"repo_id": "example/base", "revision": "a" * 40, "allow_patterns": ["*.json"]}]}
    _publish(tmp_path, model_store.entry_key(task), plan)
    with pytest.raises(ModelFilesError, match="dependencies"):
        model_store.plan_model(task, tmp_path)


def test_legacy_source_and_transport_context_do_not_change_pinned_identity(task, plan, tmp_path):
    path = _publish(tmp_path, model_store.entry_key(task), plan)
    original = path.read_bytes()
    task.requested_revision = "release-v1"
    with patch.object(model_store, "store_lock", side_effect=AssertionError("cache hit took write lock")), patch(
            "acprof.container.download_model._prepare_repository_plan", side_effect=AssertionError("cache hit requested metadata")):
        result = model_store.plan_model(task, tmp_path)
    assert "source" not in result
    assert result == json.loads(original)
    assert path.read_bytes() == original


def test_budget_and_receipt_use_the_verified_plan_selected_for_reuse(task, plan, tmp_path):
    def snapshot(**kwargs):
        path = tmp_path / "hf/models--example--model/snapshots" / task.model_revision
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_bytes(b"{}")
        return str(path)

    with patch("huggingface_hub.snapshot_download", side_effect=snapshot):
        original = model_store.prepare_model(task, copy.deepcopy(plan), tmp_path)
    candidate = copy.deepcopy(plan)
    candidate["files"].append({"path": "extra.bin", "size": 4})
    candidate["selected_bytes"] = 6
    seal_plan(candidate)
    with patch("huggingface_hub.snapshot_download", side_effect=snapshot) as download:
        reused = model_store.prepare_model(task, candidate, tmp_path, planned_download_bytes=0)
    assert reused == original
    download.assert_not_called()
