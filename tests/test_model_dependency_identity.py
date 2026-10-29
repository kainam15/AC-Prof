"""Each offline loader repository must have one unambiguous pinned identity."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub import try_to_load_from_cache

from acprof.container.model_files import ModelFilesError, plan_download, seal_plan, validate_plan
from acprof.host import model_store
from acprof.host.detect import TaskInfo


def _repository(repo_id, revision, source="huggingface"):
    plan = plan_download(model_id=repo_id, revision=revision, family="nlp",
        backend="transformers_pipeline", policy="full", files={"config.json": {"size": 2}},
        read_json=lambda _: {})
    plan.update(source=source, endpoint="https://modelscope.cn" if source == "modelscope" else "https://huggingface.co")
    return seal_plan(plan)


def _plan(source="huggingface", *, primary_id="example/model", dependency_id="example/base", revision="b" * 40):
    primary = _repository(primary_id, "a" * 40, source)
    primary["dependencies"] = [{"repo_id": dependency_id, "revision": revision,
                                "download": _repository(dependency_id, revision)}]
    primary["total_selected_bytes"] = 4
    return seal_plan(primary)


def _task(plan):
    task = TaskInfo(model_id=plan["model_id"], model_revision=plan["model_revision"],
        pipeline_tag="feature-extraction", task_family="nlp", runtime_backend="transformers_pipeline",
        library_name="transformers", detection_method="manual", requested_revision=None)
    task.model_source = plan["source"]
    task.model_download_policy = "full"
    task.model_spec = {"schema_version": 1, "format": "transformers-pipeline",
        "task": "feature-extraction", "pipeline_task": "feature-extraction",
        "dependencies": [{k: v for k, v in item.items() if k != "download"} for item in plan["dependencies"]]}
    return task


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
@pytest.mark.parametrize("revision", ["a" * 40, "b" * 40])
def test_resealed_plan_cannot_redeclare_main_repository(source, revision):
    plan = _plan(source, dependency_id="example/model", revision=revision)
    with pytest.raises(ModelFilesError, match="offline loader"):
        validate_plan(plan)


@pytest.mark.parametrize("collides_with", ["main", "dependency"])
def test_distinct_ids_cannot_alias_the_same_loader_directory(collides_with):
    plan = _plan(primary_id="example-/model" if collides_with == "main" else "example/model",
                 dependency_id="example/-model")
    if collides_with == "dependency":
        plan["dependencies"].append({"repo_id": "example-/model", "revision": "c" * 40,
                                     "download": _repository("example-/model", "c" * 40)})
        plan["total_selected_bytes"] = 6
        seal_plan(plan)
    with pytest.raises(ModelFilesError, match="offline loader"):
        validate_plan(plan)


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
def test_planning_rejects_main_collision_before_metadata_or_cache_creation(tmp_path, source):
    plan = _plan(source, dependency_id="example/model")
    root = tmp_path / "store"
    with patch("acprof.container.download_model._prepare_repository_plan",
               side_effect=AssertionError("unexpected repository request")) as metadata:
        with pytest.raises(ValueError, match="offline loader"):
            model_store.plan_model(_task(plan), root)
    metadata.assert_not_called()
    assert not root.exists()


@pytest.mark.parametrize("invalid", ["main-collision", "dependency-source"])
def test_preparation_rejects_ambiguous_plan_before_payload_or_cache_creation(tmp_path, invalid):
    plan = _plan(dependency_id="example/model" if invalid == "main-collision" else "example/base")
    if invalid == "dependency-source":
        child = plan["dependencies"][0]["download"]
        child["source"] = "modelscope"
        seal_plan(child)
        seal_plan(plan)
    root = tmp_path / "store"
    with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected payload request")) as download:
        with pytest.raises(ModelFilesError, match="offline loader|dependency.*source"):
            model_store.prepare_model(_task(plan), plan, root)
    download.assert_not_called()
    assert not root.exists()


@pytest.mark.parametrize("invalid", ["main-collision", "dependency-source"])
def test_reading_resealed_verified_plan_rejects_conflict_without_rewriting(tmp_path, invalid):
    plan = _plan(dependency_id="example/model" if invalid == "main-collision" else "example/base")
    child = plan["dependencies"][0]["download"]
    if invalid == "dependency-source":
        child["source"] = "modelscope"
    for repo in (child, plan):
        repo["verification"] = "sha256"
        repo["files"][0]["sha256"] = hashlib.sha256(b"{}").hexdigest()
        seal_plan(repo)
    key = "a" * 64
    path = tmp_path / "entries" / key / "model_download_plan.json"
    path.parent.mkdir(parents=True)
    payload = json.dumps(plan).encode()
    path.write_bytes(payload)
    with pytest.raises(ModelFilesError, match="offline loader|dependency.*source"):
        model_store.read_entry(key, tmp_path)
    assert path.read_bytes() == payload


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
@pytest.mark.parametrize("legacy_dependency", [False, True])
@pytest.mark.parametrize("dependency_revision", ["a" * 40, "b" * 40])
def test_distinct_repositories_keep_independent_offline_refs_and_sources(tmp_path, source, legacy_dependency, dependency_revision):
    plan = _plan(source, revision=dependency_revision)
    if legacy_dependency:
        child = plan["dependencies"][0]["download"]
        child.pop("source")
        seal_plan(child)
        seal_plan(plan)
    original = copy.deepcopy(plan)
    validate_plan(plan)
    assert plan == original

    def snapshot(repo_id, revision, **kwargs):
        path = tmp_path / "hf" / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_bytes(b"{}" if repo_id == "example/model" else b"[]")
        return str(path)

    def modelscope_file(model_id, name, revision, root, **kwargs):
        path = root / "modelscope" / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
        return str(path)

    # ModelScope's payload path requires its expected digest in the input plan.
    if source == "modelscope":
        plan["files"][0]["lfs_sha256"] = hashlib.sha256(b"{}").hexdigest()
        seal_plan(plan)
    with patch("huggingface_hub.snapshot_download", side_effect=snapshot), patch(
            "acprof.model_repository.modelscope_file", side_effect=modelscope_file):
        record = model_store.prepare_model(_task(plan), plan, tmp_path)
    cache = tmp_path / "entries" / record["entry_id"] / "hf"
    for repo_id, revision, expected in (("example/model", "a" * 40, b"{}"), ("example/base", dependency_revision, b"[]")):
        path = try_to_load_from_cache(repo_id, "config.json", cache_dir=str(cache))
        assert isinstance(path, str)
        assert Path(path).parent.name == revision
        assert Path(path).read_bytes() == expected
    assert record["model_download"]["source"] == source
    assert record["model_download"]["dependencies"][0]["download"]["source"] == "huggingface"
    assert record["model_artifact_bytes"] == 4
    model_store.verify_entry({"model_store": record}, tmp_path)
