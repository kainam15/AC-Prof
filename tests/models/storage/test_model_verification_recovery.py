"""Failed preparation must leave its input plan usable for an explicit retry."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container.download_model import verify_download
from acprof.container.model_files import ModelFilesError, plan_download, seal_plan, validate_plan
from acprof.host.model_store import store as model_store

CONTENT = b"{}"
SHA256 = hashlib.sha256(CONTENT).hexdigest()


def repository_plan(model_id="example/retry", revision="a" * 40, *, files=None):
    plan = plan_download(model_id=model_id, revision=revision, family="nlp",
        backend="transformers_pipeline", policy="full",
        files=files or {"config.json": {"size": 2, "lfs_sha256": SHA256}}, read_json=lambda _: {})
    plan["endpoint"] = "https://huggingface.co"
    return seal_plan(plan)


@pytest.mark.parametrize("failure", ["missing", "size", "sha256", "lfs", "blob", "read", "cancel"])
def test_failed_repository_verification_preserves_plan_and_allows_retry(tmp_path, failure):
    files = {"a.json": {"size": None}, "z.json": {"size": 2}}
    if failure == "sha256":
        files["z.json"]["sha256"] = SHA256
    elif failure == "lfs":
        files["z.json"]["lfs_sha256"] = SHA256
    elif failure == "blob":
        files["z.json"]["blob_id"] = hashlib.sha1(b"blob 2\0" + CONTENT).hexdigest()
    plan = repository_plan(files=files)
    before = copy.deepcopy(plan)
    (tmp_path / "a.json").write_bytes(CONTENT)
    last = tmp_path / "z.json"
    if failure != "missing":
        last.write_bytes(b"x" if failure == "size" else b"xx" if failure in {"sha256", "lfs", "blob"} else CONTENT)
    error = KeyboardInterrupt("verification cancelled") if failure == "cancel" else OSError("payload read failed")
    open_path = Path.open

    def read(path, *args, **kwargs):
        if path == last and failure in {"read", "cancel"}:
            raise error
        return open_path(path, *args, **kwargs)

    expected = type(error) if failure in {"read", "cancel"} else ModelFilesError
    match = {"missing": "missing", "size": "size mismatch", "sha256": "SHA256 mismatch",
             "lfs": "SHA256 mismatch", "blob": "Git blob mismatch", "read": "payload read failed",
             "cancel": "verification cancelled"}[failure]
    with patch.object(Path, "open", read), pytest.raises(expected, match=match):
        verify_download(tmp_path, plan)
    assert plan == before
    validate_plan(plan)

    last.write_bytes(CONTENT)
    verified = verify_download(tmp_path, plan)
    validate_plan(verified)
    assert verified["verification"] == "sha256"
    assert verified["selected_bytes"] == 4
    assert [(item["size"], item["sha256"]) for item in verified["files"]] == [(2, SHA256), (2, SHA256)]


@pytest.fixture
def preparation(tmp_path):
    plan = repository_plan()
    dependency = repository_plan("example/base", "b" * 40)
    plan["dependencies"] = [{"repo_id": "example/base", "revision": "b" * 40, "download": dependency}]
    plan["total_selected_bytes"] = 4
    seal_plan(plan)
    task = SimpleNamespace(model_id=plan["model_id"], model_revision=plan["model_revision"],
        task_family=plan["task_family"], runtime_backend=plan["backend"], model_adapter="family-default",
        model_download_policy=plan["requested_policy"], model_resolution={},
        model_spec={"dependencies": [{"repo_id": "example/base", "revision": "b" * 40}]})
    snapshots = {}
    for repo in (plan, dependency):
        target = tmp_path / "hf" / ("models--" + repo["model_id"].replace("/", "--")) / "snapshots" / repo["model_revision"]
        target.mkdir(parents=True)
        (target / "config.json").write_bytes(CONTENT)
        snapshots[repo["model_id"]] = target
    return task, plan, tmp_path, snapshots


@pytest.mark.parametrize("stage", ["dependency_download", "dependency_verification", "publication"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_failed_preparation_preserves_input_and_retries_without_partial_verification(preparation, stage, error_type):
    task, plan, root, snapshots = preparation
    before = copy.deepcopy(plan)
    failure = error_type("preparation interrupted")
    open_path = Path.open

    def snapshot(**kwargs):
        if stage == "dependency_download" and kwargs["repo_id"] == "example/base":
            raise failure
        return str(snapshots[kwargs["repo_id"]])

    def read(path, *args, **kwargs):
        if stage == "dependency_verification" and path == snapshots["example/base"] / "config.json":
            raise failure
        return open_path(path, *args, **kwargs)

    publish = model_store.publish_view

    def publish_view(*args, **kwargs):
        if stage == "publication":
            raise failure
        return publish(*args, **kwargs)

    with patch("huggingface_hub.snapshot_download", side_effect=snapshot), patch.object(Path, "open", read), \
            patch.object(model_store, "publish_view", side_effect=publish_view):
        with pytest.raises(error_type, match="preparation interrupted") as caught:
            model_store.prepare_model(task, plan, root)
    assert caught.value is failure
    assert plan == before
    validate_plan(plan)
    assert model_store.read_entry(model_store.entry_key(task), root) is None
    assert all((target / "config.json").read_bytes() == CONTENT for target in snapshots.values())

    with patch("huggingface_hub.snapshot_download", side_effect=lambda **kw: str(snapshots[kw["repo_id"]])) as download:
        record = model_store.prepare_model(task, plan, root, planned_download_bytes=0)
    assert all(call.kwargs["local_files_only"] for call in download.call_args_list)
    assert plan == before
    assert model_store.read_entry(record["entry_id"], root) == record["model_download"]
    model_store.verify_entry({"model_store": record}, root)


def test_successful_preparation_returns_complete_independent_plan(preparation):
    task, plan, root, snapshots = preparation
    before = copy.deepcopy(plan)
    with patch("huggingface_hub.snapshot_download", side_effect=lambda **kw: str(snapshots[kw["repo_id"]])):
        record = model_store.prepare_model(task, plan, root)
    assert plan == before
    verified = record["model_download"]
    validate_plan(verified)
    assert verified["total_selected_bytes"] == 4
    for repo in (verified, verified["dependencies"][0]["download"]):
        assert repo["verification"] == "sha256"
        assert repo["files"][0]["sha256"] == SHA256
        assert repo["download_provenance"]["cache_hit"] is True
    verified["dependencies"][0]["download"]["files"][0]["size"] = 99
    assert plan == before
    assert model_store.read_entry(record["entry_id"], root)["total_selected_bytes"] == 4
