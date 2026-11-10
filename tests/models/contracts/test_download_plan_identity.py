"""Download planning source changes invalidate both cached weights and model layers."""

import pytest

import acprof.installation as installation
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import model_fingerprint
from acprof.host.model_store.store import entry_key
from tests.support.runtime_fixture import copy_dependency_tree


@pytest.fixture
def download_plan_context(tmp_path, monkeypatch):
    # Runtime profile selection reads its pinned dependency locks from this resource root.
    copy_dependency_tree(tmp_path)
    # Keep these fixture inputs explicit so omitting a production source stays detectable.
    files = {
        "acprof/models/download_plan.py": b"PLANNER_VERSION = 1\n",
        "acprof/models/files.py": b"CHECKPOINT_VERSION = 1\n",
        "acprof/models/spec.py": b"SPEC_VERSION = 1\n",
        "acprof/container/download_model.py": b"DOWNLOADER_VERSION = 1\n",
        "acprof/container/compat/transformers_model_types.json": b"{}\n",
        "acprof/hub/huggingface/endpoints.py": b"ENDPOINT_VERSION = 1\n",
        "dockerfiles/runtime-model.Dockerfile": b"FROM runtime\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    task = TaskInfo(
        model_id="example/bert", model_revision="a" * 40, pipeline_tag="fill-mask",
        task_family="nlp", runtime_backend="transformers_pipeline",
        library_name="transformers", detection_method="test",
    )
    monkeypatch.setattr(installation, "resource_root", lambda: tmp_path)
    return tmp_path, task


@pytest.mark.parametrize("relative", [
    "acprof/models/download_plan.py",
    "acprof/models/files.py",
    "acprof/models/spec.py",
], ids=["planner", "checkpoint-files", "model-spec"])
def test_download_plan_sources_invalidate_store_and_model_identity(download_plan_context, relative):
    root, task = download_plan_context
    runtime_id = "sha256:" + "b" * 64
    store_key = entry_key(task)
    image_key = model_fingerprint(task, runtime_id, root)
    assert entry_key(task) == store_key
    assert model_fingerprint(task, runtime_id, root) == image_key

    source = root / relative
    original = source.read_bytes()
    source.write_bytes(original + b"# changed planning behavior\n")
    assert entry_key(task) != store_key
    assert model_fingerprint(task, runtime_id, root) != image_key

    source.write_bytes(original)
    assert entry_key(task) == store_key
    assert model_fingerprint(task, runtime_id, root) == image_key
