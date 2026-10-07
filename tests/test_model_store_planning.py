"""Planning metadata remains owned until it has been consumed, even during GC."""
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from acprof.container.model_files import ModelFilesError
from acprof.host import model_store
from acprof.host.detect import TaskInfo

_MAIN_ID = "example/model"
_DEPENDENCY_ID = "example/dependency"
_MAIN_REVISION = "a" * 40
_DEPENDENCY_REVISION = "b" * 40
_FILES = {
    _MAIN_ID: {
        "config.json": b'{"model_type":"bert"}',
        "model.safetensors": b"weights",
    },
    _DEPENDENCY_ID: {
        "model.safetensors.index.json": b'{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}',
        "model-00001-of-00001.safetensors": b"dependency",
    },
}


def _planning_task(source, *, with_dependency=False):
    task = TaskInfo(model_id=_MAIN_ID, model_revision=_MAIN_REVISION,
        pipeline_tag="feature-extraction", task_family="nlp",
        runtime_backend="transformers_pipeline", library_name="transformers",
        detection_method="manual", requested_revision=None)
    task.model_source = source
    task.model_download_policy = "auto"
    if with_dependency:
        task.model_spec = {
            "schema_version": 1, "format": "transformers-pipeline",
            "task": "feature-extraction", "pipeline_task": "feature-extraction",
            "dependencies": [{"repo_id": _DEPENDENCY_ID, "revision": _DEPENDENCY_REVISION}],
        }
    return task


def _prune_in_another_thread(root):
    # A cancellation callback gives lock contention a bounded, observable outcome.
    # There are no scheduler sleeps and no synchronous recursive flock acquisition.
    cancel = Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model_store.prune_store, root=root, apply=True,
            cancel=cancel, on_wait=cancel.set)
        try:
            return future.result(timeout=5)
        finally:
            cancel.set()


def _install_metadata_downloads(monkeypatch, root, on_download):
    written = []

    def information(repo_id):
        return SimpleNamespace(
            sha=_MAIN_REVISION if repo_id == _MAIN_ID else _DEPENDENCY_REVISION,
            siblings=[SimpleNamespace(rfilename=name, size=len(data),
                lfs={"sha256": hashlib.sha256(data).hexdigest()})
                for name, data in _FILES[repo_id].items()],
        )

    def download(repo_id, name, revision, source):
        data = _FILES[repo_id][name]
        assert revision == information(repo_id).sha
        repository = "models--" + repo_id.replace("/", "--")
        digest = hashlib.sha256(data).hexdigest()
        if source == "huggingface":
            target = root / "hf" / repository / "blobs" / digest
        else:
            target = root / "modelscope" / repository / "snapshots" / revision / name
        partial = target.with_name(target.name + ".incomplete")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(data)
        written.append((partial, data))
        on_download(repo_id, "partial", partial, data)
        partial.replace(target)
        written[-1] = (target, data)
        on_download(repo_id, "complete", target, data)
        if source == "huggingface":
            path = root / "hf" / repository / "snapshots" / revision / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(os.path.relpath(target, path.parent))
            return str(path)
        return str(target)

    def hf_file(repo_id, name, *, revision, cache_dir, endpoint):
        assert Path(cache_dir) == root / "hf"
        return download(repo_id, name, revision, "huggingface")

    def modelscope_file(repo_id, name, revision, store_root, *, sha256):
        assert store_root == root
        assert sha256 == hashlib.sha256(_FILES[repo_id][name]).hexdigest()
        return download(repo_id, name, revision, "modelscope")

    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    monkeypatch.setenv("HF_FALLBACK_ENDPOINTS", "")
    monkeypatch.setattr("huggingface_hub.HfApi.model_info",
        lambda self, repo_id, **kwargs: information(repo_id))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_file)
    monkeypatch.setattr("acprof.model_repository.modelscope_info",
        lambda repo_id, revision: information(repo_id))
    monkeypatch.setattr("acprof.model_repository.modelscope_file", modelscope_file)
    return written


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
@pytest.mark.parametrize("repository", ["main", "dependency"])
def test_planning_metadata_survives_concurrent_prune(tmp_path, monkeypatch, source, repository):
    task = _planning_task(source, with_dependency=repository == "dependency")
    protected_id = _MAIN_ID if repository == "main" else _DEPENDENCY_ID
    protected_phases = []

    def while_downloading(repo_id, phase, path, data):
        if repo_id != protected_id:
            return
        try:
            _prune_in_another_thread(tmp_path)
        except model_store.ModelStoreCancelled:
            pass
        assert path.is_file(), f"GC removed active {phase} metadata: {path}"
        assert path.read_bytes() == data
        protected_phases.append(phase)

    written = _install_metadata_downloads(monkeypatch, tmp_path, while_downloading)
    plan = model_store.plan_model(task, tmp_path)

    assert protected_phases == ["partial", "complete"]
    assert plan["source"] == source
    assert plan["model_revision"] == _MAIN_REVISION
    if repository == "dependency":
        dependency = plan["dependencies"][0]["download"]
        assert dependency["model_revision"] == _DEPENDENCY_REVISION
        assert dependency["source"] == "huggingface"
    # Planning does not pin an unused snapshot after consuming its metadata.
    result = _prune_in_another_thread(tmp_path)
    assert result["reclaimable_bytes"] == sum(len(data) for _, data in written)
    assert all(not path.exists() for path, _ in written)


@pytest.mark.parametrize("error_type", [ModelFilesError, KeyboardInterrupt])
def test_planning_failure_releases_metadata_for_prune(tmp_path, monkeypatch, error_type):
    task = _planning_task("huggingface")

    def interrupt_download(repo_id, phase, path, data):
        assert phase == "partial"
        raise error_type("metadata planning interrupted")

    written = _install_metadata_downloads(monkeypatch, tmp_path, interrupt_download)
    with pytest.raises(error_type, match="metadata planning interrupted"):
        model_store.plan_model(task, tmp_path)

    result = _prune_in_another_thread(tmp_path)
    assert result["reclaimable_bytes"] == len(_FILES[_MAIN_ID]["config.json"])
    assert all(not path.exists() for path, _ in written)
