"""Cache lookups observe files once and tolerate concurrent cache removal."""
import errno
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container.download_model import verify_download
from acprof.host import detect
from acprof.host.model_store import store as model_store
from acprof.hub.repository import modelscope_snapshot
from acprof.models.download_plan import PLAN_FILENAME, plan_download, seal_plan
from acprof.models.files import ModelFilesError

MODEL_ID = "fixture/model"
REVISION = "a" * 40
ENTRY_ID = "b" * 64


def _write_verified_entry(root, entry_id, payloads, context, *, source="huggingface"):
    entry = root / "entries" / entry_id
    snapshot = entry / "hf/models--fixture--model/snapshots" / REVISION
    snapshot.mkdir(parents=True)
    for name, content in payloads.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    plan = plan_download(model_id=MODEL_ID, revision=REVISION, family="nlp",
        backend="transformers_pipeline", policy="full",
        files={name: {"size": len(content)} for name, content in payloads.items()}, read_json=lambda _: {})
    plan["endpoint"] = "https://huggingface.co"
    if source == "modelscope":
        plan.update(source=source, endpoint="https://modelscope.cn")
    plan["requested_revision"] = None
    plan["repository_context"] = context
    # Seal the new context before verifying and publishing the synthetic entry.
    seal_plan(plan)
    verify_download(snapshot, plan)
    (entry / PLAN_FILENAME).write_text(json.dumps(plan))
    return SimpleNamespace(root=root, entry=entry, snapshot=snapshot, plan=plan, payloads=payloads)


@pytest.fixture
def cached_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path))
    payloads = {"config.json": b'{"model_type":"bert"}',
                **{f"model-{index:05d}.safetensors": b"fixture" for index in range(128)}}
    context = {"schema_version": 1, "repository_files": list(payloads),
        "pipeline_tag": "feature-extraction", "library_name": "transformers", "config": {"model_type": "bert"},
        "transformers_info": {}, "card_data": {}, "tags": [], "safetensors": None}
    return _write_verified_entry(tmp_path, ENTRY_ID, payloads, context)


@pytest.mark.parametrize(("architecture", "declares_task", "status", "selected"), [
    ("BertForMaskedLM", False, "candidate", "fill-mask"),
    ("CustomEncoderModel", False, "needs_configuration", None),
    ("CustomEncoderModel", True, "candidate", "tabular-classification"),
])
def test_old_modelscope_context_uses_only_pinned_task_evidence(
    tmp_path, monkeypatch, architecture, declares_task, status, selected,
):
    from acprof.cli.inspect import main

    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path))
    config = json.dumps({"model_type": "bert", "architectures": [architecture]}).encode()
    payloads = {"config.json": config, "model.safetensors": b"x"}
    if declares_task:
        payloads["iris.onnx"] = b"fixture"
        payloads["acprof_model.json"] = json.dumps({"schema_version": 1, "format": "onnxruntime",
            "task": "tabular-classification", "model_file": "iris.onnx", "feature_dim": 4}).encode()
    context = {"schema_version": 1, "repository_files": list(payloads), "pipeline_tag": "text-classification",
               "library_name": "transformers", "config": {}, "transformers_info": {}, "tags": [],
               "card_data": {}, "safetensors": None}
    entry = _write_verified_entry(tmp_path, ENTRY_ID, payloads, context, source="modelscope")
    manifest = entry.entry / PLAN_FILENAME
    original = manifest.read_bytes()

    args = [MODEL_ID, "--model-source", "modelscope", "--revision", REVISION]
    with patch("acprof.hub.repository.modelscope_api") as api:
        output = tmp_path / "inspection"
        code = main([*args, "--output-dir", str(output)])
        report = json.loads((output / "model_resolution.json").read_text())
        assert (code, report["status"]) == (0 if selected else 2, status)
        if selected:
            assert report["task"] == selected
        raw = report["provenance"]["raw_evidence"]
        assert raw["metadata"]["config.json"] == json.loads(config)
        assert raw["hub"]["pipeline_tag"] is None
        assert report["model_revision"] == REVISION
        if declares_task:
            assert report["contract"]["fields"]["library"] == {
                "value": "transformers", "state": "derived", "sources": ["modelscope.default_library"]}
        if not selected:
            assert main([*args, "--task", "fill-mask"]) == 0
        api.assert_not_called()
    assert manifest.read_bytes() == original


@pytest.mark.parametrize(("layout", "stale_library", "task", "backend"), [
    ("bert", "diffusers", "fill-mask", "transformers_pipeline"),
    ("unrelated-index", "diffusers", "fill-mask", "transformers_pipeline"),
    ("ddpm", "transformers", "unconditional-image-generation", "diffusers"),
])
def test_modelscope_library_of_cached_snapshot(tmp_path, monkeypatch, layout, stale_library, task, backend):
    from acprof.cli.inspect import main

    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path))
    payloads = {"config.json": b'{"model_type":"bert","architectures":["BertForMaskedLM"]}', "model.safetensors": b"x"}
    if layout == "unrelated-index":
        payloads["model_index.json"] = b'{"evaluation_results":[]}'
    if layout == "ddpm":
        payloads = {"model_index.json": b'{"_class_name":"DDPMPipeline","unet":["diffusers","UNet2DModel"]}',
                    "unet/config.json": b"{}", "unet/diffusion_pytorch_model.safetensors": b"x"}
    tags = ["library:diffusers"] if stale_library == "diffusers" else []
    context = {"schema_version": 1, "repository_files": list(payloads), "pipeline_tag": None,
               "library_name": stale_library, "config": {}, "transformers_info": {}, "tags": tags,
               "card_data": {}, "safetensors": None}
    saved = _write_verified_entry(tmp_path, ENTRY_ID, payloads, context, source="modelscope")
    manifest = saved.entry / PLAN_FILENAME
    original = manifest.read_bytes()
    with patch("acprof.hub.repository.modelscope_api") as sdk:
        for explicit in (False, True):
            out = tmp_path / ("explicit" if explicit else "auto")
            args = [MODEL_ID, "--model-source", "modelscope", "--revision", REVISION, "--output-dir", str(out)]
            assert main(args + (["--task", task] if explicit else [])) == 0
            result = json.loads((out / "model_resolution.json").read_text())
            assert (result["task"], result["backend"]) == (task, backend)
            assert result["library"] == ("diffusers" if layout == "ddpm" else "transformers")
            origin = "model_index.json" if layout == "ddpm" else "modelscope.default_library"
            assert result["provenance"]["raw_evidence"]["hub"]["library_source"] == origin
            assert result["model_revision"] == REVISION
        sdk.assert_not_called()
    assert manifest.read_bytes() == original


def _count_payload_stats(monkeypatch, paths):
    calls = Counter()
    original = Path.stat

    def count(path, *args, **kwargs):
        if path in paths:
            calls[path] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", count)
    return calls


def test_cached_inventory_observes_each_payload_once(cached_entry, monkeypatch):
    paths = {cached_entry.snapshot / name for name in cached_entry.payloads}
    calls = _count_payload_stats(monkeypatch, paths)
    info = model_store.cached_model_info(MODEL_ID, REVISION, source="huggingface")
    assert info.sha == REVISION
    assert {item.rfilename for item in info.siblings} == set(cached_entry.payloads)
    assert calls == Counter({path: 1 for path in paths})


@pytest.mark.parametrize("legacy", [False, True], ids=["with-inventory", "legacy"])
def test_detection_checks_only_entries_that_can_answer_the_lookup(cached_entry, monkeypatch, legacy):
    # A filtered dependency and an older entry hold different files at the same commit.
    optional = {"tokenizer_config.json": b'{"tokenizer_class":"BertTokenizer"}',
                "processor_config.json": b"{}"}
    context = cached_entry.plan["repository_context"]
    context["repository_files"].extend(optional)
    if legacy:
        del cached_entry.plan["repository_context"]
    (cached_entry.entry / PLAN_FILENAME).write_text(json.dumps(seal_plan(cached_entry.plan)))

    older = _write_verified_entry(cached_entry.root, "c" * 64, optional, context)
    os.utime(older.entry, (1, 1))
    os.utime(cached_entry.entry, (2, 2))

    newest_paths = {cached_entry.snapshot / name for name in cached_entry.payloads}
    older_paths = {older.snapshot / name for name in optional}
    calls = _count_payload_stats(monkeypatch, newest_paths | older_paths)
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "huggingface")
    with patch("huggingface_hub.HfApi.model_info") as api, patch("huggingface_hub.hf_hub_download") as download:
        task = detect.detect_task(MODEL_ID)
    api.assert_not_called()
    download.assert_not_called()
    assert task.model_revision == REVISION
    assert not task.metadata_errors
    assert task.repository_metadata == {
        "config.json": {"model_type": "bert"},
        **{name: json.loads(content) for name, content in optional.items()},
    }
    # One scan per served file or inventory, including inventory fallback to the older entry.
    newest_scans, older_scans = (1, 3) if legacy else (2, 2)
    assert calls.total() <= len(newest_paths) * newest_scans + len(older_paths) * older_scans


def test_cached_lookups_require_complete_payload_even_when_config_matches(cached_entry):
    (cached_entry.snapshot / "model-00127.safetensors").unlink()
    assert model_store.cached_metadata(MODEL_ID, "config.json", REVISION, source="huggingface") is None
    assert model_store.cached_model_info(MODEL_ID, REVISION, source="huggingface") is None


def test_prune_between_metadata_lookup_and_open_is_a_cache_miss(cached_entry, monkeypatch):
    original = model_store.read_metadata_file

    def prune_before_open(path, **kwargs):
        cancel = Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model_store.prune_store, root=cached_entry.root,
                                     apply=True, cancel=cancel, on_wait=cancel.set)
            try:
                result = future.result(timeout=5)
            finally:
                cancel.set()
        assert result["entries"] == [ENTRY_ID]
        assert not cached_entry.entry.exists()
        return original(path, **kwargs)

    monkeypatch.setattr(model_store, "read_metadata_file", prune_before_open)
    assert model_store.cached_metadata(MODEL_ID, "config.json", REVISION, source="huggingface") is None


def test_entry_directory_removed_before_enumeration_is_a_cache_miss(cached_entry, monkeypatch):
    entries = cached_entry.root / "entries"
    original = Path.iterdir

    def remove_before_list(path):
        if path == entries:
            shutil.rmtree(entries)
        return original(path)

    monkeypatch.setattr(Path, "iterdir", remove_before_list)
    assert model_store.cached_model_info(MODEL_ID, REVISION, source="huggingface") is None


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
def test_source_snapshot_observes_payload_once(tmp_path, monkeypatch, source):
    snapshot = (modelscope_snapshot(tmp_path, MODEL_ID, REVISION) if source == "modelscope"
                else tmp_path / "hf/models--fixture--model/snapshots" / REVISION)
    snapshot.mkdir(parents=True)
    path = snapshot / "config.json"
    path.write_bytes(b"{}")
    calls = []
    original = Path.stat

    def count(candidate, *args, **kwargs):
        if candidate == path:
            calls.append(candidate)
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", count)
    with patch("huggingface_hub.try_to_load_from_cache", return_value=str(path)):
        result = model_store.cached_file(tmp_path,
            {"source": source, "model_id": MODEL_ID, "model_revision": REVISION},
            {"path": "config.json", "size": 2})
    assert result is not None
    assert (result.path, result.size) == (path, 2)
    assert len(calls) == 1


def test_model_source_reuses_file_size_if_gc_removes_the_observed_path(tmp_path, monkeypatch):
    snapshot = modelscope_snapshot(tmp_path, MODEL_ID, REVISION)
    snapshot.mkdir(parents=True)
    path = snapshot / "config.json"
    path.write_bytes(b"{}")
    plan = {
        "source": "modelscope",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "endpoint": "https://modelscope.cn",
        "selected_bytes": 2,
        "files": [{"path": "config.json", "size": 2}],
    }
    cached_file = model_store.cached_file

    def remove_after_observation(*args, **kwargs):
        observed = cached_file(*args, **kwargs)
        if observed is not None:
            observed.path.unlink()
        return observed

    monkeypatch.setattr(model_store, "cached_file", remove_after_observation)

    sources = model_store.model_sources(plan, tmp_path)

    assert [(source.cached_bytes, source.estimated_bytes) for source in sources] == [(2, 0)]
    assert not path.exists()


def test_metadata_can_use_another_matching_entry_after_removal(cached_entry, monkeypatch):
    second = cached_entry.root / "entries" / ("c" * 64)
    shutil.copytree(cached_entry.entry, second)
    os.utime(second, (1, 1))
    os.utime(cached_entry.entry, (2, 2))
    original = model_store.read_metadata_file

    def remove_first(path, **kwargs):
        if path.is_relative_to(cached_entry.entry):
            path.unlink()
        return original(path, **kwargs)

    monkeypatch.setattr(model_store, "read_metadata_file", remove_first)
    metadata = model_store.cached_metadata(MODEL_ID, "config.json", REVISION, source="huggingface")
    assert metadata.content == cached_entry.payloads["config.json"]
    assert metadata.path.is_relative_to(second)


@pytest.mark.parametrize("failure", [PermissionError("fixture permission"), OSError(errno.EIO, "fixture I/O"),
                                     ModelFilesError("fixture SHA256 mismatch")])
def test_metadata_errors_are_not_hidden_by_another_entry_or_network(cached_entry, monkeypatch, failure):
    second = cached_entry.root / "entries" / ("c" * 64)
    shutil.copytree(cached_entry.entry, second)
    os.utime(second, (1, 1))
    os.utime(cached_entry.entry, (2, 2))

    def fail_first(path, **kwargs):
        assert path.is_relative_to(cached_entry.entry)
        raise failure

    monkeypatch.setattr(model_store, "read_metadata_file", fail_first)
    with patch("huggingface_hub.hf_hub_download") as download, pytest.raises(type(failure)) as caught:
        detect._download_metadata(MODEL_ID, "config.json", REVISION, source="huggingface")
    assert caught.value is failure
    download.assert_not_called()


@pytest.mark.parametrize("phase", ["enumerate", "stat"])
def test_inventory_permission_errors_are_not_cache_misses(cached_entry, monkeypatch, phase):
    method = "iterdir" if phase == "enumerate" else "stat"
    target = cached_entry.root / "entries" if phase == "enumerate" else cached_entry.snapshot / "config.json"
    original = getattr(Path, method)

    def inaccessible(path, *args, **kwargs):
        if path == target:
            raise PermissionError("fixture permission")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, method, inaccessible)
    with pytest.raises(PermissionError, match="fixture permission"):
        model_store.cached_model_info(MODEL_ID, REVISION, source="huggingface")


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
@pytest.mark.parametrize("state", ["missing", "wrong_size", "directory", "fifo", "broken_link", "unknown_size"])
def test_source_cache_requires_a_regular_file_and_expected_size(tmp_path, source, state):
    snapshot = (modelscope_snapshot(tmp_path, MODEL_ID, REVISION) if source == "modelscope"
                else tmp_path / "hf/models--fixture--model/snapshots" / REVISION)
    snapshot.mkdir(parents=True)
    path = snapshot / "config.json"
    if state in {"wrong_size", "unknown_size"}:
        path.write_bytes(b"{}")
    elif state == "directory":
        path.mkdir()
    elif state == "fifo":
        os.mkfifo(path)
    elif state == "broken_link":
        path.symlink_to(snapshot / "absent")
    size = None if state == "unknown_size" else 3
    with patch("huggingface_hub.try_to_load_from_cache", return_value=str(path)):
        result = model_store.cached_file(tmp_path,
            {"source": source, "model_id": MODEL_ID, "model_revision": REVISION},
            {"path": "config.json", "size": size})
    expected = path if state == "unknown_size" and source == "huggingface" else None
    assert (result.path if result is not None else None) == expected
