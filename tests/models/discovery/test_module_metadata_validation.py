"""Invalid module declarations never become evidence or expand metadata reads."""
import json
from types import SimpleNamespace

import pytest

from acprof.host.detect import detect_task
from acprof.host.preparation.task_support import TaskSupportError, require_task_support

REVISION = "a" * 40
MODEL_ID = "fixture/module-validation"
CONFIG = {"model_type": "bert", "architectures": ["BertModel"]}


@pytest.fixture
def repository(tmp_path, monkeypatch):
    metadata = {
        "config.json": CONFIG,
        "tokenizer_config.json": {"model_max_length": 128},
        "modules.json": [],
        "0_Transformer/config.json": CONFIG,
        "0_Transformer/sentence_bert_config.json": {"max_seq_length": 128},
        "1_Pooling/config.json": {"word_embedding_dimension": 16, "pooling_mode_mean_tokens": True},
    }
    downloaded = []
    hub = SimpleNamespace(
        pipeline_tag="feature-extraction", library_name="sentence-transformers",
        sha=REVISION, config={},
        siblings=[SimpleNamespace(rfilename=name) for name in metadata],
    )

    def download(*, repo_id, filename, revision, dry_run=False, **_kwargs):
        assert repo_id == MODEL_ID
        assert revision == hub.sha
        path = tmp_path / filename
        content = json.dumps(metadata[filename]).encode("utf-8")
        if dry_run:
            return SimpleNamespace(commit_hash=hub.sha, file_size=len(content))
        downloaded.append(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *_args, **_kwargs: hub)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return metadata, downloaded, hub


def module(path, kind="Transformer", index=0):
    return {"idx": index, "name": str(index), "path": path, "type": f"sentence_transformers.models.{kind}"}


@pytest.mark.parametrize("invalid_module", [
    None, 7, {}, {"path": None}, {"path": "/outside"},
    {"path": "../outside"}, {"path": "nested/../outside"}, {"path": "folder\\name"},
], ids=["null", "scalar", "missing-path", "null-path", "absolute", "parent", "nested-parent", "backslash"])
def test_invalid_later_module_does_not_publish_or_expand_metadata(repository, invalid_module):
    metadata, downloaded, _hub = repository
    metadata["modules.json"] = [module("0_Transformer"), invalid_module]

    task = detect_task(MODEL_ID, revision=REVISION)

    assert downloaded == ["config.json", "modules.json", "tokenizer_config.json"]
    assert task.repository_metadata == {
        "config.json": CONFIG,
        "tokenizer_config.json": metadata["tokenizer_config.json"],
    }
    assert len(task.metadata_errors) == 1
    assert "modules.json" in task.metadata_errors[0]
    assert "module path must stay inside the model snapshot" in task.metadata_errors[0]
    with pytest.raises(TaskSupportError, match="module path must stay inside the model snapshot"):
        require_task_support(task)


def test_oversized_module_list_is_not_published_as_model_evidence(repository):
    metadata, downloaded, _hub = repository
    metadata["modules.json"] = [module("0_Transformer"), module("1_Pooling", "Pooling", 1),
                                *[module("", "Normalize", index) for index in range(2, 65)]]

    task = detect_task(MODEL_ID, revision=REVISION)

    assert "modules.json" not in task.repository_metadata
    assert downloaded == ["config.json", "modules.json", "tokenizer_config.json"]
    assert len(task.metadata_errors) == 1
    with pytest.raises(TaskSupportError, match="modules.json exceeds 64 modules"):
        require_task_support(task)


@pytest.mark.parametrize("root", ["", ".", "0_Transformer"])
def test_valid_modules_preserve_root_paths_and_deduplicate_metadata_reads(repository, root):
    metadata, downloaded, _hub = repository
    metadata["modules.json"] = [module(root), module("1_Pooling", "Pooling", 1), module(root, "Normalize", 2)]

    task = detect_task(MODEL_ID, revision=REVISION)

    assert task.metadata_errors == ()
    assert task.repository_metadata["modules.json"] == metadata["modules.json"]
    expected = ["config.json", "modules.json", "tokenizer_config.json"]
    if root == "0_Transformer":
        expected += ["0_Transformer/config.json", "0_Transformer/sentence_bert_config.json"]
        assert task.repository_metadata[expected[-2]] == CONFIG
        assert task.repository_metadata[expected[-1]] == {"max_seq_length": 128}
    expected.append("1_Pooling/config.json")
    assert downloaded == expected
    require_task_support(task)


def test_module_count_boundary_and_corrected_retry_remain_supported(repository):
    metadata, downloaded, hub = repository
    metadata["modules.json"] = [module("0_Transformer"), module("1_Pooling", "Pooling", 1),
                                *[module("", "Normalize", index) for index in range(2, 65)]]
    rejected = detect_task(MODEL_ID, revision=REVISION)
    assert rejected.metadata_errors

    metadata["modules.json"] = [module("0_Transformer"), module("1_Pooling", "Pooling", 1),
                                *[module("", "Normalize", index) for index in range(2, 64)]]
    hub.sha = "b" * 40
    downloaded.clear()
    corrected = detect_task(MODEL_ID, revision=hub.sha)

    assert corrected.model_revision == hub.sha
    assert corrected.metadata_errors == ()
    assert len(corrected.repository_metadata["modules.json"]) == 64
    assert downloaded.count("0_Transformer/config.json") == 1
    require_task_support(corrected)
