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


def test_inspect_tracks_the_configuration_read_by_each_sentence_module(repository, tmp_path, monkeypatch):
    from acprof.cli.main import main
    from acprof.models.evidence import content_digest

    metadata, downloaded, hub = repository
    modules = [module("words", "WordEmbeddings"), module("./conv/", "CNN", 1), module("memory", "LSTM", 2),
               module("1_Pooling", "Pooling", 3), module("normalize", "Normalize", 4)]
    modules[2]["type"] = "sentence_transformers.sentence_transformer.modules.lstm.LSTM"
    metadata["modules.json"] = modules
    configs = {
        "words/wordembedding_config.json": {
            "tokenizer_class": "sentence_transformers.sentence_transformer.modules.tokenizer.whitespace.WhitespaceTokenizer",
            "max_seq_length": 128,
        },
        "conv/cnn_config.json": {"in_embedding_dimension": 16, "out_channels": 16, "kernel_sizes": [3]},
        "memory/lstm_config.json": {"embedding_dimension": 16, "hidden_dim": 8},
    }
    unused = ["sentence_bert_config.json", "conv/config.json", "words/sentence_bert_config.json", "normalize/config.json"]
    metadata.update(configs)
    metadata.update(dict.fromkeys(unused, []))
    weights = [directory + "/model.safetensors" for directory in ("words", "conv", "memory")]
    hub.siblings = [SimpleNamespace(rfilename=name) for name in (*metadata, *weights)]
    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", lambda _cwd: None)

    def inspect(revision):
        hub.sha = revision
        output = tmp_path / revision
        assert main(["inspect", MODEL_ID, "--revision", revision, "--output-dir", str(output)]) == 0
        report = json.loads((output / "model_resolution.json").read_text())
        assert report["runtime_validation"]["status"] == "not_run"
        assert report["execution"]["backend"] == "sentence_transformers"
        return [item["configuration_sha256"] for item in report["execution"]["components"]]

    initial = inspect(REVISION)
    assert initial[:3] == [content_digest({name: data}) for name, data in configs.items()]
    assert all(downloaded.count(name) == 1 for name in configs)
    assert set(downloaded).isdisjoint(unused + weights)
    metadata["memory/lstm_config.json"] = {**configs["memory/lstm_config.json"], "num_layers": 2}
    changed = inspect("b" * 40)
    assert [i for i, (old, new) in enumerate(zip(initial, changed, strict=True)) if old != new] == [2]


def test_static_embedding_keeps_tokenizer_as_an_asset_and_dropout_reads_its_settings(repository):
    metadata, downloaded, hub = repository
    metadata["modules.json"] = [module("vectors", "StaticEmbedding"), module("regularize", "Dropout", 1)]
    settings = "regularize/config.json"
    metadata[settings] = {"dropout": 0.1}
    assets = ["vectors/tokenizer.json", "vectors/model.safetensors"]
    hub.siblings = [SimpleNamespace(rfilename=name) for name in (*metadata, *assets)]

    task = detect_task(MODEL_ID, revision=REVISION)
    require_task_support(task)
    assert task.repository_metadata[settings] == {"dropout": 0.1}
    assert set(downloaded).isdisjoint(assets)


def test_inspect_reports_missing_pooling_file_before_loading_weights(repository, tmp_path, monkeypatch):
    from acprof.cli.main import main

    metadata, downloaded, hub = repository
    metadata["modules.json"] = [module("0_Transformer"), module("pool", "Pooling", 1)]
    hub.siblings.append(SimpleNamespace(rfilename="0_Transformer/model.safetensors"))
    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", lambda _cwd: None)
    output = tmp_path / "inspection"

    assert main(["inspect", MODEL_ID, "--revision", REVISION, "--output-dir", str(output)]) == 2
    report = json.loads((output / "model_resolution.json").read_text())
    assert report["status"] == "blocked"
    assert report["failure"]["reason_code"] == "artifact_incomplete"
    assert "pool/config.json" in report["failure"]["detail"]
    assert report["runtime_validation"]["status"] == "not_run"
    assert "0_Transformer/model.safetensors" not in downloaded


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
