"""Invalid metadata is diagnosed at the read boundary and never normalized away."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.detect import _read_json_metadata, _repository_metadata, detect_task
from acprof.hub.repository import MetadataFile
from acprof.models.metadata_validation import MetadataValidationError

REVISION = "a" * 40


def _file(raw: bytes, name: str = "config.json") -> MetadataFile:
    return MetadataFile(Path("/fixture/snapshots") / REVISION / name, raw)


@pytest.mark.parametrize("number", ["Infinity", "-Infinity", "NaN", "1e999"])
def test_nonfinite_metadata_reports_pointer_and_original_bytes_digest(number):
    raw = ('{"nested":{"a/b~c":[' + number + ']}}').encode()
    metadata = _file(raw)
    with pytest.raises(MetadataValidationError) as caught:
        _read_json_metadata(metadata)
    diagnostic = caught.value.diagnostics[0]
    assert diagnostic["file"] == "config.json"
    assert diagnostic["json_pointer"] == "/nested/a~1b~0c/0"
    assert diagnostic["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert diagnostic["reason_code"] == "metadata_invalid"
    assert metadata.content == raw


def test_invalid_json_keeps_source_digest():
    raw = b'{"model_type": "bert",}'
    with pytest.raises(MetadataValidationError) as caught:
        _read_json_metadata(_file(raw))
    assert caught.value.diagnostics[0]["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert caught.value.diagnostics[0]["file"] == "config.json"


def test_excessive_metadata_nesting_is_a_structured_failure():
    raw = b"[" * 5000 + b"0" + b"]" * 5000
    with pytest.raises(MetadataValidationError, match="nesting") as caught:
        _read_json_metadata(_file(raw))
    diagnostic = caught.value.diagnostics[0]
    assert diagnostic["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert diagnostic["file"] == "config.json"
    assert diagnostic["reason_code"] == "metadata_invalid"


def test_detect_task_returns_blocked_metadata_result_without_fallback():
    raw = b'{"model_type":"bert","architectures":["BertForMaskedLM"],"training":{"limit":Infinity}}'
    hub = SimpleNamespace(
        pipeline_tag="fill-mask", library_name="transformers", sha=REVISION,
        config={"model_type": "bert"}, tags=[],
        siblings=[SimpleNamespace(rfilename=name) for name in ("config.json", "model.safetensors")],
    )
    with patch("acprof.host.model_store.store.cached_model_info", return_value=None), patch(
        "huggingface_hub.HfApi.model_info", return_value=hub,
    ), patch("acprof.host.detect._download_metadata", return_value=_file(raw)), patch(
        "acprof.host.detect._detect_from_config", side_effect=AssertionError("invalid metadata cannot authorize fallback"),
    ), patch(
        "acprof.models.candidates.discover_model_candidates", side_effect=AssertionError("invalid metadata must not be hashed"),
    ):
        task = detect_task("fixture/nonfinite", override_tag="fill-mask")
    assert task.model_resolution["status"] == "blocked"
    assert task.model_resolution["failure"]["reason_code"] == "metadata_invalid"
    assert task.metadata_diagnostics[0]["json_pointer"] == "/training/limit"
    assert task.metadata_sources["config.json"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "config.json" not in task.repository_metadata
    json.dumps(asdict(task), allow_nan=False)


def test_invalid_fallback_config_returns_blocked_result():
    raw = b'{"model_type":"bert","architectures":["BertForMaskedLM"],"limit":Infinity}'
    with patch("acprof.host.detect._detect_from_hub", return_value=None), patch(
        "acprof.host.detect._download_metadata", return_value=_file(raw),
    ):
        task = detect_task("fixture/nonfinite", override_tag="fill-mask")
    assert task.model_resolution["status"] == "blocked"
    assert task.model_resolution["failure"]["reason_code"] == "metadata_invalid"
    assert task.metadata_diagnostics[0]["json_pointer"] == "/limit"


@pytest.mark.parametrize(("declaration", "pointer"), [
    ({"unet": [{}, "UNet2DModel"]}, "/unet/0"),
    ({"unet": [[], "UNet2DModel"]}, "/unet/0"),
    ({"unet": ["diffusers", {}]}, "/unet/1"),
    ({"unet": ["diffusers", []]}, "/unet/1"),
    ({"_class_name": {}, "unet": ["diffusers", "UNet2DModel"]}, "/_class_name"),
    ({"_class_name": [], "unet": ["diffusers", "UNet2DModel"]}, "/_class_name"),
])
def test_invalid_diffusers_declaration_blocks_before_candidate_discovery(declaration, pointer):
    raw = json.dumps({"_class_name": "DDPMPipeline", **declaration}).encode()
    hub = SimpleNamespace(
        pipeline_tag="text-to-image", library_name="diffusers", sha=REVISION,
        config={}, tags=[], siblings=[SimpleNamespace(rfilename="model_index.json")],
    )
    with patch("acprof.host.model_store.store.cached_model_info", return_value=None), patch(
        "huggingface_hub.HfApi.model_info", return_value=hub,
    ), patch("acprof.host.detect._download_metadata", return_value=_file(raw, "model_index.json")), patch(
        "acprof.host.detect._detect_from_config", side_effect=AssertionError("invalid metadata cannot authorize fallback"),
    ), patch(
        "acprof.models.candidates.discover_model_candidates",
        side_effect=AssertionError("invalid metadata reached candidate discovery"),
    ) as discover:
        task = detect_task("fixture/invalid-diffusers", revision=REVISION)
    assert task.model_resolution["status"] == "blocked"
    assert task.model_resolution["failure"]["reason_code"] == "metadata_invalid"
    assert task.model_resolution["candidates"] == []
    diagnostic = task.metadata_diagnostics[0]
    assert diagnostic["file"] == "model_index.json"
    assert diagnostic["json_pointer"] == pointer
    assert diagnostic["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "model_index.json" not in task.repository_metadata
    discover.assert_not_called()


def test_repository_metadata_retains_raw_hash_and_reads_checkpoint_index():
    raw = b'{  "model_type" : "bert" }\n'
    index = b'{"weight_map":{"layer":"model-00001-of-00001.safetensors"}}'
    hub = SimpleNamespace(siblings=[SimpleNamespace(rfilename=name) for name in (
        "config.json", "model.safetensors.index.json", "model-00001-of-00001.safetensors",
    )])
    documents = {"config.json": _file(raw), "model.safetensors.index.json": _file(index, "model.safetensors.index.json")}
    with patch("acprof.host.detect._download_metadata", side_effect=lambda _, name, *args, **kwargs: documents[name]):
        result = _repository_metadata("fixture/model", REVISION, hub)
    assert result["repository_metadata"]["model.safetensors.index.json"]["weight_map"] == {
        "layer": "model-00001-of-00001.safetensors",
    }
    assert result["metadata_sources"]["config.json"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["metadata_sources"]["config.json"]["encoding"] == "raw"
    assert not result["metadata_diagnostics"]


def test_unused_weight_index_does_not_override_preferred_complete_weights():
    raw = b'{"model_type":"bert"}'
    hub = SimpleNamespace(siblings=[SimpleNamespace(rfilename=name) for name in (
        "config.json", "model.safetensors", "pytorch_model.bin.index.json",
    )])
    with patch("acprof.host.detect._download_metadata", return_value=_file(raw)) as download:
        result = _repository_metadata("fixture/model", REVISION, hub)
    assert [entry.args[1] for entry in download.call_args_list] == ["config.json"]
    assert not result["metadata_errors"]


def test_index_discovery_preserves_root_and_sentence_transformer_candidates():
    documents = {
        "config.json": _file(b'{"model_type":"bert"}'),
        "modules.json": _file(b'[{"path":"encoder","type":"sentence_transformers.models.Transformer"}]', "modules.json"),
        "encoder/config.json": _file(b'{"model_type":"bert"}', "encoder/config.json"),
        "model.safetensors.index.json": _file(
            b'{"weight_map":{"layer":"model-00001-of-00001.safetensors"}}', "model.safetensors.index.json",
        ),
        "encoder/model.safetensors.index.json": _file(
            b'{"weight_map":{"layer":"model-00001-of-00001.safetensors"}}', "encoder/model.safetensors.index.json",
        ),
    }
    files = [*documents, "model-00001-of-00001.safetensors", "encoder/model-00001-of-00001.safetensors"]
    hub = SimpleNamespace(siblings=[SimpleNamespace(rfilename=name) for name in files])
    with patch("acprof.host.detect._download_metadata", side_effect=lambda _, name, *args, **kwargs: documents[name]):
        result = _repository_metadata("fixture/model", REVISION, hub)
    assert "model.safetensors.index.json" in result["repository_metadata"]
    assert "encoder/model.safetensors.index.json" in result["repository_metadata"]
    assert not result["metadata_errors"]
