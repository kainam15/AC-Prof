"""Hub parameter metadata remains descriptive and exact through public task discovery."""
import hashlib
import json
from inspect import Parameter, signature
from unittest.mock import patch

import pytest
from huggingface_hub import ModelInfo

from acprof.host.detect import TaskInfo, detect_task
from acprof.hub.repository import MetadataFile, model_source_scope


@pytest.fixture
def discover(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("acprof.host.model_store.store.cached_model_info", lambda *args, **kwargs: None)

    def from_hub(parameters, total, *, config=None, tags=(), omit_safetensors=False):
        repo = "fixture/parameter-evidence"
        revision = "d" * 40
        info = ModelInfo(id=repo, sha=revision, pipeline_tag="fill-mask", library_name="transformers",
                         config={"architectures": ["BertForMaskedLM"], **(config or {})}, tags=list(tags),
                         siblings=[{"rfilename": "README.md"}],
                         safetensors=None if omit_safetensors else {"parameters": parameters, "total": total})
        with model_source_scope("huggingface"), patch("huggingface_hub.HfApi.model_info", return_value=info):
            return detect_task(repo, revision=revision)

    return from_hub


@pytest.mark.parametrize(("parameters", "total", "count", "known", "reason"), [
    ({"F32": 2, "I8": None}, 102, 102, {"FP32": 2}, "invalid count for INT8"),
    ({"F32": 2}, 102, 102, {"FP32": 2}, "sum to 2, declared total is 102"),
    ({"F32": 9, "float32": -2}, 7, 7, {"FP32": 9}, "invalid count for FP32"),
    ({"I8": True}, True, None, {}, "invalid count for INT8"),
    ({"F32": 3.4}, 3.4, None, {}, "invalid count for FP32"),
    ({"F32": 2}, -1, None, {"FP32": 2}, "invalid total"),
    ({"F32": 2}, float(2**53), None, {"FP32": 2}, "invalid total"),
    ({"F32": None, "": 1}, 3, 3, {}, "missing dtype name"),
    ({}, 6, 6, {}, "missing or empty"),
])
def test_invalid_histograms_keep_known_counts_and_declared_total(discover, parameters, total, count, known, reason):
    task = discover(parameters, total)
    assert task.parameter_count == count
    assert task.parameter_dtype_counts == known
    assert task.parameter_bytes is None
    assert reason in task.parameter_bytes_reason
    assert task.precision_dtype is None
    assert task.quantized is None


@pytest.mark.parametrize(("counts", "total", "size", "precision", "quantized"), [
    ({"F16": 2, "float16": "3", "I64": 1}, 6.0, 18, "FP16", False),
    ({}, 0, 0, None, None),
    ({"F16": 0, "F32": 0}, 0, 0, None, None),
    ({"I8": 2, "F32": 2}, 4, 10, None, None),
    ({"F32": 2, "I8": 2}, 4, 10, None, None),
    ({"I8": 0, "F16": 3}, 3, 6, "FP16", False),
    ({"CUSTOM": 0, "F16": 3}, 3, 6, "FP16", False),
    ({"I8": 2}, 2, 2, "INT8", True),
    ({"F32": float(2**53 - 1)}, str(2**53 - 1), 4 * (2**53 - 1), "FP32", False),
    ({"F32": str(2**53 + 1)}, 2**53 + 1, 4 * (2**53 + 1), "FP32", False),
])
def test_complete_histogram_needs_unique_positive_dominant_dtype(discover, counts, total, size, precision, quantized):
    task = discover(counts, total)
    assert task.parameter_bytes == size
    assert task.parameter_bytes_reason is None
    assert task.precision_dtype == precision
    assert task.quantized is quantized


def test_unknown_dtype_keeps_recorded_identity_without_quantization_claim(discover):
    task = discover({"CUSTOM": 3}, 3)
    assert task.parameter_bytes is None
    assert "CUSTOM" in task.parameter_bytes_reason
    assert task.precision_dtype == "CUSTOM"
    assert task.quantized is None


@pytest.mark.parametrize(("unnamed", "expected"), [
    ({"": "40", " ": 2.0}, 42),
    ({"": 0, " ": False, "\t": -1}, None),
    ({}, None),
])
def test_unnamed_elements_remain_separate_from_dtype_counts(discover, unnamed, expected):
    task = discover({"F16": 2, **unnamed}, 2)
    assert task.parameter_count == 2
    assert task.parameter_dtype_counts == {"FP16": 2}
    if expected is None:
        assert "unnamed_dtype_elements" not in task.hub_metadata
    else:
        assert task.hub_metadata["unnamed_dtype_elements"] == expected
        assert task.parameter_bytes is None
        assert "missing dtype name" in task.parameter_bytes_reason
        assert task.precision_dtype is None


def test_config_dtype_fallback_and_quantization_evidence(discover):
    fallback = discover(None, None, config={"torch_dtype": "float16"}, omit_safetensors=True)
    assert fallback.precision_dtype == "FP16"
    assert fallback.quantized is False
    invalid = discover(None, None, config={"dtype": {"unexpected": "mapping"}}, omit_safetensors=True)
    assert invalid.precision_dtype is None
    assert invalid.quantized is None
    partial = discover({"F32": 2, "F16": None}, 20, tags=["gptq"], config={"torch_dtype": "float32"})
    assert partial.precision_dtype is None
    assert partial.quantized is True
    assert partial.quantization_method == "gptq"


def test_parameter_bytes_diagnostic_keeps_taskinfo_positional_api():
    assert signature(TaskInfo).parameters["parameter_bytes_reason"].kind is Parameter.KEYWORD_ONLY


def test_modelscope_fallback_preserves_local_config_provenance(tmp_path, monkeypatch):
    repo = "fixture/config-source"
    revision = "c" * 40
    config = MetadataFile(tmp_path / "snapshots" / revision / "config.json", b'{"architectures":["BertForMaskedLM"]}')
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("acprof.host.model_store.store.cached_model_info", lambda *args, **kwargs: None)
    monkeypatch.setattr("acprof.host.model_store.store.cached_metadata",
                        lambda model, name, *args, **kwargs: config if name == "config.json" else None)
    with model_source_scope("modelscope"), patch("acprof.hub.repository.modelscope_info", side_effect=OSError("offline")):
        fallback = detect_task(repo, revision=revision)
    assert fallback.model_source == "modelscope"
    assert fallback.detection_method == "config_infer"
    assert fallback.metadata_sources["config.json"]["revision"] == revision
    assert fallback.model_metadata_source is None

    card = ModelInfo(id=repo, sha=revision, pipeline_tag="fill-mask", library_name="transformers",
                     siblings=[{"rfilename":"README.md"}], tags=["license:mit"])
    with model_source_scope("modelscope"), patch("acprof.hub.repository.modelscope_info", return_value=card):
        hub = detect_task(repo, revision=revision)
    assert hub.model_source == "modelscope"
    assert hub.model_metadata_source == "modelscope_hub"
    assert hub.model_license == "mit"


@pytest.fixture
def discover_pinned(tmp_path, monkeypatch):
    repo, revision = "fixture/weight-metadata", "c" * 40
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("acprof.host.model_store.store.cached_model_info", lambda *args, **kwargs: None)

    def discover(file_config, summary_config, *, source="huggingface", safetensors=None, offline=False, tags=()):
        document = {"architectures": ["BertForMaskedLM"], "model_type": "bert", **file_config}
        content = json.dumps(document).encode()
        file = MetadataFile(tmp_path / "snapshots" / revision / "config.json", content)
        summary = ModelInfo(id=repo, sha=revision, pipeline_tag="fill-mask", library_name="transformers",
                            config=summary_config.copy(), tags=list(tags), siblings=[{"rfilename": "config.json"}],
                            safetensors=safetensors)
        endpoint = ("huggingface_hub.HfApi.model_info" if source == "huggingface"
                    else "acprof.hub.repository.modelscope_info")
        monkeypatch.setattr("acprof.host.model_store.store.cached_metadata",
                            lambda model, name, *a, **kw: file if name == "config.json" else None)
        with model_source_scope(source), patch(endpoint, return_value=summary, side_effect=OSError("offline") if offline else None):
            task = detect_task(repo, revision=revision)
        assert summary.config == summary_config
        assert task.repository_metadata["config.json"] == document
        assert task.metadata_sources["config.json"]["sha256"] == hashlib.sha256(content).hexdigest()
        assert task.metadata_sources["config.json"]["revision"] == revision
        return task

    return discover


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
@pytest.mark.parametrize(("summary", "file_quant", "offline", "expected"), [
    (None, "gptq", False, "gptq"),
    ("awq", "gptq", False, "gptq"),
    ("awq", None, False, "awq"),
    (None, None, False, None),
    (None, "gptq", True, "gptq"),
])
def test_quantization_follows_pinned_config(discover_pinned, source, summary, file_quant, offline, expected):
    def quantization(method):
        return {"quant_method": method, "bits": 4}

    file_config = {"torch_dtype": "float16"}
    if file_quant:
        file_config["quantization_config"] = quantization(file_quant)
    task = discover_pinned(file_config, {"quantization_config": quantization(summary)} if summary else {},
                           source=source, offline=offline, safetensors={"parameters": {"F16": 6, "I32": 2}, "total": 8})
    assert task.quantization_config == (quantization(expected) if expected else {})
    assert task.quantization_method == expected
    assert task.quantized is bool(expected)
    assert task.precision_dtype == "FP16"
    assert task.model_source == source
    assert task.model_metadata_source == (None if offline else f"{source}_hub")
    assert task.parameter_count == (None if offline else 8)


@pytest.mark.parametrize(("safetensors", "expected"), [
    (None, "BF16"),
    ({"parameters": {"F32": 6}, "total": 6}, "FP32"),
    ({"parameters": {"F32": 6}, "total": 8}, None),
])
@pytest.mark.parametrize(("api_dtype", "file_dtype"), [
    ({"torch_dtype": "float16"}, {"dtype": "bfloat16"}),
    ({"dtype": "float16"}, {"torch_dtype": "bfloat16"}),
    ({}, {"torch_dtype": "float16", "dtype": "bfloat16"}),
])
def test_pinned_dtype_alias_and_safetensors_authority(discover_pinned, safetensors, expected, api_dtype, file_dtype):
    task = discover_pinned(file_dtype, api_dtype, safetensors=safetensors)
    assert task.precision_dtype == expected


@pytest.mark.parametrize(("configuration", "tags", "method"), [
    ({"quant_method": " gptq "}, (), "gptq"),
    ({"quant_method": ["gptq"]}, (), None),
    ({"quant_method": {"name": "gptq"}}, (), None),
    ({"quant_method": 4, "quantization_method": "gptq"}, (), "gptq"),
    ({"quant_method": "   ", "quant_type": "nf4"}, (), "nf4"),
    ({"quant_method": False, "quantization_method": {}}, ("awq",), "awq"),
])
def test_quantization_method_requires_a_recorded_name(discover_pinned, configuration, tags, method):
    task = discover_pinned({"quantization_config": configuration}, {}, tags=tags)
    assert task.quantized is True
    assert task.quantization_config == configuration
    assert task.quantization_method == method


# Condensed from moonshotai/Kimi-K2.5@94b44e57fb4797fde1c1872bed70aef3c93a07f6.
_KIMI_QUANTIZATION = {"quant_method": "compressed-tensors", "format": "pack-quantized",
                      "config_groups": {"group_0": {"weights": {"num_bits": 4}}},
                      "ignore": ["re:vision_tower.*", "re:mm_projector.*"]}
_DECODER = {"dtype": "int8", "quantization_config": _KIMI_QUANTIZATION}
_TOP = {"quant_method": "awq", "bits": 4}


@pytest.mark.parametrize(("file_config", "hub_config", "expected"), [
    ({"text_config": _DECODER}, {}, _KIMI_QUANTIZATION),
    ({"decoder": _DECODER}, {}, _KIMI_QUANTIZATION),
    ({"generator": _DECODER}, {}, _KIMI_QUANTIZATION),
    ({"quantization_config": _TOP, "text_config": _DECODER}, {}, _TOP),
    ({"quantization_config": None, "text_config": _DECODER}, {}, _KIMI_QUANTIZATION),
    ({"text_config": _DECODER}, {"quantization_config": _TOP}, _KIMI_QUANTIZATION),
    ({}, {"text_config": _DECODER}, _KIMI_QUANTIZATION),
    ({"vision_config": _DECODER, "text_encoder": _DECODER}, {}, {}),
    ({"text_config": _DECODER, "decoder": {}}, {}, {}),
    ({"text_config": {"quantization_config": ["fp8"]}}, {"quantization_config": _TOP}, {}),
    ({"quantization_config": [], "text_config": _DECODER}, {}, {}),
    ({"text_config": _DECODER, "decoder": {}}, {"quantization_config": _TOP}, {}),
])
def test_decoder_quantization_follows_one_declared_text_config(discover_pinned, file_config, hub_config, expected):
    task = discover_pinned({"dtype": "bfloat16", **file_config}, hub_config)
    assert task.quantization_config == expected
    assert task.quantization_method == expected.get("quant_method")
    if expected:
        assert task.quantized is True
    assert task.precision_dtype == "BF16"
    assert "text_config" not in task.model_config
    assert task.model_config.get("quantization_config") == file_config.get("quantization_config", hub_config.get("quantization_config"))


def test_cached_decoder_quantization_keeps_config_provenance(discover_pinned):
    task = discover_pinned({"text_config": _DECODER}, {}, source="modelscope", offline=True)
    assert task.quantization_config == _KIMI_QUANTIZATION
    assert task.quantized is True
    assert task.precision_dtype is None
    assert task.model_metadata_source is None


def test_nested_hub_quantization_keeps_nonfinite_diagnostic_path(discover_pinned):
    task = discover_pinned({}, {"text_config": {"quantization_config": {"bits": float("inf")}}})
    assert not task.quantization_config
    assert task.quantized is None
    assert ("hub.config", "/text_config/quantization_config/bits") in {
        (item["file"], item["json_pointer"]) for item in task.metadata_diagnostics
    }


@pytest.mark.parametrize("file_config", [
    {"text_config": _DECODER, "decoder": {}},
    {"text_config": {"quantization_config": ["fp8"]}},
])
@pytest.mark.parametrize(("tags", "expected"), [((), None), (("awq",), True)])
def test_inconclusive_decoder_declaration_keeps_independent_evidence(discover_pinned, file_config, tags, expected):
    task = discover_pinned({"dtype": "float16", **file_config}, {}, tags=tags)
    assert task.precision_dtype == "FP16"
    assert task.quantized is expected
    assert task.quantization_config == {}
    assert task.quantization_method == ("awq" if tags else None)


def test_unselected_decoder_summary_cannot_invalidate_pinned_declaration(discover_pinned):
    task = discover_pinned({"text_config": _DECODER},
                           {"text_config": {"quantization_config": {"bits": float("inf")}}})
    assert task.quantization_config == _KIMI_QUANTIZATION
    assert not task.metadata_diagnostics
