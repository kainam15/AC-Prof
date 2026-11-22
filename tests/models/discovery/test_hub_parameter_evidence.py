"""Hub parameter metadata remains descriptive and exact through public task discovery."""
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
