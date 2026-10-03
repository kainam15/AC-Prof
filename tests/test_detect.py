import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host import detect


def test_hub_detection_routes_text_to_image_to_diffusers() -> None:
    hub_info = SimpleNamespace(
        pipeline_tag="text-to-image",
        library_name="diffusers",
        sha="diffusion-revision",
        safetensors=None,
        card_data={},
        config={},
        tags=["diffusers"],
    )

    with patch("huggingface_hub.HfApi.model_info", return_value=hub_info):
        info = detect._detect_from_hub(
            "stable-diffusion-v1-5/stable-diffusion-v1-5"
        )

    assert (info) is not None
    assert info is not None
    assert (info.pipeline_tag) == ("text-to-image")
    assert (info.task_family) == ("diffusion")
    assert (info.runtime_backend) == ("diffusers")
    assert (info.model_revision) == ("diffusion-revision")

def test_hub_detection_collects_model_metadata() -> None:
    hub_info = SimpleNamespace(
        pipeline_tag="fill-mask",
        library_name="transformers",
        sha="0123456789abcdef",
        safetensors=SimpleNamespace(
            parameters={"F32": 110_106_428},
            total=110_106_428,
        ),
        card_data={"license": "apache-2.0"},
        config={"architectures": ["BertForMaskedLM"]},
        tags=["transformers", "license:apache-2.0"],
    )

    with patch("huggingface_hub.HfApi.model_info", return_value=hub_info):
        info = detect._detect_from_hub(
            "google-bert/bert-base-uncased"
        )

    assert (info) is not None
    assert info is not None
    assert (info.parameter_count) == (110_106_428)
    assert (info.parameter_bytes) == (440_425_712)
    assert (info.precision_dtype) == ("FP32")
    assert (info.parameter_dtype_counts) == ({"FP32": 110_106_428})
    assert not (info.quantized)
    assert (info.model_license) == ("apache-2.0")
    assert (info.model_metadata_source) == ("huggingface_hub")

def test_hub_detection_preserves_quantization_metadata() -> None:
    hub_info = SimpleNamespace(
        pipeline_tag="text-generation",
        library_name="transformers",
        sha="fedcba9876543210",
        safetensors=SimpleNamespace(
            parameters={"I8": 7_000_000_000},
            total=7_000_000_000,
        ),
        card_data={},
        config={
            "architectures": ["ExampleForCausalLM"],
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 8,
            },
        },
        tags=["gptq", "license:mit"],
    )

    with patch("huggingface_hub.HfApi.model_info", return_value=hub_info):
        info = detect._detect_from_hub("example/quantized-model")

    assert (info) is not None
    assert info is not None
    assert (info.parameter_bytes) == (7_000_000_000)
    assert (info.precision_dtype) == ("INT8")
    assert (info.quantized)
    assert (info.quantization_method) == ("gptq")
    assert (info.quantization_config) == ({"quant_method": "gptq", "bits": 8})
    assert (info.model_license) == ("mit")

def test_parameter_bytes_is_null_when_any_dtype_width_is_unknown() -> None:
    assert (detect._parameter_bytes_from_dtype_counts(
            {"FP16": 10, "INT64": 2}
        )) == (36)
    metadata = detect._hub_model_metadata(
        SimpleNamespace(
            safetensors=SimpleNamespace(
                parameters={"F16": 10, "CUSTOM": 2},
                total=12,
            ),
            card_data={},
            config={},
            tags=[],
        )
    )

    assert (metadata["parameter_dtype_counts"]) == ({"FP16": 10, "CUSTOM": 2})
    assert (metadata["parameter_bytes"]) is None

def test_config_fallback_reads_config_json_without_transformers_dependency() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir, "snapshots", "a" * 40, "config.json")
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            json.dumps({"architectures": ["BertForMaskedLM"]}),
            encoding="utf-8",
        )

        with patch("huggingface_hub.hf_hub_download", return_value=str(config_path)):
            info = detect._detect_from_config("google-bert/bert-base-uncased")

    assert (info) is not None
    assert info is not None
    assert (info.pipeline_tag) == ("fill-mask")
    assert (info.task_family) == ("nlp")
    assert (info.runtime_backend) == ("transformers_pipeline")
    assert (info.library_name) == ("transformers")
    assert (info.detection_method) == ("config_infer")

def test_config_fallback_does_not_analyze_a_file_without_a_pinned_snapshot():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "config.json")
        path.write_text(json.dumps({"architectures": ["BertForMaskedLM"]}))
        diagnostics = []
        with patch("huggingface_hub.hf_hub_download", return_value=str(path)):
            result = detect._detect_from_config("unseen/model", diagnostics)
        assert (result) is None
        assert ("SHA") in (";".join(diagnostics))

def test_detect_task_reports_auto_detection_failure_reasons() -> None:
    stderr = io.StringIO()

    with patch("huggingface_hub.HfApi.model_info", side_effect=RuntimeError("hub timeout")), patch(
        "huggingface_hub.hf_hub_download", side_effect=OSError("config missing")
    ), patch("sys.stderr", stderr):
        with pytest.raises(ValueError) as raised:
            detect.detect_task("missing/model")

    assert (stderr.getvalue()) == ("")
    message = str(raised.value)
    assert ("missing/model") in (message)
    assert ("hub_api: RuntimeError: hub timeout") in (message)
    assert ("config_json: OSError: config missing") in (message)
    assert ("AutoConfig:") not in (message)
    assert ("--task-family") not in (message)
