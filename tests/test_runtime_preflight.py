"""Frozen regressions from the 2026-09-27 compatibility audit; no model weights."""
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.task_support import TaskSupportError, require_task_support


def model(task, family, model_type, **kwargs):
    return TaskInfo(
        "fixture/model", task, family, "transformers_pipeline", "transformers",
        "a" * 40, "fixture", model_config={"model_type": model_type}, **kwargs,
    )


def test_explicit_remote_profile_respects_manifest_and_load_option_denials():
    from acprof.container.load_policy import load_policy
    from acprof.extensions import CATALOG
    from acprof.runtime_profiles import PROFILES
    profile = replace(PROFILES["cv-cpu"], profile_id="fixture-custom", trust_remote_code=True)
    extension = CATALOG.get_extension("cv", "transformers_pipeline", "family-default", "image-classification")
    with patch.dict(PROFILES, {profile.profile_id: profile}), patch.dict(os.environ, {"ACPROF_RUNTIME_PROFILE": profile.profile_id}), patch.dict(
        sys.modules, {"torch": SimpleNamespace(float32="float32"), "transformers": SimpleNamespace(__version__="4.57.6")}):
        assert (load_policy("fixture/model", "image-classification", "transformers_pipeline", "cpu")["trust_remote_code"])
        assert not (load_policy("fixture/model", "image-classification", "transformers_pipeline", "cpu", {"trust_remote_code": False})["trust_remote_code"])
        with patch.object(CATALOG, "get_extension", return_value=replace(extension, trust_remote_code=False)):
            assert not (load_policy("fixture/model", "image-classification", "transformers_pipeline", "cpu")["trust_remote_code"])
            with pytest.raises(ValueError) as caught:
                load_policy("fixture/model", "image-classification", "transformers_pipeline", "cpu", {"trust_remote_code": True})
        assert (caught.value.failure.reason_code) == ("remote_code_disallowed")

def test_precision_intersects_adapter_support_and_scopes_sam_evidence():
    from acprof.extensions import CATALOG
    from acprof.precision import resolve_precision
    from acprof.runtime_profiles import PROFILES
    extension = CATALOG.get_extension("cv", "transformers_pipeline", "family-default", "mask-generation")
    args = dict(task="mask-generation", model_type="sam2")
    assert (resolve_precision(PROFILES["cv-cpu"], extension, device="cpu", version="4.57.6", **args)["dtype"]) == ("FP32")
    assert (resolve_precision(PROFILES["cv-cu128"], extension, device="gpu", version="5.6.0", **args)["dtype"]) == ("FP16")
    scoped = replace(extension, precision_policy={"model_type_overrides": {"sam2": {"preferred_dtype": "FP32"}}})
    assert (resolve_precision(PROFILES["cv-cu128"], scoped, device="gpu", version="5.6.0", **args)["dtype"]) == ("FP32")
    with pytest.raises(ValueError) as caught:
        resolve_precision(PROFILES["cv-cu128"], replace(extension, dtypes=("FP32",)), device="gpu", version="5.6.0", **args)
    assert (caught.value.failure.reason_code) == ("precision_mismatch")

@pytest.mark.parametrize('sources_case', range(3))
def test_dependency_versions_dynamic_imports_and_training_requirements(sources_case):
    from acprof.runtime_dependencies import dependency_preflight
    from acprof.runtime_profiles import PROFILES
    cases = (({"requirements.txt": "transformers<4"}, "runtime_dependency_incompatible"),
             ({"custom.py": "import importlib\nimportlib.import_module(name)"}, "runtime_dependency_unknown"),
             ({"requirements-dev.txt": "unknown-development-only-package"}, None))
    (sources, reason) = tuple(cases)[sources_case]
    task = model("image-classification", "cv", "vit", repository_files=tuple(sources))
    if "custom.py" in sources:
        task.model_config["auto_map"] = {"AutoModel": "custom.Model"}
    if reason:
        with pytest.raises(ValueError) as caught:
            dependency_preflight(task, PROFILES["cv-cpu"], read_source=sources.__getitem__)
        assert (caught.value.failure.reason_code) == (reason)
    else:
        assert (dependency_preflight(task, PROFILES["cv-cpu"], read_source=sources.__getitem__)["dependencies"]) == ([])

def test_cv_profile_false_reaches_pipeline_as_false():
    from acprof.container.handlers.cv import CVHandler
    pipeline = Mock(return_value=SimpleNamespace())
    fake_torch = SimpleNamespace(float16="float16", float32="float32", bfloat16="bfloat16")
    with patch.dict(sys.modules, {"torch": fake_torch, "transformers": SimpleNamespace(pipeline=pipeline, __version__="4.57.6")}), patch.dict(
            os.environ, {"ACPROF_RUNTIME_PROFILE": "cv-cpu"}):
        CVHandler().load("fixture/model", "image-classification", "transformers_pipeline", "cpu")
    assert (pipeline.call_args.kwargs["trust_remote_code"]) is (False)

def test_installed_transformers_must_match_the_selected_lock():
    from acprof.container.handlers.cv import CVHandler
    pipeline = Mock()
    with patch.dict(sys.modules, {"torch": SimpleNamespace(), "transformers": SimpleNamespace(pipeline=pipeline, __version__="5.6.0")}), patch.dict(
        os.environ, {"ACPROF_RUNTIME_PROFILE": "cv-cpu"}):
        with pytest.raises(ValueError) as caught:
            CVHandler().load("fixture/model", "image-classification", "transformers_pipeline", "cpu")
    assert (caught.value.failure.reason_code) == ("runtime_dependency_incompatible")
    pipeline.assert_not_called()

def test_sam_gpu_defaults_to_fp32_but_explicit_fp16_is_rejected():
    from acprof.container.handlers.cv import CVHandler
    for model_type in ("sam", "sam2"):
        pipeline = Mock(return_value=SimpleNamespace())
        fake_torch = SimpleNamespace(float16="float16", float32="float32", bfloat16="bfloat16")
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            "torch": fake_torch, "transformers": SimpleNamespace(pipeline=pipeline, __version__="4.57.6"),
        }), patch.dict(os.environ, {"ACPROF_RUNTIME_PROFILE": "cv-cu128"}):
            Path(directory, "config.json").write_text(json.dumps({"model_type": model_type}))
            CVHandler().load(directory, "mask-generation", "transformers_pipeline", "cuda")
            assert (pipeline.call_args.kwargs["torch_dtype"]) == ("float32")
            pipeline.reset_mock()
            with pytest.raises(ValueError) as caught:
                CVHandler().load(directory, "mask-generation", "transformers_pipeline", "cuda", load_options={"dtype": "FP16"})
            assert (caught.value.failure.reason_code) == ("precision_mismatch")
            pipeline.assert_not_called()

def test_unversioned_typing_backport_uses_target_python_stdlib():
    from acprof.runtime_dependencies import dependency_preflight
    from acprof.runtime_profiles import PROFILES
    task = model("image-classification", "cv", "vit", repository_files=("requirements.txt",))
    report = dependency_preflight(task, PROFILES["cv-cpu"], read_source=lambda _: "typing\ntyping-extensions\n")
    typing = next(item for item in report["dependencies"] if item["distribution"] == "typing")
    assert (typing["status"]) == ("stdlib")
    assert (typing["source"]) == ("requirements.txt")
    assert (typing["python_version"]) == ("3.10")
    with pytest.raises(ValueError):
        dependency_preflight(task, PROFILES["cv-cpu"], read_source=lambda _: "typing==3.7.4.3")

def test_remote_rmbg_missing_scikit_image_is_preflight_failure():
    task = model("image-segmentation", "cv", "birefnet",
                 repository_files=("config.json", "modeling_rmbg.py"))
    task.model_config["auto_map"] = {"AutoModelForImageSegmentation": "modeling_rmbg.RMBG"}
    with patch("acprof.host.detect.read_model_source", return_value="from skimage import transform\n"):
        with pytest.raises(TaskSupportError) as caught:
            require_task_support(task)
    assert (caught.value.failure.reason_code) == ("runtime_dependency_missing")
    assert (caught.value.failure.evidence["dependencies"][0]["distribution"]) == ("scikit-image")

def test_manga_ocr_tokenizer_dependency_is_checked_without_remote_code():
    task = model("image-to-text", "cv", "vision-encoder-decoder",
                 repository_metadata={"tokenizer_config.json": {"tokenizer_class": "BertJapaneseTokenizer"}})
    with pytest.raises(TaskSupportError) as caught:
        require_task_support(task)
    assert (caught.value.failure.reason_code) == ("runtime_dependency_missing")
    assert (caught.value.failure.evidence["dependencies"][0]["distribution"]) == ("fugashi")

def test_glm_ocr_task_is_checked_in_selected_runtime():
    task = model("image-to-text", "cv", "glm_ocr")
    with pytest.raises(TaskSupportError) as caught:
        require_task_support(task)
    assert (caught.value.failure.reason_code) == ("runtime_task_unsupported")
    assert (caught.value.failure.stage) == ("preflight")
    assert (caught.value.failure.runtime_profile) == ("cv-transformers560-cu128")
    assert (caught.value.failure.evidence["transformers_version"]) == ("5.6.0")

def test_explicit_transformers5_cannot_borrow_old_pipeline_registry():
    task = model("image-to-text", "cv", "blip", runtime_profile_id="cv-transformers560-cpu")
    with pytest.raises(TaskSupportError) as caught:
        require_task_support(task)
    assert (caught.value.failure.reason_code) == ("runtime_task_unsupported")

def test_structured_torchscript_requires_contract_before_runtime():
    task = TaskInfo("fixture/graph", "graph-ml", "structured", "torchscript", "pytorch",
                    "a" * 40, "fixture")
    with pytest.raises(TaskSupportError) as caught:
        require_task_support(task)
    assert (caught.value.failure.reason_code) == ("model_contract_required")
    assert (task.model_resolution["status"]) == ("needs_configuration")
