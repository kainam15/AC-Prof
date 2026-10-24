"""ELECTRA replaced-token-detection is an executable model-head contract, not a generic Hub label."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from acprof.container.handlers.nlp import NLPHandler
from acprof.container.validation import OutputValidationError, validate_output
from acprof.extensions import CATALOG, UnsupportedExtensionError
from acprof.host.collection_workflow import PreparationWorkflow
from acprof.host.detect import _detect_from_hub
from acprof.host.task_support import require_task_support
from acprof.model_candidates import apply_model_candidate, discover_model_candidates
from acprof.model_evidence import content_digest


def _hub(architecture, tag="pretraining", transformer_tag=None, auto_model=None):
    return SimpleNamespace(
        pipeline_tag=tag, library_name="transformers", sha="a" * 40,
        config={"architectures": [architecture], "model_type": "electra"},
        transformers_info=SimpleNamespace(
            pipeline_tag=transformer_tag if transformer_tag is not None else tag,
            auto_model=auto_model,
        ),
        siblings=[],
    )


def _detect(hub):
    with patch("huggingface_hub.HfApi.model_info", return_value=hub):
        return _detect_from_hub("fixture/discriminator")


def test_known_head_becomes_runnable_without_model_id_special_case():
    task = _detect(_hub("ElectraForPreTraining"))
    result = apply_model_candidate(task, discover_model_candidates(task))
    assert (task.pipeline_tag, task.task_family, task.runtime_backend) == (
        "replaced-token-detection", "nlp", "transformers_model")
    assert task.hub_metadata["pipeline_tag"] == "pretraining"
    assert result["status"] == "candidate"
    assert result["semantics"]["status"] == "inferred"
    assert result["provenance"]["sources"]["hub"]["sha256"] == content_digest(task.hub_metadata)
    assert not result["conflicts"]
    task.model_resolution = result
    assert PreparationWorkflow.questions(task, SimpleNamespace(task=None, model_spec=None)) == []
    require_task_support(task, devices=("cpu",))
    assert task.model_adapter == "family-default"
    assert CATALOG.resolve(task="replaced-token-detection", library="transformers").backend == "transformers_model"


def test_missing_primary_hub_tag_uses_registered_electra_head():
    # Real Hub shape: the top-level pipeline_tag is absent while its nested
    # TransformersInfo reports the generic, unregistered `pretraining` label.
    task = _detect(_hub("ElectraForPreTraining", tag=None, transformer_tag="pretraining",
                        auto_model="AutoModelForPreTraining"))
    result = apply_model_candidate(task, discover_model_candidates(task))
    assert (task.pipeline_tag, task.task_family, task.runtime_backend) == (
        "replaced-token-detection", "nlp", "transformers_model")
    assert result["status"] == "candidate"
    assert result["semantics"]["status"] == "inferred"
    assert not result["conflicts"] and not result["missing"]
    assert task.hub_metadata["pipeline_tag"] is None
    assert result["provenance"]["sources"]["hub"]["sha256"] == content_digest(task.hub_metadata)
    assert any(item.get("kind") == "unsupported_task_hint" and item.get("value") == "pretraining"
               for item in result["provenance"]["observations"])
    task.model_resolution = result
    assert PreparationWorkflow.questions(task, SimpleNamespace(task=None, model_spec=None)) == []
    require_task_support(task, devices=("cpu",))


def test_missing_primary_hub_tag_cannot_invent_an_unknown_pretraining_head():
    task = _detect(_hub("SomeUnknownForPreTraining", tag=None,
                        transformer_tag="pretraining", auto_model="AutoModelForPreTraining"))
    result = apply_model_candidate(task, discover_model_candidates(task))
    assert result["status"] == "needs_configuration"
    assert result["semantics"]["status"] == "unresolved"
    assert task.runtime_backend == "unknown"


def test_missing_primary_hub_tag_does_not_hide_supported_task_disagreement():
    task = _detect(_hub("ElectraForPreTraining", tag=None,
                        transformer_tag="fill-mask", auto_model="AutoModelForMaskedLM"))
    result = apply_model_candidate(task, discover_model_candidates(task))
    assert result["status"] == "ambiguous"
    assert result["semantics"]["status"] == "conflict"
    assert result["conflicts"]


def test_unknown_pretraining_and_other_electra_head_are_not_silently_reclassified():
    unknown = _detect(_hub("SomeUnknownForPreTraining"))
    assert unknown.pipeline_tag == "pretraining"
    assert unknown.runtime_backend == "unknown"
    masked = _detect(_hub("ElectraForMaskedLM", tag="fill-mask"))
    assert masked.pipeline_tag == "fill-mask"
    assert masked.runtime_backend == "transformers_pipeline"
    with pytest.raises(UnsupportedExtensionError):
        CATALOG.get_extension("nlp", "transformers_pipeline", task="replaced-token-detection")


def test_rtd_model_loader_rejects_wrong_architecture_before_weights(monkeypatch):
    import acprof.container.handlers.nlp as nlp

    config = SimpleNamespace(model_type="electra", architectures=["ElectraForMaskedLM"])
    from_pretrained = Mock(return_value=config)
    tokenizer = Mock()
    model = Mock()
    fake_transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=from_pretrained),
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
        ElectraForPreTraining=SimpleNamespace(from_pretrained=model),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(nlp, "load_policy", lambda *args, **kwargs:
                        {"dtype": "fp32", "trust_remote_code": False})
    with pytest.raises(ValueError, match="ElectraForPreTraining checkpoint"):
        NLPHandler().load("fixture/invalid", "replaced-token-detection", "transformers_model", "cpu")
    tokenizer.assert_not_called()
    model.assert_not_called()


class _Tensor:
    def __init__(self, values):
        self.values = values

    def to(self, device):
        return self


class _Tokenizer:
    def __init__(self):
        self.called_with = None

    def __call__(self, texts, **kwargs):
        self.called_with = texts
        assert kwargs == {"return_tensors": "pt", "padding": True, "truncation": False}
        return {"input_ids": _Tensor(texts), "attention_mask": _Tensor(texts)}


def test_rtd_inference_output_and_shape_validation(monkeypatch):
    tokenizer = _Tokenizer()
    model = Mock(device="cpu")
    model.return_value = SimpleNamespace(logits=np.ones((2, 5), dtype="float32"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    ctx = {"pipeline": SimpleNamespace(model=model, tokenizer=tokenizer),
           "task_type": "replaced-token-detection", "backend": "transformers_model"}
    processed = {"text": "a short example", "batch_size": 2, "_effective_input_scale": 3,
                 "_truncated_by_limit": False}
    raw = {"text": "a short example", "batch_size": 2, "input_scale": 3}
    handler = NLPHandler()
    output = handler.predict(ctx, processed)
    response = handler.postprocess(ctx, output)
    assert tokenizer.called_with == ["a short example", "a short example"]
    assert response == {"task": "replaced-token-detection", "output_type": "token_scores",
                        "output_shape": [2, 5], "n_results": 2}
    evidence = validate_output(ctx, raw, processed, output, response)
    assert "replaced_token_logits" in evidence["task"]["checks"]
    with pytest.raises(OutputValidationError, match="reported output shape differs"):
        validate_output(ctx, raw, processed, output, {**response, "output_shape": [1, 5]})
    with pytest.raises(ValueError, match="requires .batch, sequence."):
        handler.postprocess(ctx, np.ones((2, 5, 2), dtype="float32"))
