"""Offline real Transformers custom pipeline bridge; tiny random weights, CPU only."""
import importlib.util
import json
import os
import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.runtime


@pytest.mark.skipif(not (all(importlib.util.find_spec(name) for name in ("torch", "transformers"))), reason="requires the NLP container")
class TestCustomPipelineRuntime:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from acprof.runtime_profiles import PROFILES
        base = PROFILES[os.environ.get("ACPROF_RUNTIME_PROFILE", "nlp-cpu")]
        # Only generated test code opts in; production remote-code policy stays strict.
        allowed = replace(base, profile_id="fixture-nlp-custom", trust_remote_code=True)
        denied = replace(base, profile_id="fixture-nlp-denied", trust_remote_code=False)
        profiles = patch.dict(PROFILES, {allowed.profile_id: allowed, denied.profile_id: denied})
        profiles.start()
        self._request.addfinalizer(partial(profiles.stop))

    @staticmethod
    def snapshot(root: Path, *, broken=False):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

        torch.manual_seed(123)
        torch.set_num_threads(1)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]
        (root / "vocab.txt").write_text("\n".join(vocab))
        tokenizer = BertTokenizerFast(vocab_file=str(root / "vocab.txt"), model_max_length=32)
        tokenizer.save_pretrained(root)
        config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
                            num_labels=2)
        config.custom_pipelines = {"acme-classify": {"impl": "custom_pipeline.Classifier",
                                                    "pt": ["AutoModelForSequenceClassification"]}}
        BertForSequenceClassification(config).save_pretrained(root)
        method = ("    def _forward(self, model_inputs):\n"
                  "        raise RuntimeError('fixture prediction failure')\n") if broken else "    pass\n"
        (root / "custom_pipeline.py").write_text(
            "from transformers import TextClassificationPipeline\n\nclass Classifier(TextClassificationPipeline):\n" + method)
        # The declaration is carried by the image rather than altering the Hub snapshot.
        from acprof.model_spec import encode_model_spec
        return encode_model_spec({"schema_version": 1, "format": "transformers-pipeline",
                                  "task": "text-classification", "pipeline_task": "acme-classify"})

    @staticmethod
    def environment(root: Path, spec: str):
        return {"ACPROF_MODEL_SPEC_B64": spec, "MODEL_LOCAL_PATH": str(root), "MODEL_ID": "local/custom",
                "MODEL_REVISION": "a" * 40, "TASK_FAMILY": "nlp", "TASK_TYPE": "text-classification",
                "RUNTIME_BACKEND": "transformers_pipeline", "USE_GPU": "0", "TORCH_NUM_THREADS": "1",
                "ACPROF_RUNTIME_PROFILE": "fixture-nlp-custom",
                "ACPROF_MODEL_ADAPTER": "family-default", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    def test_custom_pipeline_rejects_a_profile_without_remote_code_permission(self):
        from acprof.container.handlers.nlp import NLPHandler
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoded = self.snapshot(root)
            environment = {**self.environment(root, encoded), "ACPROF_RUNTIME_PROFILE": "fixture-nlp-denied"}
            with patch.dict(os.environ, environment), pytest.raises(ValueError, match="trust_remote_code=True"):
                NLPHandler().load(str(root), "text-classification", "transformers_pipeline", "cpu")

    def test_custom_pipeline_matches_native_predictions_and_independent_validation(self):
        from transformers import pipeline

        from acprof.container.handlers.nlp import NLPHandler
        from acprof.container.runtime_validate import validate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoded = self.snapshot(root)
            with patch.dict(os.environ, self.environment(root, encoded)):
                handler = NLPHandler()
                context = handler.load(str(root), "text-classification", "transformers_pipeline", "cpu")
                assert (type(context["pipeline"]).__name__) == ("Classifier")
                assert (context["task_type"]) == ("text-classification")
                payload = {"text": "hello world", "batch_size": 1, "input_scale": 2}
                actual = handler.predict(context, handler.preprocess(context, payload))
                expected = pipeline("text-classification", model=str(root), device="cpu", trust_remote_code=False)("hello world")
                assert (actual[0]["label"]) == (expected[0]["label"])
                assert (actual[0]["score"]) == (expected[0]["score"]) or round(abs((actual[0]["score"]) - (expected[0]["score"])), 6) == 0
                report = validate(payload)
                assert (report["status"]) == ("ok")
                assert (report["validation"]["task"]["status"]) == ("verified")
                assert ("classification_labels_scores") in (report["validation"]["task"]["checks"])
                assert (report["model_spec"]["pipeline_task"]) == ("acme-classify")
                assert (report["workload_contract"]["input"]["actual_scale"]) == (2)
                assert (all(item["status"] == "verified" for item in report["stages"]))

    def test_custom_code_failure_keeps_the_prediction_stage_and_cause(self):
        from acprof.container.runtime_validate import validate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoded = self.snapshot(root, broken=True)
            with patch.dict(os.environ, self.environment(root, encoded)):
                stages = []
                with pytest.raises(RuntimeError, match="fixture prediction failure"):
                    validate({"text": "hello world", "batch_size": 1, "input_scale": 2}, stages=stages)
                assert (stages[-1]["stage"]) == ("predict")
                assert (stages[-1]["status"]) == ("error")

    def test_custom_auto_map_loads_unknown_architecture_from_local_snapshot(self):
        from acprof.container.handlers.nlp import NLPHandler
        from acprof.container.runtime_validate import validate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snapshot(root)
            config = json.loads((root / "config.json").read_text())
            config.pop("custom_pipelines")
            config.update(model_type="acprof_fixture", architectures=["CustomClassifier"], auto_map={
                "AutoConfig": "custom_model.CustomConfig",
                "AutoModelForSequenceClassification": "custom_model.CustomClassifier",
            })
            (root / "config.json").write_text(json.dumps(config))
            (root / "custom_model.py").write_text(
                "from transformers import BertConfig, BertForSequenceClassification\n\n"
                "class CustomConfig(BertConfig):\n    model_type = 'acprof_fixture'\n\n"
                "class CustomClassifier(BertForSequenceClassification):\n    config_class = CustomConfig\n")
            with patch.dict(os.environ, self.environment(root, "")):
                context = NLPHandler().load(str(root), "text-classification", "transformers_pipeline", "cpu")
                assert (type(context["pipeline"].model).__name__) == ("CustomClassifier")
                report = validate({"text": "hello world", "batch_size": 1, "input_scale": 2})
                assert (report["status"]) == ("ok")
                assert (report["validation"]["task"]["status"]) == ("verified")
                assert (report["workload_contract"]["input"]["actual_scale"]) == (2)
