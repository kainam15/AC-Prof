"""Offline shared custom pipeline protocol with tiny random Torch weights."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_custom_multimodal import pipeline_spec
from test_multimodal_handler import audio_payload, image_payload

CUSTOM_MODEL = '''from transformers import BertConfig, BertForSequenceClassification

class AudioConfig(BertConfig):
    model_type = "fixture_unseen_multimodal"

class AudioModel(BertForSequenceClassification):
    config_class = AudioConfig
'''

CUSTOM_PIPELINE = '''import numpy as np
import torch
from transformers import AutoTokenizer, Pipeline

class AudioPipeline(Pipeline):
    def __init__(self, model, tokenizer=None, **kwargs):
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model.config._name_or_path, local_files_only=True)
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.events = []

    def _sanitize_parameters(self, **kwargs):
        return {}, kwargs, {}

    def preprocess(self, payload):
        self.events.append("preprocess")
        inputs = dict(self.tokenizer(payload["question"], return_tensors="pt"))
        if "waveform" in payload:
            assert payload["rate"] == 16000
            inputs["audio_values"] = torch.tensor(payload["waveform"]).reshape(1, -1)
        elif "picture" in payload:
            inputs["pixel_values"] = torch.tensor(np.asarray(payload["picture"]).copy()).float().unsqueeze(0)
        else:
            assert payload["fps"] == 2.0
            inputs["pixel_values_videos"] = torch.tensor(payload["frames"].copy()).float()
        return inputs

    def _forward(self, inputs, limit, temperature):
        self.events.append("predict")
        assert temperature == 0.0 and limit > 0
        media = inputs.pop("audio_values", None)
        if media is None:
            media = inputs.pop("pixel_values", None)
        if media is None:
            media = inputs.pop("pixel_values_videos")
        scores = self.model(**inputs).logits + media.float().mean()
        return {"scores": scores, "samples": media.numel(), "limit": limit}

    def postprocess(self, outputs):
        self.events.append("postprocess")
        label = int(outputs["scores"].argmax())
        return f"answer {label} samples {outputs['samples']} limit {outputs['limit']}"
'''


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "transformers")),
                     "requires the custom multimodal container")
class CustomMultimodalRuntimeTests(unittest.TestCase):
    @staticmethod
    def snapshot(root: Path):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

        torch.manual_seed(123)
        torch.set_num_threads(1)
        (root / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwhat\nis\nsaid\n")
        BertTokenizerFast(vocab_file=str(root / "vocab.txt"), model_max_length=32).save_pretrained(root)
        config = BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=32, max_position_embeddings=32, num_labels=2)
        BertForSequenceClassification(config).save_pretrained(root)
        data = json.loads((root / "config.json").read_text())
        data.update(model_type="fixture_unseen_multimodal", auto_map={
            "AutoConfig": "custom_model.AudioConfig", "AutoModel": "custom_model.AudioModel",
        }, custom_pipelines={"listen-and-answer": {
            "impl": "custom_pipeline.AudioPipeline", "pt": ["AutoModel"], "type": "multimodal",
        }})
        (root / "config.json").write_text(json.dumps(data))
        (root / "custom_model.py").write_text(CUSTOM_MODEL)
        (root / "custom_pipeline.py").write_text(CUSTOM_PIPELINE)

    def test_full_pipeline_loads_transitive_imports_before_inference(self):
        from acprof.container.runtime_validate import validate
        from acprof.model_spec import encode_model_spec

        with tempfile.TemporaryDirectory(prefix="acprof_nested_pipeline_") as directory:
            root = Path(directory)
            self.snapshot(root)
            (root / "pipeline_base.py").write_text(CUSTOM_PIPELINE)
            (root / "pipeline_bridge.py").write_text("from .pipeline_base import AudioPipeline\n")
            (root / "custom_pipeline.py").write_text("from .pipeline_bridge import AudioPipeline\n")
            spec = pipeline_spec()
            spec.pop("dependencies")
            environment = {"ACPROF_MODEL_SPEC_B64": encode_model_spec(spec), "MODEL_LOCAL_PATH": directory,
                           "MODEL_ID": "fixture/nested", "MODEL_REVISION": "a" * 40,
                           "TASK_TYPE": "audio-text-to-text", "TASK_FAMILY": "multimodal",
                           "RUNTIME_BACKEND": "transformers_pipeline", "USE_GPU": "0", "TORCH_NUM_THREADS": "1",
                           "ACPROF_MODEL_ADAPTER": "family-default", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            with patch.dict(os.environ, environment):
                result = validate({"samples": [{"text": "What is said?", "audio_base64": audio_payload(),
                                               "sampling_rate": 16000}], "params": {"max_new_tokens": 1}})
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["validation"]["task"]["status"], "verified")
            self.assertIn("limit 1", result["response"]["texts"][0])

    def test_modalities_match_official_pipeline_and_runtime_validation(self):
        import numpy as np
        import torch
        from PIL import Image

        from acprof.container.handlers.multimodal import MultimodalHandler
        from acprof.container.runtime_validate import validate
        from acprof.model_spec import encode_model_spec

        device = os.getenv("ACPROF_TEST_DEVICE", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            self.fail("CUDA runtime verification requested but unavailable")
        for task in ("audio-text-to-text", "image-text-to-text", "video-text-to-text"):
            with self.subTest(task=task), tempfile.TemporaryDirectory(prefix="acprof_custom_") as directory:
                root = Path(directory)
                self.snapshot(root)
                spec = pipeline_spec()
                spec.pop("dependencies")
                spec["task"] = task
                sample = {"text": "What is said?"}
                if task == "audio-text-to-text":
                    sample.update(audio_base64=audio_payload(), sampling_rate=16000)
                elif task == "image-text-to-text":
                    sample["image_base64"] = image_payload()
                    spec["multimodal"]["inputs"] = {"picture": "image", "question": "text"}
                else:
                    sample.update(video_frames_base64=[image_payload(), image_payload()], fps=2.0)
                    spec["multimodal"]["inputs"] = {"frames": "video", "fps": "fps", "question": "text"}
                payload = {"samples": [sample], "params": {"max_new_tokens": 4}}
                environment = {"ACPROF_MODEL_SPEC_B64": encode_model_spec(spec), "MODEL_LOCAL_PATH": directory,
                               "MODEL_ID": "fixture/unseen", "MODEL_REVISION": "a" * 40,
                               "TASK_TYPE": task, "TASK_FAMILY": "multimodal", "RUNTIME_BACKEND": "transformers_pipeline",
                               "USE_GPU": "1" if device == "cuda" else "0", "TORCH_NUM_THREADS": "1",
                               "ACPROF_MODEL_ADAPTER": "family-default", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                with patch.dict(os.environ, environment):
                    handler = MultimodalHandler()
                    context = handler.load(directory, task, "transformers_pipeline", device)
                    self.assertEqual(type(context["pipeline"].model).__name__, "AudioModel")
                    processed = handler.preprocess(context, payload)
                    self.assertEqual(context["pipeline"].events, ["preprocess"])
                    first = handler.postprocess(context, handler.predict(context, processed))
                    second = handler.postprocess(context, handler.predict(context, processed))
                    self.assertEqual(first, second)
                    self.assertEqual(context["pipeline"].events, ["preprocess", "predict", "postprocess", "predict", "postprocess"])
                    self.assertIn("limit 4", first["texts"][0])
                    direct_payload = {"question": sample["text"]}
                    if task == "audio-text-to-text":
                        direct_payload.update(waveform=np.zeros(160, dtype=np.float32), rate=16000)
                    elif task == "image-text-to-text":
                        direct_payload["picture"] = Image.new("RGB", (8, 8), "white")
                    else:
                        direct_payload.update(frames=np.full((2, 8, 8, 3), 255, dtype=np.uint8), fps=2.0)
                    expected = context["pipeline"](direct_payload, limit=4, temperature=0.0)
                    self.assertEqual(first["texts"], [expected])
                    # Real operators remain visible to the existing Torch profiling path.
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities) as profiler:
                        handler.predict(context, processed)
                    self.assertTrue(any(event.key.startswith("aten::") for event in profiler.key_averages()))
                    if device == "cuda":
                        self.assertTrue(any(event.device_type == torch.autograd.DeviceType.CUDA for event in profiler.events()))
                    report = validate(payload)
                    self.assertEqual(report["status"], "ok")
                    self.assertEqual(report["validation"]["task"]["status"], "verified")
                    self.assertTrue(all(stage["status"] == "verified" for stage in report["stages"]))


@unittest.skipUnless(importlib.util.find_spec("transformers"), "requires the custom multimodal container")
class LocalPipelineDependencyRuntimeTests(unittest.TestCase):
    @staticmethod
    def snapshot(root: Path):
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"custom_pipelines": {"listen-and-answer": {
            "impl": "entry.FixturePipeline", "pt": ["AutoModel"], "type": "multimodal"}}}))
        (root / "entry.py").write_text("from .bridge import FixturePipeline\n")
        (root / "bridge.py").write_text("from .leaf import FixturePipeline\n")
        (root / "leaf.py").write_text(
            "class FixturePipeline:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        raise AssertionError('basic probe must not instantiate a pipeline')\n"
            "    def preprocess(self, inputs):\n"
            "        raise AssertionError('basic probe must not preprocess')\n"
            "    def _forward(self, inputs, limit, temperature):\n"
            "        raise AssertionError('basic probe must not infer')\n"
            "    def postprocess(self, outputs):\n"
            "        raise AssertionError('basic probe must not postprocess')\n")

    @staticmethod
    def probe(root: Path, cache: Path, *, native: bool = False):
        from acprof.model_spec import encode_model_spec

        spec = pipeline_spec()
        spec.pop("dependencies")
        environment = {**os.environ, "ACPROF_MODEL_SPEC_B64": encode_model_spec(spec),
                       "MODEL_LOCAL_PATH": str(root), "MODEL_ID": "fixture/import-only",
                       "TASK_TYPE": "audio-text-to-text", "HF_MODULES_CACHE": str(cache),
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        script = "import json; from acprof.container.model_probe import validate_basic; "
        if native:
            # Acceptance of a candidate happens before publishing capability=true.
            # Only the declaration is replaced; the installed upstream loader runs
            # unmodified, and importing the AC-Prof copy shim is forbidden.
            script += (
                "import sys; from unittest.mock import patch; "
                "patch.dict(sys.modules, {'acprof.container.compat.transformers_dynamic': None}).start(); "
                "patch('acprof.container.local_pipeline.transformers_capabilities', return_value={"
                "'local_dynamic_transitive_imports': True, 'local_dynamic_symlink_safe': True}).start(); "
            )
        return subprocess.run([sys.executable, "-c", script + "print(json.dumps(validate_basic({})))"],
            env=environment, capture_output=True, text=True, timeout=60)

    def test_basic_probe_loads_transitive_imports_from_empty_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root, cache = Path(directory) / "model-snapshot", Path(directory) / "cache"
            self.snapshot(root)
            result = self.probe(root, cache)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["inference"], "not_run")
            self.assertEqual(report["preprocess"], "not_run")
            self.assertEqual({item["stage"] for item in report["stages"]}, {"import", "signature"})
            self.assertEqual(list(root.glob("__pycache__")), [])

    def test_missing_transitive_import_fails_at_snapshot_before_entry_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root, cache = Path(directory) / "model-snapshot", Path(directory) / "cache"
            self.snapshot(root)
            (root / "leaf.py").unlink()
            marker = Path(directory) / "executed"
            (root / "entry.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\nfrom .bridge import FixturePipeline\n")
            result = self.probe(root, cache)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileNotFoundError", result.stderr)
            self.assertIn(str(root / "leaf.py"), result.stderr)
            self.assertFalse(marker.exists())

    def test_snapshot_symlinks_keep_relative_names_and_original_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, cache = base / "pinned-snapshot", base / "cache"
            self.snapshot(root)
            blobs = base / "blobs"
            blobs.mkdir()
            originals = {}
            for index, source in enumerate(sorted(root.glob("*.py"))):
                blob = blobs / str(index)
                source.rename(blob)
                originals[blob] = blob.read_bytes()
                source.symlink_to(blob)
            alias = base / "model-snapshot"
            alias.symlink_to(root, target_is_directory=True)
            result = self.probe(alias, cache)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual({path: path.read_bytes() for path in originals}, originals)
            self.assertTrue(all(path.is_symlink() for path in root.glob("*.py")))

    def test_circular_relative_imports_terminate(self):
        with tempfile.TemporaryDirectory() as directory:
            root, cache = Path(directory) / "model-snapshot", Path(directory) / "cache"
            self.snapshot(root)
            (root / "bridge.py").write_text("READY = True\nfrom .leaf import FixturePipeline\n")
            leaf = root / "leaf.py"
            leaf.write_text("from .bridge import READY\nassert READY\n" + leaf.read_text())
            result = self.probe(root, cache)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_import_exception_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root, cache = Path(directory) / "model-snapshot", Path(directory) / "cache"
            self.snapshot(root)
            leaf = root / "leaf.py"
            leaf.write_text("raise RuntimeError('fixture dependency import failed')\n" + leaf.read_text())
            result = self.probe(root, cache)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("RuntimeError: fixture dependency import failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
