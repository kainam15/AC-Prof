import sys
import types
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from acprof.container.handlers.cv import CVHandler
from acprof.host.detect import TaskInfo
from acprof.host.model_schema import _model_io_formats
from acprof.host.task_support import require_task_support
from acprof.workloads.cv import CVWorkloadGenerator


class TestCVCaption:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.handler = CVHandler()
        self.tokenizer = Mock()
        self.tokenizer.encode.side_effect = lambda text, **kw: {
            "a cat": [10, 20], "蓝色的猫": [30, 31, 32], "": [],
        }[text]
        self.pipe = Mock(tokenizer=self.tokenizer)
        self.ctx = {"pipeline": self.pipe, "task_type": "image-to-text"}

    def test_caption_text_and_counts_describe_all_returned_sequences(self):
        result = self.handler.postprocess(self.ctx, [
            {"generated_text": "a cat"}, {"generated_text": "蓝色的猫"},
        ])
        assert (result) == ({
            "task": "image-to-text", "output_type": "caption",
            "captions": ["a cat", "蓝色的猫"], "n_results": 2,
            "output_length": 9, "output_token_count": 5,
        })
        self.tokenizer.encode.assert_any_call("a cat", add_special_tokens=False)

    def test_empty_generated_text_is_a_valid_zero_length_caption(self):
        result = self.handler.postprocess(self.ctx, {"generated_text": ""})
        assert (result["captions"]) == ([""])
        assert (result["output_length"]) == (0)
        assert (result["output_token_count"]) == (0)

    @pytest.mark.parametrize('tokenizer_case', range(2), ids=['None', "Mock(encode=Mock(side_effect=ValueError('no tokenizer')))"])
    def test_unavailable_token_count_is_null_without_losing_caption(self, tokenizer_case):
        tokenizer = tuple((None, Mock(encode=Mock(side_effect=ValueError('no tokenizer')))))[tokenizer_case]
        self.pipe.tokenizer = tokenizer
        result = self.handler.postprocess(self.ctx, [{"generated_text": "a cat"}])
        assert (result["captions"]) == (["a cat"])
        assert (result["output_length"]) == (5)
        assert (result["output_token_count"]) is None

    def test_callable_tokenizer_and_partial_count_failure(self):
        self.pipe.tokenizer = lambda text, **kw: {"input_ids": [10, 20]}
        result = self.handler.postprocess(self.ctx, [{"generated_text": "a cat"}])
        assert (result["output_token_count"]) == (2)
        self.pipe.tokenizer = self.tokenizer
        self.tokenizer.encode.side_effect = [[10, 20], ValueError("second caption failed")]
        result = self.handler.postprocess(self.ctx, [
            {"generated_text": "a cat"}, {"generated_text": "蓝色的猫"},
        ])
        assert (result["output_token_count"]) is None
        assert (result["output_length"]) == (9)

    @pytest.mark.parametrize('output', (None, [], 'a cat', [{'label': 'cat'}], [{'generated_text': None}]))
    def test_malformed_caption_output_fails_instead_of_reporting_detection(self, output):
        with pytest.raises(ValueError, match="caption"):
            self.handler.postprocess(self.ctx, output)

    def test_generation_parameters_reach_pipeline(self):
        params = {"max_new_tokens": 12, "generate_kwargs": {"do_sample": False}}
        processed = {"image": "decoded-image", "params": params}
        self.handler.predict(self.ctx, processed)
        self.pipe.assert_called_once_with("decoded-image", **params)
        assert (processed["params"]) == (params)

    def test_invalid_generation_parameters_are_rejected(self):
        with pytest.raises(ValueError, match="params must be an object"):
            self.handler.predict(self.ctx, {"image": "image", "params": []})

    def test_other_cv_tasks_keep_their_output_and_call_contract(self):
        for task, output_type in (("image-classification", "classification"), ("object-detection", "detection")):
            ctx = {"pipeline": self.pipe, "task_type": task}
            result = self.handler.postprocess(ctx, [{"label": "cat", "score": 0.9}])
            assert (result) == ({"task": task, "output_type": output_type, "n_results": 1})
            self.handler.predict(ctx, {"image": "image", "params": {}})
            self.pipe.assert_called_with("image")

    def test_unknown_pipeline_error_is_preserved_without_string_patch(self):
        torch = types.ModuleType("torch")
        torch.float16, torch.float32 = "fp16", "fp32"
        transformers = types.ModuleType("transformers")
        transformers.__version__ = "4.57.6"
        transformers.pipeline = Mock(side_effect=KeyError("Unknown task image-to-text, available tasks are []"))
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            with pytest.raises(KeyError, match="Unknown task image-to-text"):
                self.handler.load("example/model", "image-to-text", "transformers_pipeline", "cpu")
            transformers.pipeline.side_effect = KeyError("broken_model_config")
            with pytest.raises(KeyError, match="broken_model_config"):
                self.handler.load("example/model", "image-to-text", "transformers_pipeline", "cpu")

    def test_caption_task_and_output_schema_are_available(self):
        info = TaskInfo(
            model_id="example/caption-model", pipeline_tag="image-to-text",
            task_family="cv", runtime_backend="transformers_pipeline",
            library_name="transformers", model_revision="revision", detection_method="unit",
        )
        require_task_support(info)
        _, output = _model_io_formats(info)
        properties = output["json_schema"]["properties"]
        assert (properties["output_type"]["enum"]) == (["caption"])
        assert (properties["captions"]) == ({"type": "array", "items": {"type": "string"}})
        assert (properties["output_token_count"]) == ({"type": ["integer", "null"]})
        assert ("output_length") in (output["json_schema"]["required"])

    def test_caption_workload_does_not_silently_ignore_batch_size(self):
        with pytest.raises(ValueError, match="batch_size=1"):
            CVWorkloadGenerator("example/model", "image-to-text", 2)
        generator = CVWorkloadGenerator("example/model", "image-to-text", 1)
        payload = generator.generate(0.5)
        assert (payload) == (generator.generate(0.5))
        assert ("image_base64") in (payload)


class TestCVExtendedHandler:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.handler = CVHandler()

    def test_zero_shot_tasks_forward_candidates_and_parameters(self):
        for task in ("zero-shot-image-classification", "zero-shot-object-detection"):
            pipe = Mock()
            self.handler.predict({"pipeline": pipe, "task_type": task}, {
                "image": "decoded", "candidate_labels": ["cat", "person"],
                "params": {"threshold": 0.2},
            })
            pipe.assert_called_once_with("decoded", candidate_labels=["cat", "person"], threshold=0.2)

    def test_zero_shot_missing_labels_are_rejected(self):
        with pytest.raises(ValueError, match="candidate_labels"):
            self.handler.predict({"pipeline": Mock(), "task_type": "zero-shot-object-detection"},
                                 {"image": "decoded", "params": {}})

    @pytest.mark.parametrize('payload', ({}, {'image_base64': 'garbage'}))
    def test_missing_or_invalid_images_do_not_turn_into_dummy_measurements(self, payload):
        with pytest.raises(ValueError):
            self.handler.preprocess({"task_type": "image-classification"}, payload)

    @pytest.mark.parametrize('task_case', range(4))
    def test_summaries_match_depth_segmentation_masks_and_feature_tasks(self, task_case):
        samples = [
            ("depth-estimation", {"predicted_depth": np.zeros((1, 4, 5)), "depth": Image.new("L", (5, 4))}, "depth", 1),
            ("image-segmentation", [{"mask": Image.new("L", (5, 4))}] * 2, "segmentation", 2),
            ("mask-generation", {"masks": [np.zeros((4, 5))] * 3}, "masks", 3),
            ("image-feature-extraction", [[[1.0, 2.0], [3.0, 4.0]]], "features", 1),
        ]
        (task, output, output_type, count) = tuple(samples)[task_case]
        result = self.handler.postprocess({"task_type": task}, output)
        assert (result["output_type"]) == (output_type)
        assert (result["n_results"]) == (count)
        if task == "depth-estimation":
            assert (result["depth_shape"]) == ([1, 4, 5])
        if task == "image-feature-extraction":
            assert (result["feature_shape"]) == ([1, 2, 2])

    def test_video_frame_count_mismatch_fails_before_model_forward(self):
        generator = CVWorkloadGenerator("example/model", "video-classification", 1)
        ctx = {"task_type": "video-classification", "model": Mock(config=types.SimpleNamespace(num_frames=8))}
        with pytest.raises(ValueError, match="num_frames.*8|8.*num_frames"):
            self.handler.preprocess(ctx, generator.generate(0.05))

    def test_probe_metadata_preserves_multiplier_before_pixel_rounding(self):
        payload = CVWorkloadGenerator("example/model", "image-classification", 1).generate(0.3)
        processed = self.handler.preprocess({"task_type": "image-classification"}, payload)
        assert (processed["_effective_input_scale"]) == (0.3)
        assert not (processed["_truncated_by_limit"])
        payload["input_scale"] = 0.5
        with pytest.raises(ValueError, match="input_scale.*resolution|resolution.*input_scale"):
            self.handler.preprocess({"task_type": "image-classification"}, payload)

    def test_keypoint_summary_counts_valid_points_and_persons(self):
        ctx = {"task_type": "keypoint-detection", "processor": Mock(), "keypoint_kind": "vitpose"}
        ctx["processor"].post_process_pose_estimation.return_value = [[
            {"keypoints": np.ones((17, 2))}, {"keypoints": np.ones((17, 2))},
        ]]
        result = self.handler.postprocess(ctx, {"outputs": "raw", "boxes": [[[0, 0, 100, 100]]]})
        assert (result["output_type"]) == ("keypoints")
        assert (result["n_results"]) == (2)
        assert (result["keypoint_count"]) == (34)
