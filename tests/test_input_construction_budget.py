import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.container.handlers.diffusion import DiffusionHandler
from acprof.container.handlers.nlp import NLPHandler
from acprof.failures import failure_from_exception
from acprof.input_budget import InputBudget, InputBudgetExceeded, check_nlp_input
from acprof.workloads.diffusion import DiffusionWorkloadGenerator
from acprof.workloads.nlp import NLPWorkloadGenerator
from tests.test_nlp_handler import FakeTokenizer


def test_nlp_rejects_batch_before_tokenization():
    tokenizer = FakeTokenizer("[MASK]")
    tokenizer.encode = Mock(wraps=tokenizer.encode)
    pipe = SimpleNamespace(tokenizer=tokenizer)
    with pytest.raises(ValueError, match="input_budget_exceeded"):
        NLPHandler().preprocess({"task_type": "fill-mask", "pipeline": pipe},
                                {"text": "hello [MASK]", "batch_size": 10**9})
    pipe.tokenizer.encode.assert_not_called()


@pytest.mark.parametrize("task", ["sentence-similarity", "text-ranking", "zero-shot-classification"])
def test_nlp_joint_candidate_batch_budget_is_checked_before_expansion(task):
    tokenizer = FakeTokenizer("[MASK]")
    tokenizer.encode = Mock(wraps=tokenizer.encode)
    pipe = SimpleNamespace(tokenizer=tokenizer)
    payload = {"text": "hello", "query": "q", "documents": ["a", "b"],
               "candidate_labels": ["a", "b"], "batch_size": 3,
               "input_budget": {"max_batch_items": 4}}
    with pytest.raises(ValueError, match="input_budget_exceeded"):
        NLPHandler().preprocess({"task_type": task, "pipeline": pipe}, payload)
    pipe.tokenizer.encode.assert_not_called()


def test_qa_empty_context_fails_in_preprocess():
    tokenizer = FakeTokenizer("[MASK]")
    with pytest.raises(ValueError, match="context.*token|token.*context"):
        NLPHandler().preprocess({"task_type": "question-answering",
                                 "pipeline": SimpleNamespace(tokenizer=tokenizer)},
                                {"question": "who", "context": ""})


def test_diffusion_host_rejects_pixels_before_resize():
    generator = DiffusionWorkloadGenerator("fixture", "image-to-image", 1)
    with patch("PIL.Image.Image.resize", side_effect=AssertionError("must reject before resize")):
        with pytest.raises(ValueError, match="input_budget_exceeded"):
            generator.generate(65536)


def test_diffusion_container_rejects_joint_batch_pixels():
    with pytest.raises(ValueError, match="input_budget_exceeded"):
        DiffusionHandler().preprocess({"task_type": "text-to-image"},
                                     {"prompt": ["a", "b"], "resolution": 8192})


def test_explicit_budget_roundtrips_host_plan_and_container(tmp_path):
    path = tmp_path / "workload.json"
    path.write_text(json.dumps({"schema_version": 1, "input_budget": {"max_batch_items": 5000}}))
    generator = NLPWorkloadGenerator("fixture", "fill-mask", 4097, str(path))
    payload = generator.generate(8)
    assert payload["input_budget"] == generator.plan_metadata()["input_budget"]
    result = NLPHandler().preprocess({"task_type": "fill-mask", "pipeline": SimpleNamespace(
        tokenizer=FakeTokenizer("[MASK]"))}, payload)
    assert result["batch_size"] == 4097


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "100", float("inf")])
def test_budget_override_requires_positive_integer(value):
    with pytest.raises(ValueError, match="positive integers"):
        InputBudget.from_value({"max_pixels": value})


def test_joint_text_budget_checks_expanded_content_and_is_not_oom():
    raw = {"text": "abcde", "batch_size": 2, "input_budget": {"max_text_characters": 9}}
    with pytest.raises(InputBudgetExceeded) as caught:
        check_nlp_input(raw, "feature-extraction")
    failure = failure_from_exception(caught.value, stage="preprocess")
    assert failure.reason_code == "input_budget_exceeded"
    assert failure.retryability == "after_configuration"
    raw["input_budget"]["max_text_characters"] = 10
    check_nlp_input(raw, "feature-extraction")


def test_huge_host_text_is_rejected_before_building_word_list():
    generator = NLPWorkloadGenerator("fixture", "fill-mask", 1)
    with pytest.raises(InputBudgetExceeded):
        generator.generate(10**10)


def test_video_frame_budget_is_checked_before_synthetic_frames(tmp_path):
    path = tmp_path / "video.json"
    path.write_text(json.dumps({"schema_version": 1, "params": {"num_frames": 10**9}}))
    with patch("PIL.Image.new", side_effect=AssertionError("must reject before frame allocation")):
        with pytest.raises(InputBudgetExceeded):
            DiffusionWorkloadGenerator("fixture", "video-to-video", 1, str(path))


def test_diffusion_pixel_override_is_frozen_and_honored_on_both_sides(tmp_path):
    path = tmp_path / "diffusion.json"
    path.write_text(json.dumps({"schema_version": 1, "input_budget": {"max_pixels": 2 * 8192**2}}))
    generator = DiffusionWorkloadGenerator("fixture", "text-to-image", 2, str(path))
    payload = generator.generate(8192)
    assert payload["input_budget"] == generator.plan_metadata()["input_budget"]
    assert DiffusionHandler().preprocess({"task_type": "text-to-image"}, payload)["resolution"] == 8192
