"""Request-wide image limits must fail before any expanded pixel allocation."""

import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from acprof.container.handlers.cv import CVHandler
from acprof.container.handlers.diffusion import _decode_images
from acprof.container.handlers.multimodal import MultimodalHandler
from acprof.image_inputs import decode_image_groups, load_image_groups
from acprof.input_budget import InputBudget, InputBudgetExceeded, positive_integer
from acprof.workloads.cv import CVWorkloadGenerator
from acprof.workloads.multimodal import MultimodalWorkloadGenerator


def png(size=(8, 8)):
    stream = io.BytesIO()
    Image.new("RGB", size, "red").save(stream, format="PNG")
    return stream.getvalue()


def encoded(size=(8, 8)):
    return base64.b64encode(png(size)).decode("ascii")


def forbid_conversion(monkeypatch):
    conversion = Mock(side_effect=AssertionError("pixels expanded before budget check"))
    monkeypatch.setattr(Image.Image, "convert", conversion)
    return conversion


@pytest.mark.parametrize("value", [2, 2.0, np.int64(2)])
def test_positive_integer_has_one_lossless_numeric_policy(value):
    assert positive_integer(value, "count") == 2


@pytest.mark.parametrize("value", [True, False, 0, -1, 2.5, "2", None, float("nan"), float("inf")])
def test_positive_integer_rejects_invalid_numbers(value):
    with pytest.raises(ValueError, match="count must be a positive integer"):
        positive_integer(value, "count")


def test_strict_integer_fields_preserve_type_requirement_and_large_integers():
    assert positive_integer(np.int64(2), "rate", allow_integral_float=False) == 2
    assert positive_integer(10 ** 1000, "count") == 10 ** 1000
    with pytest.raises(ValueError, match="rate"):
        positive_integer(2.0, "rate", allow_integral_float=False)


@pytest.mark.parametrize(("groups", "limit"), [
    ({"image": [encoded()]}, 63),
    ({"image": [encoded()], "video": [encoded()]}, 100),
    ({"video": [encoded(), encoded()]}, 127),
])
def test_request_pixels_rejected_before_conversion(groups, limit, monkeypatch):
    conversion = forbid_conversion(monkeypatch)
    with pytest.raises(InputBudgetExceeded, match="input_budget_exceeded: pixels"):
        decode_image_groups(groups, budget=InputBudget(max_pixels=limit))
    conversion.assert_not_called()


def test_request_batch_rejected_before_base64_decode(monkeypatch):
    decode = Mock(side_effect=AssertionError("decoded an over-budget batch"))
    monkeypatch.setattr(base64, "b64decode", decode)
    with pytest.raises(InputBudgetExceeded, match="batch_items=2"):
        decode_image_groups({"video": ["invalid", "invalid"]}, budget=InputBudget(max_batch_items=1))
    decode.assert_not_called()


def test_mixed_resolutions_at_exact_pixel_boundary_remain_usable():
    groups = decode_image_groups({"image": [encoded()], "video": [encoded((4, 4))]},
                                 budget=InputBudget(max_pixels=80))
    try:
        assert groups["image"][0].getpixel((0, 0)) == (255, 0, 0)
        assert groups["video"][0].size == (4, 4)
    finally:
        for images in groups.values():
            for image in images:
                image.close()


def test_video_dimensions_checked_before_conversion(monkeypatch):
    frames = [encoded(), encoded((4, 4))]
    conversion = forbid_conversion(monkeypatch)
    with pytest.raises(ValueError, match="same resolution"):
        decode_image_groups({"video": frames}, budget=InputBudget(), same_size={"video"})
    conversion.assert_not_called()


def test_header_resources_closed_after_budget_failure(monkeypatch):
    data = png()
    original_open = Image.open
    opened = []

    def record_open(stream):
        source = original_open(stream)
        opened.append((stream, source))
        return source

    monkeypatch.setattr(Image, "open", record_open)
    with pytest.raises(InputBudgetExceeded):
        load_image_groups({"image": [data, data]}, budget=InputBudget(max_pixels=100))
    assert len(opened) == 2
    assert all(stream.closed and source.fp is None for stream, source in opened)


def test_already_converted_images_closed_when_later_decode_fails(monkeypatch):
    data = png()
    original_convert = Image.Image.convert
    converted = []

    def fail_second(source, mode):
        if converted:
            raise OSError("broken second image")
        image = original_convert(source, mode)
        converted.append(image)
        return image

    monkeypatch.setattr(Image.Image, "convert", fail_second)
    with pytest.raises(ValueError, match="valid image"):
        load_image_groups({"video": [data, data]}, budget=InputBudget())
    with pytest.raises(ValueError, match="closed"):
        converted[0].getpixel((0, 0))


def test_cv_video_counts_all_frames_before_processor(monkeypatch):
    frames = [encoded(), encoded()]
    conversion = forbid_conversion(monkeypatch)
    processor = Mock()
    ctx = {"task_type": "video-classification", "model": SimpleNamespace(config=SimpleNamespace(num_frames=2)),
           "processor": processor}
    with pytest.raises(InputBudgetExceeded, match="pixels=128"):
        CVHandler().preprocess(ctx, {"frames_base64": frames, "input_budget": {"max_pixels": 100}})
    conversion.assert_not_called()
    processor.assert_not_called()


def test_onnx_image_checks_source_and_resize_budgets(monkeypatch):
    with patch.dict("sys.modules", {"onnxruntime": SimpleNamespace()}):
        from acprof.container.handlers.onnx_tasks import ONNXImageClassificationHandler

    image = encoded()
    conversion = forbid_conversion(monkeypatch)
    handler = ONNXImageClassificationHandler()
    for config, limit in [({"mode": "RGB"}, 63),
                          ({"mode": "RGB", "resize": {"width": 20, "height": 20}}, 100)]:
        with pytest.raises(InputBudgetExceeded, match="pixels"):
            handler.preprocess({"image_processing": config},
                               {"image_base64": image, "input_budget": {"max_pixels": limit}})
    conversion.assert_not_called()


def test_multimodal_image_and_video_share_request_budget(monkeypatch):
    image, frame = encoded(), encoded()
    conversion = forbid_conversion(monkeypatch)
    raw = {"samples": [{"text": "Describe", "image_base64": image,
                        "video_frames_base64": [frame], "fps": 2}],
           "input_budget": {"max_pixels": 100}}
    with pytest.raises(InputBudgetExceeded, match="pixels=128"):
        MultimodalHandler().preprocess({"task_type": "any-to-any"}, raw)
    conversion.assert_not_called()


def test_diffusion_source_video_budget_cannot_be_bypassed_by_resize(monkeypatch):
    frames = [encoded(), encoded()]
    conversion = forbid_conversion(monkeypatch)
    with pytest.raises(InputBudgetExceeded, match="pixels=128"):
        _decode_images(frames, "frames_base64", 4, budget=InputBudget(max_pixels=100))
    conversion.assert_not_called()


def test_cv_generator_checks_output_before_synthetic_allocation(monkeypatch):
    generator = CVWorkloadGenerator("model", "video-classification", 1)
    allocate = Mock(side_effect=AssertionError("allocated synthetic image"))
    monkeypatch.setattr(Image, "new", allocate)
    with pytest.raises(InputBudgetExceeded, match="pixels"):
        generator.generate(100)
    allocate.assert_not_called()


@pytest.mark.parametrize("family", ["cv", "multimodal"])
def test_explicit_budget_travels_with_payload_and_plan(tmp_path, family):
    asset = tmp_path / "image.png"
    asset.write_bytes(png())
    spec = tmp_path / "workload.json"
    spec.write_text(json.dumps({"schema_version": 1, "image_path": str(asset),
                                "input_budget": {"max_pixels": 100}}))
    if family == "cv":
        generator = CVWorkloadGenerator("model", "image-classification", 1, str(spec))
        raw = generator.generate(8 / 224)
        processed = CVHandler().preprocess({"task_type": "image-classification"}, raw)
        assert processed["image"].size == (8, 8)
    else:
        generator = MultimodalWorkloadGenerator("model", "image-text-to-text", 1, str(spec))
        raw = generator.generate(8)
        assert len(raw["samples"]) == 1
    assert raw["input_budget"] == generator.plan_metadata()["input_budget"]
    assert raw["input_budget"]["max_pixels"] == 100
    with pytest.raises(InputBudgetExceeded):
        generator.generate(1 if family == "cv" else 11)


def test_multimodal_generator_bounds_all_visual_outputs_before_resize(tmp_path, monkeypatch):
    asset = tmp_path / "image.png"
    asset.write_bytes(png())
    spec = tmp_path / "workload.json"
    spec.write_text(json.dumps({"schema_version": 1, "modalities": ["image", "video"],
        "scale_modality": "image", "image_path": str(asset), "video_frames": [str(asset)],
        "video_num_frames": 1, "image_resolution": 8, "input_budget": {"max_pixels": 128}}))
    generator = MultimodalWorkloadGenerator("model", "any-to-any", 1, str(spec))
    resize = Mock(side_effect=AssertionError("resized before aggregate budget"))
    monkeypatch.setattr(Image.Image, "resize", resize)
    with pytest.raises(InputBudgetExceeded, match="pixels=145"):
        generator.generate(9)
    resize.assert_not_called()


def test_host_assets_share_budget_before_any_conversion(tmp_path, monkeypatch):
    asset = tmp_path / "image.png"
    asset.write_bytes(png())
    spec = tmp_path / "workload.json"
    spec.write_text(json.dumps({"schema_version": 1, "modalities": ["image", "video"],
        "scale_modality": "image", "image_path": str(asset), "video_frames": [str(asset)],
        "input_budget": {"max_pixels": 100}}))
    conversion = forbid_conversion(monkeypatch)
    with pytest.raises(InputBudgetExceeded, match="pixels=128"):
        MultimodalWorkloadGenerator("model", "any-to-any", 1, str(spec))
    conversion.assert_not_called()

def test_decoder_pixel_limit_is_an_explicit_input_error(monkeypatch):
    data = png()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    conversion = forbid_conversion(monkeypatch)
    with pytest.raises(ValueError, match="decoder pixel limit") as failure:
        load_image_groups({"image": [data]}, budget=InputBudget())
    assert isinstance(failure.value.__cause__, Image.DecompressionBombError)
    conversion.assert_not_called()


def test_onnx_grayscale_mode_retains_source_dimensions():
    image = decode_image_groups({"image": [encoded()]}, budget=InputBudget(), mode="L")["image"][0]
    try:
        assert image.mode == "L"
        assert image.size == (8, 8)
        assert image.getpixel((0, 0)) == 76
    finally:
        image.close()


def test_multimodal_synthetic_video_rejected_before_allocation(monkeypatch):
    generator = MultimodalWorkloadGenerator("model", "video-text-to-text", 1)
    allocate = Mock(side_effect=AssertionError("allocated synthetic video"))
    monkeypatch.setattr(Image, "new", allocate)
    with pytest.raises(InputBudgetExceeded):
        generator.generate(1_000_000_000)
    allocate.assert_not_called()
