import base64
import io
import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from acprof.workloads.cv import CVWorkloadGenerator


@pytest.mark.parametrize('task', ('image-classification', 'mask-generation', 'video-classification'))
def test_every_task_rejects_unimplemented_batching(task):
    with pytest.raises(ValueError, match="batch_size=1"):
        CVWorkloadGenerator("example/model", task, 2)

def test_zero_shot_labels_are_present_and_recorded():
    generator = CVWorkloadGenerator("example/model", "zero-shot-object-detection", 1)
    payload = generator.generate(0.25)
    assert (payload["candidate_labels"]) == (["cat", "dog", "car", "person"])
    assert (generator.plan_metadata()["candidate_labels"]) == (payload["candidate_labels"])

def test_video_uses_distinct_reproducible_frames_at_legacy_resolution():
    generator = CVWorkloadGenerator("example/model", "video-classification", 1)
    payload = generator.generate(0.125)
    assert (payload) == (generator.generate(0.125))
    assert (len(payload["frames_base64"])) == (16)
    assert (len(set(payload["frames_base64"]))) > (1)
    with Image.open(io.BytesIO(base64.b64decode(payload["frames_base64"][0]))) as frame:
        assert (frame.size) == ((28, 28))
    metadata = generator.input_metadata(0.125, payload)
    assert (metadata["num_frames"]) == (16)
    assert (len(metadata["frame_sha256"])) == (16)
    assert (generator.default_input_scales()) is None

def test_spec_tracks_local_assets_and_scales_normalized_pose_boxes():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        Image.new("RGB", (12, 8), "red").save(path / "source.png")
        (path / "spec.json").write_text(json.dumps({
            "schema_version": 1, "image_path": "source.png",
            "input_scales": [0.5, 1], "boxes": [[0.25, 0.125, 0.5, 0.75]],
            "params": {"dataset_index": 0},
        }))
        generator = CVWorkloadGenerator("example/model", "keypoint-detection", 1,
                                        workload_spec_path=str(path / "spec.json"))
        payload = generator.generate(0.5)
        assert (payload["boxes"]) == ([[28, 14, 56, 84]])
        assert (generator.default_input_scales()) == ([0.5, 1])
        assert (generator.plan_metadata()["params"]) == ({"dataset_index": 0})
        assert (len(generator.plan_metadata()["source_sha256"][0])) == (64)

def test_video_manifest_never_resamples_or_drops_frames():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        for index, color in enumerate(("red", "blue")):
            Image.new("RGB", (8, 8), color).save(path / f"{index}.png")
        (path / "spec.json").write_text(json.dumps({"video_frames": ["0.png", "1.png"]}))
        generator = CVWorkloadGenerator("example/model", "video-classification", 1,
                                        workload_spec_path=str(path / "spec.json"))
        assert (len(generator.generate(0.1)["frames_base64"])) == (2)
        (path / "spec.json").write_text(json.dumps({
            "video_frames": ["0.png", "1.png"], "num_frames": 3,
        }))
        with pytest.raises(ValueError, match="num_frames"):
            CVWorkloadGenerator("example/model", "video-classification", 1,
                                workload_spec_path=str(path / "spec.json"))

@pytest.mark.parametrize('scale_case', range(5), ids=['0', '-1', "float('nan')", "float('inf')", 'True'])
def test_bad_scales_are_rejected(scale_case):
    generator = CVWorkloadGenerator("example/model", "image-classification", 1)
    scale = tuple((0, -1, float('nan'), float('inf'), True))[scale_case]
    with pytest.raises(ValueError):
        generator.generate(scale)

@pytest.mark.parametrize('spec', ({'typo': 1}, {'candidate_labels': []}, {'params': {'batch_size': 3}}, {'video_frames': ['missing.png']}, {'input_scales': [0.5, 0.5]}))
def test_spec_rejects_unsupported_or_mismatched_inputs(spec):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spec.json"
        path.write_text(json.dumps(spec))
        with pytest.raises(ValueError):
            CVWorkloadGenerator("example/model", "zero-shot-image-classification", 1,
                                workload_spec_path=str(path))
