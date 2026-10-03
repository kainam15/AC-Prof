"""Shared tiny CV snapshots for separate CPU and CUDA runtime checks."""
import json
import tempfile
from pathlib import Path

import pytest

from acprof.container.handlers.cv import CVHandler
from acprof.workloads.cv import CVWorkloadGenerator


class CVRuntimeFixture:
    @pytest.fixture(scope="class", autouse=True)
    def _class_setup(self, request):
        import torch

        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(previous)

    def _exercise(self, model, processor, task, spec=None, device="cpu"):
        handler = CVHandler()
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            model.save_pretrained(snapshot)
            processor.save_pretrained(snapshot)
            context = handler.load(str(snapshot), task, "transformers_model", device)
            if task == "mask-generation":
                from acprof.container.load_policy import actual_dtype
                assert (actual_dtype(context)) == ("torch.float32")
            manifest = None
            if spec is not None:
                manifest = Path(directory) / "workload.json"
                manifest.write_text(json.dumps(spec))
            generator = CVWorkloadGenerator("local/tiny", task, 1,
                                            workload_spec_path=str(manifest) if manifest else None)
            processed = handler.preprocess(context, generator.generate(0.25))
            result = handler.predict(context, processed)
            return handler.postprocess(context, result)

    def _sam_mask_pipeline(self, device):
        from transformers import (
            SamConfig,
            SamImageProcessor,
            SamMaskDecoderConfig,
            SamModel,
            SamPromptEncoderConfig,
            SamVisionConfig,
        )

        model = SamModel(SamConfig(
            vision_config=SamVisionConfig(hidden_size=32, output_channels=32, num_hidden_layers=1,
                                          num_attention_heads=4, image_size=64, patch_size=16,
                                          global_attn_indexes=[0], num_pos_feats=16, mlp_dim=64),
            prompt_encoder_config=SamPromptEncoderConfig(hidden_size=32, image_size=64, patch_size=16),
            mask_decoder_config=SamMaskDecoderConfig(hidden_size=32, mlp_dim=64, num_hidden_layers=1,
                                                     num_attention_heads=4, iou_head_hidden_dim=32),
        ))
        processor = SamImageProcessor(size={"longest_edge": 64}, pad_size={"height": 64, "width": 64})
        result = self._exercise(model, processor, "mask-generation", {"params": {
            "points_per_batch": 4, "points_per_crop": 2, "pred_iou_thresh": 0.0,
            "stability_score_thresh": 0.0,
        }}, device=device)
        assert (result["output_type"]) == ("masks")
        assert isinstance(result["n_results"], int)
