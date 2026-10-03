import copy
import hashlib
import tempfile
from pathlib import Path

import pytest

from acprof.container.download_model import verify_download
from acprof.container.model_files import ModelFilesError, plan_download, validate_plan


class TestModelFilePlan:
    def plan(self, names, metadata=None, **kwargs):
        metadata = {"config.json": {"model_type": "bert"}, **(metadata or {})}
        return plan_download(
            model_id="example/model", revision="a" * 40, family=kwargs.pop("family", "nlp"),
            backend=kwargs.pop("backend", "transformers_pipeline"),
            files={name: {"size": 1} for name in names}, read_json=metadata.__getitem__,
            native_model_types={"bert", "whisper", "t5"}, **kwargs,
        )

    def selected(self, plan):
        validate_plan(plan)
        return {entry["path"] for entry in plan["files"]}

    def test_safetensors_preserves_processor_resources_and_default_variant(self):
        plan = self.plan([
            "config.json", "model.safetensors", "model.fp16.safetensors", "pytorch_model.bin",
            "flax_model.msgpack", "tf_model.h5", "tokenizer.model", "tokenizer.json", "processor_config.json",
            "chat_template.jinja", "special_audio.bin", "LICENSE", "auxiliary/config.json",
        ])
        assert (self.selected(plan)) == ({
            "config.json", "model.safetensors", "tokenizer.model", "tokenizer.json", "processor_config.json",
            "chat_template.jinja", "special_audio.bin", "LICENSE", "auxiliary/config.json",
        })
        assert (plan["weights"][0]["variant"]) is None

    def test_bin_only_checkpoint_is_preserved(self):
        plan = self.plan(["config.json", "pytorch_model.bin", "flax_model.msgpack"])
        assert (self.selected(plan)) == ({"config.json", "pytorch_model.bin"})
        assert (plan["weights"][0]["format"]) == ("bin")

    def test_shard_index_selects_every_referenced_shard(self):
        index = "model.safetensors.index.json"
        shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        plan = self.plan(["config.json", index, *shards, "pytorch_model.bin"], {
            index: {"weight_map": {"a": shards[0], "b": shards[1], "c": shards[0]}},
        })
        assert (self.selected(plan)) == ({"config.json", index, *shards})

    def test_missing_safe_shard_does_not_silently_fall_back_to_bin(self):
        index = "model.safetensors.index.json"
        with pytest.raises(ModelFilesError, match="missing checkpoint shards"):
            self.plan(["config.json", index, "pytorch_model.bin"], {index: {"weight_map": {"a": "missing.safetensors"}}})

    @pytest.mark.parametrize('weight_map', ({}, {'a': '../outside.safetensors'}, {'a': 3}))
    def test_invalid_shard_paths_and_empty_index_are_rejected(self, weight_map):
        index = "model.safetensors.index.json"
        with pytest.raises(ModelFilesError):
            self.plan(["config.json", index], {index: {"weight_map": weight_map}})

    @pytest.mark.parametrize('config', ({'model_type': 'new_model'}, {'model_type': 'bert', 'auto_map': {'AutoModel': 'custom.Model'}}, {'model_type': 'bert', 'quantization_config': {'quant_method': 'gptq'}}, {'model_type': 'bert', 'custom_pipelines': {'custom': {'impl': 'custom.Pipeline'}}}, {'model_type': ['custom', 'Model']}))
    def test_unknown_custom_and_quantized_models_keep_full_snapshot(self, config):
        names = ["config.json", "model.safetensors", "pytorch_model.bin", "custom.py", "extra.bin"]
        plan = self.plan(names, {"config.json": config})
        assert (self.selected(plan)) == (set(names))
        assert (plan["effective_policy"]) == ("full")

    def test_full_policy_does_not_parse_model_configuration(self):
        names = ["config.json", "model.safetensors", "flax_model.msgpack"]
        plan = self.plan(names, {"config.json": None}, policy="full")
        assert (self.selected(plan)) == (set(names))
        assert (plan["reason"]) == ("explicit_full")

    def test_registered_custom_adapter_keeps_all_its_assets(self):
        names = ["config.json", "model.safetensors", "pytorch_model.bin", "processing.py"]
        plan = self.plan(names, adapter="moss-transcribe-diarize")
        assert (self.selected(plan)) == (set(names))
        assert (plan["reason"]) == ("custom_adapter")

    def test_sentence_transformers_keeps_dense_modules(self):
        names = ["modules.json", "0_Transformer/config.json", "0_Transformer/model.safetensors",
                 "0_Transformer/pytorch_model.bin", "1_Pooling/config.json", "2_Dense/pytorch_model.bin"]
        plan = self.plan(names, {
            "modules.json": [{"path": "0_Transformer", "type": "sentence_transformers.models.Transformer"},
                             {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                             {"path": "2_Dense", "type": "sentence_transformers.models.Dense"}],
            "0_Transformer/config.json": {"model_type": "bert"},
        }, backend="sentence_transformers")
        assert (self.selected(plan)) == (set(names) - {"0_Transformer/pytorch_model.bin"})

    def test_diffusers_selects_all_components_without_changing_precision_or_ema(self):
        names = ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
                 "unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.non_ema.bin",
                 "text_encoder/config.json", "text_encoder/pytorch_model.bin", "scheduler/scheduler_config.json",
                 "tokenizer/vocab.json", "v1-5-pruned.ckpt", "v1-5-pruned.safetensors"]
        plan = self.plan(names, {"model_index.json": {
            "_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"],
            "text_encoder": ["transformers", "CLIPTextModel"], "scheduler": ["diffusers", "PNDMScheduler"],
            "tokenizer": ["transformers", "CLIPTokenizer"], "safety_checker": [None, None],
        }}, family="diffusion", backend="diffusers")
        assert (self.selected(plan)) == (set(names) - {
            "unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.non_ema.bin",
            "v1-5-pruned.ckpt", "v1-5-pruned.safetensors",
        })
        assert (len(plan["weights"])) == (2)

    @pytest.mark.parametrize('pipeline', ('NewPipeline', ['custom', 'Pipeline']))
    def test_unknown_diffusers_pipeline_keeps_complete_repository(self, pipeline):
        names = ["model_index.json", "unet/diffusion_pytorch_model.safetensors", "extra.ckpt"]
        plan = self.plan(names, {"model_index.json": {"_class_name": pipeline}}, backend="diffusers")
        assert (self.selected(plan)) == (set(names))

    @pytest.mark.parametrize('pipeline,scheduler', (('DDPMPipeline', 'DDPMScheduler'), ('DDIMPipeline', 'DDIMScheduler')))
    def test_ddpm_root_components_preserve_selected_weights_and_scheduler(self, pipeline, scheduler):
        names = ["model_index.json", "config.json", "scheduler_config.json",
                 "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin",
                 "diffusion_pytorch_model.fp16.safetensors", "README.md"]
        plan = self.plan(names, {"model_index.json": {
            "_class_name": pipeline, "unet": ["diffusers", "UNet2DModel"],
            "scheduler": ["diffusers", scheduler],
        }}, family="diffusion", backend="diffusers")
        assert (plan["effective_policy"]) == ("selected")
        assert (self.selected(plan)) == ({
            "model_index.json", "config.json", "scheduler_config.json",
            "diffusion_pytorch_model.safetensors", "README.md",
        })
        assert (plan["weights"][0]["component"]) == (".")
        assert (plan["weights"][0]["files"]) == (["diffusion_pytorch_model.safetensors"])

    @pytest.mark.parametrize('missing', ('scheduler_config.json', 'config.json', 'diffusion_pytorch_model.safetensors'))
    def test_ddpm_missing_components_still_fail(self, missing):
        with pytest.raises(ModelFilesError):
            self.plan({"model_index.json", "config.json", "scheduler_config.json",
                       "diffusion_pytorch_model.safetensors"} - {missing}, {"model_index.json": {
                "_class_name": "DDPMPipeline", "unet": ["diffusers", "UNet2DModel"],
                "scheduler": ["diffusers", "DDPMScheduler"],
            }}, backend="diffusers")

    def test_diffusers_component_directory_takes_precedence_over_root(self):
        names = ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.bin",
                 "config.json", "diffusion_pytorch_model.safetensors", "scheduler_config.json"]
        plan = self.plan(names, {"model_index.json": {
            "_class_name": "DDPMPipeline", "unet": ["diffusers", "UNet2DModel"],
            "scheduler": ["diffusers", "DDPMScheduler"],
        }}, backend="diffusers")
        assert (plan["weights"][0]["component"]) == ("unet")
        assert ("unet/diffusion_pytorch_model.bin") in (self.selected(plan))

    @pytest.mark.parametrize('spec', ((['custom'], 'Model'), ('diffusers', {'custom': 'Model'}), ('custom',), {'custom': 'Model'}))
    def test_nonstandard_diffusers_component_keeps_complete_repository(self, spec):
        names = ["model_index.json", "unet/diffusion_pytorch_model.safetensors", "extra.ckpt"]
        plan = self.plan(names, {"model_index.json": {
            "_class_name": "StableDiffusionPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "custom": list(spec) if isinstance(spec, tuple) else spec,
        }}, backend="diffusers")
        assert (self.selected(plan)) == (set(names))
        assert (plan["effective_policy"]) == ("full")

    def test_structured_manifest_selects_exact_artifact(self):
        plan = self.plan(["acprof_model.json", "policy.pt", "training.pth", "README.md"], {
            "acprof_model.json": {"schema_version": 1, "format": "torchscript", "model_file": "policy.pt"},
        }, family="structured", backend="torchscript")
        assert (self.selected(plan)) == ({"acprof_model.json", "policy.pt", "README.md"})

    def test_plan_hash_detects_changed_selection(self):
        plan = self.plan(["config.json", "model.safetensors"])
        plan["files"].pop()
        with pytest.raises(ModelFilesError, match="hash mismatch"):
            validate_plan(plan)

    def test_verification_records_actual_hash_and_rejects_corrupt_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            content = b"weights"
            (Path(temporary) / "model.safetensors").write_bytes(content)
            plan = self.plan(["model.safetensors"], policy="full")
            plan["files"][0].update(size=len(content), lfs_sha256=hashlib.sha256(content).hexdigest())
            verified = verify_download(temporary, copy.deepcopy(plan))
            assert (verified["files"][0]["sha256"]) == (hashlib.sha256(content).hexdigest())
            validate_plan(verified)
            (Path(temporary) / "model.safetensors").write_bytes(b"garbage")
            with pytest.raises(ModelFilesError, match="SHA256 mismatch"):
                verify_download(temporary, plan)
