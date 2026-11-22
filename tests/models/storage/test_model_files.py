import copy
import hashlib
import tempfile
from pathlib import Path

import pytest

from acprof.container.download_model import verify_download
from acprof.models.download_plan import plan_download, seal_plan, validate_plan
from acprof.models.files import ModelFilesError


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

    def test_native_cross_encoder_keeps_torch_checkpoint_and_tokenizer(self):
        keep = {"model.safetensors", "config.json", "tokenizer.json", "vocab.txt",
                "special_tokens_map.json", "openvino/other.bin", "README.md"}
        exports = {"pytorch_model.bin", "flax_model.msgpack", "onnx/model.onnx",
                   "openvino/openvino_model.bin", "openvino/openvino_model.xml",
                   "openvino/openvino_model_qint8_quantized.bin", "openvino/openvino_model_qint8_quantized.xml"}
        plan = self.plan(keep | exports, backend="cross_encoder")
        assert self.selected(plan) == keep
        assert plan["weights"][0]["files"] == ["model.safetensors"]
        assert plan["selected_bytes"] == len(keep)

    @pytest.mark.parametrize("options", [{"policy": "full"}, {"adapter": "different"}])
    def test_explicit_cross_encoder_snapshot_or_adapter_keeps_all_files(self, options):
        names = {"model.safetensors", "pytorch_model.bin", "config.json", "onnx/model.onnx"}
        plan = self.plan(names, backend="cross_encoder", **options)
        assert self.selected(plan) == names
        assert plan["effective_policy"] == "full"

    def test_modular_cross_encoder_preserves_declared_component_weights(self):
        names = {"config.json", "model.safetensors", "pytorch_model.bin", "modules.json",
                 "1_Dense/config.json", "1_Dense/pytorch_model.bin"}
        plan = self.plan(names, {"modules.json": [{"idx": 0, "path": "1_Dense",
                         "type": "sentence_transformers.base.modules.dense.Dense"}]}, backend="cross_encoder")
        assert self.selected(plan) == names
        assert plan["effective_policy"] == "full"

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

    def test_native_seq2seq_excludes_onnx_and_rust_exports(self):
        plan = self.plan(["config.json", "model.safetensors", "spiece.model", "tokenizer.json",
                          "onnx/encoder_model.onnx", "onnx/decoder_model.onnx", "rust_model.ot"],
                         {"config.json": {"model_type": "t5"}}, backend="transformers_model")
        assert self.selected(plan) == {"config.json", "model.safetensors", "spiece.model", "tokenizer.json"}

    def test_shard_index_selects_every_referenced_shard(self):
        index = "model.safetensors.index.json"
        shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        plan = self.plan(["config.json", index, *shards, "pytorch_model.bin"], {
            index: {"weight_map": {"a": shards[0], "b": shards[1], "c": shards[0]}},
        })
        assert (self.selected(plan)) == ({"config.json", index, *shards})

    @pytest.mark.parametrize("backend", ["transformers_pipeline", "cross_encoder"])
    def test_missing_safe_shard_does_not_silently_fall_back_to_bin(self, backend):
        index = "model.safetensors.index.json"
        with pytest.raises(ModelFilesError, match="missing checkpoint shards"):
            self.plan(["config.json", index, "pytorch_model.bin"],
                      {index: {"weight_map": {"a": "missing.safetensors"}}}, backend=backend)

    @pytest.mark.parametrize('weight_map', [{}, {'a': '../outside.safetensors'}, {'a': 3}])
    def test_invalid_shard_paths_and_empty_index_are_rejected(self, weight_map):
        index = "model.safetensors.index.json"
        with pytest.raises(ModelFilesError):
            self.plan(["config.json", index], {index: {"weight_map": weight_map}})

    @pytest.mark.parametrize('config', [{'model_type': 'new_model'}, {'model_type': 'bert', 'auto_map': {'AutoModel': 'custom.Model'}}, {'model_type': 'bert', 'quantization_config': {'quant_method': 'gptq'}}, {'model_type': 'bert', 'custom_pipelines': {'custom': {'impl': 'custom.Pipeline'}}}, {'model_type': ['custom', 'Model']}])
    @pytest.mark.parametrize("backend", ["transformers_pipeline", "cross_encoder"])
    def test_unknown_custom_and_quantized_models_keep_full_snapshot(self, config, backend):
        names = ["config.json", "model.safetensors", "pytorch_model.bin", "custom.py", "extra.bin"]
        plan = self.plan(names, {"config.json": config}, backend=backend)
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

    @pytest.mark.parametrize("backend", ["transformers_pipeline", "transformers_model", "chronos"])
    def test_root_loader_plans_root_weights_in_a_mixed_repository(self, backend):
        # The module chain's shard is missing. Its index must not be read for a root loader.
        names = {"config.json", "model.safetensors", "pytorch_model.bin", "modules.json",
                 "encoder/config.json", "encoder/model.safetensors.index.json", "1_Pooling/config.json"}
        plan = self.plan(names, backend=backend)
        assert plan["weights"] == [{
            "component": ".", "format": "safetensors", "variant": None, "files": ["model.safetensors"],
        }]
        assert self.selected(plan) == names - {"pytorch_model.bin"}

    @pytest.mark.parametrize("backend", ["transformers_pipeline", "sentence_transformers"])
    def test_nested_encoder_cannot_substitute_for_a_missing_root_shard(self, backend):
        names = {"config.json", "model.safetensors.index.json", "pytorch_model.bin", "modules.json",
                 "encoder/config.json", "encoder/model.safetensors", "encoder/pytorch_model.bin"}
        metadata = {
            "modules.json": [{"path": "encoder", "type": "sentence_transformers.models.Transformer"}],
            "encoder/config.json": {"model_type": "bert"},
            "model.safetensors.index.json": {"weight_map": {"layer.weight": "missing-root.safetensors"}},
        }
        if backend == "transformers_pipeline":
            with pytest.raises(ModelFilesError, match="missing checkpoint shards.*missing-root.safetensors"):
                self.plan(names, metadata, backend=backend)
        else:
            plan = self.plan(names, metadata, backend=backend)
            assert plan["weights"] == [{
                "component": "encoder", "format": "safetensors", "variant": None,
                "files": ["encoder/model.safetensors"],
            }]
            assert self.selected(plan) == names - {"encoder/pytorch_model.bin"}

    @pytest.mark.parametrize("safetensors", [False, True])
    def test_sentence_transformers_keeps_dense_modules(self, safetensors):
        names = ["modules.json", "0_Transformer/config.json", "0_Transformer/model.safetensors",
                 "0_Transformer/pytorch_model.bin", "1_Pooling/config.json", "2_Dense/config.json", "2_Dense/pytorch_model.bin"]
        if safetensors:
            names.append("2_Dense/model.safetensors")
        plan = self.plan(names, {
            "modules.json": [{"path": "0_Transformer", "type": "sentence_transformers.models.Transformer"},
                             {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                             {"path": "2_Dense", "type": "sentence_transformers.models.Dense"}],
            "0_Transformer/config.json": {"model_type": "bert"},
        }, backend="sentence_transformers")
        excluded = {"0_Transformer/pytorch_model.bin"}
        if safetensors:
            excluded.add("2_Dense/pytorch_model.bin")
        assert self.selected(plan) == set(names) - excluded
        assert plan["weights"][1]["files"] == [
            "2_Dense/model.safetensors" if safetensors else "2_Dense/pytorch_model.bin"]

    @pytest.mark.parametrize(("kind", "required_file", "weighted"), [
        ("Pooling", "config.json", False), ("WordWeights", "config.json", False),
        ("BoW", "config.json", False), ("WordEmbeddings", "wordembedding_config.json", True),
        ("StaticEmbedding", "tokenizer.json", True), ("WeightedLayerPooling", "config.json", True),
        ("LSTM", "lstm_config.json", True), ("CNN", "cnn_config.json", True),
    ])
    def test_sentence_module_requires_files_from_its_declared_directory(self, kind, required_file, weighted):
        standalone = kind in {"BoW", "WordEmbeddings", "StaticEmbedding"}
        modules = [] if standalone else [{"path": "", "type": "sentence_transformers.models.Transformer"}]
        modules.append({"path": "stage", "type": "sentence_transformers.models." + kind})
        metadata = {"modules.json": modules}
        names = {"modules.json", "config.json", "model.safetensors", "other/" + required_file,
                 "stage/other.json", "stage/pytorch_model.bin", "stage/model.safetensors.index.json"}
        with pytest.raises(ModelFilesError) as error:
            self.plan(names, metadata, backend="sentence_transformers")
        assert kind in str(error.value)
        assert "stage/" + required_file in str(error.value)

        names.add("stage/" + required_file)
        if weighted:
            with pytest.raises(ModelFilesError, match="stage/model.safetensors"):
                self.plan(names - {"stage/pytorch_model.bin"}, metadata, backend="sentence_transformers")
        plan = self.plan(names, metadata, backend="sentence_transformers")
        if standalone:
            assert plan["effective_policy"] == "full"
            assert self.selected(plan) == names
        elif weighted:
            assert plan["weights"][1]["files"] == ["stage/pytorch_model.bin"]
        else:
            assert [w["component"] for w in plan["weights"]] == ["."]

    def test_sentence_modules_accept_relative_directory_aliases_and_optional_configuration(self):
        names = {"modules.json", "config.json", "model.safetensors", "pool/config.json"}
        modules = [{"path": "", "type": "sentence_transformers.models.Transformer"},
                   {"path": "./pool//", "type": "sentence_transformers.models.Pooling"},
                   {"path": "dropout", "type": "sentence_transformers.models.Dropout"},
                   {"path": "normalize", "type": "sentence_transformers.models.Normalize"}]
        plan = self.plan(names, {"modules.json": modules}, backend="sentence_transformers")
        assert plan["effective_policy"] == "selected"
        assert self.selected(plan) == names

    @pytest.mark.parametrize("declaration", [{}, {"path": None}, {"path": "sub/../pool"}, {"path": "/pool"}])
    def test_sentence_module_requires_an_explicit_snapshot_relative_directory(self, declaration):
        names = {"modules.json", "config.json", "model.safetensors"}
        modules = [{"path": "", "type": "sentence_transformers.models.Transformer"},
                   {"type": "sentence_transformers.models.Pooling", **declaration}]
        with pytest.raises(ModelFilesError, match="invalid model file path"):
            self.plan(names, {"modules.json": modules}, backend="sentence_transformers")

    def test_custom_sentence_module_keeps_snapshot_even_if_pooling_directory_is_empty(self):
        names = {"modules.json", "config.json", "model.safetensors", "pytorch_model.bin"}
        modules = [{"path": "", "type": "sentence_transformers.models.Transformer"},
                   {"path": "pool", "type": "sentence_transformers.models.Pooling"},
                   {"path": "custom", "type": "vendor.Custom"}]
        plan = self.plan(names, {"modules.json": modules}, backend="sentence_transformers")
        assert (plan["effective_policy"], plan["reason"]) == ("full", "custom_sentence_transformers_module")
        assert self.selected(plan) == names

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

    @pytest.mark.parametrize('pipeline', ['NewPipeline', ['custom', 'Pipeline']])
    def test_unknown_diffusers_pipeline_keeps_complete_repository(self, pipeline):
        names = ["model_index.json", "unet/diffusion_pytorch_model.safetensors", "extra.ckpt"]
        plan = self.plan(names, {"model_index.json": {"_class_name": pipeline}}, backend="diffusers")
        assert (self.selected(plan)) == (set(names))

    @pytest.mark.parametrize(("pipeline", "scheduler"), [('DDPMPipeline', 'DDPMScheduler'), ('DDIMPipeline', 'DDIMScheduler')])
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

    @pytest.mark.parametrize('missing', ['scheduler_config.json', 'config.json', 'diffusion_pytorch_model.safetensors'])
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

    @pytest.mark.parametrize('spec', [(['custom'], 'Model'), ('diffusers', {'custom': 'Model'}), ('custom',), {'custom': 'Model'}])
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
            plan["selected_bytes"] = len(content)
            seal_plan(plan)
            verified = verify_download(temporary, copy.deepcopy(plan))
            assert (verified["files"][0]["sha256"]) == (hashlib.sha256(content).hexdigest())
            validate_plan(verified)
            (Path(temporary) / "model.safetensors").write_bytes(b"garbage")
            with pytest.raises(ModelFilesError, match="SHA256 mismatch"):
                verify_download(temporary, plan)
