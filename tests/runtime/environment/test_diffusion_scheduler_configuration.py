"""Scheduler selection uses only the already loaded local configuration."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.container.handlers.diffusion import DiffusionHandler


def runtime(monkeypatch):
    import sys

    class DDIMScheduler:
        @classmethod
        def from_config(cls, config):
            instance = cls()
            instance.config = copy.deepcopy(config)
            return instance

    class EulerDiscreteScheduler:
        config = {"prediction_type": "epsilon", "num_train_timesteps": 1000}
        compatibles = [DDIMScheduler]

    class StableDiffusionXLPipeline:
        def __init__(self):
            self.scheduler = EulerDiscreteScheduler()
            self.unet = object()

        def to(self, device):
            return self

    pipeline = StableDiffusionXLPipeline()
    loader = Mock(return_value=pipeline)
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        DiffusionPipeline=SimpleNamespace(from_pretrained=loader), DDIMScheduler=DDIMScheduler,
    ))
    monkeypatch.setattr("acprof.container.handlers.diffusion.load_policy",
                        lambda *args, **kwargs: {"dtype": "fixture-float32"})
    return pipeline, loader


def test_scheduler_is_selected_from_local_config_without_another_pretrained_call(monkeypatch, tmp_path):
    pipeline, loader = runtime(monkeypatch)
    original = copy.deepcopy(pipeline.scheduler.config)
    context = DiffusionHandler().load(str(tmp_path), "text-to-image", "diffusers", "cpu",
                                     load_options={"scheduler": "DDIMScheduler"})
    assert type(context["pipeline"].scheduler).__name__ == "DDIMScheduler"
    assert context["pipeline"].scheduler.config == original
    assert loader.call_count == 1
    assert loader.call_args.kwargs["local_files_only"] is True


def test_scheduler_requires_the_loaded_scheduler_to_declare_compatibility(monkeypatch, tmp_path):
    pipeline, _loader = runtime(monkeypatch)
    pipeline.scheduler.compatibles = []
    with pytest.raises(ValueError, match="compatible"):
        DiffusionHandler().load(str(tmp_path), "text-to-image", "diffusers", "cpu",
                                load_options={"scheduler": "DDIMScheduler"})


def test_unregistered_scheduler_is_not_dynamically_imported(monkeypatch, tmp_path):
    runtime(monkeypatch)
    with pytest.raises(ValueError, match="scheduler"):
        DiffusionHandler().load(str(tmp_path), "text-to-image", "diffusers", "cpu",
                                load_options={"scheduler": "UnregisteredScheduler"})
