"""Check the input contract of the native two-tower CLIP checkpoint."""
from dataclasses import replace

import pytest

from acprof.extensions import CATALOG, ExtensionCatalog
from acprof.host.detect import TaskInfo
from acprof.models.backend_candidates import describe_backend_candidates
from acprof.models.candidates import apply_model_candidate, discover_model_candidates
from acprof.models.resolution import resolve_model_interface


def checkpoint(tag="zero-shot-image-classification", **config):
    config = {"model_type": "clip", "architectures": ["CLIPModel"], **config}
    return TaskInfo("fixture/ordinary-name", tag, CATALOG.task_families[tag], "transformers_pipeline",
                    "transformers", "a" * 40, "hub_api", model_config=config,
                    repository_metadata={"config.json": config}, repository_files=("config.json", "model.safetensors"),
                    hub_metadata={"pipeline_tag": tag, "tags": [tag],
                                  "transformers_info": {"pipeline_tag": tag, "auto_model": "AutoModelForZeroShotImageClassification"}
                                  if tag == "zero-shot-image-classification" else {}})


def test_paired_clip_does_not_offer_generic_single_input_workloads():
    task = checkpoint()
    task.model_resolution = apply_model_candidate(task, discover_model_candidates(task))
    assert task.task_discovery["status"] == "ready"
    assert [profile["interface"] for profile in task.task_discovery["tasks"]] == ["zero-shot-image-classification"]
    assert task.runtime_backend == "transformers_pipeline"


@pytest.mark.parametrize("tag", ["feature-extraction", "image-feature-extraction"])
def test_explicit_single_input_choice_cannot_claim_support_for_full_clip(tag):
    task = checkpoint(tag)
    candidates = describe_backend_candidates(task, task=tag)
    for backend in ("transformers_pipeline", "transformers_model"):
        candidate = next(item for item in candidates if item["backend"] == backend)
        assert not candidate["selectable"]
        assert "CLIPModel" in candidate["failure"]["detail"]
        # A caller restoring the frozen task can arrive directly at interface resolution.
        with pytest.raises(ValueError, match="CLIPModel"):
            resolve_model_interface(replace(task, runtime_backend=backend))


@pytest.mark.parametrize(("tag", "model_type", "head"), [
    ("feature-extraction", "clip_text_model", "CLIPTextModel"),
    ("image-feature-extraction", "clip_vision_model", "CLIPVisionModel"),
])
def test_separately_saved_tower_uses_its_ordinary_native_pipeline(tag, model_type, head):
    task = checkpoint(tag, model_type=model_type, architectures=[head])
    candidates = describe_backend_candidates(task, task=tag)
    assert next(item for item in candidates if item["backend"] == "transformers_pipeline")["selectable"]


@pytest.mark.parametrize("config", [
    {"custom_pipelines": {"feature-extraction": {"impl": "custom.Extractor"}}},
    {"auto_map": {"AutoModel": "custom.TextTower"}},
])
def test_custom_loader_uses_existing_configuration_review(config):
    task = checkpoint("feature-extraction", **config)
    route = CATALOG.resolve(task=task.pipeline_tag, library=task.library_name, config=task.model_config)
    assert route.backend == "transformers_pipeline"
    assert CATALOG.select_extension(task).adapter == "family-default"


@pytest.mark.parametrize("adapter", ["tower-extractor", "family-default"])
def test_registered_separate_handler_can_supply_its_own_tower_interface(adapter):
    catalog = ExtensionCatalog()
    original = CATALOG.get_extension("nlp", "transformers_pipeline", task="feature-extraction")
    catalog.add(replace(original, extension_id="tower-extractor", model_types=("clip",),
                        adapter=adapter, handler_entrypoint="fixture.adapter:TowerHandler"))
    task = checkpoint("feature-extraction")
    route = catalog.resolve(task=task.pipeline_tag, backend=task.runtime_backend, config=task.model_config)
    assert route.declaration.adapter == adapter
    assert catalog.select_extension(task).adapter == adapter
