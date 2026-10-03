"""Static decisions keep correlated evidence and execution observations separate."""
import copy

import pytest

from acprof.host.detect import TaskInfo
from acprof.model_resolution import discover_model_candidates, require_resolved_candidate


def candidate(*, tag="unknown", config=None, hub=None):
    task = TaskInfo("example/model", tag, "unknown", "transformers_pipeline",
                    "transformers", "a" * 40, "hub_api")
    task.model_config = config or {}
    task.repository_metadata = {"config.json": task.model_config}
    task.hub_metadata = hub or {}
    task.model_resolution = discover_model_candidates(task)
    return task


def test_transformers_info_resolves_missing_task_without_model_import():
    task = candidate(hub={"transformers_info": {"pipeline_tag": "text-generation",
                                                 "auto_model": "AutoModelForCausalLM"}})
    assert (task.pipeline_tag) == ("text-generation")
    assert (task.model_resolution["semantics"]["status"]) == ("declared")
    assert (task.model_resolution["runtime_validation"]["status"]) == ("not_run")

def test_conflicting_hub_observations_abstain_even_when_runtime_passed():
    task = candidate(tag="fill-mask", hub={"transformers_info": {"pipeline_tag": "text-generation"}})
    task.model_resolution["runtime_validation"] = {"status": "verified"}
    with pytest.raises(ValueError, match="ambiguous|conflict"):
        require_resolved_candidate(task)

def test_explicit_choice_preserves_conflict_evidence():
    task = candidate(tag="fill-mask", hub={"transformers_info": {"pipeline_tag": "text-generation"}})
    task.model_resolution = discover_model_candidates(task, override_tag="text-generation")
    require_resolved_candidate(task)
    assert (task.model_resolution["semantics"]["status"]) == ("explicit")
    assert (task.model_resolution["provenance"]["overridden_conflicts"])

def test_correlated_observations_share_source_and_have_no_vote_score():
    task = candidate(tag="text-generation", config={"architectures": ["LlamaForCausalLM"]},
                     hub={"transformers_info": {"pipeline_tag": "text-generation",
                                                 "auto_model": "AutoModelForCausalLM"}})
    provenance = task.model_resolution["provenance"]
    hub = [item for item in provenance["observations"] if item["source_id"] == "hub"]
    assert (len(hub)) >= (2)
    assert (provenance["sources"]["hub"]["derived_from"]) == (["repository_snapshot"])
    assert ("confidence_score") not in (provenance)

def test_static_identity_changes_with_evidence_but_not_runtime():
    first = candidate(tag="text-generation", config={"model_type": "llama"})
    second = candidate(tag="text-generation", config={"model_type": "gpt2"})
    identity = first.model_resolution["provenance"]["identity_sha256"]
    assert (identity) != (second.model_resolution["provenance"]["identity_sha256"])
    first.model_resolution["runtime_validation"] = {"status": "verified"}
    assert (identity) == (first.model_resolution["provenance"]["identity_sha256"])

def test_bare_auto_model_does_not_prove_a_task():
    task = candidate(hub={"transformers_info": {"auto_model": "AutoModel"}})
    with pytest.raises(ValueError):
        require_resolved_candidate(task)

def test_custom_pipeline_loader_hint_cannot_supply_missing_task_semantics():
    task = candidate(config={"custom_pipelines": {
        "custom-task": {"impl": "pipeline.CustomPipeline", "pt": ["AutoModel"]},
    }}, hub={"transformers_info": {"auto_model": "AutoModel", "pipeline_tag": "feature-extraction"}})
    with pytest.raises(ValueError, match="task semantics are unknown"):
        require_resolved_candidate(task)
    assert not (task.model_resolution["candidates"])

@pytest.mark.parametrize('config', ({}, {'custom_pipelines': {'custom-task': {'impl': 'pipeline.CustomPipeline', 'pt': ['AutoModelForCausalLM']}}}))
def test_generic_tag_conflicts_without_matching_custom_pipeline_loader(config):
    task = candidate(tag="text-generation", config=config, hub={
        "transformers_info": {"auto_model": "AutoModel", "pipeline_tag": "feature-extraction"},
    })
    with pytest.raises(ValueError, match="conflict"):
        require_resolved_candidate(task)

@pytest.mark.parametrize('config,hub', (({'architectures': ['BertForMaskedLM']}, {}), ({}, {'transformers_info': {'auto_model': 'AutoModelForMaskedLM'}})))
def test_incompatible_declared_head_requires_explicit_resolution(config, hub):
    task = candidate(tag="text-generation", config=config, hub=hub)
    with pytest.raises(ValueError, match="conflict|ambiguous"):
        require_resolved_candidate(task)

def test_shared_loader_is_not_a_translation_semantic_conflict():
    task = candidate(tag="translation", hub={"transformers_info": {"auto_model": "AutoModelForSeq2SeqLM"}})
    require_resolved_candidate(task)
    assert (task.pipeline_tag) == ("translation")

@pytest.mark.parametrize('tag_case', range(7))
def test_library_loader_tag_does_not_override_a_compatible_workload(tag_case):
    cases = (
        ("summarization", "text2text-generation", "AutoModelForSeq2SeqLM", "BartForConditionalGeneration"),
        ("translation", "text2text-generation", "AutoModelForSeq2SeqLM", "T5ForConditionalGeneration"),
        ("zero-shot-classification", "text-classification", "AutoModelForSequenceClassification", "BartForSequenceClassification"),
        ("text-ranking", "text-classification", "AutoModelForSequenceClassification", "BertForSequenceClassification"),
        ("table-question-answering", "text2text-generation", "AutoModelForSeq2SeqLM", "BartForConditionalGeneration"),
        ("image-to-text", "image-text-to-text", "AutoModelForImageTextToText", "BlipForConditionalGeneration"),
        ("image-to-text", "image-text-to-text", "AutoModelForMultimodalLM", "BlipForConditionalGeneration"),
    )
    (tag, library_tag, loader, architecture) = tuple(cases)[tag_case]
    task = candidate(tag=tag, config={"architectures": [architecture]}, hub={
        "transformers_info": {"pipeline_tag": library_tag, "auto_model": loader},
    })
    require_resolved_candidate(task)
    assert (task.pipeline_tag) == (tag)
    observations = task.model_resolution["provenance"]["observations"]
    hint = next(item for item in observations if item["field"] == "hub.transformers_info.pipeline_tag")
    assert (hint["kind"]) == ("loader_hint")
    assert (hint["value"]) == (library_tag)
    assert (hint["reason"])

@pytest.mark.parametrize('tag,library_tag,loader', (('summarization', 'translation', 'AutoModelForSeq2SeqLM'), ('text-ranking', 'zero-shot-classification', 'AutoModelForSequenceClassification'), ('text-ranking', 'text-generation', 'AutoModelForCausalLM'), ('summarization', 'text2text-generation', 'AutoModelForMaskedLM'), ('summarization', 'text2text-generation', None)))
def test_shared_loader_does_not_erase_two_distinct_task_declarations(tag, library_tag, loader):
    task = candidate(tag=tag, hub={"transformers_info": {
        "pipeline_tag": library_tag, "auto_model": loader,
    }})
    with pytest.raises(ValueError, match="conflict"):
        require_resolved_candidate(task)

@pytest.mark.parametrize('architecture,loader,library_tag', (('BertModel', 'AutoModel', 'feature-extraction'), ('MPNetForMaskedLM', 'AutoModelForMaskedLM', 'fill-mask'), ('Qwen3ForCausalLM', 'AutoModelForCausalLM', 'text-generation')))
def test_sentence_transformer_modules_establish_the_encoder_interface(architecture, loader, library_tag):
    task = candidate(tag="sentence-similarity", config={"architectures": [architecture]}, hub={
        "transformers_info": {"auto_model": loader, "pipeline_tag": library_tag},
    })
    task.library_name = "sentence-transformers"
    task.repository_metadata["modules.json"] = [
        {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    ]
    task.model_resolution = discover_model_candidates(task)
    require_resolved_candidate(task)
    assert (task.runtime_backend) == ("sentence_transformers")
    task.repository_metadata.pop("modules.json")
    task.model_resolution = discover_model_candidates(task)
    with pytest.raises(ValueError, match="conflict"):
        require_resolved_candidate(task)

def test_native_registry_reverse_lookup_resolves_architecture():
    task = candidate(config={"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"})
    assert (task.pipeline_tag) == ("text-generation")
    assert (any(item["field"] == "config.architectures" for item in
                        task.model_resolution["provenance"]["observations"]))

def test_runtime_observation_cannot_rewrite_static_semantics():
    from acprof.model_contract import record_runtime_validation
    task = candidate(tag="text-generation")
    before = copy.deepcopy(task.model_resolution)
    record_runtime_validation(task, {"mode": "full", "status": "ok", "image_id": "sha256:" + "b" * 64,
                                     "build_fingerprint": "c" * 64, "payload_sha256": "d" * 64,
                                     "devices": {"off": {"status": "ok"}}})
    assert (task.model_resolution["semantics"]) == (before["semantics"])
    assert (task.model_resolution["provenance"]) == (before["provenance"])
    assert (task.model_resolution["runtime_validation"]["status"]) == ("verified")
