"""Separate a reranker's paired scoring interface from its classification head."""
from types import SimpleNamespace

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.preparation.task_selection import select_tasks
from acprof.models.candidates import apply_model_candidate, discover_model_candidates

BGE_ID = "BAAI/bge-reranker-v2-m3"
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MSMARCO_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MSMARCO_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def model(model_id, revision, tag, *, source="huggingface", **config):
    config = {"architectures": ["XLMRobertaForSequenceClassification"], "model_type": "xlm-roberta",
              "id2label": {"0": "LABEL_0"}, **config}
    return TaskInfo(model_id, tag, "nlp", "unknown", "sentence-transformers", revision, "hub_api",
                    model_source=source, model_config=config, repository_metadata={"config.json": config},
                    repository_files=("config.json", "model.safetensors"), hub_metadata={
                        "pipeline_tag": tag, "tags": [tag, "text-classification", "transformers"],
                        "transformers_info": {"pipeline_tag": "text-classification",
                                              "auto_model": "AutoModelForSequenceClassification"},
                    })


def resolve(task, override=None):
    task.model_resolution = apply_model_candidate(task, discover_model_candidates(task, override_tag=override))
    return task.task_discovery


def test_declared_ranking_is_not_blocked_by_the_shared_loader_tag():
    task = model(MSMARCO_ID, MSMARCO_REVISION, "text-ranking",
                 architectures=["BertForSequenceClassification"], model_type="bert")
    discovery = resolve(task)
    assert discovery["status"] == "ready"
    assert [item["task_id"] for item in discovery["tasks"]] == ["text-ranking"]
    assert task.runtime_backend == "cross_encoder"
    assert any(obs.get("value") == "text-classification" and not obs["sufficient"]
               for obs in task.model_resolution["evidence"] if obs["source"] == "hub.tags")


def test_reviewed_bge_checkpoint_selects_pairs_while_retaining_hub_observation():
    task = model(BGE_ID, BGE_REVISION, "text-classification")
    discovery = resolve(task)
    assert discovery["status"] == "ready"
    assert [item["interface"] for item in discovery["tasks"]] == ["text-ranking"]
    assert task.runtime_backend == "cross_encoder"
    provenance = task.model_resolution["provenance"]
    assert provenance["sources"]["reviewed_model_card"]["url"].endswith(f"/{BGE_REVISION}/README.md")
    assert any(obs["field"] == "hub_metadata" and obs["task"] == "text-classification"
               for obs in provenance["observations"])


@pytest.mark.parametrize(("model_id", "revision", "source", "changes"), [
    ("fixture/bge-reranker-v2-m3", BGE_REVISION, "huggingface", {}),
    (BGE_ID, "a" * 40, "huggingface", {}),
    (BGE_ID, BGE_REVISION, "modelscope", {}),
    (BGE_ID, BGE_REVISION, "huggingface", {"id2label": {"0": "NEGATIVE", "1": "POSITIVE"}}),
    (BGE_ID, BGE_REVISION, "huggingface", {"model_type": "bert"}),
])
def test_model_name_single_logit_or_other_snapshot_cannot_grant_ranking(model_id, revision, source, changes):
    task = model(model_id, revision, "text-classification", source=source, **changes)
    assert [x["interface"] for x in resolve(task)["tasks"]] == ["text-classification"]


def test_legacy_record_without_model_source_keeps_generic_classification():
    fields = vars(model(BGE_ID, BGE_REVISION, "text-classification")).copy()
    del fields["model_source"]
    legacy = SimpleNamespace(**fields)
    assert [x["interface"] for x in resolve(legacy)["tasks"]] == ["text-classification"]


def test_missing_pinned_config_cannot_promote_hub_summary_to_reviewed_evidence():
    task = model(BGE_ID, BGE_REVISION, "text-classification")
    task.repository_metadata = {}
    assert [x["interface"] for x in resolve(task)["tasks"]] == ["text-classification"]


def test_explicit_classification_stays_available_for_reviewed_checkpoint():
    task = model(BGE_ID, BGE_REVISION, "text-classification")
    assert [x["interface"] for x in resolve(task, "text-classification")["tasks"]] == ["text-classification"]
    assert task.runtime_backend == "transformers_pipeline"


def test_authored_second_workload_can_still_be_selected():
    task = model("fixture/ranker", "a" * 40, "text-ranking", task_specific_params={"text-classification": {}})
    discovery = resolve(task)
    assert discovery["status"] == "selection_required"
    assert {x["interface"] for x in discovery["tasks"]} == {"text-ranking", "text-classification"}
    options = SimpleNamespace(tasks="text-ranking", workload_spec="", task="", backend="", task_family="", model_spec="")
    selected = select_tasks(task, options)
    assert [item["interface"] for item in selected.selected_tasks] == ["text-ranking"]
