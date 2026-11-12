"""Static inspection failures remain distinct from unsupported tasks and measurement."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from acprof.failures import FAILURE_PREFIX, Failure
from acprof.host.preparation.collection_workflow import PreparationWorkflow
from acprof.tui.i18n import translate
from acprof.tui.run.monitor_view import monitor_header_view
from acprof.tui.run.progress import ProgressSnapshot, RunProgressTracker
from acprof.tui.run.results import RunResult, evaluate_process_completion
from acprof.tui.run.stages import Stage
from acprof.tui.ui.view_models import reason_presentation


@pytest.mark.parametrize("reason", [
    "task_selection_required", "evidence_conflict", "model_contract_required",
    "runtime_unavailable", "backend_unsupported", "metadata_invalid",
    "artifact_incomplete", "artifact_layout_unknown", "artifact_layout_unsupported",
    "runtime_dependency_missing", "precision_mismatch",
])
def test_static_failures_do_not_claim_the_task_is_unsupported(reason):
    failure = Failure("preflight", reason, "fixture inspection detail")
    tracker = RunProgressTracker()
    state = tracker.feed(FAILURE_PREFIX + json.dumps(failure.to_dict()))
    assert state.stage is Stage.FAILED
    assert state.measurement_status == "not_started"
    assert state.runtime_status == "not_started"
    assert not state.measurement_active
    assert state.failure["reason_code"] == reason
    assert "fixture inspection detail" in state.detail
    assert monitor_header_view(state).summary.level == "error"


def test_real_task_unsupported_keeps_its_specific_stage():
    failure = Failure("preflight", "runtime_task_unsupported", "fixture unsupported task")
    state = RunProgressTracker().feed(FAILURE_PREFIX + json.dumps(failure.to_dict()))
    assert state.stage is Stage.TASK_UNSUPPORTED


@pytest.mark.parametrize(("reason", "english", "chinese"), [
    ("task_selection_required", "inference profile", "推理配置组"),
    ("evidence_conflict", "conflicts", "冲突"),
    ("model_contract_required", "configuration", "配置"),
    ("runtime_unavailable", "runtime", "运行环境"),
    ("backend_unsupported", "backend", "后端"),
    ("metadata_invalid", "metadata", "元数据"),
    ("artifact_incomplete", "incomplete", "不完整"),
    ("artifact_layout_unknown", "unverified", "未确认"),
    ("artifact_layout_unsupported", "model artifact", "模型制品"),
])
def test_inspection_reasons_have_specific_bilingual_remedies(reason, english, chinese):
    view = reason_presentation(reason, "raw evidence remains")
    assert view.level in {"error", "warning"}
    assert english in translate(view.problem, "en").lower()
    assert chinese in translate(view.problem, "zh")
    assert translate(view.next_step, "en")
    assert view.detail == reason + "\nraw evidence remains"


def test_blocked_review_does_not_claim_resolution_succeeded():
    task = SimpleNamespace(
        model_id="fixture/model", model_revision="a" * 40,
        pipeline_tag="text-generation", runtime_backend="transformers_model", model_adapter="family-default",
        model_resolution={"status": "blocked", "failure": {
            "reason_code": "runtime_unavailable", "detail": "No compatible runtime is registered.",
            "evidence": {"next_action": "Register a compatible runtime."},
        }},
    )
    view = PreparationWorkflow.review_details(task, [])
    assert set(view["fields"]) == {"Model", "Revision"}
    assert "No compatible runtime is registered." in view["summary"]
    assert "Register a compatible runtime." in view["summary"]


@pytest.mark.parametrize("kind", ["run", "probe"])
def test_preflight_failure_completion_keeps_measurement_not_started(kind):
    failure = Failure("preflight", "metadata_invalid", "config.json:/limit is non-finite")
    snapshot = ProgressSnapshot(stage=Stage.FAILED, detail=failure.detail, failure=failure.to_dict(), errors=1)
    completed = evaluate_process_completion(kind, 2, snapshot, ProgressSnapshot())
    assert completed.final_snapshot.stage is Stage.FAILED
    assert completed.final_snapshot.measurement_status == "not_started"
    assert completed.final_snapshot.detail == failure.detail
    assert completed.final_snapshot.failure == failure.to_dict()


@pytest.mark.parametrize("updates", [
    {"measurement_status": "running", "measurement_active": True},
    {"total_cases": 1, "current_case": 1},
])
def test_started_or_planned_measurement_failure_keeps_existing_completion_semantics(updates):
    failure = Failure("preflight", "runtime_unavailable", "fixture late failure")
    snapshot = replace(
        ProgressSnapshot(stage=Stage.FAILED, detail=failure.detail, failure=failure.to_dict()), **updates,
    )
    completed = evaluate_process_completion(
        "run", 2, snapshot, ProgressSnapshot(), run_result=RunResult(belongs_to_attempt=True),
    )
    assert completed.final_snapshot.measurement_status == "failed"
    assert not completed.final_snapshot.measurement_active
