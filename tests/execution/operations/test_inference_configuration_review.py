"""Offline pre-download review, exact confirmation and immutable resume intent."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.preparation import inference_configuration as preparation
from acprof.host.preparation.task_selection import select_tasks
from acprof.models import configuration_resolution as resolution
from acprof.models.candidates import apply_model_candidate, discover_model_candidates
from acprof.models.inference_configuration import (
    configuration_identity,
    validate_configuration_binding,
)
from acprof.runtime.profiles import select_runtime_profile


def requests():
    return {
        "schema_version": 1, "preset": "custom",
        "backend": {"mode": "explicit", "value": "transformers_pipeline"},
        "runtime_profile": {"mode": "explicit", "value": "fixture-profile"},
        "loading": {"off": {"dtype": {"mode": "explicit", "value": "FP32"}}},
        "tasks": {"group.v1": {
            "max_new_tokens": {"mode": "explicit", "value": 1},
            "do_sample": {"mode": "explicit", "value": False},
        }},
    }


def candidate():
    value = {
        "schema_version": 1, "state": "resolved", "rule_version": "acprof-inference-v1",
        "model": {"source": "huggingface", "model_id": "fixture/model", "revision": "a" * 40},
        "backend": {"id": "transformers_pipeline", "adapter": "family-default",
                    "extension_id": "torch-nlp", "runtime_profile": "fixture-profile"},
        "loading": {"off": {"dtype": "FP32"}},
        "tasks": {"group.v1": {
            "interface": "text-generation", "fingerprint": "b" * 64,
            "generation": {"max_new_tokens": 1, "do_sample": False},
        }},
        "requests": requests(), "decisions": [], "workload": {},
        "validation": {"static": "validated", "runtime": "not_run"},
    }
    value["identity_sha256"] = configuration_identity(value)
    return value


def task():
    return SimpleNamespace(model_id="fixture/model", model_revision="a" * 40,
                           inference_configuration={})


class Workflow:
    def __init__(self, *responses, interactive=True):
        self.interactive = interactive
        self.responses = iter(responses)
        self.calls = []

    def ask(self, stage, kind, **fields):
        self.calls.append((stage, kind, fields))
        return next(self.responses)


@pytest.fixture
def resolved(monkeypatch):
    initial = task()
    selected = copy.copy(initial)
    proposed = candidate()
    draft = {
        "questions": [{"path": "tasks.group.v1.max_new_tokens", "request": {
            "mode": "explicit", "value": 1,
        }}],
        "parameter_advanced": [], "requests": requests(),
    }
    build = Mock(return_value=draft)
    apply = Mock(return_value=requests())
    resolve = Mock(return_value=(selected, proposed))
    monkeypatch.setattr(resolution, "build_configuration_draft", build)
    monkeypatch.setattr(resolution, "requests_from_answers", apply)
    monkeypatch.setattr(resolution, "resolve_inference_configuration", resolve)
    return initial, selected, proposed, draft, build, resolve


def args(tmp_path, **updates):
    return SimpleNamespace(output_dir=str(tmp_path), inference_config="",
                           confirm_inference="", **updates)


def offline_selected_model(interface="text-generation"):
    config = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}
    initial = TaskInfo(
        "fixture/configuration", interface, "nlp", "transformers_model", "transformers",
        "a" * 40, "fixture", model_config=config,
        repository_metadata={"config.json": config},
        repository_files=("config.json", "model.safetensors"),
    )
    initial.model_resolution = apply_model_candidate(
        initial, discover_model_candidates(
            initial, override_tag=interface, override_backend="transformers_model",
        ),
    )
    selection = SimpleNamespace(
        task=interface, tasks="", backend="transformers_model", model_spec="", workload_spec="",
    )
    initial = select_tasks(initial, selection, Workflow(interactive=False))
    initial.runtime_profile_id = "nlp-cpu"
    return initial


def test_cli_real_resolver_candidate_requests_and_confirmation_roundtrip(tmp_path, capsys):
    initial = offline_selected_model()
    original = copy.deepcopy(initial)
    options = args(tmp_path)
    workload = {"batch_size": 1, "input_scales": "8,16"}
    with pytest.raises(preparation.InferenceConfigurationReviewRequired) as waiting:
        preparation.prepare_inference_configuration(
            initial, options, Workflow(interactive=False), devices=["off"], workload=workload,
        )
    review = waiting.value
    saved = json.loads(review.candidate_path.read_text(encoding="utf-8"))
    saved_requests = json.loads(review.requests_path.read_text(encoding="utf-8"))
    assert saved["state"] == "resolved"
    assert saved["requests"] == saved_requests
    assert configuration_identity(saved) == review.identity_sha256
    assert review.candidate_path.parent == tmp_path / ".inference-configurations"
    assert "--inference-config" in capsys.readouterr().out
    assert initial == original

    options.inference_config = str(review.requests_path)
    options.confirm_inference = review.identity_sha256
    confirmed = preparation.prepare_inference_configuration(
        initial, options, Workflow(interactive=False), devices=["off"], workload=workload,
    )
    assert confirmed.inference_configuration == {**saved, "state": "frozen"}
    assert confirmed.inference_configuration["validation"] == {"static": "validated", "runtime": "not_run"}
    assert confirmed.selected_tasks[0]["config"]["generation"]["max_new_tokens"] == 64
    validate_configuration_binding(confirmed.inference_configuration, confirmed)
    assert initial == original


@pytest.mark.parametrize("interfaces", [
    ("feature-extraction", "text-generation"),
    ("text-generation", "feature-extraction"),
])
def test_field_contract_rejects_mixed_loading_declarations_in_both_orders(monkeypatch, interfaces):
    initial = offline_selected_model()
    profile = select_runtime_profile(initial)
    template = initial.selected_tasks[0]
    initial.selected_tasks = [
        {**copy.deepcopy(template), "task_id": interface, "interface": interface}
        for interface in interfaces
    ]
    original = copy.deepcopy(initial)
    # GPT2LMHeadModel discovery already rejects feature extraction. Isolate the
    # field contract while retaining real per-interface catalog declarations.
    monkeypatch.setattr(resolution, "_route_options", lambda *_: (
        [initial.runtime_backend], [profile.profile_id],
    ))
    with pytest.raises(ValueError, match=(
        "^selected tasks have different loading declarations; configure separate task collections$"
    )):
        resolution._fields(initial, initial, profile, ["off"], {})
    assert initial == original


def test_cli_without_confirmation_saves_a_reviewable_candidate_before_freezing(resolved, tmp_path, capsys, monkeypatch):
    initial, _, proposed, _, _, _ = resolved
    freeze = Mock(side_effect=AssertionError("unconfirmed configuration must not freeze"))
    monkeypatch.setattr(preparation, "freeze_inference_configuration", freeze)
    with pytest.raises(preparation.InferenceConfigurationReviewRequired) as waiting:
        preparation.prepare_inference_configuration(
            initial, args(tmp_path), Workflow(interactive=False), devices=["off"],
        )
    assert waiting.value.identity_sha256 == proposed["identity_sha256"]
    assert json.loads(waiting.value.candidate_path.read_text())["state"] == "resolved"
    saved = json.loads(waiting.value.requests_path.read_text())
    assert saved["tasks"]["group.v1"]["max_new_tokens"] == {"mode": "explicit", "value": 1}
    assert "--confirm-inference" in capsys.readouterr().out
    freeze.assert_not_called()
    assert initial.inference_configuration == {}


def test_cli_exact_confirmation_freezes_without_changing_the_digest(resolved, tmp_path):
    initial, _, proposed, _, _, _ = resolved
    options = args(tmp_path)
    options.confirm_inference = proposed["identity_sha256"]
    result = preparation.prepare_inference_configuration(
        initial, options, Workflow(interactive=False), devices=["off"],
    )
    assert result.inference_configuration["state"] == "frozen"
    assert result.inference_configuration["identity_sha256"] == proposed["identity_sha256"]
    assert proposed["state"] == "resolved"
    assert initial.inference_configuration == {}


def test_cli_changed_confirmation_requires_review_of_new_content(resolved, tmp_path):
    initial, _, proposed, _, _, _ = resolved
    options = args(tmp_path)
    options.confirm_inference = "c" * 64
    with pytest.raises(preparation.InferenceConfigurationReviewRequired) as waiting:
        preparation.prepare_inference_configuration(
            initial, options, Workflow(interactive=False), devices=["off"],
        )
    assert waiting.value.confirmation_mismatch
    assert waiting.value.identity_sha256 == proposed["identity_sha256"]
    assert initial.inference_configuration == {}


def test_interactive_confirmation_is_separate_and_edit_reuses_the_original_selection(resolved, tmp_path):
    initial, _, proposed, _, build, resolve = resolved
    workflow = Workflow(
        {"action": "answer", "answers": {}}, {"action": "edit"},
        {"action": "answer", "answers": {}},
        {"action": "confirm", "confirmation_digest": proposed["identity_sha256"]},
    )
    result = preparation.prepare_inference_configuration(
        initial, args(tmp_path), workflow, devices=["off"],
    )
    assert [call[2]["purpose"] for call in workflow.calls] == [
        "inference_configuration", "inference_confirmation",
        "inference_configuration", "inference_confirmation",
    ]
    assert all(call.args[0] is initial for call in build.call_args_list)
    assert all(call.args[0] is initial for call in resolve.call_args_list)
    assert result.inference_configuration["state"] == "frozen"
    assert initial.inference_configuration == {}


def test_failed_resolution_keeps_refreshed_auto_values_editable(resolved, tmp_path):
    initial, selected, proposed, draft, build, resolve = resolved
    refreshed = copy.deepcopy(draft)
    refreshed["questions"][0].update(
        request={"mode": "auto"}, resolved_value=12, enabled=True,
    )
    build.side_effect = [draft, refreshed, refreshed]
    resolve.side_effect = [ValueError("another active field is unset"), (selected, proposed)]
    workflow = Workflow(
        {"action": "answer", "answers": {}}, {"action": "answer", "answers": {}},
        {"action": "confirm", "confirmation_digest": proposed["identity_sha256"]},
    )
    preparation.prepare_inference_configuration(initial, args(tmp_path), workflow, devices=["off"])
    retry = workflow.calls[1][2]
    assert retry["purpose"] == "inference_configuration"
    assert retry["questions"][0]["request"] == {"mode": "auto"}
    assert retry["questions"][0]["resolved_value"] == 12
    assert retry["questions"][0]["enabled"] is True
    assert "unset" in retry["detail"]


def test_interactive_stale_digest_does_not_freeze(resolved, tmp_path):
    initial = resolved[0]
    workflow = Workflow({"action": "answer", "answers": {}},
                        {"action": "confirm", "confirmation_digest": "c" * 64})
    with pytest.raises(ValueError, match="stale"):
        preparation.prepare_inference_configuration(
            initial, args(tmp_path), workflow, devices=["off"],
        )
    assert initial.inference_configuration == {}


def test_backend_request_refreshes_the_draft_before_resolution(resolved, tmp_path):
    initial, selected, proposed, draft, build, resolve = resolved
    changed_requests = requests()
    changed_requests["backend"] = {"mode": "auto"}
    changed = {**draft, "requests": changed_requests}
    build.side_effect = [draft, changed, changed]
    resolution.requests_from_answers.return_value = changed_requests
    workflow = Workflow(
        {"action": "answer", "answers": {}}, {"action": "answer", "answers": {}},
        {"action": "confirm", "confirmation_digest": proposed["identity_sha256"]},
    )
    preparation.prepare_inference_configuration(initial, args(tmp_path), workflow, devices=["off"])
    assert [call[2]["purpose"] for call in workflow.calls] == [
        "inference_configuration", "inference_configuration", "inference_confirmation",
    ]
    resolve.assert_called_once_with(initial, changed_requests, devices=["off"], workload=None)
    assert selected.inference_configuration == {}


def frozen():
    value = candidate()
    value["state"] = "frozen"
    return value


def test_resume_partial_matching_requests_do_not_resolve_auto(tmp_path, monkeypatch):
    options = args(tmp_path)
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({
        "schema_version": 1, "tasks": {"group.v1": {
            "max_new_tokens": {"mode": "explicit", "value": 1},
        }},
    }), encoding="utf-8")
    options.inference_config = str(path)
    value = frozen()
    options.confirm_inference = value["identity_sha256"]
    monkeypatch.setattr(resolution, "resolve_inference_configuration",
                        Mock(side_effect=AssertionError("resume must not re-resolve")))
    preparation.validate_resume_inference_requests(value, options)
    resolution.resolve_inference_configuration.assert_not_called()


@pytest.mark.parametrize("replacement", [
    {"tasks": {"group.v1": {"max_new_tokens": {"mode": "explicit", "value": True}}}},
    {"tasks": {"group.v1": {"max_new_tokens": {"mode": "auto"}}}},
    {"tasks": {"group.v1": {"unknown": {"mode": "explicit", "value": 1}}}},
    {"preset": "auto"},
])
def test_resume_rejects_changed_types_modes_unknown_paths_or_presets(tmp_path, replacement):
    options = args(tmp_path)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps({"schema_version": 1, **replacement}), encoding="utf-8")
    options.inference_config = str(path)
    with pytest.raises(ValueError, match="frozen"):
        preparation.validate_resume_inference_requests(frozen(), options)


def test_resume_rejects_a_different_confirmation_without_reading_a_file(tmp_path):
    options = args(tmp_path)
    options.confirm_inference = "f" * 64
    with pytest.raises(ValueError, match="confirm_inference"):
        preparation.validate_resume_inference_requests(frozen(), options)


@pytest.mark.parametrize("raw", ['{"value": NaN}', '[]', '{invalid'])
def test_request_loader_reports_invalid_json_with_its_path(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid.json"):
        preparation.read_inference_requests(path)


def test_request_loader_rejects_excessive_depth_before_copy(tmp_path):
    path = tmp_path / "deep.json"
    path.write_text('{"nested":' * 600 + "0" + "}" * 600, encoding="utf-8")
    with pytest.raises(ValueError, match="deep.json"):
        preparation.read_inference_requests(path)
