"""Shared parameter declarations reach inspection and existing task review."""
import copy
import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from acprof.host.detect import TaskInfo
from acprof.models.configuration import describe_configuration_options, parameter_declaration
from acprof.tasks.parameters import (
    apply_parameter_review,
    describe_task_parameters,
    parameter_review,
)
from acprof.tasks.workloads import parse_task


def model():
    config = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"], "n_layer": 2}
    info = TaskInfo(
        "fixture/configuration", "text-generation", "nlp", "transformers_model",
        "transformers", "a" * 40, "hub_api", model_config=config,
        repository_files=("config.json", "model.safetensors", "generation_config.json"),
        repository_metadata={
            "config.json": config, "generation_config.json": {"max_new_tokens": 48},
        },
        hub_metadata={"pipeline_tag": "text-generation"},
    )
    info.selected_tasks = [parse_task(
        "text-generation", {}, encoder_decoder=False, source="fixture",
        default_generation={"max_new_tokens": 48},
    )]
    return info


def test_inspection_describes_consumers_and_sources_without_freezing_or_mutating():
    info = model()
    before = copy.deepcopy(info)
    options = describe_configuration_options(info)

    assert info == before
    assert options["phase"] == "discovery"
    assert options["frozen"] is False
    assert options["runtime_validation"] == "not_run"
    task = options["task_options"][0]
    fields = {item["name"]: item for item in task["parameters"]}
    assert fields["temperature"]["kind"] == "float"
    assert fields["temperature"]["scope"] == "inference"
    assert {"stage": "predict", "argument": "params.temperature"} in fields["temperature"]["consumers"]
    assert "n_layer" not in fields
    assert task["source_discovery"]["parameters"]["max_new_tokens"]["default_candidates"]
    json.dumps(options, allow_nan=False)


def test_explicit_profile_constraint_is_passed_without_reusing_auto_selected_profile(monkeypatch):
    info = model()
    info.runtime_profile_id = "prior-auto-profile"
    candidates = Mock(return_value=[])
    monkeypatch.setattr(
        "acprof.models.backend_candidates.describe_backend_candidates", candidates,
    )
    describe_configuration_options(info)
    assert candidates.call_args.args[0].runtime_profile_id == ""
    describe_configuration_options(info, runtime_profile_constraint="requested-profile")
    assert candidates.call_args.args[0].runtime_profile_id == "requested-profile"
    assert info.runtime_profile_id == "prior-auto-profile"


def test_unregistered_backend_has_no_editable_parameter_declarations():
    info = model()
    info.runtime_backend = "unregistered-engine"
    options = describe_configuration_options(info)
    assert options["task_options"][0]["parameters"] == []
    assert options["diagnostics"][0]["reason_code"] == "parameter_declaration_unavailable"
    assert describe_task_parameters(info, info.selected_tasks[0]) == []


def test_task_review_uses_manifest_constraint_instead_of_a_second_range_table(monkeypatch):
    info = model()
    declaration = parameter_declaration(info, info.selected_tasks[0])
    restricted = replace(declaration, parameters=tuple(
        replace(spec, maximum=64) if spec.name == "max_new_tokens" else spec
        for spec in declaration.parameters
    ))
    monkeypatch.setattr("acprof.tasks.parameters.parameter_declaration", lambda *args: restricted)
    review = parameter_review(info, info.selected_tasks)
    answers = {field["path"]: field["value"]
               for field in review["questions"] + review["parameter_advanced"]}
    answers["task_0.max_new_tokens"] = 65
    with pytest.raises(ValueError, match="max_new_tokens"):
        apply_parameter_review(info, info.selected_tasks, answers)
    assert info.selected_tasks[0]["config"]["generation"]["max_new_tokens"] == 48


def test_inspect_exports_discovery_and_preserves_runtime_evidence_boundary(monkeypatch, tmp_path, capsys):
    from acprof.cli.inspect import main
    from acprof.models.candidates import apply_model_candidate, discover_model_candidates

    info = model()
    info.selected_tasks = []
    info.model_resolution = apply_model_candidate(
        info, discover_model_candidates(info, override_tag="text-generation"),
    )
    monkeypatch.setattr("acprof.host.detect.detect_task", lambda *args, **kwargs: copy.deepcopy(info))
    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", Mock())
    assert main([
        info.model_id, "--task", "text-generation", "--tasks", "all",
        "--explain", "--output-dir", str(tmp_path),
    ]) == 0
    saved = json.loads((tmp_path / "model_resolution.json").read_text())
    assert saved["configuration_options"]["phase"] == "discovery"
    assert saved["configuration_options"]["frozen"] is False
    assert saved["configuration_options"]["task_options"][0]["parameters"]
    output = capsys.readouterr().out
    assert "Configuration options: static discovery" in output
    assert "scope=inference; consumer=predict:params.temperature" in output
    assert "Default candidate: 48 <- generation_config.json:/max_new_tokens" in output
    assert "Measurement: not_started" in output


def test_deep_metadata_is_diagnosed_without_mutating_or_crashing():
    info = model()
    nested = 1
    for _ in range(600):
        nested = {"nested": nested}
    info.repository_metadata["config.json"]["nested"] = nested
    options = describe_configuration_options(info)
    source = options["task_options"][0]["source_discovery"]
    assert source["sources"]["config.json"]["state"] == "invalid"
    assert any("depth" in entry["detail"] for entry in source["diagnostics"])
    assert not any(item["selectable"] for item in options["backend_candidates"])
    assert info.repository_metadata["config.json"]["nested"] is nested
    json.dumps(options, allow_nan=False)
