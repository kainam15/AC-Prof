"""Comparison keeps execution configuration differences distinct from shared workloads."""
from __future__ import annotations

import copy

from acprof.analysis.comparison import load_comparison_snapshot
from acprof.models.inference_configuration import configuration_identity
from acprof.tasks.workloads import parse_task
from tests.support.comparison_fixtures import ComparisonFixture
from tests.support.inference_configuration_fixture import frozen_task


def configured_fixture(tmp_path):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    configuration = copy.deepcopy(frozen_task().inference_configuration)
    selected = parse_task("tabular-regression", {}, encoder_decoder=False, source="fixture")
    configuration["model"]["model_id"] = "example/torch"
    configuration["backend"].update(id="torchscript", extension_id="torch-structured",
                                     runtime_profile="structured-cpu")
    configuration["loading"] = {"off": {}}
    configuration["tasks"] = {selected["task_id"]: {
        "interface": selected["interface"], "fingerprint": selected["fingerprint"], "generation": {},
    }}
    configuration["requests"] = {}
    configuration["identity_sha256"] = configuration_identity(configuration)
    fixture.change_json(fixture.left, "input_scale_plan.json", lambda plan: plan.update(
        task_workload=selected, inference_configuration_sha256=configuration["identity_sha256"]))
    fixture.change_json(fixture.left, "static_meta.json", lambda metadata: metadata.update(
        schema_version=8, model_revision="a" * 40, runtime_backend="torchscript",
        runtime_environment={"adapter": "family-default", "profile_id": "structured-cpu"},
        inference_configuration=configuration))
    return fixture, configuration


def test_comparison_exposes_frozen_identity_loading_and_task_values(tmp_path):
    fixture, configuration = configured_fixture(tmp_path)

    snapshot = load_comparison_snapshot(fixture.left)

    assert snapshot["identity"]["inference_configuration_sha256"] == configuration["identity_sha256"]
    assert snapshot["identity"]["inference_loading"] == {"off": {}}
    assert snapshot["identity"]["inference_generation"] == {"tabular-regression": {}}
    assert not any("inference_configuration:" in issue for issue in snapshot["issues"])
    legacy = load_comparison_snapshot(fixture.right)
    assert legacy["identity"]["inference_configuration_sha256"] is None
    assert legacy["inference_configuration_validation"] is None


def test_comparison_rejects_conflicting_static_identity(tmp_path):
    fixture, _configuration = configured_fixture(tmp_path)
    fixture.change_json(fixture.left, "static_meta.json",
                        lambda metadata: metadata.update(model_name="example/another-model"))

    snapshot = load_comparison_snapshot(fixture.left)

    assert not snapshot["valid"]
    assert any("model.model_id differs" in issue for issue in snapshot["issues"])
    assert snapshot["identity"]["inference_configuration_sha256"] is None
