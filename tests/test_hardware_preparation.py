"""Image preparation is isolated from offline hardware acceptance."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml

from acprof.experiment import RunConfig
from scripts import check_hardware, prepare_hardware

ROOT = Path(__file__).resolve().parents[1]


def cases(output: Path):
    return [
        (name, RunConfig(model=f"demo/{name}", revision=revision, task="fill-mask",
                         cpus="2", mems="8", gpus="off,on", input_scales="64"))
        for name, revision in (("first", "a" * 40), ("second", "b" * 40))
    ]


def task_for(model: str, revision: str):
    return SimpleNamespace(model_id=model, model_revision=revision, pipeline_tag="fill-mask",
                           model_download_policy="")


def image_for(name: str):
    return SimpleNamespace(tag="sha256:" + "c" * 64, name=f"acprof-nlp-{name}:prepared")


@contextmanager
def observed_lock(order):
    order.append("lock_enter")
    try:
        yield
    finally:
        order.append("lock_exit")


def test_prepares_every_case_and_only_then_publishes_pinned_selection(tmp_path):
    selected = cases(tmp_path)
    output = tmp_path / "prepare"
    output.mkdir()
    workflow = Mock()
    workflow.resolve.side_effect = [task_for(config.model, config.revision) for _, config in selected]
    order = []
    def build(task, _root):
        order.append(f"image:{task.model_id}")
        assert task.model_download_policy == "auto"
        return image_for(task.model_id)
    with patch.object(prepare_hardware, "PreparationWorkflow", return_value=workflow), patch.object(
        prepare_hardware, "MeasurementLock", side_effect=lambda: observed_lock(order)
    ), patch.object(prepare_hardware, "prepare_image", side_effect=build), patch.object(
        check_hardware, "run_config"
    ) as run:
        assert prepare_hardware.prepare_cases(selected, output)
        run.assert_not_called()
    matrix = json.loads((output / "prepared-matrix.json").read_text())
    assert matrix["schema_version"] == 1
    assert [(c["name"], c["revision"]) for c in matrix["cases"]] == [
        ("first", "a" * 40), ("second", "b" * 40)]
    assert json.loads((output / "preparation.json").read_text())["successful"] is True
    assert order == ["lock_enter", "image:demo/first", "image:demo/second", "lock_exit"]
    assert [call.args[0].gpus for call in workflow.resolve.call_args_list] == ["off,on", "off,on"]


def test_custom_branch_is_pinned_for_offline_validation(tmp_path):
    output = tmp_path / "preparation"
    task = task_for("demo/custom", "d" * 40)
    workflow = Mock()
    workflow.resolve.return_value = task
    with patch.object(prepare_hardware, "PreparationWorkflow", return_value=workflow), patch.object(
        prepare_hardware, "MeasurementLock", side_effect=lambda: observed_lock([])
    ), patch.object(prepare_hardware, "prepare_image", return_value=image_for("custom")) as build:
        assert prepare_hardware.main([
            "--model", "demo/custom", "--revision", "main", "--task", "fill-mask",
            "--gpus", "off", "--output-dir", str(output)
        ]) == 0
    assert workflow.resolve.call_count == 2
    assert workflow.resolve.call_args_list[0].args[0].revision == "main"
    assert workflow.resolve.call_args_list[1].args[0].revision == "d" * 40
    build.assert_called_once_with(task, str(prepare_hardware.ROOT))
    matrix = json.loads((output / "prepared-matrix.json").read_text())
    assert matrix["cases"] == [{
        "name": "custom", "model": "demo/custom", "revision": "d" * 40,
        "task": "fill-mask", "input_scales": "64"
    }]
    # The exact output of preparation satisfies the validator's pinned matrix contract.
    assert len(check_hardware.load_hardware_matrix(
        output / "prepared-matrix.json", RunConfig(gpus="off"), tmp_path / "measurements"
    )) == 1


def test_failing_second_case_does_not_publish_partial_selection(tmp_path):
    output = tmp_path / "preparation"
    output.mkdir()
    selected = cases(tmp_path)
    workflow = Mock()
    workflow.resolve.side_effect = [task_for(config.model, config.revision) for _, config in selected]
    with patch.object(prepare_hardware, "PreparationWorkflow", return_value=workflow), patch.object(
        prepare_hardware, "MeasurementLock", side_effect=lambda: observed_lock([])
    ), patch.object(prepare_hardware, "prepare_image",
                    side_effect=[image_for("first"), RuntimeError("missing model")]):
        assert not prepare_hardware.prepare_cases(selected, output)
    report = json.loads((output / "preparation.json").read_text())
    assert report["successful"] is False
    assert [c["status"] for c in report["cases"]] == ["passed", "failed"]
    assert "missing model" in report["cases"][1]["error"]
    assert not (output / "prepared-matrix.json").exists()


def test_prepare_rejects_wrong_resolved_revision(tmp_path):
    output = tmp_path / "preparation"
    output.mkdir()
    workflow = Mock()
    workflow.resolve.return_value = task_for("demo/first", "f" * 40)
    with patch.object(prepare_hardware, "PreparationWorkflow", return_value=workflow), patch.object(
        prepare_hardware, "MeasurementLock", side_effect=lambda: observed_lock([])
    ), patch.object(prepare_hardware, "prepare_image", return_value=image_for("first")):
        assert not prepare_hardware.prepare_cases(cases(tmp_path)[:1], output)
    assert "differs" in json.loads((output / "preparation.json").read_text())["cases"][0]["error"]
    assert not (output / "prepared-matrix.json").exists()


def test_interrupted_preparation_remains_cancelled(tmp_path):
    output = tmp_path / "preparation"
    output.mkdir()
    workflow = Mock()
    workflow.resolve.return_value = task_for("demo/first", "a" * 40)
    with patch.object(prepare_hardware, "PreparationWorkflow", return_value=workflow), patch.object(
        prepare_hardware, "MeasurementLock", side_effect=lambda: observed_lock([])
    ), patch.object(prepare_hardware, "prepare_image", side_effect=KeyboardInterrupt), pytest.raises(
        KeyboardInterrupt
    ):
        prepare_hardware.prepare_cases(cases(tmp_path), output)
    report = json.loads((output / "preparation.json").read_text())
    assert [c["status"] for c in report["cases"]] == ["cancelled", "not_run"]
    assert not (output / "prepared-matrix.json").exists()


@pytest.mark.parametrize("variable", ["ACPROF_REQUIRE_PREPARED_IMAGE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_prepare_rejects_forced_offline_mode_before_any_writes(tmp_path, monkeypatch, variable):
    output = tmp_path / "prepare"
    monkeypatch.setenv(variable, "1")
    with pytest.raises(SystemExit) as failure:
        prepare_hardware.main(["--model", "demo/custom", "--output-dir", str(output)])
    assert failure.value.code == 2
    assert not output.exists()


def test_busy_measurement_lock_does_not_touch_model_or_hash_sources(tmp_path):
    output = tmp_path / "preparation"
    output.mkdir()
    with patch.object(prepare_hardware, "MeasurementLock", side_effect=RuntimeError("lock busy")), patch.object(
        prepare_hardware, "host_identity"
    ) as identity, patch.object(prepare_hardware, "prepare_image") as build:
        assert not prepare_hardware.prepare_cases(cases(tmp_path)[:1], output)
    identity.assert_not_called()
    build.assert_not_called()
    assert not (output / "preparation.json").exists()
    assert not (output / "prepared-matrix.json").exists()


def test_hardware_workflow_prepares_before_offline_measurement():
    workflow = yaml.safe_load((ROOT / ".github/workflows/hardware.yml").read_text())
    job = workflow["jobs"]["profile"]
    stages = job["steps"]
    preparation = next(i for i, step in enumerate(stages)
                       if "scripts/prepare_hardware.py" in step.get("run", ""))
    measurement = next(i for i, step in enumerate(stages)
                       if "scripts/check_hardware.py" in step.get("run", ""))
    assert preparation < measurement
    assert stages[preparation]["env"]["ACPROF_REQUIRE_PREPARED_IMAGE"] == "0"
    assert stages[preparation]["env"]["HF_HUB_OFFLINE"] == "0"
    assert stages[preparation]["env"]["ACPROF_MAX_DOWNLOAD"] == ""
    assert stages[preparation]["env"]["ACPROF_MODEL_MAX_DOWNLOAD"] == "4GB"
    assert "prepared-matrix.json" in stages[measurement]["run"]
    assert "examples/hardware-validation.json" not in stages[measurement]["run"]
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "acprof"]
    assert "workflow_dispatch" in job["if"]
    assert "refs/heads/main" in job["if"]
