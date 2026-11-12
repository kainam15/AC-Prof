"""Inspection and collection share workload selection without running a model."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.cli.inspect import main
from acprof.host.detect import TaskInfo
from acprof.host.preparation.task_selection import select_tasks
from acprof.models.candidates import apply_model_candidate, discover_model_candidates


@pytest.fixture
def multiple_profiles():
    config = {
        "model_type": "gpt2", "architectures": ["GPT2LMHeadModel"],
        "task_specific_params": {
            "summarization": {"num_beams": 4},
            "translation_en_to_de": {"num_beams": 1},
        },
    }
    task = TaskInfo(
        "fixture/multiple-profiles", "text-generation", "nlp", "transformers_model",
        "transformers", "a" * 40, "hub_api", model_config=config,
        repository_files=("config.json", "model.safetensors"),
        repository_metadata={"config.json": config},
        hub_metadata={"pipeline_tag": "text-generation"},
    )
    task.model_resolution = apply_model_candidate(task, discover_model_candidates(
        task, override_tag="text-generation"))
    return task


@pytest.fixture
def inspection(monkeypatch, multiple_profiles):
    monkeypatch.setattr("acprof.host.detect.detect_task", lambda *args, **kwargs: copy.deepcopy(multiple_profiles))
    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", Mock())
    support = Mock()
    probe = Mock()
    monkeypatch.setattr("acprof.host.preparation.task_support.require_task_support", support)
    monkeypatch.setattr("acprof.host.preparation.interface_probe.probe_interface", probe)
    return support, probe


def report(directory):
    return json.loads((directory / "model_resolution.json").read_text())


def test_explicit_interface_keeps_real_profile_choice(multiple_profiles, inspection, tmp_path, capsys):
    support, probe = inspection
    code = main([multiple_profiles.model_id, "--task", "text-generation", "--output-dir", str(tmp_path)])
    assert code == 2
    saved = report(tmp_path)
    assert saved["status"] == "selection_required"
    assert saved["failure"]["reason_code"] == "task_selection_required"
    assert not saved.get("selected_tasks")
    assert "--tasks" in capsys.readouterr().err
    support.assert_not_called()
    probe.assert_not_called()


def test_inspect_selects_the_same_explicit_workload_as_collection(
    multiple_profiles, inspection, tmp_path,
):
    support, probe = inspection
    options = SimpleNamespace(task="text-generation", tasks="summarization", backend="", model_spec="")
    expected = select_tasks(copy.deepcopy(multiple_profiles), options, SimpleNamespace(interactive=False))
    code = main([
        multiple_profiles.model_id, "--task", "text-generation", "--tasks", "summarization",
        "--output-dir", str(tmp_path),
    ])
    assert code == 0
    saved = report(tmp_path)
    assert saved["status"] == "candidate"
    assert saved["selected_tasks"] == expected.selected_tasks
    assert saved["selected_task"] == saved["selected_tasks"][0]
    assert saved["selected_task"]["config"]["generation"]["num_beams"] == 4
    assert saved["selected_task"]["execution_profile"]["semantic_task_ids"] == ["summarization"]
    assert multiple_profiles.selected_tasks == []
    support.assert_called_once()
    assert support.call_args.args[0].selected_tasks == expected.selected_tasks
    probe.assert_not_called()


@pytest.mark.parametrize("selection", ["unknown-profile", "summarization,summarization"])
def test_invalid_profile_choice_returns_a_diagnostic(selection, multiple_profiles, inspection, tmp_path, capsys):
    support, probe = inspection
    code = main([multiple_profiles.model_id, "--tasks", selection, "--output-dir", str(tmp_path)])
    assert code == 2
    saved = report(tmp_path)
    assert saved["status"] == "selection_required"
    assert saved["failure"]["reason_code"] == "task_selection_required"
    assert "invalid or duplicate task selection" in capsys.readouterr().err
    support.assert_not_called()
    probe.assert_not_called()


def test_all_profiles_are_checked_and_exported(multiple_profiles, inspection, tmp_path):
    support, probe = inspection
    code = main([multiple_profiles.model_id, "--tasks", "all", "--output-dir", str(tmp_path)])
    assert code == 0
    saved = report(tmp_path)
    assert len(saved["selected_tasks"]) == 2
    assert set(saved["task_resolutions"]) == {item["task_id"] for item in saved["selected_tasks"]}
    assert support.call_count == 2
    assert {call.args[0].selected_tasks[0]["config"]["generation"]["num_beams"]
            for call in support.call_args_list} == {1, 4}
    assert all(item["runtime_validation"]["status"] == "not_run"
               for item in saved["task_resolutions"].values())
    probe.assert_not_called()


def test_unknown_secondary_profile_is_not_hidden_by_supported_first_profile(
    multiple_profiles, inspection, tmp_path,
):
    support, _ = inspection
    checked = []

    def check(task, **kwargs):
        checked.append(task.selected_tasks[0]["task_id"])
        task.model_resolution["compatibility"] = {
            "status": "supported" if len(checked) == 1 else "unknown",
            "source": "locked_transformers_mapping", "evidence_level": "static",
            "runtime_profile": f"fixture-profile-{len(checked)}",
            "transformers_version": f"fixture-version-{len(checked)}",
            "checkpoint_verified": False,
        }

    support.side_effect = check
    assert main([multiple_profiles.model_id, "--tasks", "all", "--output-dir", str(tmp_path)]) == 0
    saved = report(tmp_path)
    assert saved["compatibility"] == {
        "status": "unknown", "source": "selected_task_profiles",
        "evidence_level": "static", "checkpoint_verified": False,
    }
    assert saved["task_resolutions"][checked[0]]["compatibility"]["status"] == "supported"
    assert saved["task_resolutions"][checked[1]]["compatibility"]["status"] == "unknown"
    assert saved["task_resolutions"][checked[1]]["compatibility"]["runtime_profile"] == "fixture-profile-2"


def test_a_failed_secondary_profile_blocks_the_combined_result(multiple_profiles, inspection, tmp_path):
    from acprof.failures import Failure
    from acprof.host.preparation.task_support import TaskSupportError

    support, probe = inspection
    def check(task, **kwargs):
        if task.selected_tasks[0]["config"]["generation"]["num_beams"] == 4:
            raise TaskSupportError(Failure("preflight", "runtime_unavailable", "fixture beam runtime unavailable"))
    support.side_effect = check
    code = main([multiple_profiles.model_id, "--tasks", "all", "--output-dir", str(tmp_path)])
    assert code == 2
    saved = report(tmp_path)
    assert saved["status"] == "blocked"
    assert saved["failure"]["reason_code"] == "runtime_unavailable"
    assert support.call_count == 2
    failed = [item for item in saved["task_resolutions"].values() if item.get("failure")]
    assert len(failed) == 1
    assert failed[0]["selected_task"]["config"]["generation"]["num_beams"] == 4
    probe.assert_not_called()


def test_interface_probe_requires_one_profile(multiple_profiles, inspection, tmp_path, capsys):
    _, probe = inspection
    code = main([
        multiple_profiles.model_id, "--tasks", "all", "--probe-interface", "--output-dir", str(tmp_path),
    ])
    assert code == 2
    assert "--tasks" in capsys.readouterr().err
    probe.assert_not_called()


def test_explicit_choice_retries_a_previous_selection_failure(multiple_profiles, inspection):
    from acprof.host.preparation.task_inspection import inspect_selected_tasks

    support, _ = inspection
    options = SimpleNamespace(task="text-generation", tasks="", backend="", model_spec="")
    waiting, error = inspect_selected_tasks(multiple_profiles, options)
    assert error.failure.reason_code == "task_selection_required"
    options.tasks = "summarization"
    selected, error = inspect_selected_tasks(waiting, options)
    assert error is None
    assert selected.model_resolution["status"] == "candidate"
    assert "failure" not in selected.model_resolution
    assert selected.selected_tasks[0]["config"]["generation"]["num_beams"] == 4
    support.assert_called_once()


def test_model_lookup_error_retains_exit_one(monkeypatch, capsys):
    from acprof.host.preparation.model_errors import ModelLookupError

    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", Mock())
    monkeypatch.setattr("acprof.host.detect.detect_task", Mock(side_effect=ModelLookupError(
        "fixture/missing", "network_error", detail="fixture network unavailable",
    )))
    assert main(["fixture/missing"]) == 1
    assert "[model-lookup][ERROR]" in capsys.readouterr().err


def test_metadata_error_is_reported_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", Mock())
    monkeypatch.setattr("acprof.host.detect.detect_task", Mock(side_effect=ValueError("config.json:/limit is non-finite")))
    assert main(["fixture/invalid"]) == 2
    output = capsys.readouterr()
    assert "config.json:/limit is non-finite" in output.err
    assert "Traceback" not in output.err
