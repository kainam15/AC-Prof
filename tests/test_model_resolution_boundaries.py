"""Static discovery, loader capability and interface validation stay separate."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize(("task", "backend"), [(None, None), ("fill-mask", None), (None, "transformers_pipeline")])
def test_absent_cli_overrides_have_one_resolution_identity(task, backend):
    from acprof.host.detect import TaskInfo
    from acprof.model_candidates import discover_model_candidates

    info = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline", "transformers",
                    "a" * 40, "hub_api", model_config={"model_type": "bert"},
                    hub_metadata={"pipeline_tag": "fill-mask"})
    omitted = discover_model_candidates(info, override_tag=task, override_backend=backend)
    empty = discover_model_candidates(info, override_tag=task or "", override_backend=backend or "")
    assert empty["selection"] == omitted["selection"]
    assert empty["provenance"] == omitted["provenance"]
    explicit = discover_model_candidates(info, override_tag="fill-mask", override_backend="transformers_pipeline")
    assert explicit["provenance"]["identity_sha256"] != omitted["provenance"]["identity_sha256"]


def test_resolution_layers_are_importable_without_model_runtimes():
    script = """
import sys
from acprof.model_support import transformers_capabilities, supports_transformers_task
from acprof.model_candidates import apply_model_candidate, discover_model_candidates
from acprof.model_resolution import resolve_model_interface, require_resolved_candidate
assert callable(transformers_capabilities)
assert callable(supports_transformers_task)
assert callable(discover_model_candidates)
assert callable(resolve_model_interface)
assert callable(require_resolved_candidate)
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], text=True,
                            capture_output=True, check=False, timeout=25)
    assert result.returncode == 0, result.stderr


def test_discovery_does_not_mutate_task_until_explicit_application():
    from acprof.host.detect import TaskInfo
    from acprof.model_candidates import apply_model_candidate, discover_model_candidates

    task = TaskInfo('fixture/model', 'unknown', 'unknown', 'transformers_pipeline',
                    'transformers', 'a' * 40, 'hub_api',
                    hub_metadata={'transformers_info': {
                        'pipeline_tag': 'text-generation', 'auto_model': 'AutoModelForCausalLM'}})
    original = (task.pipeline_tag, task.task_family, task.runtime_backend)
    resolution = discover_model_candidates(task)
    assert (task.pipeline_tag, task.task_family, task.runtime_backend) == original
    assert resolution['selection']['task'] == 'text-generation'
    apply_model_candidate(task, resolution)
    assert (task.pipeline_tag, task.task_family, task.runtime_backend) == (
        'text-generation', 'nlp', resolution['selection']['backend'])


def test_retired_cross_layer_aliases_are_not_reexported():
    import acprof.model_candidates as candidates
    import acprof.model_resolution as resolution
    import acprof.model_support as support

    assert not hasattr(resolution, "discover_model_candidates")
    assert not hasattr(resolution, "transformers_capabilities")
    assert not hasattr(resolution, "audio_text_loader")
    assert not hasattr(candidates, "resolve_model_interface")
    assert not hasattr(support, "discover_model_candidates")
    assert not hasattr(candidates, "discover_declared_tasks")


def test_resume_checks_interface_identity_before_restoring_runtime():
    from dataclasses import asdict
    from types import SimpleNamespace
    from unittest.mock import patch

    from acprof.host.collection_workflow import PreparationWorkflow
    from acprof.host.detect import TaskInfo
    from acprof.host.run_state import RunState
    from acprof.host.task_selection import select_tasks
    from acprof.model_candidates import apply_model_candidate, discover_model_candidates

    task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline", "transformers",
                    "a" * 40, "hub_api", model_config={"architectures": ["BertForMaskedLM"]})
    task.model_resolution = apply_model_candidate(task, discover_model_candidates(task))
    task = select_tasks(task, SimpleNamespace(task="", tasks="", backend="", model_spec=""), PreparationWorkflow())
    task.model_config["architectures"] = ["RobertaForMaskedLM"]
    state = object.__new__(RunState)
    state.data = {"runtime": {"task": asdict(task)}}
    with patch("acprof.host.runtimes.service.execution_runtime") as runtime, pytest.raises(ValueError, match="identity"):
        state.restore_runtime()
    runtime.assert_not_called()
