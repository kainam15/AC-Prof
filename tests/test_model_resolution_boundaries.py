"""Static discovery, loader capability and interface validation stay separate."""
import subprocess
import sys


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
