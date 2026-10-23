"""Static discovery, loader capability and interface validation stay separate."""
import subprocess
import sys


def test_resolution_layers_are_importable_without_model_runtimes():
    script = """
import sys
from acprof.model_support import transformers_capabilities, supports_transformers_task
from acprof.model_candidates import discover_model_candidates
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


def test_retired_cross_layer_aliases_are_not_reexported():
    import acprof.model_candidates as candidates
    import acprof.model_resolution as resolution
    import acprof.model_support as support

    assert not hasattr(resolution, "discover_model_candidates")
    assert not hasattr(resolution, "transformers_capabilities")
    assert not hasattr(resolution, "audio_text_loader")
    assert not hasattr(candidates, "resolve_model_interface")
    assert not hasattr(support, "discover_model_candidates")
