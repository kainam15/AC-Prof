"""Service entrypoints must work without the checkout or editable installation."""

import subprocess
import sys
from pathlib import Path

import pytest

from acprof.source_identity import stage_service_context

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("module", [
    "acprof.input_budget",
    "acprof.config",
    "acprof.container.execution",
    "acprof.container.load_policy",
    "acprof.container.runtime_manifest",
    "acprof.container.compute_profile_runner",
    "acprof.container.handlers.nlp",
])
def test_service_entrypoint_imports_without_checkout(tmp_path, module):
    context = tmp_path / "service"
    stage_service_context(PROJECT_ROOT, context)
    # -I excludes cwd/PYTHONPATH; -S also excludes editable-install .pth hooks.
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", """
import importlib
import sys
from pathlib import Path

context = Path(sys.argv[1])
sys.path.insert(0, str(context))
importlib.import_module(sys.argv[2])
for name, module in sys.modules.copy().items():
    if name == "acprof" or name.startswith("acprof."):
        assert Path(module.__file__).is_relative_to(context), name
""", str(context), module],
        cwd=tmp_path, capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
