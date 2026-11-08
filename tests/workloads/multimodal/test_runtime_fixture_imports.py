"""Runtime fixture collection must not pull host-only dependencies into containers."""
import subprocess
import sys
from pathlib import Path


def test_custom_multimodal_runtime_collects_without_host_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib
import importlib.abc
import sys

class RejectHostImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "acprof.host" or fullname.startswith("acprof.host."):
            raise ImportError("runtime fixture imported host dependency: " + fullname)
        return None

sys.meta_path.insert(0, RejectHostImports())
importlib.import_module("tests.workloads.multimodal.test_custom_multimodal_runtime")
"""],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True,
        timeout=15, check=False,
    )
    assert result.returncode == 0, result.stderr
